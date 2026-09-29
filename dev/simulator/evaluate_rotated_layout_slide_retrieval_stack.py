"""评测真实文本任务检索、几何 Demo 选择与冻结 predictor 的误差传播。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Any

import numpy as np
import torch

from src.components.retriever.text_retriever import (
    TextCandidate,
    TextRetriever,
    TextRetrieverConfig,
)
from dev.simulator.evaluate_rotated_layout_slide_demo_policy_confirmation import (
    _cross_transport,
    _load_data,
    _load_policy,
)
from dev.simulator.evaluate_rotated_layout_slide_transport import (
    OPERATIONS,
    SPLITS,
    _metrics,
    _strip_rows,
)
from dev.simulator.train_rotated_layout_slide_demo_policy import _predict


OPERATION_IDS = {name: identifier for identifier, name in OPERATIONS.items()}
ALLOWED_CATEGORIES = {"canonical", "paraphrase", "compositional"}


@dataclass(frozen=True)
class LanguagePair:
    """同一物体集合下语义相反的一对 query。"""

    name: str
    category: str
    toward: str
    away: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "LanguagePair":
        return cls(**value)

    def __post_init__(self) -> None:
        strings = (self.name, self.category, self.toward, self.away)
        if any(not isinstance(value, str) or not value.strip() for value in strings):
            raise ValueError("language pair 字段不能为空")
        if self.category not in ALLOWED_CATEGORIES:
            raise ValueError(f"未知语言类别：{self.category}")
        if self.toward == self.away:
            raise ValueError("toward/away query 不能相同")


@dataclass(frozen=True)
class RetrievalStackConfig:
    """冻结的文本—几何—动作端到端压力测试协议。"""

    schema_version: str
    seed: int
    bootstrap_resamples: int
    selective_margin_threshold: float
    minimum_canonical_task_accuracy: float
    minimum_paraphrase_task_accuracy: float
    minimum_end_to_end_direction_accuracy: float
    maximum_policy_to_oracle_mse_ratio: float
    task_prototypes: dict[str, str]
    language_pairs: tuple[LanguagePair, ...]

    @classmethod
    def from_json(cls, path: Path) -> "RetrievalStackConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["language_pairs"] = tuple(
            LanguagePair.from_dict(value) for value in raw["language_pairs"]
        )
        return cls(**raw)

    def __post_init__(self) -> None:
        if self.schema_version != "rotated-layout-slide-retrieval-stack-v1":
            raise ValueError("未知 retrieval stack schema")
        if set(self.task_prototypes) != set(OPERATION_IDS):
            raise ValueError("task_prototypes 必须且只能包含 toward/away")
        if any(
            not isinstance(text, str) or not text.strip()
            for text in self.task_prototypes.values()
        ):
            raise ValueError("task prototype 不能为空")
        if self.seed < 0 or self.bootstrap_resamples <= 0:
            raise ValueError("seed/bootstrap_resamples 非法")
        if self.selective_margin_threshold < 0:
            raise ValueError("selective margin threshold 不能为负")
        probabilities = (
            self.minimum_canonical_task_accuracy,
            self.minimum_paraphrase_task_accuracy,
            self.minimum_end_to_end_direction_accuracy,
        )
        if any(not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("accuracy threshold 必须位于 [0,1]")
        if self.maximum_policy_to_oracle_mse_ratio <= 0:
            raise ValueError("MSE ratio threshold 必须为正")
        if not self.language_pairs:
            raise ValueError("至少需要一组 language pair")
        names = [item.name for item in self.language_pairs]
        if len(names) != len(set(names)):
            raise ValueError("language pair name 必须唯一")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _bootstrap_mean(
    values: np.ndarray,
    *,
    seed: int,
    resamples: int,
) -> dict[str, float | int]:
    """对独立 language-pair 单元进行非参数 bootstrap。"""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("bootstrap 输入必须是一维有限非空数组")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(resamples, len(values)))
    samples = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "units": len(values),
    }


def _score_record(result: Any) -> dict[str, Any]:
    hits = result.hits
    if len(hits) != len(OPERATION_IDS):
        raise ValueError("文本任务索引必须返回两个 task bucket")
    return {
        "predicted_operation": hits[0].candidate.candidate_id,
        "margin": float(hits[0].score - hits[1].score),
        "query_relation_effect": result.query.relation_effect,
        "query_relation_effect_confidence": (
            result.query.relation_effect_confidence
        ),
        "ranking": [
            {
                "operation": hit.candidate.candidate_id,
                "score": hit.score,
                "raw_text_score": hit.raw_text_score,
                "object_score": hit.object_score,
                "relation_score": hit.relation_score,
                "goal_operation": hit.parsed.goal_operation,
                "goal_operation_confidence": hit.parsed.goal_operation_confidence,
                "relation_effect": hit.parsed.relation_effect,
                "relation_effect_confidence": (
                    hit.parsed.relation_effect_confidence
                ),
            }
            for hit in hits
        ],
    }


def _text_queries(
    retriever: TextRetriever,
    config: RetrievalStackConfig,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]], list[float]]:
    records: list[dict[str, Any]] = []
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    latencies = []
    for pair in config.language_pairs:
        for true_operation in OPERATION_IDS:
            text = getattr(pair, true_operation)
            started = time.perf_counter_ns()
            result = retriever.retrieve(text, top_k=len(OPERATION_IDS))
            latencies.append((time.perf_counter_ns() - started) / 1e6)
            score = _score_record(result)
            row = {
                "language_pair": pair.name,
                "category": pair.category,
                "true_operation": true_operation,
                "query": text,
                **score,
            }
            row["correct"] = row["predicted_operation"] == true_operation
            row["accepted"] = row["margin"] >= config.selective_margin_threshold
            records.append(row)
            lookup[(pair.name, true_operation)] = row
    return records, lookup, latencies


def _task_metrics(
    records: list[dict[str, Any]],
    config: RetrievalStackConfig,
) -> dict[str, Any]:
    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        correctness = np.asarray([row["correct"] for row in rows], dtype=float)
        accepted = [row for row in rows if row["accepted"]]
        return {
            "queries": len(rows),
            "accuracy": float(correctness.mean()),
            "margin_mean": float(np.mean([row["margin"] for row in rows])),
            "margin_min": float(np.min([row["margin"] for row in rows])),
            "selective_coverage": len(accepted) / len(rows),
            "selective_accuracy": (
                float(np.mean([row["correct"] for row in accepted]))
                if accepted
                else None
            ),
        }

    output = {"overall": summarize(records)}
    for category in sorted(ALLOWED_CATEGORIES):
        rows = [row for row in records if row["category"] == category]
        if rows:
            output[category] = summarize(rows)

    pair_accuracy = []
    for pair in config.language_pairs:
        rows = [row for row in records if row["language_pair"] == pair.name]
        pair_accuracy.append(float(np.mean([row["correct"] for row in rows])))
    output["language_pair_bootstrap"] = _bootstrap_mean(
        np.asarray(pair_accuracy),
        seed=config.seed,
        resamples=config.bootstrap_resamples,
    )
    return output


def _nearest_demo(
    query_state: np.ndarray,
    bank_data: dict[str, np.ndarray],
    bank_indices: np.ndarray,
    operation_id: int,
    state_std: np.ndarray,
) -> tuple[int, float]:
    candidates = bank_indices[
        bank_data["operation_id"][bank_indices] == operation_id
    ]
    if not len(candidates):
        raise ValueError(f"operation={operation_id} 没有候选 Demo")
    normalized = (bank_data["state"][candidates] - query_state) / state_std
    distances = np.mean(normalized.astype(np.float64) ** 2, axis=1)
    nearest = int(np.argmin(distances))
    return int(candidates[nearest]), float(distances[nearest])


def _append_context(
    storage: dict[str, list[np.ndarray | int | str | bool | float]],
    *,
    query_data: dict[str, np.ndarray],
    bank_data: dict[str, np.ndarray],
    query_index: int,
    demo_index: int,
    language_pair: LanguagePair,
    text_correct: bool,
    accepted: bool,
    geometry_distance: float,
) -> None:
    storage["query_state"].append(query_data["state"][query_index])
    storage["demo_state"].append(bank_data["state"][demo_index])
    storage["anchor"].append(
        _cross_transport(query_data, bank_data, query_index, demo_index)
    )
    storage["target"].append(query_data["action"][query_index])
    storage["query_row"].append(query_index)
    storage["demo_row"].append(demo_index)
    storage["query_pair_id"].append(int(query_data["pair_id"][query_index]))
    storage["language_pair"].append(language_pair.name)
    storage["category"].append(language_pair.category)
    storage["text_correct"].append(text_correct)
    storage["accepted"].append(accepted)
    storage["geometry_distance"].append(geometry_distance)


def _empty_context() -> dict[str, list[Any]]:
    return {
        name: []
        for name in (
            "query_state",
            "demo_state",
            "anchor",
            "target",
            "query_row",
            "demo_row",
            "query_pair_id",
            "language_pair",
            "category",
            "text_correct",
            "accepted",
            "geometry_distance",
        )
    }


def _stack_context(storage: dict[str, list[Any]]) -> dict[str, np.ndarray]:
    array_fields = {"query_state", "demo_state", "anchor", "target"}
    integer_fields = {"query_row", "demo_row", "query_pair_id"}
    boolean_fields = {"text_correct", "accepted"}
    output = {}
    for name, values in storage.items():
        if name in array_fields:
            output[name] = np.stack(values)
        elif name in integer_fields:
            output[name] = np.asarray(values, dtype=np.int64)
        elif name in boolean_fields:
            output[name] = np.asarray(values, dtype=bool)
        elif name == "geometry_distance":
            output[name] = np.asarray(values, dtype=np.float64)
        else:
            output[name] = np.asarray(values)
    return output


def _build_contexts(
    *,
    query_data: dict[str, np.ndarray],
    bank_data: dict[str, np.ndarray],
    bank_indices: np.ndarray,
    state_std: np.ndarray,
    config: RetrievalStackConfig,
    text_lookup: dict[tuple[str, str], dict[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    selected = _empty_context()
    oracle = _empty_context()
    for language_pair in config.language_pairs:
        for query_index in range(len(query_data["pair_id"])):
            true_id = int(query_data["operation_id"][query_index])
            true_operation = OPERATIONS[true_id]
            text = text_lookup[(language_pair.name, true_operation)]
            predicted_id = OPERATION_IDS[text["predicted_operation"]]
            selected_demo, selected_distance = _nearest_demo(
                query_data["state"][query_index],
                bank_data,
                bank_indices,
                predicted_id,
                state_std,
            )
            oracle_demo, oracle_distance = _nearest_demo(
                query_data["state"][query_index],
                bank_data,
                bank_indices,
                true_id,
                state_std,
            )
            common = {
                "query_data": query_data,
                "bank_data": bank_data,
                "query_index": query_index,
                "language_pair": language_pair,
                "text_correct": bool(text["correct"]),
                "accepted": bool(text["accepted"]),
            }
            _append_context(
                selected,
                demo_index=selected_demo,
                geometry_distance=selected_distance,
                **common,
            )
            _append_context(
                oracle,
                demo_index=oracle_demo,
                geometry_distance=oracle_distance,
                **common,
            )
    return _stack_context(selected), _stack_context(oracle)


def _subset_metrics(
    predictions: dict[str, np.ndarray],
    target: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any] | None:
    if not bool(mask.any()):
        return None
    return {
        name: _strip_rows(_metrics(value[mask], target[mask]))
        for name, value in predictions.items()
    }


def _end_to_end_metrics(
    *,
    selected: dict[str, np.ndarray],
    oracle: dict[str, np.ndarray],
    selected_policy: np.ndarray,
    oracle_policy: np.ndarray,
    no_demo: np.ndarray,
    config: RetrievalStackConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    target = selected["target"]
    predictions = {
        "text_selected_transport": selected["anchor"],
        "text_selected_policy": selected_policy,
        "oracle_nearest_transport": oracle["anchor"],
        "oracle_nearest_policy": oracle_policy,
        "no_demo_policy": no_demo,
    }
    masks = {
        "overall": np.ones(len(target), dtype=bool),
        "text_correct": selected["text_correct"],
        "text_wrong": ~selected["text_correct"],
        "selective_accepted": selected["accepted"],
    }
    for category in sorted(ALLOWED_CATEGORIES):
        masks[category] = selected["category"] == category
    metrics = {
        name: _subset_metrics(predictions, target, mask)
        for name, mask in masks.items()
    }

    selected_error = np.mean(
        (selected_policy.astype(np.float64) - target.astype(np.float64)) ** 2,
        axis=(1, 2),
    )
    oracle_error = np.mean(
        (oracle_policy.astype(np.float64) - target.astype(np.float64)) ** 2,
        axis=(1, 2),
    )
    difference = selected_error - oracle_error
    per_language_pair = np.asarray(
        [
            difference[selected["language_pair"] == pair.name].mean()
            for pair in config.language_pairs
        ]
    )
    comparison = _bootstrap_mean(
        per_language_pair,
        seed=config.seed + 1,
        resamples=config.bootstrap_resamples,
    )
    comparison["definition"] = "text-selected policy MSE minus oracle policy MSE"
    diagnostics = {
        "policy_minus_oracle_language_pair_bootstrap": comparison,
        "selected_geometry_distance_mean": float(
            selected["geometry_distance"].mean()
        ),
        "oracle_geometry_distance_mean": float(oracle["geometry_distance"].mean()),
        "selected_demo_equals_oracle_rate": float(
            np.mean(selected["demo_row"] == oracle["demo_row"])
        ),
        "selected_demo_equals_oracle_given_correct_text": float(
            np.mean(
                selected["demo_row"][selected["text_correct"]]
                == oracle["demo_row"][selected["text_correct"]]
            )
        ),
    }
    return metrics, diagnostics


def run(
    *,
    project_root: Path,
    config_path: Path,
    bank_root: Path,
    query_root: Path,
    training_root: Path,
    output_path: Path,
    gliner_model_dir: Path,
    minilm_model_dir: Path,
    text_config_path: Path | None,
    text_config: TextRetrieverConfig | None,
    config: RetrievalStackConfig,
) -> None:
    if output_path.exists():
        raise FileExistsError(f"输出已存在，拒绝覆盖：{output_path}")
    torch.set_num_threads(1)
    bank_data, bank_report = _load_data(bank_root)
    query_data, query_report = _load_data(query_root)
    training_report_path = training_root / "report.json"
    checkpoint_path = training_root / "checkpoint.pt"
    training_report = json.loads(training_report_path.read_text(encoding="utf-8"))
    model, checkpoint = _load_policy(checkpoint_path, training_report)
    if _sha256(bank_root / "branch_samples.npz") != checkpoint["data_sha256"]:
        raise ValueError("checkpoint 与 Demo bank 数据 hash 不匹配")

    bank_indices = np.where(bank_data["split_id"] == SPLITS["train"])[0]
    if len(set(bank_data["operation_id"][bank_indices].tolist())) != 2:
        raise ValueError("train Demo bank 必须同时包含两个 operation")
    state_std = model.state_std.detach().cpu().numpy()

    model_load_started = time.perf_counter()
    text_retriever = TextRetriever.from_local_models(
        gliner_model_dir=gliner_model_dir,
        minilm_model_dir=minilm_model_dir,
        device="cpu",
        config=text_config,
    )
    text_model_load_seconds = time.perf_counter() - model_load_started
    index_started = time.perf_counter()
    text_retriever.build_index(
        [
            TextCandidate(operation, config.task_prototypes[operation])
            for operation in OPERATION_IDS
        ]
    )
    text_index_ms = (time.perf_counter() - index_started) * 1000.0
    text_records, text_lookup, text_latencies = _text_queries(
        text_retriever, config
    )
    task_metrics = _task_metrics(text_records, config)

    geometry_started = time.perf_counter()
    selected, oracle = _build_contexts(
        query_data=query_data,
        bank_data=bank_data,
        bank_indices=bank_indices,
        state_std=state_std,
        config=config,
        text_lookup=text_lookup,
    )
    geometry_ms_per_context = (
        (time.perf_counter() - geometry_started) * 1000.0 / len(selected["target"])
    )
    batch_size = int(checkpoint["train_config"]["batch_size"])
    selected_policy = _predict(model, selected, batch_size=batch_size)
    oracle_policy = _predict(model, oracle, batch_size=batch_size)
    no_demo = _predict(
        model,
        selected,
        batch_size=batch_size,
        demo_mask=0.0,
    )
    action_metrics, diagnostics = _end_to_end_metrics(
        selected=selected,
        oracle=oracle,
        selected_policy=selected_policy,
        oracle_policy=oracle_policy,
        no_demo=no_demo,
        config=config,
    )
    overall = action_metrics["overall"]
    canonical_accuracy = task_metrics["canonical"]["accuracy"]
    noncanonical = [
        row for row in text_records if row["category"] != "canonical"
    ]
    noncanonical_accuracy = float(
        np.mean([row["correct"] for row in noncanonical])
    )
    policy_mse = overall["text_selected_policy"]["action_mse"]
    oracle_mse = overall["oracle_nearest_policy"]["action_mse"]
    criteria = {
        "R1_canonical_task_accuracy": (
            canonical_accuracy >= config.minimum_canonical_task_accuracy
        ),
        "R2_noncanonical_task_accuracy": (
            noncanonical_accuracy >= config.minimum_paraphrase_task_accuracy
        ),
        "R3_end_to_end_direction_accuracy": (
            overall["text_selected_policy"]["direction_accuracy"]
            >= config.minimum_end_to_end_direction_accuracy
        ),
        "R4_policy_to_oracle_mse_ratio": (
            policy_mse / max(oracle_mse, 1e-12)
            <= config.maximum_policy_to_oracle_mse_ratio
        ),
        "R5_no_demo_exact_zero": (
            float(np.max(np.abs(no_demo))) == 0.0
        ),
    }
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(project_root),
        "config": asdict(config),
        "config_sha256": _sha256(config_path),
        "text_retriever_config": asdict(text_retriever.config),
        "text_retriever_config_sha256": (
            _sha256(text_config_path) if text_config_path is not None else None
        ),
        "protocol": {
            "demo_bank_rows": len(bank_indices),
            "demo_bank_scene_pairs": int(
                len(np.unique(bank_data["pair_id"][bank_indices]))
            ),
            "query_rows": len(query_data["pair_id"]),
            "query_scene_pairs": int(len(np.unique(query_data["pair_id"]))),
            "language_pairs": len(config.language_pairs),
            "language_queries": len(text_records),
            "action_contexts": len(selected["target"]),
            "task_text_used_only_by_retriever": True,
            "geometry_used_only_after_task_bucket_selection": True,
            "predictor_checkpoint_frozen": True,
            "query_split_labels_ignored": True,
        },
        "artifacts": {
            "checkpoint_sha256": _sha256(checkpoint_path),
            "training_report_sha256": _sha256(training_report_path),
            "bank_data_sha256": bank_report["files"]["branch_samples.npz"],
            "query_data_sha256": query_report["files"]["branch_samples.npz"],
        },
        "text_retrieval": {
            "metrics": task_metrics,
            "noncanonical_accuracy": noncanonical_accuracy,
            "queries": text_records,
        },
        "action_prediction": {
            "metrics": action_metrics,
            "diagnostics": diagnostics,
        },
        "latency": {
            "platform": platform.platform(),
            "text_model_load_seconds": text_model_load_seconds,
            "text_index_ms": text_index_ms,
            "text_query_median_ms": float(np.median(text_latencies)),
            "text_query_p95_ms": float(np.quantile(text_latencies, 0.95)),
            "geometry_selection_ms_per_context": geometry_ms_per_context,
        },
        "model_footprint": {
            "gliner_checkpoint_bytes": _directory_bytes(gliner_model_dir),
            "minilm_checkpoint_bytes": _directory_bytes(minilm_model_dir),
            "predictor_parameters": sum(
                parameter.numel() for parameter in model.parameters()
            ),
        },
        "criteria": criteria,
        "all_criteria_passed": all(criteria.values()),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(f"{output_path.suffix}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--query-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gliner-model-dir", type=Path, required=True)
    parser.add_argument("--minilm-model-dir", type=Path, required=True)
    parser.add_argument("--text-retriever-config", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    text_config_path = (
        arguments.text_retriever_config.resolve()
        if arguments.text_retriever_config is not None
        else None
    )
    run(
        project_root=arguments.project_root.resolve(),
        config_path=arguments.config.resolve(),
        bank_root=arguments.bank_root.resolve(),
        query_root=arguments.query_root.resolve(),
        training_root=arguments.training_root.resolve(),
        output_path=arguments.output.resolve(),
        gliner_model_dir=arguments.gliner_model_dir.resolve(),
        minilm_model_dir=arguments.minilm_model_dir.resolve(),
        text_config_path=text_config_path,
        text_config=(
            TextRetrieverConfig(
                **json.loads(text_config_path.read_text(encoding="utf-8"))
            )
            if text_config_path is not None
            else None
        ),
        config=RetrievalStackConfig.from_json(arguments.config.resolve()),
    )

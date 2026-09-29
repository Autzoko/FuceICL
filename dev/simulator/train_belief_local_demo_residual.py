"""训练并门控以 raw retrieved Demo action 为硬先验的轻量 residual Predictor。"""

from __future__ import annotations

import argparse
import atexit
import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from dev.predictor.layout_equivariant_demo_policy import (
    LayoutEquivariantDemoPolicy,
    LayoutEquivariantDemoPolicyConfig,
)
from dev.simulator.evaluate_rotated_layout_slide_transport import (
    SPLITS,
    _bootstrap_difference,
    _metrics,
    _scene_means,
    _sha256,
    _strip_rows,
)
from dev.simulator.train_rotated_layout_slide_demo_policy import (
    _latency,
    _predict,
    _structural_audit,
)


@dataclass(frozen=True)
class BeliefLocalResidualConfig:
    """冻结的 raw-Demo residual 训练、验证与可靠性门槛。"""

    schema_version: str
    expected_data_sha256: str
    expected_preprocess_report_sha256: str
    seed: int
    train_neighbors: int
    hidden_dim: int
    maximum_translation_residual_m: float
    maximum_rotation_residual_rad: float
    translation_action_scale_m: float
    rotation_action_scale_rad: float
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    state_std_floor: float
    early_stopping_patience: int
    bootstrap_resamples: int
    minimum_validation_relative_improvement: float
    minimum_validation_direction_accuracy: float
    maximum_wrong_operation_direction_accuracy: float
    minimum_shuffled_relative_improvement: float
    maximum_parameters: int
    latency_warmup: int
    latency_iterations: int
    maximum_latency_p95_ms: float

    @classmethod
    def from_json(cls, path: Path) -> "BeliefLocalResidualConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def __post_init__(self) -> None:
        if self.schema_version != "belief-local-demo-residual-v1":
            raise ValueError("未知 belief-local residual schema")
        if not self.expected_data_sha256 or not self.expected_preprocess_report_sha256:
            raise ValueError("必须冻结输入数据与预处理报告 hash")
        positive = (
            self.train_neighbors,
            self.hidden_dim,
            self.maximum_translation_residual_m,
            self.maximum_rotation_residual_rad,
            self.translation_action_scale_m,
            self.rotation_action_scale_rad,
            self.epochs,
            self.batch_size,
            self.learning_rate,
            self.gradient_clip_norm,
            self.state_std_floor,
            self.early_stopping_patience,
            self.bootstrap_resamples,
            self.minimum_validation_relative_improvement,
            self.minimum_validation_direction_accuracy,
            self.minimum_shuffled_relative_improvement,
            self.maximum_parameters,
            self.latency_warmup,
            self.latency_iterations,
            self.maximum_latency_p95_ms,
        )
        if min(positive) <= 0 or self.seed < 0 or self.weight_decay < 0:
            raise ValueError("belief-local residual 配置非法")
        probabilities = (
            self.minimum_validation_relative_improvement,
            self.minimum_validation_direction_accuracy,
            self.maximum_wrong_operation_direction_accuracy,
            self.minimum_shuffled_relative_improvement,
        )
        if any(not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("概率或相对改善门槛必须位于 [0,1]")


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def _load_data(
    root: Path,
    config: BeliefLocalResidualConfig,
) -> tuple[dict[str, np.ndarray], dict[str, Any], Path, Path]:
    data_path = root / "progress_chunks.npz"
    report_path = root / "preprocess_report.json"
    if _sha256(data_path) != config.expected_data_sha256:
        raise ValueError("progress chunk data hash 与冻结配置不一致")
    if _sha256(report_path) != config.expected_preprocess_report_sha256:
        raise ValueError("preprocess report hash 与冻结配置不一致")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("files", {}).get("progress_chunks.npz") != _sha256(data_path):
        raise ValueError("preprocess report 中的数据 hash 不一致")
    if report.get("protocol", {}).get("test_split_used", True):
        raise ValueError("上游预处理错误标记为使用 test split")
    with np.load(data_path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    return data, report, data_path, report_path


def _rank_demos(
    data: dict[str, np.ndarray],
    *,
    query_indices: np.ndarray,
    bank_indices: np.ndarray,
    state_std: np.ndarray,
    same_operation: bool,
    neighbors: int,
    exclude_same_pair: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按 belief-state 距离选 Demo；top-K 中每个 scene pair 最多一个。"""
    query_rows: list[int] = []
    demo_rows: list[int] = []
    distances: list[float] = []
    for query_index in query_indices:
        query_operation = int(data["operation_id"][query_index])
        operation_matches = data["operation_id"][bank_indices] == query_operation
        eligible = bank_indices[
            operation_matches if same_operation else ~operation_matches
        ]
        if exclude_same_pair:
            eligible = eligible[
                data["pair_id"][eligible] != data["pair_id"][query_index]
            ]
        delta = (data["state"][eligible] - data["state"][query_index]) / state_std
        squared_distance = np.mean(delta.astype(np.float64) ** 2, axis=1)
        order = np.argsort(squared_distance, kind="stable")
        used_pairs: set[int] = set()
        selected = 0
        for local_index in order:
            demo_index = int(eligible[local_index])
            pair_id = int(data["pair_id"][demo_index])
            if pair_id in used_pairs:
                continue
            used_pairs.add(pair_id)
            query_rows.append(int(query_index))
            demo_rows.append(demo_index)
            distances.append(float(squared_distance[local_index]))
            selected += 1
            if selected == neighbors:
                break
        if selected != neighbors:
            raise ValueError("可用的不同 scene-pair Demo 数量不足")
    return (
        np.asarray(query_rows, dtype=np.int64),
        np.asarray(demo_rows, dtype=np.int64),
        np.asarray(distances, dtype=np.float64),
    )


def _contexts(
    data: dict[str, np.ndarray],
    query_rows: np.ndarray,
    demo_rows: np.ndarray,
) -> dict[str, np.ndarray]:
    if query_rows.shape != demo_rows.shape or query_rows.ndim != 1:
        raise ValueError("query/demo row shape 不一致")
    return {
        "query_state": data["state"][query_rows],
        "demo_state": data["state"][demo_rows],
        # 不做旋转或重定位，raw Demo action 就是硬锚点。
        "anchor": data["action"][demo_rows],
        "target": data["action"][query_rows],
        "query_row": query_rows,
        "demo_row": demo_rows,
    }


def _shuffled_demo_rows(
    data: dict[str, np.ndarray],
    query_rows: np.ndarray,
    correct_demo_rows: np.ndarray,
    bank_indices: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """在同 operation bank 中随机错配，且逐 query 排除正确 Demo。"""
    shuffled = np.empty_like(correct_demo_rows)
    rng = np.random.default_rng(seed)
    for position, (query_index, correct_index) in enumerate(
        zip(query_rows, correct_demo_rows, strict=True)
    ):
        operation = int(data["operation_id"][query_index])
        eligible = bank_indices[
            (data["operation_id"][bank_indices] == operation)
            & (bank_indices != correct_index)
        ]
        if not len(eligible):
            raise ValueError("同 operation bank 不足以构造错配")
        shuffled[position] = int(rng.choice(eligible))
    if bool(np.any(shuffled == correct_demo_rows)):
        raise ValueError("shuffled Demo 仍含原始配对")
    return shuffled


def _relative_improvement(candidate: float, baseline: float) -> float:
    return float((baseline - candidate) / max(baseline, 1e-12))


def _evaluate(
    *,
    model: LayoutEquivariantDemoPolicy,
    correct: dict[str, np.ndarray],
    shuffled: dict[str, np.ndarray],
    wrong: dict[str, np.ndarray],
    data: dict[str, np.ndarray],
    config: BeliefLocalResidualConfig,
) -> dict[str, Any]:
    predictions = {
        "raw_nearest_copy": correct["anchor"],
        "demo_residual": _predict(
            model, correct, batch_size=config.batch_size
        ),
        "same_operation_shuffled_residual": _predict(
            model, shuffled, batch_size=config.batch_size
        ),
        "wrong_operation_residual": _predict(
            model, wrong, batch_size=config.batch_size
        ),
        "no_demo": _predict(
            model,
            correct,
            batch_size=config.batch_size,
            demo_mask=0.0,
        ),
    }
    raw_metrics = {
        name: _metrics(prediction, correct["target"])
        for name, prediction in predictions.items()
    }
    pair_ids = data["pair_id"][correct["query_row"]]
    scene_mse = {
        name: _scene_means(metrics["per_row_action_mse"], pair_ids)[1]
        for name, metrics in raw_metrics.items()
    }
    residual_vs_raw = _bootstrap_difference(
        scene_mse["demo_residual"],
        scene_mse["raw_nearest_copy"],
        seed=config.seed + 101,
        resamples=config.bootstrap_resamples,
    )
    residual_vs_shuffled = _bootstrap_difference(
        scene_mse["demo_residual"],
        scene_mse["same_operation_shuffled_residual"],
        seed=config.seed + 102,
        resamples=config.bootstrap_resamples,
    )
    stripped = {name: _strip_rows(value) for name, value in raw_metrics.items()}
    return {
        "queries": len(correct["query_row"]),
        "query_scene_pairs": len(np.unique(pair_ids)),
        "metrics": stripped,
        "residual_minus_raw_scene_pair_bootstrap": residual_vs_raw,
        "residual_minus_shuffled_scene_pair_bootstrap": residual_vs_shuffled,
        "relative_mse_improvement_over_raw": _relative_improvement(
            stripped["demo_residual"]["action_mse"],
            stripped["raw_nearest_copy"]["action_mse"],
        ),
        "relative_mse_improvement_over_shuffled": _relative_improvement(
            stripped["demo_residual"]["action_mse"],
            stripped["same_operation_shuffled_residual"]["action_mse"],
        ),
    }


def run(
    *,
    project_root: Path,
    config_path: Path,
    data_root: Path,
    output_root: Path,
    config: BeliefLocalResidualConfig,
) -> None:
    if output_root.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output_root}")
    _seed_everything(config.seed)
    data, preprocess, data_path, preprocess_path = _load_data(data_root, config)
    train_indices = np.where(data["split_id"] == SPLITS["train"])[0]
    validation_indices = np.where(data["split_id"] == SPLITS["val"])[0]
    state_mean = data["state"][train_indices].mean(axis=0).astype(np.float32)
    state_std = data["state"][train_indices].std(axis=0)
    state_std = np.maximum(state_std, config.state_std_floor).astype(np.float32)

    train_query, train_demo, train_distance = _rank_demos(
        data,
        query_indices=train_indices,
        bank_indices=train_indices,
        state_std=state_std,
        same_operation=True,
        neighbors=config.train_neighbors,
        exclude_same_pair=True,
    )
    val_query, val_demo, val_distance = _rank_demos(
        data,
        query_indices=validation_indices,
        bank_indices=train_indices,
        state_std=state_std,
        same_operation=True,
        neighbors=1,
        exclude_same_pair=True,
    )
    wrong_query, wrong_demo, wrong_distance = _rank_demos(
        data,
        query_indices=validation_indices,
        bank_indices=train_indices,
        state_std=state_std,
        same_operation=False,
        neighbors=1,
        exclude_same_pair=True,
    )
    if not np.array_equal(val_query, wrong_query):
        raise ValueError("correct/wrong validation query 顺序不一致")
    shuffled_demo = _shuffled_demo_rows(
        data,
        val_query,
        val_demo,
        train_indices,
        seed=config.seed + 17,
    )
    train_contexts = _contexts(data, train_query, train_demo)
    validation_contexts = _contexts(data, val_query, val_demo)
    shuffled_contexts = _contexts(data, val_query, shuffled_demo)
    wrong_contexts = _contexts(data, wrong_query, wrong_demo)

    policy_config = LayoutEquivariantDemoPolicyConfig(
        state_dim=data["state"].shape[1],
        action_horizon=data["action"].shape[1],
        hidden_dim=config.hidden_dim,
        maximum_translation_residual_m=config.maximum_translation_residual_m,
        maximum_rotation_residual_rad=config.maximum_rotation_residual_rad,
        translation_action_scale_m=config.translation_action_scale_m,
        rotation_action_scale_rad=config.rotation_action_scale_rad,
    )
    model = LayoutEquivariantDemoPolicy(
        policy_config,
        state_mean=torch.from_numpy(state_mean),
        state_std=torch.from_numpy(state_std),
    )
    dataset = TensorDataset(
        torch.from_numpy(train_contexts["query_state"]).float(),
        torch.from_numpy(train_contexts["demo_state"]).float(),
        torch.from_numpy(train_contexts["anchor"]).float(),
        torch.from_numpy(train_contexts["target"]).float(),
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    training_log: list[dict[str, float | int]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_validation_mse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    for epoch in range(config.epochs):
        model.train()
        losses = []
        for query, demo, anchor, target in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(query, demo, anchor, torch.ones(len(query)))
            translation_loss = torch.mean(
                ((prediction[..., :3] - target[..., :3])
                 / config.maximum_translation_residual_m) ** 2
            )
            rotation_loss = torch.mean(
                ((prediction[..., 3:6] - target[..., 3:6])
                 / config.maximum_rotation_residual_rad) ** 2
            )
            loss = translation_loss + rotation_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip_norm
            )
            optimizer.step()
            losses.append(float(loss.detach()))
        validation_prediction = _predict(
            model, validation_contexts, batch_size=config.batch_size
        )
        validation_mse = float(
            np.mean(
                (validation_prediction - validation_contexts["target"]) ** 2
            )
        )
        training_log.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
                "validation_action_mse": validation_mse,
            }
        )
        if validation_mse < best_validation_mse - 1e-12:
            best_validation_mse = validation_mse
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= config.early_stopping_patience:
            break
    model.load_state_dict(best_state)

    validation = _evaluate(
        model=model,
        correct=validation_contexts,
        shuffled=shuffled_contexts,
        wrong=wrong_contexts,
        data=data,
        config=config,
    )
    structural = _structural_audit(model, validation_contexts)
    latency = _latency(
        model,
        validation_contexts,
        warmup=config.latency_warmup,
        iterations=config.latency_iterations,
    )
    excluded_same_scene = bool(
        np.all(data["pair_id"][train_query] != data["pair_id"][train_demo])
    )
    criteria = {
        "R1_relative_improvement_over_raw": (
            validation["relative_mse_improvement_over_raw"]
            >= config.minimum_validation_relative_improvement
        ),
        "R2_scene_paired_significance_over_raw": (
            validation["residual_minus_raw_scene_pair_bootstrap"]["ci95_high"]
            < 0.0
        ),
        "R3_direction_accuracy": (
            validation["metrics"]["demo_residual"]["direction_accuracy"]
            >= config.minimum_validation_direction_accuracy
        ),
        "R4_wrong_operation_direction_accuracy": (
            validation["metrics"]["wrong_operation_residual"][
                "direction_accuracy"
            ]
            <= config.maximum_wrong_operation_direction_accuracy
        ),
        "R5_shuffled_demo_dependence": (
            validation["relative_mse_improvement_over_shuffled"]
            >= config.minimum_shuffled_relative_improvement
            and validation["residual_minus_shuffled_scene_pair_bootstrap"][
                "ci95_high"
            ]
            < 0.0
        ),
        "R6_structural_constraints": (
            structural["identity_anchor_max_abs_error"] == 0.0
            and structural["no_demo_max_abs_value"] == 0.0
            and structural["translation_residual_max_abs_m"]
            <= config.maximum_translation_residual_m + 1e-8
            and structural["rotation_residual_max_abs_rad"]
            <= config.maximum_rotation_residual_rad + 1e-8
            and structural["gripper_residual_max_abs"] == 0.0
        ),
        "R7_deployment_budget": (
            structural["parameters"] <= config.maximum_parameters
            and latency["p95_ms"] <= config.maximum_latency_p95_ms
        ),
        "R8_provenance_and_split": excluded_same_scene,
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(project_root),
        "config": asdict(config),
        "config_sha256": _sha256(config_path),
        "data_sha256": _sha256(data_path),
        "preprocess_report_sha256": _sha256(preprocess_path),
        "protocol": {
            "anchor": "raw retrieved Demo action; no transport",
            "train_neighbors_distinct_scene_pairs": config.train_neighbors,
            "validation_neighbors": 1,
            "training_contexts": len(train_query),
            "validation_queries": len(val_query),
            "same_scene_training_demo_excluded": excluded_same_scene,
            "progress_feature_used_for_retrieval": False,
            "operation_feature_used_by_model": False,
            "future_query_fields_used": False,
            "test_split_used": False,
            "best_epoch_selected_only_by_validation": best_epoch,
        },
        "retrieval_distance": {
            "train_median": float(np.median(train_distance)),
            "validation_median": float(np.median(val_distance)),
            "wrong_operation_median": float(np.median(wrong_distance)),
        },
        "training": {
            "epochs_run": len(training_log),
            "best_validation_action_mse": best_validation_mse,
            "final_train_loss": training_log[-1]["train_loss"],
        },
        "validation": validation,
        "structural_audit": structural,
        "latency": latency,
        "criteria": criteria,
        "validation_gate_passed": all(criteria.values()),
        "upstream_preprocess_schema_version": preprocess["schema_version"],
    }

    temporary = output_root.with_name(
        f".{output_root.name}.incomplete-{os.getpid()}"
    )
    temporary.mkdir(parents=True)

    def cleanup() -> None:
        shutil.rmtree(temporary, ignore_errors=True)

    atexit.register(cleanup)
    checkpoint_path = temporary / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(policy_config),
            "train_config": asdict(config),
            "data_sha256": _sha256(data_path),
            "validation_gate_passed": report["validation_gate_passed"],
        },
        checkpoint_path,
    )
    log_path = temporary / "training_log.jsonl"
    with log_path.open("w", encoding="utf-8") as stream:
        for row in training_log:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    retrieval_path = temporary / "retrieval_contexts.npz"
    with retrieval_path.open("wb") as stream:
        np.savez_compressed(
            stream,
            train_query_row=train_query,
            train_demo_row=train_demo,
            train_distance=train_distance,
            validation_query_row=val_query,
            validation_demo_row=val_demo,
            validation_distance=val_distance,
            shuffled_demo_row=shuffled_demo,
            wrong_demo_row=wrong_demo,
            wrong_distance=wrong_distance,
        )
    report["files"] = {
        "checkpoint.pt": _sha256(checkpoint_path),
        "training_log.jsonl": _sha256(log_path),
        "retrieval_contexts.npz": _sha256(retrieval_path),
    }
    report_path = temporary / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_root)
    atexit.unregister(cleanup)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not report["validation_gate_passed"]:
        raise RuntimeError("belief-local Demo residual 未通过 validation 门控")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    resolved_config = arguments.config.resolve()
    run(
        project_root=arguments.project_root.resolve(),
        config_path=resolved_config,
        data_root=arguments.data_root.resolve(),
        output_root=arguments.output_root.resolve(),
        config=BeliefLocalResidualConfig.from_json(resolved_config),
    )

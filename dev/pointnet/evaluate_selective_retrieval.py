"""评估轻量 Retriever 的 episode-held-out precision–coverage 与拒绝能力。"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from dev.pointnet.compare_retrievers import (
    ComparisonConfig,
    _read_jsonl,
    _sha256,
    encode_contexts,
    load_model,
)
from dev.pointnet.dataset import PointNetContextStore
from dev.pointnet.evaluate_end_to_end_retriever import (
    _base_candidate_mask,
    _load_text_scores,
    _rank_with_episode_cap,
    _text_candidate_masks,
)


@dataclass(frozen=True)
class ReliabilityConfig:
    """选择性检索的固定校准协议。"""

    seed: int = 20260928
    text_group_budget: int = 10
    text_score_weight: float = 0.15
    target_precision: float = 0.90
    minimum_calibration_accepts: int = 20
    bootstrap_resamples: int = 5000
    batch_size: int = 64
    num_workers: int = 4

    def __post_init__(self) -> None:
        if min(
            self.text_group_budget,
            self.minimum_calibration_accepts,
            self.bootstrap_resamples,
            self.batch_size,
        ) <= 0:
            raise ValueError("budget/count/bootstrap/batch 必须为正")
        if self.num_workers < 0:
            raise ValueError("num_workers 不能为负")
        if not 0.0 < self.target_precision <= 1.0:
            raise ValueError("target_precision 必须位于 (0, 1]")
        if not 0.0 <= self.text_score_weight <= 1.0:
            raise ValueError("text_score_weight 必须位于 [0, 1]")


def _git_commit(project_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _episode_key(record: Mapping[str, Any]) -> str:
    return ":".join(
        (
            str(record["task"]),
            str(record["variation"]),
            str(record["episode"]),
        )
    )


def _split_episode_groups(
    records: Sequence[Mapping[str, Any]],
    eligible_indices: Sequence[int],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """每个 task 内按稳定 hash 交替分 calibration/test episodes。"""
    groups_by_task: dict[str, set[str]] = defaultdict(set)
    for index in eligible_indices:
        record = records[index]
        groups_by_task[str(record["task"])].add(_episode_key(record))
    calibration_groups: set[str] = set()
    for task, groups in groups_by_task.items():
        ranked = sorted(
            groups,
            key=lambda group: hashlib.sha256(
                f"{seed}:{task}:{group}".encode("utf-8")
            ).hexdigest(),
        )
        calibration_groups.update(ranked[::2])
    calibration = np.asarray(
        [
            position
            for position, record_index in enumerate(eligible_indices)
            if _episode_key(records[record_index]) in calibration_groups
        ],
        dtype=np.int64,
    )
    evaluation = np.asarray(
        [
            position
            for position, record_index in enumerate(eligible_indices)
            if _episode_key(records[record_index]) not in calibration_groups
        ],
        dtype=np.int64,
    )
    if not len(calibration) or not len(evaluation):
        raise ValueError("calibration/test episode split 为空")
    return calibration, evaluation


def _fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    max_iterations: int = 50,
    l2: float = 1e-3,
) -> dict[str, np.ndarray]:
    """用小型 Newton logistic calibrator 融合无需额外前向的置信特征。"""
    mean = features.mean(axis=0)
    std = np.maximum(features.std(axis=0), 1e-6)
    normalized = (features - mean) / std
    design = np.concatenate(
        (np.ones((len(features), 1)), normalized),
        axis=1,
    )
    targets = labels.astype(np.float64)
    if len(np.unique(targets)) < 2:
        raise ValueError("calibration labels 只有一个类别")
    weights = np.zeros(design.shape[1], dtype=np.float64)
    regularizer = np.eye(design.shape[1], dtype=np.float64) * l2
    regularizer[0, 0] = 1e-8
    for _ in range(max_iterations):
        logits = np.clip(design @ weights, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        gradient = design.T @ (probabilities - targets) / len(targets)
        gradient += regularizer @ weights
        curvature = probabilities * (1.0 - probabilities)
        hessian = design.T @ (curvature[:, None] * design) / len(targets)
        hessian += regularizer
        step = np.linalg.solve(hessian, gradient)
        weights -= step
        if float(np.linalg.norm(step)) < 1e-8:
            break
    return {"mean": mean, "std": std, "weights": weights}


def _logistic_score(features: np.ndarray, model: Mapping[str, np.ndarray]) -> np.ndarray:
    normalized = (features - model["mean"]) / model["std"]
    design = np.concatenate(
        (np.ones((len(features), 1)), normalized),
        axis=1,
    )
    logits = np.clip(design @ model["weights"], -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-logits))


def _select_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    target_precision: float,
    minimum_accepts: int,
) -> dict[str, float | int | None]:
    best: dict[str, float | int | None] | None = None
    for threshold in np.unique(scores)[::-1]:
        accepted = scores >= threshold
        count = int(accepted.sum())
        if count < minimum_accepts:
            continue
        precision = float(labels[accepted].mean())
        if precision < target_precision:
            continue
        candidate = {
            "threshold": float(threshold),
            "accepted": count,
            "coverage": count / len(labels),
            "precision": precision,
        }
        if best is None or count > int(best["accepted"]):
            best = candidate
    if best is not None:
        return best
    return {
        "threshold": None,
        "accepted": 0,
        "coverage": 0.0,
        "precision": None,
    }


def _evaluate_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float | None,
) -> dict[str, float | int | None]:
    accepted = np.zeros(len(labels), dtype=bool) if threshold is None else scores >= threshold
    count = int(accepted.sum())
    true_positives = int(labels[accepted].sum()) if count else 0
    total_positives = int(labels.sum())
    return {
        "accepted": count,
        "coverage": count / len(labels),
        "precision": true_positives / count if count else None,
        "correct_recall": true_positives / max(total_positives, 1),
    }


def _precision_at_coverages(
    scores: np.ndarray,
    labels: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    order = np.argsort(-scores, kind="stable")
    result = {}
    for fraction in (0.10, 0.25, 0.50, 0.75, 1.00):
        count = min(len(labels), max(1, int(round(fraction * len(labels)))))
        chosen = order[:count]
        result[f"coverage_{fraction:.2f}"] = {
            "accepted": count,
            "precision": float(labels[chosen].mean()),
        }
    return result


def _bootstrap_precision(
    *,
    scores: np.ndarray,
    labels: np.ndarray,
    group_ids: np.ndarray,
    threshold: float | None,
    resamples: int,
    seed: int,
) -> dict[str, float | int | None]:
    if threshold is None:
        return {"ci95_low": None, "ci95_high": None, "valid_resamples": 0}
    unique = np.unique(group_ids)
    members = [np.flatnonzero(group_ids == group) for group in unique]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(resamples):
        chosen = rng.integers(0, len(members), size=len(members))
        indices = np.concatenate([members[index] for index in chosen])
        accepted = scores[indices] >= threshold
        if accepted.any():
            values.append(float(labels[indices][accepted].mean()))
    if not values:
        return {"ci95_low": None, "ci95_high": None, "valid_resamples": 0}
    return {
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
        "valid_resamples": len(values),
    }


@torch.inference_mode()
def run(
    *,
    project_root: Path,
    data_root: Path,
    checkpoint_path: Path,
    text_scores_path: Path,
    output_path: Path,
    device: torch.device,
    config: ReliabilityConfig,
) -> None:
    if output_path.exists():
        raise FileExistsError(f"输出已存在，拒绝覆盖：{output_path}")
    dataset_hash = _sha256(data_root / "summary.json")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("dataset_summary_sha256") != dataset_hash:
        raise ValueError("Retriever checkpoint 与数据不匹配")
    store = PointNetContextStore(data_root, "val", cache_size=4)
    records = store.records
    pair_rows = _read_jsonl(data_root / "pairs-val.jsonl")
    pairs = {str(row["query_id"]): row for row in pair_rows}
    id_to_index = {
        str(record["chunk_id"]): index for index, record in enumerate(records)
    }
    text_scores, chunk_groups, text_metadata = _load_text_scores(
        text_scores_path,
        records,
    )
    if text_metadata.get("manifest_sha256") != _sha256(
        data_root / "manifest-val.jsonl"
    ):
        raise ValueError("文本分数与 val manifest 不匹配")
    text_mask = _text_candidate_masks(
        text_scores,
        chunk_groups,
        (config.text_group_budget,),
    )[config.text_group_budget]
    retriever = load_model(checkpoint_path, device)
    embeddings = encode_contexts(
        retriever,
        store,
        ComparisonConfig(
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        ),
        device,
    )
    pointnet_scores = ((embeddings @ embeddings.T) + 1.0) / 2.0
    normalized_text = ((text_scores + 1.0) / 2.0).clamp(0.0, 1.0)
    combined_scores = (
        config.text_score_weight * normalized_text
        + (1.0 - config.text_score_weight) * pointnet_scores
    )

    eligible_indices = []
    labels = []
    features = []
    candidate_counts = []
    for query_index, record in enumerate(records):
        pair = pairs[str(record["chunk_id"])]
        base_mask = _base_candidate_mask(records, query_index)
        positives = {
            id_to_index[identifier]
            for identifier in pair["positive_ids"]
            if identifier in id_to_index and bool(base_mask[id_to_index[identifier]])
        }
        if not positives:
            continue
        candidate_mask = base_mask & text_mask[query_index]
        ranked = _rank_with_episode_cap(
            combined_scores[query_index],
            candidate_mask,
            records,
            episode_cap=1,
        )
        if not ranked:
            continue
        top = ranked[0]
        second = ranked[1] if len(ranked) > 1 else ranked[0]
        eligible_indices.append(query_index)
        labels.append(float(top in positives))
        candidate_counts.append(int(candidate_mask.sum()))
        features.append(
            (
                float(combined_scores[query_index, top]),
                float(
                    combined_scores[query_index, top]
                    - combined_scores[query_index, second]
                ),
                float(pointnet_scores[query_index, top]),
                float(
                    pointnet_scores[query_index, top]
                    - pointnet_scores[query_index, second]
                ),
                float(normalized_text[query_index, top]),
                float(abs(pointnet_scores[query_index, top] - normalized_text[query_index, top])),
            )
        )

    feature_names = (
        "combined_top1",
        "combined_margin",
        "pointnet_top1",
        "pointnet_margin_on_combined_top2",
        "text_top1",
        "cross_modal_score_gap",
    )
    features_array = np.asarray(features, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=np.float64)
    calibration_indices, test_indices = _split_episode_groups(
        records,
        eligible_indices,
        config.seed,
    )
    logistic = _fit_logistic(
        features_array[calibration_indices],
        labels_array[calibration_indices],
    )
    confidence_scores = {
        "combined_top1": features_array[:, 0],
        "combined_margin": features_array[:, 1],
        "pointnet_margin": features_array[:, 3],
        "logistic": _logistic_score(features_array, logistic),
    }
    test_group_ids = np.asarray(
        [_episode_key(records[eligible_indices[index]]) for index in test_indices]
    )
    methods = {}
    for offset, (name, scores) in enumerate(confidence_scores.items()):
        calibration = _select_threshold(
            scores[calibration_indices],
            labels_array[calibration_indices],
            target_precision=config.target_precision,
            minimum_accepts=config.minimum_calibration_accepts,
        )
        threshold = calibration["threshold"]
        test_scores = scores[test_indices]
        test_labels = labels_array[test_indices]
        evaluation = _evaluate_threshold(test_scores, test_labels, threshold)
        evaluation["precision_episode_bootstrap"] = _bootstrap_precision(
            scores=test_scores,
            labels=test_labels,
            group_ids=test_group_ids,
            threshold=threshold if isinstance(threshold, float) else None,
            resamples=config.bootstrap_resamples,
            seed=config.seed + offset,
        )
        methods[name] = {
            "calibration": calibration,
            "test": evaluation,
            "test_precision_at_fixed_coverages": _precision_at_coverages(
                test_scores,
                test_labels,
            ),
        }

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "label": "top-1 candidate belongs to pair-label positives",
            "split": "task-stratified, episode-held-out calibration/test",
            "selection": "threshold chosen only on calibration episodes",
            "runtime_cost": "scalar features from existing text + PointNet scores",
        },
        "config": asdict(config),
        "evaluator_git_commit": _git_commit(project_root),
        "dataset_summary_sha256": dataset_hash,
        "retriever_checkpoint_sha256": _sha256(checkpoint_path),
        "text_scores_sha256": _sha256(text_scores_path),
        "feature_names": feature_names,
        "logistic_calibrator": {
            "mean": logistic["mean"].tolist(),
            "std": logistic["std"].tolist(),
            "weights": logistic["weights"].tolist(),
        },
        "eligible_queries": len(eligible_indices),
        "calibration_queries": len(calibration_indices),
        "test_queries": len(test_indices),
        "full_top1_precision": float(labels_array.mean()),
        "candidate_chunks_mean": float(np.mean(candidate_counts)),
        "methods": methods,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(f"{output_path.suffix}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--text-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA，但当前节点没有可用 GPU")
    run(
        project_root=arguments.project_root.resolve(),
        data_root=arguments.data_root.resolve(),
        checkpoint_path=arguments.checkpoint.resolve(),
        text_scores_path=arguments.text_scores.resolve(),
        output_path=arguments.output.resolve(),
        device=device,
        config=ReliabilityConfig(),
    )


if __name__ == "__main__":
    main()

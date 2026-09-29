"""评测 paired predictor 的 query-only 与 Demo-copy 基线。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np


SPLITS = {"train": 0, "val": 1, "test": 2}
OPERATIONS = {0: "left", 1: "right"}


@dataclass(frozen=True)
class DemoBaselineConfig:
    """冻结的 Demo 必要性基线协议。"""

    schema_version: str
    seed: int
    bootstrap_resamples: int
    state_std_floor: float
    minimum_correct_demo_direction_accuracy: float
    maximum_wrong_demo_direction_accuracy: float

    @classmethod
    def from_json(cls, path: Path) -> "DemoBaselineConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def __post_init__(self) -> None:
        if self.schema_version != "bidirectional-slide-demo-baselines-v1":
            raise ValueError("未知 Demo baseline schema")
        if self.seed < 0 or self.bootstrap_resamples <= 0:
            raise ValueError("seed/bootstrap_resamples 非法")
        if self.state_std_floor <= 0.0:
            raise ValueError("state_std_floor 必须为正")
        probabilities = (
            self.minimum_correct_demo_direction_accuracy,
            self.maximum_wrong_demo_direction_accuracy,
        )
        if any(not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("direction accuracy threshold 必须在 [0,1]")


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


def _metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    if prediction.shape != target.shape or target.ndim != 3 or target.shape[-1] != 7:
        raise ValueError("prediction/target 必须为相同 [N,H,7]")
    error = prediction - target
    per_row_mse = np.mean(error**2, axis=(1, 2))
    translation_error = np.linalg.norm(error[..., :3], axis=-1)
    predicted_direction = prediction[..., :3].sum(axis=1)
    target_direction = target[..., :3].sum(axis=1)
    denominator = np.linalg.norm(predicted_direction, axis=1) * np.linalg.norm(
        target_direction,
        axis=1,
    )
    cosine = np.divide(
        np.sum(predicted_direction * target_direction, axis=1),
        denominator,
        out=np.full(len(target), -1.0),
        where=denominator > 1e-12,
    )
    return {
        "action_mse": float(per_row_mse.mean()),
        "translation_rmse_mm": float(
            np.sqrt(np.mean(error[..., :3] ** 2)) * 1000.0
        ),
        "translation_l2_mean_mm": float(translation_error.mean() * 1000.0),
        "direction_cosine_mean": float(cosine.mean()),
        "direction_accuracy": float(np.mean(cosine > 0.0)),
        "per_row_action_mse": per_row_mse,
    }


def _nearest(
    query_state: np.ndarray,
    train_state: np.ndarray,
    train_indices: np.ndarray,
) -> tuple[int, float]:
    distances = np.linalg.norm(train_state[train_indices] - query_state, axis=1)
    order = np.lexsort((train_indices, distances))
    selected = int(train_indices[order[0]])
    return selected, float(distances[order[0]])


def _scene_values(
    row_values: np.ndarray,
    pair_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(pair_ids)
    values = np.asarray(
        [row_values[pair_ids == pair_id].mean() for pair_id in unique]
    )
    return unique, values


def _bootstrap_difference(
    left: np.ndarray,
    right: np.ndarray,
    *,
    seed: int,
    resamples: int,
) -> dict[str, float]:
    if left.shape != right.shape or left.ndim != 1 or not len(left):
        raise ValueError("paired bootstrap 输入非法")
    differences = left - right
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(resamples, len(differences)))
    samples = differences[indices].mean(axis=1)
    return {
        "mean": float(differences.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "unit": "canonical action MSE; first model minus second model",
        "scene_pairs": len(differences),
    }


def _strip_rows(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        key: value
        for key, value in metrics.items()
        if key != "per_row_action_mse"
    }


def _evaluate_split(
    *,
    name: str,
    split_id: int,
    data: dict[str, np.ndarray],
    normalized_state: np.ndarray,
    train_indices: np.ndarray,
    operation_means: dict[int, np.ndarray],
    global_mean: np.ndarray,
    config: DemoBaselineConfig,
) -> dict[str, Any]:
    query_indices = np.where(data["split_id"] == split_id)[0]
    target = data["action"][query_indices]
    correct = np.empty_like(target)
    wrong = np.empty_like(target)
    geometry_only = np.empty_like(target)
    oracle_mean = np.empty_like(target)
    selections = []
    for output_index, query_index in enumerate(query_indices):
        operation = int(data["operation_id"][query_index])
        correct_pool = train_indices[
            data["operation_id"][train_indices] == operation
        ]
        wrong_pool = train_indices[
            data["operation_id"][train_indices] != operation
        ]
        correct_index, correct_distance = _nearest(
            normalized_state[query_index], normalized_state, correct_pool
        )
        wrong_index, wrong_distance = _nearest(
            normalized_state[query_index], normalized_state, wrong_pool
        )
        geometry_index, geometry_distance = _nearest(
            normalized_state[query_index], normalized_state, train_indices
        )
        correct[output_index] = data["action"][correct_index]
        wrong[output_index] = data["action"][wrong_index]
        geometry_only[output_index] = data["action"][geometry_index]
        oracle_mean[output_index] = operation_means[operation]
        selections.append(
            {
                "query_row": int(query_index),
                "query_pair_id": int(data["pair_id"][query_index]),
                "query_operation": OPERATIONS[operation],
                "correct_demo_row": correct_index,
                "correct_demo_pair_id": int(data["pair_id"][correct_index]),
                "correct_distance": correct_distance,
                "wrong_demo_row": wrong_index,
                "wrong_demo_pair_id": int(data["pair_id"][wrong_index]),
                "wrong_distance": wrong_distance,
                "geometry_only_demo_row": geometry_index,
                "geometry_only_demo_operation": OPERATIONS[
                    int(data["operation_id"][geometry_index])
                ],
                "geometry_only_distance": geometry_distance,
            }
        )

    query_only = np.broadcast_to(global_mean, target.shape).copy()
    pair_midpoint = np.empty_like(target)
    for pair_id in np.unique(data["pair_id"][query_indices]):
        mask = data["pair_id"][query_indices] == pair_id
        if int(mask.sum()) != 2:
            raise ValueError(f"split={name} pair={pair_id} 不是两条分支")
        pair_midpoint[mask] = target[mask].mean(axis=0)
    predictions = {
        "query_only_train_mean": query_only,
        "paired_test_midpoint_lower_bound": pair_midpoint,
        "geometry_only_demo_copy": geometry_only,
        "correct_demo_copy": correct,
        "wrong_demo_copy": wrong,
        "operation_mean_oracle": oracle_mean,
    }
    raw_metrics = {
        model: _metrics(prediction, target)
        for model, prediction in predictions.items()
    }
    pair_ids = data["pair_id"][query_indices]
    scene_mse = {
        model: _scene_values(metrics["per_row_action_mse"], pair_ids)[1]
        for model, metrics in raw_metrics.items()
    }
    comparisons = {
        "correct_minus_query_only": _bootstrap_difference(
            scene_mse["correct_demo_copy"],
            scene_mse["query_only_train_mean"],
            seed=config.seed + 1 + split_id,
            resamples=config.bootstrap_resamples,
        ),
        "correct_minus_wrong": _bootstrap_difference(
            scene_mse["correct_demo_copy"],
            scene_mse["wrong_demo_copy"],
            seed=config.seed + 11 + split_id,
            resamples=config.bootstrap_resamples,
        ),
    }
    criteria = {
        "correct_demo_beats_query_only": (
            comparisons["correct_minus_query_only"]["ci95_high"] < 0.0
        ),
        "correct_demo_beats_wrong_demo": (
            comparisons["correct_minus_wrong"]["ci95_high"] < 0.0
        ),
        "correct_demo_direction_accuracy": (
            raw_metrics["correct_demo_copy"]["direction_accuracy"]
            >= config.minimum_correct_demo_direction_accuracy
        ),
        "wrong_demo_direction_accuracy": (
            raw_metrics["wrong_demo_copy"]["direction_accuracy"]
            <= config.maximum_wrong_demo_direction_accuracy
        ),
    }
    return {
        "rows": len(query_indices),
        "scene_pairs": len(np.unique(pair_ids)),
        "metrics": {
            model: _strip_rows(metrics) for model, metrics in raw_metrics.items()
        },
        "comparisons": comparisons,
        "criteria": criteria,
        "all_criteria_passed": all(criteria.values()),
        "selections": selections,
    }


def run(
    *,
    project_root: Path,
    config_path: Path,
    data_root: Path,
    output_path: Path,
    config: DemoBaselineConfig,
) -> None:
    if output_path.exists():
        raise FileExistsError(f"输出已存在，拒绝覆盖：{output_path}")
    data_path = data_root / "branch_samples.npz"
    preprocess_report_path = data_root / "preprocess_report.json"
    preprocess_report = json.loads(
        preprocess_report_path.read_text(encoding="utf-8")
    )
    with np.load(data_path) as archive:
        data = {key: archive[key] for key in archive.files}
    required = {
        "pair_id",
        "split_id",
        "operation_id",
        "state",
        "action",
    }
    if not required.issubset(data):
        missing = sorted(required - data.keys())
        raise KeyError(f"predictor data 缺少字段：{missing}")
    train_indices = np.where(data["split_id"] == SPLITS["train"])[0]
    state_mean = data["state"][train_indices].mean(axis=0)
    state_std = data["state"][train_indices].std(axis=0)
    state_std = np.maximum(state_std, config.state_std_floor)
    normalized_state = (data["state"] - state_mean) / state_std
    operation_means = {
        operation: data["action"][
            train_indices[data["operation_id"][train_indices] == operation]
        ].mean(axis=0)
        for operation in OPERATIONS
    }
    global_mean = data["action"][train_indices].mean(axis=0)
    results = {
        split: _evaluate_split(
            name=split,
            split_id=SPLITS[split],
            data=data,
            normalized_state=normalized_state,
            train_indices=train_indices,
            operation_means=operation_means,
            global_mean=global_mean,
            config=config,
        )
        for split in ("val", "test")
    }
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(project_root),
        "config": asdict(config),
        "config_sha256": _sha256(config_path),
        "data_sha256": _sha256(data_path),
        "preprocess_report_sha256": _sha256(preprocess_report_path),
        "preprocess_action_lower_bound": preprocess_report[
            "canonical_observation_only_mse_lower_bound"
        ],
        "protocol": {
            "retrieval_distance": "train-standardized 21D observation state L2",
            "operation_used_only_for": (
                "constructing correct/wrong Demo and operation-mean oracle"
            ),
            "model_inputs_exclude_operation": True,
        },
        "results": results,
        "primary_test_all_criteria_passed": results["test"][
            "all_criteria_passed"
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(f"{output_path.suffix}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    printable = {
        "results": {
            split: {
                key: value
                for key, value in result.items()
                if key != "selections"
            }
            for split, result in results.items()
        },
        "primary_test_all_criteria_passed": report[
            "primary_test_all_criteria_passed"
        ],
    }
    print(json.dumps(printable, indent=2, sort_keys=True), flush=True)
    if not report["primary_test_all_criteria_passed"]:
        raise RuntimeError("Demo baseline 未通过全部 test 预注册判据")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(
        project_root=arguments.project_root.resolve(),
        config_path=arguments.config.resolve(),
        data_root=arguments.data_root.resolve(),
        output_path=arguments.output.resolve(),
        config=DemoBaselineConfig.from_json(arguments.config.resolve()),
    )

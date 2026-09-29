"""评测多进度几何检索与无参数 layout-equivariant Demo transport。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from dev.simulator.evaluate_rotated_layout_slide_transport import (
    OPERATIONS,
    SPLITS,
    _bootstrap_difference,
    _metrics,
    _point_yaw,
    _scene_means,
    _strip_rows,
    _transport,
    _wrap_angle,
)
from dev.simulator.preprocess_rotated_layout_slide_predictor import _sha256


@dataclass(frozen=True)
class ProgressTransportConfig:
    """冻结的 progress-local transport 基线与门槛。"""

    schema_version: str
    seed: int
    bootstrap_resamples: int
    state_std_floor: float
    maximum_progress_mae: float
    minimum_progress_within_tolerance_fraction: float
    progress_tolerance: float
    minimum_direction_accuracy: float
    minimum_relative_mse_improvement: float
    large_yaw_gap_degrees: float
    minimum_large_yaw_rows: int
    minimum_large_yaw_direction_accuracy: float
    minimum_large_yaw_relative_mse_improvement: float
    minimum_shuffled_relative_mse_improvement: float

    @classmethod
    def from_json(cls, path: Path) -> "ProgressTransportConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def __post_init__(self) -> None:
        if self.schema_version != "rotated-layout-slide-progress-transport-v1":
            raise ValueError("未知 progress transport schema")
        positive = (
            self.bootstrap_resamples,
            self.state_std_floor,
            self.maximum_progress_mae,
            self.progress_tolerance,
            self.large_yaw_gap_degrees,
            self.minimum_large_yaw_rows,
        )
        if min(positive) <= 0 or self.seed < 0:
            raise ValueError("progress transport 配置非法")
        probabilities = (
            self.minimum_progress_within_tolerance_fraction,
            self.minimum_direction_accuracy,
            self.minimum_relative_mse_improvement,
            self.minimum_large_yaw_direction_accuracy,
            self.minimum_large_yaw_relative_mse_improvement,
            self.minimum_shuffled_relative_mse_improvement,
        )
        if any(not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("progress transport rate 必须位于 [0,1]")


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _load_data(root: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    data_path = root / "progress_chunks.npz"
    report_path = root / "preprocess_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if _sha256(data_path) != report["files"]["progress_chunks.npz"]:
        raise ValueError("progress chunk data hash 不匹配")
    if report.get("protocol", {}).get("test_split_used", True):
        raise ValueError("progress preprocessing 错误标记为使用 test split")
    with np.load(data_path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    return data, report


def _nearest(
    data: dict[str, np.ndarray],
    query_index: int,
    candidates: np.ndarray,
    state_std: np.ndarray,
) -> tuple[int, float]:
    delta = (data["state"][candidates] - data["state"][query_index]) / state_std
    distances = np.mean(delta.astype(np.float64) ** 2, axis=1)
    local = int(np.argmin(distances))
    return int(candidates[local]), float(distances[local])


def _relative_improvement(method: float, baseline: float) -> float:
    return (baseline - method) / max(baseline, 1e-12)


def _subset_metrics(
    predictions: dict[str, np.ndarray],
    target: np.ndarray,
    mask: np.ndarray,
) -> dict[str, dict[str, float]]:
    if not bool(mask.any()):
        raise ValueError("评价子集为空")
    return {
        name: _strip_rows(_metrics(value[mask], target[mask]))
        for name, value in predictions.items()
    }


def run(
    *,
    project_root: Path,
    config_path: Path,
    data_root: Path,
    output_root: Path,
    config: ProgressTransportConfig,
) -> None:
    if output_root.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output_root}")
    data, preprocess = _load_data(data_root)
    train = np.where(data["split_id"] == SPLITS["train"])[0]
    validation = np.where(data["split_id"] == SPLITS["val"])[0]
    if not len(train) or not len(validation):
        raise ValueError("train/validation split 为空")
    state_std = data["state"][train].std(axis=0)
    state_std = np.maximum(state_std, config.state_std_floor).astype(np.float32)
    rng = np.random.default_rng(config.seed)
    names = (
        "operation_train_mean",
        "raw_nearest_copy",
        "point_transport",
        "same_operation_shuffled_transport",
        "wrong_operation_transport",
        "no_demo",
    )
    outputs: dict[str, list[np.ndarray]] = {name: [] for name in names}
    targets = []
    query_rows = []
    demo_rows = []
    shuffled_rows = []
    geometry_distances = []
    progress_errors = []
    yaw_gaps = []
    for query_index in validation:
        operation_id = int(data["operation_id"][query_index])
        pair_id = int(data["pair_id"][query_index])
        correct_candidates = train[
            (data["operation_id"][train] == operation_id)
            & (data["pair_id"][train] != pair_id)
        ]
        wrong_candidates = train[
            (data["operation_id"][train] != operation_id)
            & (data["pair_id"][train] != pair_id)
        ]
        if not len(correct_candidates) or not len(wrong_candidates):
            raise ValueError("operation bucket 候选为空")
        demo_index, distance = _nearest(
            data, int(query_index), correct_candidates, state_std
        )
        wrong_index, _ = _nearest(
            data, int(query_index), wrong_candidates, state_std
        )
        shuffled_candidates = correct_candidates[
            correct_candidates != demo_index
        ]
        shuffled_index = int(rng.choice(shuffled_candidates))
        query_yaw = _point_yaw(
            data["state"][query_index], data["tcp_pose"][query_index]
        )

        def transported(index: int) -> np.ndarray:
            demo_yaw = _point_yaw(
                data["state"][index], data["tcp_pose"][index]
            )
            return _transport(
                data["action"][index],
                data["tcp_pose"][index],
                data["tcp_pose"][query_index],
                _wrap_angle(query_yaw - demo_yaw),
            )

        operation_mean = data["action"][
            train[data["operation_id"][train] == operation_id]
        ].mean(axis=0)
        outputs["operation_train_mean"].append(operation_mean)
        outputs["raw_nearest_copy"].append(data["action"][demo_index])
        outputs["point_transport"].append(transported(demo_index))
        outputs["same_operation_shuffled_transport"].append(
            transported(shuffled_index)
        )
        outputs["wrong_operation_transport"].append(
            transported(wrong_index)
        )
        outputs["no_demo"].append(np.zeros_like(data["action"][query_index]))
        targets.append(data["action"][query_index])
        query_rows.append(int(query_index))
        demo_rows.append(demo_index)
        shuffled_rows.append(shuffled_index)
        geometry_distances.append(distance)
        progress_errors.append(
            abs(
                float(data["progress_fraction"][query_index])
                - float(data["progress_fraction"][demo_index])
            )
        )
        demo_yaw = _point_yaw(
            data["state"][demo_index], data["tcp_pose"][demo_index]
        )
        yaw_gaps.append(
            math.degrees(abs(_wrap_angle(query_yaw - demo_yaw)))
        )

    prediction_arrays = {
        name: np.stack(values) for name, values in outputs.items()
    }
    target = np.stack(targets)
    query_rows_array = np.asarray(query_rows, dtype=np.int64)
    progress_error = np.asarray(progress_errors, dtype=np.float64)
    yaw_gap = np.asarray(yaw_gaps, dtype=np.float64)
    overall = _subset_metrics(
        prediction_arrays,
        target,
        np.ones(len(target), dtype=bool),
    )
    large_yaw = yaw_gap >= config.large_yaw_gap_degrees
    large_yaw_metrics = _subset_metrics(prediction_arrays, target, large_yaw)
    progress_metrics = {}
    for progress_index in sorted(
        set(data["progress_index"][query_rows_array].tolist())
    ):
        mask = data["progress_index"][query_rows_array] == progress_index
        progress_metrics[str(progress_index)] = {
            "queries": int(mask.sum()),
            "retrieval_progress_mae": float(progress_error[mask].mean()),
            "metrics": _subset_metrics(prediction_arrays, target, mask),
        }

    raw_mse = float(overall["raw_nearest_copy"]["action_mse"])
    point_mse = float(overall["point_transport"]["action_mse"])
    shuffled_mse = float(
        overall["same_operation_shuffled_transport"]["action_mse"]
    )
    large_raw_mse = float(
        large_yaw_metrics["raw_nearest_copy"]["action_mse"]
    )
    large_point_mse = float(
        large_yaw_metrics["point_transport"]["action_mse"]
    )
    pair_ids = data["pair_id"][query_rows_array]
    point_per_row = _metrics(
        prediction_arrays["point_transport"], target
    )["per_row_action_mse"]
    shuffled_per_row = _metrics(
        prediction_arrays["same_operation_shuffled_transport"], target
    )["per_row_action_mse"]
    _, point_pair = _scene_means(point_per_row, pair_ids)
    _, shuffled_pair = _scene_means(shuffled_per_row, pair_ids)
    bootstrap = _bootstrap_difference(
        point_pair,
        shuffled_pair,
        seed=config.seed,
        resamples=config.bootstrap_resamples,
    )
    progress_summary = {
        "mae": float(progress_error.mean()),
        "p95": float(np.quantile(progress_error, 0.95)),
        "within_tolerance_fraction": float(
            np.mean(progress_error <= config.progress_tolerance)
        ),
        "tolerance": config.progress_tolerance,
    }
    improvements = {
        "point_vs_raw": _relative_improvement(point_mse, raw_mse),
        "point_vs_shuffled": _relative_improvement(
            point_mse, shuffled_mse
        ),
        "large_yaw_point_vs_raw": _relative_improvement(
            large_point_mse, large_raw_mse
        ),
    }
    criteria = {
        "P1_progress_locality": (
            progress_summary["mae"] <= config.maximum_progress_mae
            and progress_summary["within_tolerance_fraction"]
            >= config.minimum_progress_within_tolerance_fraction
        ),
        "P2_transport_over_raw": (
            overall["point_transport"]["direction_accuracy"]
            >= config.minimum_direction_accuracy
            and improvements["point_vs_raw"]
            >= config.minimum_relative_mse_improvement
        ),
        "P3_large_yaw_transport": (
            int(large_yaw.sum()) >= config.minimum_large_yaw_rows
            and large_yaw_metrics["point_transport"]["direction_accuracy"]
            >= config.minimum_large_yaw_direction_accuracy
            and improvements["large_yaw_point_vs_raw"]
            >= config.minimum_large_yaw_relative_mse_improvement
        ),
        "P4_transport_over_shuffled": (
            improvements["point_vs_shuffled"]
            >= config.minimum_shuffled_relative_mse_improvement
            and bootstrap["ci95_high"] < 0.0
        ),
        "P5_structure_and_provenance": (
            float(np.max(np.abs(prediction_arrays["no_demo"]))) == 0.0
            and preprocess["protocol"]["test_split_used"] is False
            and len(target)
            == int(preprocess["split_row_counts"]["val"])
        ),
    }

    temporary = output_root.with_name(
        f".{output_root.name}.incomplete-{os.getpid()}"
    )
    temporary.mkdir(parents=True)
    predictions_path = temporary / "predictions.npz"
    with predictions_path.open("wb") as stream:
        np.savez_compressed(
            stream,
            query_rows=query_rows_array,
            demo_rows=np.asarray(demo_rows, dtype=np.int64),
            shuffled_rows=np.asarray(shuffled_rows, dtype=np.int64),
            geometry_distances=np.asarray(
                geometry_distances, dtype=np.float32
            ),
            progress_errors=progress_error.astype(np.float32),
            yaw_gaps_degrees=yaw_gap.astype(np.float32),
            target=target,
            **prediction_arrays,
        )
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(project_root),
        "config": asdict(config),
        "config_sha256": _sha256(config_path),
        "data_sha256": _sha256(data_root / "progress_chunks.npz"),
        "preprocess_report_sha256": _sha256(
            data_root / "preprocess_report.json"
        ),
        "protocol": {
            "typed_operation_bucket": True,
            "same_scene_demo_excluded": True,
            "progress_feature_used_for_retrieval": False,
            "test_split_used": False,
        },
        "rows": {
            "train_bank": len(train),
            "validation_queries": len(validation),
            "large_yaw_queries": int(large_yaw.sum()),
        },
        "progress_retrieval": progress_summary,
        "geometry_distance": {
            "median": float(np.median(geometry_distances)),
            "p95": float(np.quantile(geometry_distances, 0.95)),
        },
        "yaw_gap_degrees": {
            "median": float(np.median(yaw_gap)),
            "p95": float(np.quantile(yaw_gap, 0.95)),
            "maximum": float(yaw_gap.max()),
        },
        "overall": overall,
        "large_yaw": large_yaw_metrics,
        "by_progress_index": progress_metrics,
        "relative_mse_improvements": improvements,
        "point_minus_shuffled_scene_pair_bootstrap": bootstrap,
        "criteria": criteria,
        "all_criteria_passed": all(criteria.values()),
        "files": {
            "predictions.npz": _sha256(predictions_path),
        },
    }
    report_path = temporary / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_root)
    print(
        json.dumps(
            {
                "progress_retrieval": progress_summary,
                "relative_mse_improvements": improvements,
                "point_transport": overall["point_transport"],
                "large_yaw_point_transport": large_yaw_metrics[
                    "point_transport"
                ],
                "bootstrap": bootstrap,
                "criteria": criteria,
                "all_criteria_passed": all(criteria.values()),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    if not all(criteria.values()):
        raise RuntimeError("progress-local Demo transport 未通过全部门控")


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
        config=ProgressTransportConfig.from_json(resolved_config),
    )

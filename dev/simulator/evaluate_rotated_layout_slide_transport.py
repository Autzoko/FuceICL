"""评测 raw copy、layout-frame transport 与 wrong-operation Demo。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from dev.simulator.preprocess_maniskill_chunks import (
    _quaternion_wxyz_to_matrix,
)


SPLITS = {"train": 0, "val": 1, "test": 2}
OPERATIONS = {0: "toward", 1: "away"}


@dataclass(frozen=True)
class TransportBaselineConfig:
    """冻结的 layout transport baseline 协议。"""

    schema_version: str
    seed: int
    bootstrap_resamples: int
    large_yaw_gap_degrees: float
    minimum_large_gap_contexts: int
    minimum_privileged_direction_accuracy: float
    minimum_point_direction_accuracy: float
    maximum_wrong_operation_direction_accuracy: float
    maximum_point_to_privileged_mse_ratio: float

    @classmethod
    def from_json(cls, path: Path) -> "TransportBaselineConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def __post_init__(self) -> None:
        if self.schema_version != "rotated-layout-slide-transport-baselines-v1":
            raise ValueError("未知 rotated-layout transport baseline schema")
        positive = (
            self.bootstrap_resamples,
            self.large_yaw_gap_degrees,
            self.minimum_large_gap_contexts,
            self.maximum_point_to_privileged_mse_ratio,
        )
        if min(positive) <= 0 or self.seed < 0:
            raise ValueError("baseline 配置非法")
        probabilities = (
            self.minimum_privileged_direction_accuracy,
            self.minimum_point_direction_accuracy,
            self.maximum_wrong_operation_direction_accuracy,
        )
        if any(not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("accuracy threshold 必须在 [0,1]")


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


def _wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _rotation_z(angle: float) -> np.ndarray:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _point_yaw(state: np.ndarray, tcp_pose: np.ndarray) -> float:
    rotation = _quaternion_wxyz_to_matrix(tcp_pose[3:7])
    axis_world = rotation @ np.asarray(state[3:6], dtype=np.float64)
    if np.linalg.norm(axis_world[:2]) <= 1e-9:
        raise ValueError("point-derived layout axis 退化")
    return math.atan2(float(axis_world[1]), float(axis_world[0]))


def _transport(
    action: np.ndarray,
    demo_tcp_pose: np.ndarray,
    query_tcp_pose: np.ndarray,
    yaw_delta: float,
) -> np.ndarray:
    """把 Demo EEF action 经 world layout rotation 映射到 query EEF。"""
    demo_rotation = _quaternion_wxyz_to_matrix(demo_tcp_pose[3:7])
    query_rotation = _quaternion_wxyz_to_matrix(query_tcp_pose[3:7])
    frame = query_rotation.T @ _rotation_z(yaw_delta) @ demo_rotation
    output = np.asarray(action, dtype=np.float64).copy()
    output[:, :3] = action[:, :3] @ frame.T
    # axis-angle 在刚体坐标变换下作为旋转向量变换。
    output[:, 3:6] = action[:, 3:6] @ frame.T
    return output.astype(np.float32)


def _metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction/target 必须为相同 [N,H,7]")
    error = prediction.astype(np.float64) - target.astype(np.float64)
    per_row_mse = np.mean(error**2, axis=(1, 2))
    predicted_direction = prediction[..., :3].sum(axis=1)
    target_direction = target[..., :3].sum(axis=1)
    denominator = np.linalg.norm(predicted_direction, axis=1) * np.linalg.norm(
        target_direction, axis=1
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
        "direction_cosine_mean": float(cosine.mean()),
        "direction_accuracy": float(np.mean(cosine > 0.0)),
        "per_row_action_mse": per_row_mse,
    }


def _strip_rows(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        key: value
        for key, value in metrics.items()
        if key != "per_row_action_mse"
    }


def _scene_means(
    values: np.ndarray,
    query_pair_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pair_ids = np.unique(query_pair_ids)
    return pair_ids, np.asarray(
        [values[query_pair_ids == pair_id].mean() for pair_id in pair_ids]
    )


def _bootstrap_difference(
    left: np.ndarray,
    right: np.ndarray,
    *,
    seed: int,
    resamples: int,
) -> dict[str, float | int]:
    if left.shape != right.shape or left.ndim != 1 or not len(left):
        raise ValueError("paired bootstrap 输入非法")
    difference = left - right
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(difference), size=(resamples, len(difference)))
    samples = difference[indices].mean(axis=1)
    return {
        "mean": float(difference.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "scene_pairs": len(difference),
    }


def _context_predictions(
    data: dict[str, np.ndarray],
    config: TransportBaselineConfig,
) -> tuple[dict[str, np.ndarray], np.ndarray, list[dict[str, Any]]]:
    train = np.where(data["split_id"] == SPLITS["train"])[0]
    test = np.where(data["split_id"] == SPLITS["test"])[0]
    train_by_pair_operation = {
        (int(data["pair_id"][index]), int(data["operation_id"][index])): index
        for index in train
    }
    train_pairs = sorted({int(data["pair_id"][index]) for index in train})
    query_only_mean = data["action"][train].mean(axis=0)
    outputs: dict[str, list[np.ndarray]] = {
        "query_only_train_mean": [],
        "raw_frame_unaligned_copy": [],
        "privileged_layout_transport": [],
        "point_layout_transport": [],
        "wrong_operation_point_transport": [],
        "target": [],
    }
    rows = []
    for query_index in test:
        query_operation = int(data["operation_id"][query_index])
        query_privileged_yaw = float(data["axis_yaw_rad"][query_index])
        query_point_yaw = _point_yaw(
            data["state"][query_index], data["tcp_pose"][query_index]
        )
        for demo_pair_id in train_pairs:
            demo_index = train_by_pair_operation[
                (demo_pair_id, query_operation)
            ]
            wrong_index = train_by_pair_operation[
                (demo_pair_id, 1 - query_operation)
            ]
            demo_privileged_yaw = float(data["axis_yaw_rad"][demo_index])
            demo_point_yaw = _point_yaw(
                data["state"][demo_index], data["tcp_pose"][demo_index]
            )
            wrong_point_yaw = _point_yaw(
                data["state"][wrong_index], data["tcp_pose"][wrong_index]
            )
            privileged_delta = _wrap_angle(
                query_privileged_yaw - demo_privileged_yaw
            )
            point_delta = _wrap_angle(query_point_yaw - demo_point_yaw)
            wrong_delta = _wrap_angle(query_point_yaw - wrong_point_yaw)
            outputs["raw_frame_unaligned_copy"].append(
                _transport(
                    data["action"][demo_index],
                    data["tcp_pose"][demo_index],
                    data["tcp_pose"][query_index],
                    0.0,
                )
            )
            outputs["query_only_train_mean"].append(query_only_mean)
            outputs["privileged_layout_transport"].append(
                _transport(
                    data["action"][demo_index],
                    data["tcp_pose"][demo_index],
                    data["tcp_pose"][query_index],
                    privileged_delta,
                )
            )
            outputs["point_layout_transport"].append(
                _transport(
                    data["action"][demo_index],
                    data["tcp_pose"][demo_index],
                    data["tcp_pose"][query_index],
                    point_delta,
                )
            )
            outputs["wrong_operation_point_transport"].append(
                _transport(
                    data["action"][wrong_index],
                    data["tcp_pose"][wrong_index],
                    data["tcp_pose"][query_index],
                    wrong_delta,
                )
            )
            outputs["target"].append(data["action"][query_index])
            rows.append(
                {
                    "query_row": int(query_index),
                    "query_pair_id": int(data["pair_id"][query_index]),
                    "operation": OPERATIONS[query_operation],
                    "demo_row": int(demo_index),
                    "demo_pair_id": demo_pair_id,
                    "wrong_demo_row": int(wrong_index),
                    "absolute_yaw_gap_degrees": abs(
                        math.degrees(privileged_delta)
                    ),
                    "point_yaw_delta_error_degrees": abs(
                        math.degrees(_wrap_angle(point_delta - privileged_delta))
                    ),
                }
            )
    stacked = {key: np.stack(value) for key, value in outputs.items()}
    large_gap = np.asarray(
        [
            row["absolute_yaw_gap_degrees"]
            >= config.large_yaw_gap_degrees
            for row in rows
        ],
        dtype=bool,
    )
    return stacked, large_gap, rows


def run(
    *,
    project_root: Path,
    config_path: Path,
    data_root: Path,
    output_path: Path,
    config: TransportBaselineConfig,
) -> None:
    if output_path.exists():
        raise FileExistsError(f"输出已存在，拒绝覆盖：{output_path}")
    report_path = data_root / "preprocess_report.json"
    preprocess = json.loads(report_path.read_text(encoding="utf-8"))
    data_path = data_root / "branch_samples.npz"
    if _sha256(data_path) != preprocess["files"]["branch_samples.npz"]:
        raise ValueError("predictor data hash 不匹配")
    with np.load(data_path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    predictions, large_gap, rows = _context_predictions(data, config)
    if int(large_gap.sum()) < config.minimum_large_gap_contexts:
        raise ValueError("large-yaw-gap contexts 数量不足")
    target = predictions.pop("target")
    metrics_all_raw = {
        name: _metrics(prediction, target)
        for name, prediction in predictions.items()
    }
    metrics_large_raw = {
        name: _metrics(prediction[large_gap], target[large_gap])
        for name, prediction in predictions.items()
    }
    query_pair_ids = np.asarray([row["query_pair_id"] for row in rows])
    scene_mse = {
        name: _scene_means(metrics["per_row_action_mse"], query_pair_ids)[1]
        for name, metrics in metrics_all_raw.items()
    }
    large_scene_mse = {
        name: _scene_means(
            metrics["per_row_action_mse"], query_pair_ids[large_gap]
        )[1]
        for name, metrics in metrics_large_raw.items()
    }
    comparisons = {
        "point_minus_raw_all": _bootstrap_difference(
            scene_mse["point_layout_transport"],
            scene_mse["raw_frame_unaligned_copy"],
            seed=config.seed,
            resamples=config.bootstrap_resamples,
        ),
        "point_minus_raw_large_gap": _bootstrap_difference(
            large_scene_mse["point_layout_transport"],
            large_scene_mse["raw_frame_unaligned_copy"],
            seed=config.seed + 1,
            resamples=config.bootstrap_resamples,
        ),
        "point_minus_wrong_all": _bootstrap_difference(
            scene_mse["point_layout_transport"],
            scene_mse["wrong_operation_point_transport"],
            seed=config.seed + 2,
            resamples=config.bootstrap_resamples,
        ),
    }
    point_mse = metrics_all_raw["point_layout_transport"]["action_mse"]
    privileged_mse = metrics_all_raw[
        "privileged_layout_transport"
    ]["action_mse"]
    point_to_privileged_ratio = point_mse / max(privileged_mse, 1e-12)
    criteria = {
        "P1_point_transport_beats_raw_all": (
            comparisons["point_minus_raw_all"]["ci95_high"] < 0.0
        ),
        "P2_point_transport_beats_raw_large_gap": (
            comparisons["point_minus_raw_large_gap"]["ci95_high"] < 0.0
        ),
        "P3_point_transport_beats_wrong_operation": (
            comparisons["point_minus_wrong_all"]["ci95_high"] < 0.0
        ),
        "P4_privileged_direction_accuracy": (
            metrics_all_raw["privileged_layout_transport"][
                "direction_accuracy"
            ]
            >= config.minimum_privileged_direction_accuracy
        ),
        "P5_point_direction_accuracy": (
            metrics_all_raw["point_layout_transport"]["direction_accuracy"]
            >= config.minimum_point_direction_accuracy
        ),
        "P6_wrong_operation_direction_accuracy": (
            metrics_all_raw["wrong_operation_point_transport"][
                "direction_accuracy"
            ]
            <= config.maximum_wrong_operation_direction_accuracy
        ),
        "P7_point_near_privileged_mse": (
            point_to_privileged_ratio
            <= config.maximum_point_to_privileged_mse_ratio
        ),
    }
    context_mse = {
        name: metrics["per_row_action_mse"]
        for name, metrics in metrics_all_raw.items()
    }
    for index, row in enumerate(rows):
        row["action_mse"] = {
            name: float(values[index]) for name, values in context_mse.items()
        }
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(project_root),
        "config": asdict(config),
        "config_sha256": _sha256(config_path),
        "preprocess_report_sha256": _sha256(report_path),
        "protocol": {
            "test_queries": int(np.sum(data["split_id"] == SPLITS["test"])),
            "train_demo_scenes_per_query": int(
                len(np.unique(data["pair_id"][data["split_id"] == SPLITS["train"]]))
            ),
            "bootstrap_unit": "held-out query scene pair",
            "model_input_excludes": ["axis_yaw_rad", "operation_id"],
        },
        "contexts": len(rows),
        "large_gap_contexts": int(large_gap.sum()),
        "metrics_all": {
            name: _strip_rows(metrics)
            for name, metrics in metrics_all_raw.items()
        },
        "metrics_large_gap": {
            name: _strip_rows(metrics)
            for name, metrics in metrics_large_raw.items()
        },
        "point_to_privileged_mse_ratio": point_to_privileged_ratio,
        "point_yaw_delta_error_degrees": {
            "median": float(
                np.median([row["point_yaw_delta_error_degrees"] for row in rows])
            ),
            "maximum": max(
                row["point_yaw_delta_error_degrees"] for row in rows
            ),
        },
        "comparisons": comparisons,
        "criteria": criteria,
        "all_criteria_passed": all(criteria.values()),
        "context_rows": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(f"{output_path.suffix}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    summary = {key: value for key, value in report.items() if key != "context_rows"}
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if not report["all_criteria_passed"]:
        raise RuntimeError("layout transport baseline 未通过全部预注册判据")


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
        config=TransportBaselineConfig.from_json(arguments.config.resolve()),
    )

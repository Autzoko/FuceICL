"""把 rotated-layout replay 转为轻量 Demo-conditioned predictor 数据。"""

from __future__ import annotations

import argparse
import atexit
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

import h5py
import numpy as np

from dev.simulator.maniskill_action_bridge import (
    controller_to_canonical_unclipped,
)
from dev.simulator.preprocess_bidirectional_slide_predictor import (
    _metadata_by_seed,
    _sample_points,
    _trajectory,
)
from dev.simulator.preprocess_maniskill_chunks import (
    _gripper_open,
    _infer_actor_label,
    _quaternion_wxyz_to_matrix,
    _segmented_points,
)


OPERATIONS = ("toward", "away")
SPLIT_IDS = {"train": 0, "val": 1, "test": 2}
STATE_FEATURES = (
    "active_in_eef_x",
    "active_in_eef_y",
    "active_in_eef_z",
    "anchor_from_active_eef_x",
    "anchor_from_active_eef_y",
    "anchor_from_active_eef_z",
    "joint_1_position",
    "joint_2_position",
    "joint_3_position",
    "joint_4_position",
    "joint_5_position",
    "joint_6_position",
    "joint_7_position",
    "joint_1_velocity",
    "joint_2_velocity",
    "joint_3_velocity",
    "joint_4_velocity",
    "joint_5_velocity",
    "joint_6_velocity",
    "joint_7_velocity",
    "gripper_open",
)


@dataclass(frozen=True)
class RotatedLayoutPredictorDataConfig:
    """冻结的 rotated-layout predictor 数据协议。"""

    schema_version: str
    seed: int
    expected_pairs: int
    action_horizon: int
    active_points: int
    anchor_points: int
    label_probe_frames: int
    minimum_label_points: int
    maximum_centroid_error_m: float
    position_limit_m: float
    rotation_scale_rad: float
    paired_observation_tolerance: float
    maximum_layout_axis_error_degrees: float

    @classmethod
    def from_json(cls, path: Path) -> "RotatedLayoutPredictorDataConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def __post_init__(self) -> None:
        if self.schema_version != "rotated-layout-slide-predictor-data-v1":
            raise ValueError("未知 rotated-layout predictor data schema")
        positive = (
            self.expected_pairs,
            self.action_horizon,
            self.active_points,
            self.anchor_points,
            self.label_probe_frames,
            self.minimum_label_points,
            self.maximum_centroid_error_m,
            self.position_limit_m,
            self.paired_observation_tolerance,
            self.maximum_layout_axis_error_degrees,
        )
        if min(positive) <= 0 or self.seed < 0:
            raise ValueError("predictor data 配置非法")
        if self.rotation_scale_rad == 0 or self.action_horizon != 6:
            raise ValueError("rotation scale 非法或 action horizon 不是 6")


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


def _sample_seed(base_seed: int, pair_id: int, role: str) -> int:
    payload = f"{base_seed}:{pair_id}:{role}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _state(
    *,
    active_center: np.ndarray,
    anchor_center: np.ndarray,
    tcp_pose: np.ndarray,
    qpos: np.ndarray,
    qvel: np.ndarray,
) -> np.ndarray:
    if qpos.shape != (9,) or qvel.shape != (9,):
        raise ValueError("Panda qpos/qvel 必须为 9D")
    rotation = _quaternion_wxyz_to_matrix(tcp_pose[3:7])
    active_in_eef = rotation.T @ (active_center - tcp_pose[:3])
    layout_in_eef = rotation.T @ (anchor_center - active_center)
    value = np.concatenate(
        (
            active_in_eef,
            layout_in_eef,
            qpos[:7],
            qvel[:7],
            [_gripper_open(qpos)],
        )
    ).astype(np.float32)
    if value.shape != (len(STATE_FEATURES),) or not np.isfinite(value).all():
        raise ValueError("predictor state shape/value 非法")
    return value


def _actions(
    *,
    normalized: np.ndarray,
    tcp_poses: np.ndarray,
    branch: int,
    config: RotatedLayoutPredictorDataConfig,
) -> np.ndarray:
    stop = branch + config.action_horizon
    if normalized.shape[0] < stop or tcp_poses.shape[0] <= stop:
        raise ValueError("branch 后 action/observation 长度不足")
    output = np.stack(
        [
            controller_to_canonical_unclipped(
                normalized[frame],
                tcp_poses[frame],
                position_limit_m=config.position_limit_m,
                rotation_scale_rad=config.rotation_scale_rad,
            )
            for frame in range(branch, stop)
        ]
    ).astype(np.float32)
    if output.shape != (config.action_horizon, 7):
        raise ValueError("canonical action chunk shape 错误")
    return output


def _angle_degrees(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    cosine = float(np.dot(left, right) / max(denominator, 1e-12))
    return math.degrees(math.acos(np.clip(cosine, -1.0, 1.0)))


def run(
    *,
    project_root: Path,
    config_path: Path,
    replay_root: Path,
    audit_path: Path,
    output_root: Path,
    config: RotatedLayoutPredictorDataConfig,
) -> None:
    if output_root.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output_root}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("summary", {}).get("all_criteria_passed", False):
        raise ValueError("replay audit 未通过，拒绝构建 predictor 数据")
    pair_rows = audit.get("pairs", [])
    if len(pair_rows) != config.expected_pairs:
        raise ValueError("replay audit pair 数量不匹配")

    h5_paths = {
        operation: replay_root
        / operation
        / "trajectory.pointcloud.pd_ee_delta_pose.physx_cpu.h5"
        for operation in OPERATIONS
    }
    json_paths = {
        operation: path.with_suffix(".json")
        for operation, path in h5_paths.items()
    }
    episodes = {
        operation: _metadata_by_seed(json_paths[operation])
        for operation in OPERATIONS
    }
    handles = {
        operation: h5py.File(h5_paths[operation], "r")
        for operation in OPERATIONS
    }
    arrays: dict[str, list[np.ndarray | int | float]] = {
        "pair_id": [],
        "seed": [],
        "split_id": [],
        "operation_id": [],
        "branch_frame": [],
        "axis_yaw_rad": [],
        "active_points": [],
        "anchor_points": [],
        "state": [],
        "tcp_pose": [],
        "action": [],
        "controller_action": [],
    }
    rows: list[dict[str, Any]] = []
    label_diagnostics = []
    axis_errors = []
    try:
        for pair in pair_rows:
            pair_id = int(pair["pair_id"])
            seed = int(pair["seed"])
            split = str(pair["split"])
            branch = int(pair["replay_branch_frame"])
            yaw = math.radians(float(pair["axis_yaw_degrees"]))
            if split not in SPLIT_IDS:
                raise ValueError(f"未知 split：{split}")
            sampled: dict[str, dict[str, np.ndarray]] = {}
            for operation_id, operation in enumerate(OPERATIONS):
                trajectory = _trajectory(
                    handles[operation], episodes[operation][seed]
                )
                xyzw = trajectory["obs/pointcloud/xyzw"]
                segmentation = trajectory["obs/pointcloud/segmentation"]
                actor_positions = {
                    role: np.asarray(
                        trajectory[f"env_states/actors/{actor}"][:, :3],
                        dtype=np.float64,
                    )
                    for role, actor in (
                        ("active", "cube"),
                        ("anchor", "layout_anchor"),
                    )
                }
                labels = {}
                diagnostics = {}
                for role in ("active", "anchor"):
                    labels[role], diagnostics[role] = _infer_actor_label(
                        xyzw=xyzw,
                        segmentation=segmentation,
                        actor_positions=actor_positions[role],
                        config=config,
                        role=role,
                    )
                if labels["active"] == labels["anchor"]:
                    raise ValueError(f"seed={seed} active/anchor label 冲突")
                points_world = {
                    role: _segmented_points(
                        np.asarray(xyzw[branch]),
                        np.asarray(segmentation[branch]),
                        labels[role],
                    )
                    for role in ("active", "anchor")
                }
                if any(
                    len(points) < config.minimum_label_points
                    for points in points_world.values()
                ):
                    raise ValueError(f"seed={seed} segmented points 不足")
                centers = {
                    role: points.mean(axis=0)
                    for role, points in points_world.items()
                }
                tcp_poses = np.asarray(
                    trajectory["obs/extra/tcp_pose"], dtype=np.float64
                )
                qpos = np.asarray(
                    trajectory["obs/agent/qpos"], dtype=np.float64
                )
                qvel = np.asarray(
                    trajectory["obs/agent/qvel"], dtype=np.float64
                )
                normalized = np.asarray(
                    trajectory["actions"], dtype=np.float64
                )
                rotation = _quaternion_wxyz_to_matrix(tcp_poses[branch, 3:7])
                point_samples = {
                    role: _sample_points(
                        points_world[role],
                        centers[role],
                        rotation,
                        count=(
                            config.active_points
                            if role == "active"
                            else config.anchor_points
                        ),
                        seed=_sample_seed(config.seed, pair_id, role),
                    )
                    for role in ("active", "anchor")
                }
                state = _state(
                    active_center=centers["active"],
                    anchor_center=centers["anchor"],
                    tcp_pose=tcp_poses[branch],
                    qpos=qpos[branch],
                    qvel=qvel[branch],
                )
                action = _actions(
                    normalized=normalized,
                    tcp_poses=tcp_poses,
                    branch=branch,
                    config=config,
                )
                sampled[operation] = {
                    "active_points": point_samples["active"],
                    "anchor_points": point_samples["anchor"],
                    "state": state,
                    "tcp_pose": tcp_poses[branch].astype(np.float32),
                    "action": action,
                    "controller_action": normalized[
                        branch : branch + config.action_horizon
                    ].astype(np.float32),
                }
                arrays["pair_id"].append(pair_id)
                arrays["seed"].append(seed)
                arrays["split_id"].append(SPLIT_IDS[split])
                arrays["operation_id"].append(operation_id)
                arrays["branch_frame"].append(branch)
                arrays["axis_yaw_rad"].append(yaw)
                for name in (
                    "active_points",
                    "anchor_points",
                    "state",
                    "tcp_pose",
                    "action",
                    "controller_action",
                ):
                    arrays[name].append(sampled[operation][name])
                estimated_axis_world = rotation @ state[3:6]
                privileged_axis = np.asarray([math.cos(yaw), math.sin(yaw), 0.0])
                axis_error = _angle_degrees(
                    estimated_axis_world, privileged_axis
                )
                axis_errors.append(axis_error)
                rows.append(
                    {
                        "row": len(rows),
                        "pair_id": pair_id,
                        "seed": seed,
                        "split": split,
                        "operation": operation,
                        "branch_frame": branch,
                        "active_label": labels["active"],
                        "anchor_label": labels["anchor"],
                        "layout_axis_error_degrees": axis_error,
                    }
                )
                label_diagnostics.append(
                    {
                        "pair_id": pair_id,
                        "operation": operation,
                        "active": diagnostics["active"],
                        "anchor": diagnostics["anchor"],
                    }
                )

            for name in ("active_points", "anchor_points", "state", "tcp_pose"):
                error = float(
                    np.max(
                        np.abs(
                            sampled["toward"][name].astype(np.float64)
                            - sampled["away"][name].astype(np.float64)
                        )
                    )
                )
                if error > config.paired_observation_tolerance:
                    raise ValueError(
                        f"pair={pair_id} {name} paired error={error:.3e}"
                    )
    finally:
        for handle in handles.values():
            handle.close()

    integer_names = {
        "pair_id",
        "seed",
        "split_id",
        "operation_id",
        "branch_frame",
    }
    stacked = {
        name: np.asarray(values, dtype=np.int64)
        if name in integer_names
        else np.asarray(values, dtype=np.float32)
        if name == "axis_yaw_rad"
        else np.stack(values)
        for name, values in arrays.items()
    }
    if len(rows) != 2 * config.expected_pairs:
        raise ValueError("predictor rows 数量不匹配")
    paired_errors = {
        name: max(
            float(
                np.max(
                    np.abs(
                        stacked[name][2 * index]
                        - stacked[name][2 * index + 1]
                    )
                )
            )
            for index in range(config.expected_pairs)
        )
        for name in ("active_points", "anchor_points", "state", "tcp_pose")
    }
    lower_bounds = np.asarray(
        [
            np.mean(
                (
                    stacked["action"][2 * index]
                    - stacked["action"][2 * index + 1]
                )
                ** 2
            )
            / 4.0
            for index in range(config.expected_pairs)
        ]
    )
    maximum_axis_error = max(axis_errors)
    if maximum_axis_error > config.maximum_layout_axis_error_degrees:
        raise ValueError(
            f"point-derived layout axis max error={maximum_axis_error:.3f} deg"
        )

    temporary = output_root.with_name(
        f".{output_root.name}.incomplete-{os.getpid()}"
    )
    temporary.mkdir(parents=True)

    def cleanup() -> None:
        shutil.rmtree(temporary, ignore_errors=True)

    atexit.register(cleanup)
    data_path = temporary / "branch_samples.npz"
    with data_path.open("wb") as stream:
        np.savez_compressed(stream, **stacked)
    index_path = temporary / "index.jsonl"
    with index_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(project_root),
        "config": asdict(config),
        "config_sha256": _sha256(config_path),
        "replay_audit_sha256": _sha256(audit_path),
        "state_features": STATE_FEATURES,
        "model_inputs": {
            "active_points": "segmented active shape centered in current EEF",
            "anchor_points": "segmented anchor shape centered in current EEF",
            "state": "observable geometry and robot state; no goal/operation/yaw",
            "tcp_pose": "current robot observation for action frame conversion",
            "demo_action": "retrieved H6 canonical action chunk",
        },
        "audit_only_fields": ["axis_yaw_rad", "operation_id", "split_id"],
        "rows": len(rows),
        "pairs": config.expected_pairs,
        "split_row_counts": {
            split: int(np.sum(stacked["split_id"] == split_id))
            for split, split_id in SPLIT_IDS.items()
        },
        "paired_observation_max_abs_error": paired_errors,
        "layout_axis_error_degrees": {
            "maximum": maximum_axis_error,
            "median": float(np.median(axis_errors)),
        },
        "canonical_observation_only_mse_lower_bound": {
            "minimum": float(lower_bounds.min()),
            "median": float(np.median(lower_bounds)),
            "maximum": float(lower_bounds.max()),
        },
        "label_probe_error_max_m": max(
            diagnostic[role]["selected_probe_error_max_m"]
            for diagnostic in label_diagnostics
            for role in ("active", "anchor")
        ),
        "files": {
            "branch_samples.npz": _sha256(data_path),
            "index.jsonl": _sha256(index_path),
        },
    }
    report_path = temporary / "preprocess_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_root)
    atexit.unregister(cleanup)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(
        project_root=arguments.project_root.resolve(),
        config_path=arguments.config.resolve(),
        replay_root=arguments.replay_root.resolve(),
        audit_path=arguments.audit.resolve(),
        output_root=arguments.output_root.resolve(),
        config=RotatedLayoutPredictorDataConfig.from_json(
            arguments.config.resolve()
        ),
    )

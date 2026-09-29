"""把严格配对 replay 转为无 goal 泄漏的 predictor branch 数据。"""

from __future__ import annotations

import argparse
import atexit
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
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
from dev.simulator.preprocess_maniskill_chunks import (
    _gripper_open,
    _infer_actor_label,
    _quaternion_wxyz_to_matrix,
    _segmented_points,
    _shape_sigmas,
)


OPERATIONS = ("left", "right")
SPLIT_IDS = {"train": 0, "val": 1, "test": 2}
STATE_FEATURES = (
    "object_in_eef_x",
    "object_in_eef_y",
    "object_in_eef_z",
    "shape_sigma_1",
    "shape_sigma_2",
    "shape_sigma_3",
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
class PredictorDataConfig:
    """冻结的 paired predictor 数据协议。"""

    schema_version: str
    seed: int
    expected_pairs: int
    action_horizon: int
    points_per_object: int
    label_probe_frames: int
    minimum_label_points: int
    maximum_centroid_error_m: float
    position_limit_m: float
    rotation_scale_rad: float
    paired_observation_tolerance: float

    @classmethod
    def from_json(cls, path: Path) -> "PredictorDataConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def __post_init__(self) -> None:
        if self.schema_version != "bidirectional-slide-predictor-data-v1":
            raise ValueError("未知 paired predictor data schema")
        positive = (
            self.expected_pairs,
            self.action_horizon,
            self.points_per_object,
            self.label_probe_frames,
            self.minimum_label_points,
            self.maximum_centroid_error_m,
            self.position_limit_m,
            self.paired_observation_tolerance,
        )
        if min(positive) <= 0 or self.seed < 0 or self.rotation_scale_rad == 0:
            raise ValueError("predictor data 配置非法")
        if self.action_horizon != 6:
            raise ValueError("v1 固定 H=6")


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


def _metadata_by_seed(path: Path) -> dict[int, dict[str, Any]]:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    episodes = {
        int(episode["reset_kwargs"]["seed"]): episode
        for episode in metadata.get("episodes", [])
    }
    if len(episodes) != len(metadata.get("episodes", [])):
        raise ValueError(f"{path} reset seeds 不唯一")
    return episodes


def _trajectory(
    handle: h5py.File,
    episode: dict[str, Any],
) -> h5py.Group:
    key = f"traj_{int(episode['episode_id'])}"
    if key not in handle:
        raise KeyError(f"HDF5 缺少 {key}")
    value = handle[key]
    if not isinstance(value, h5py.Group):
        raise TypeError(f"{key} 不是 group")
    return value


def _sample_points(
    points_world: np.ndarray,
    center_world: np.ndarray,
    rotation_world_from_eef: np.ndarray,
    *,
    count: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    indices = rng.choice(
        len(points_world),
        size=count,
        replace=len(points_world) < count,
    )
    centered_world = points_world[indices] - center_world
    return (centered_world @ rotation_world_from_eef).astype(np.float32)


def _state(
    *,
    active_points: np.ndarray,
    tcp_pose: np.ndarray,
    qpos: np.ndarray,
    qvel: np.ndarray,
) -> np.ndarray:
    if qpos.shape != (9,) or qvel.shape != (9,):
        raise ValueError("Panda qpos/qvel 必须为 9D")
    rotation = _quaternion_wxyz_to_matrix(tcp_pose[3:7])
    center = active_points.mean(axis=0)
    object_in_eef = rotation.T @ (center - tcp_pose[:3])
    value = np.concatenate(
        (
            object_in_eef,
            _shape_sigmas(active_points),
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
    config: PredictorDataConfig,
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


def _sample_seed(base_seed: int, pair_id: int) -> int:
    payload = f"{base_seed}:{pair_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def run(
    *,
    project_root: Path,
    config_path: Path,
    replay_root: Path,
    audit_path: Path,
    output_root: Path,
    config: PredictorDataConfig,
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
    rows: list[dict[str, Any]] = []
    arrays: dict[str, list[np.ndarray | int]] = {
        "pair_id": [],
        "seed": [],
        "split_id": [],
        "operation_id": [],
        "branch_frame": [],
        "points": [],
        "state": [],
        "action": [],
        "controller_action": [],
    }
    label_diagnostics = []
    try:
        for pair in pair_rows:
            pair_id = int(pair["pair_id"])
            seed = int(pair["seed"])
            split = str(pair["split"])
            branch = int(pair["replay_branch_frame"])
            if split not in SPLIT_IDS:
                raise ValueError(f"未知 split：{split}")
            sampled: dict[str, dict[str, np.ndarray]] = {}
            for operation_id, operation in enumerate(OPERATIONS):
                trajectory = _trajectory(
                    handles[operation],
                    episodes[operation][seed],
                )
                xyzw = trajectory["obs/pointcloud/xyzw"]
                segmentation = trajectory["obs/pointcloud/segmentation"]
                actor_positions = np.asarray(
                    trajectory["env_states/actors/cube"][:, :3],
                    dtype=np.float64,
                )
                active_label, label_diagnostic = _infer_actor_label(
                    xyzw=xyzw,
                    segmentation=segmentation,
                    actor_positions=actor_positions,
                    config=config,
                    role="active cube",
                )
                points_world = _segmented_points(
                    np.asarray(xyzw[branch]),
                    np.asarray(segmentation[branch]),
                    active_label,
                )
                if len(points_world) < config.minimum_label_points:
                    raise ValueError(f"seed={seed} cube points 不足")
                tcp_poses = np.asarray(
                    trajectory["obs/extra/tcp_pose"],
                    dtype=np.float64,
                )
                qpos = np.asarray(
                    trajectory["obs/agent/qpos"],
                    dtype=np.float64,
                )
                qvel = np.asarray(
                    trajectory["obs/agent/qvel"],
                    dtype=np.float64,
                )
                normalized_actions = np.asarray(
                    trajectory["actions"],
                    dtype=np.float64,
                )
                rotation = _quaternion_wxyz_to_matrix(tcp_poses[branch, 3:7])
                center = points_world.mean(axis=0)
                point_sample = _sample_points(
                    points_world,
                    center,
                    rotation,
                    count=config.points_per_object,
                    seed=_sample_seed(config.seed, pair_id),
                )
                state = _state(
                    active_points=points_world,
                    tcp_pose=tcp_poses[branch],
                    qpos=qpos[branch],
                    qvel=qvel[branch],
                )
                action = _actions(
                    normalized=normalized_actions,
                    tcp_poses=tcp_poses,
                    branch=branch,
                    config=config,
                )
                controller = normalized_actions[
                    branch : branch + config.action_horizon
                ].astype(np.float32)
                sampled[operation] = {
                    "points": point_sample,
                    "state": state,
                    "action": action,
                    "controller_action": controller,
                }
                arrays["pair_id"].append(pair_id)
                arrays["seed"].append(seed)
                arrays["split_id"].append(SPLIT_IDS[split])
                arrays["operation_id"].append(operation_id)
                arrays["branch_frame"].append(branch)
                for name in ("points", "state", "action", "controller_action"):
                    arrays[name].append(sampled[operation][name])
                rows.append(
                    {
                        "row": len(rows),
                        "pair_id": pair_id,
                        "seed": seed,
                        "split": split,
                        "operation": operation,
                        "branch_frame": branch,
                        "active_label": active_label,
                    }
                )
                label_diagnostics.append(
                    {
                        "pair_id": pair_id,
                        "operation": operation,
                        **label_diagnostic,
                    }
                )

            for name in ("points", "state"):
                error = float(
                    np.max(
                        np.abs(
                            sampled["left"][name].astype(np.float64)
                            - sampled["right"][name].astype(np.float64)
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

    stacked = {
        name: np.asarray(values, dtype=np.int64)
        if name in {
            "pair_id",
            "seed",
            "split_id",
            "operation_id",
            "branch_frame",
        }
        else np.stack(values)
        for name, values in arrays.items()
    }
    if len(rows) != 2 * config.expected_pairs:
        raise ValueError("predictor rows 数量不匹配")
    pair_errors = {
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
        for name in ("points", "state")
    }
    action_lower_bounds = np.asarray(
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
    split_counts = {
        split: int(np.sum(stacked["split_id"] == split_id))
        for split, split_id in SPLIT_IDS.items()
    }

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
        "stored_inputs": {
            "points": (
                "segmented cube points centered at observed centroid in EEF frame"
            ),
            "state": "current observation only; no goal/operation field",
            "action": "per-step canonical EEF physical controller command",
            "controller_action": "original normalized pd_ee_delta_pose command",
        },
        "rows": len(rows),
        "pairs": config.expected_pairs,
        "split_row_counts": split_counts,
        "paired_observation_max_abs_error": pair_errors,
        "canonical_observation_only_mse_lower_bound": {
            "minimum": float(action_lower_bounds.min()),
            "median": float(np.median(action_lower_bounds)),
            "maximum": float(action_lower_bounds.max()),
        },
        "active_label_probe_error_max_m": max(
            item["selected_probe_error_max_m"] for item in label_diagnostics
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
        config=PredictorDataConfig.from_json(arguments.config.resolve()),
    )

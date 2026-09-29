"""把已审计 RotatedLayoutSlide replay 转为固定多进度 H6 chunks。"""

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

from dev.simulator.preprocess_bidirectional_slide_predictor import (
    _metadata_by_seed,
    _sample_points,
    _trajectory,
)
from dev.simulator.preprocess_maniskill_chunks import (
    _infer_actor_label,
    _quaternion_wxyz_to_matrix,
    _segmented_points,
)
from dev.simulator.preprocess_rotated_layout_slide_predictor import (
    OPERATIONS,
    SPLIT_IDS,
    STATE_FEATURES,
    _actions,
    _angle_degrees,
    _sha256,
    _state,
)


@dataclass(frozen=True)
class ProgressChunkConfig:
    """冻结的多进度 chunk 数据协议。"""

    schema_version: str
    seed: int
    expected_pairs: int
    action_horizon: int
    progress_samples: int
    active_points: int
    anchor_points: int
    label_probe_frames: int
    minimum_label_points: int
    maximum_centroid_error_m: float
    position_limit_m: float
    rotation_scale_rad: float
    maximum_layout_axis_error_degrees: float

    @classmethod
    def from_json(cls, path: Path) -> "ProgressChunkConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def __post_init__(self) -> None:
        if self.schema_version != "rotated-layout-slide-progress-chunks-v1":
            raise ValueError("未知 progress chunk schema")
        positive = (
            self.expected_pairs,
            self.action_horizon,
            self.progress_samples,
            self.active_points,
            self.anchor_points,
            self.label_probe_frames,
            self.minimum_label_points,
            self.maximum_centroid_error_m,
            self.position_limit_m,
            self.maximum_layout_axis_error_degrees,
        )
        if min(positive) <= 0 or self.seed < 0:
            raise ValueError("progress chunk 配置非法")
        if self.rotation_scale_rad == 0 or self.action_horizon != 6:
            raise ValueError("controller scale 非法或 horizon 不是 6")
        if self.progress_samples < 2:
            raise ValueError("progress_samples 至少为 2")


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _sample_seed(
    base_seed: int,
    pair_id: int,
    operation: str,
    progress_index: int,
    role: str,
) -> int:
    payload = (
        f"{base_seed}:{pair_id}:{operation}:{progress_index}:{role}".encode()
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _frames(branch: int, action_steps: int, config: ProgressChunkConfig) -> list[int]:
    last = action_steps - config.action_horizon
    if last < branch:
        raise ValueError("branch 后没有完整 action chunk")
    frames = np.rint(
        np.linspace(branch, last, config.progress_samples)
    ).astype(np.int64)
    if len(np.unique(frames)) != config.progress_samples:
        raise ValueError("trajectory 太短，progress frames 出现重复")
    return frames.tolist()


def _paths(root: Path) -> dict[str, dict[str, Path]]:
    stem = "trajectory.pointcloud.pd_ee_delta_pose.physx_cpu"
    return {
        operation: {
            "h5": root / operation / f"{stem}.h5",
            "json": root / operation / f"{stem}.json",
        }
        for operation in OPERATIONS
    }


def run(
    *,
    project_root: Path,
    config_path: Path,
    replay_root: Path,
    output_root: Path,
    config: ProgressChunkConfig,
) -> None:
    if output_root.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output_root}")
    audit_path = replay_root / "replay_audit_report.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("summary", {}).get("all_criteria_passed", False):
        raise ValueError("replay audit 未通过")
    pair_rows = audit.get("pairs", [])
    if len(pair_rows) != config.expected_pairs:
        raise ValueError("replay audit pair 数量不匹配")

    paths = _paths(replay_root)
    for operation in OPERATIONS:
        expected = audit["files"][operation]["replay"]
        if _sha256(paths[operation]["h5"]) != expected["h5_sha256"]:
            raise ValueError(f"{operation} replay HDF5 hash 不匹配")
        if _sha256(paths[operation]["json"]) != expected["json_sha256"]:
            raise ValueError(f"{operation} replay JSON hash 不匹配")
    episodes = {
        operation: _metadata_by_seed(paths[operation]["json"])
        for operation in OPERATIONS
    }
    handles = {
        operation: h5py.File(paths[operation]["h5"], "r")
        for operation in OPERATIONS
    }
    arrays: dict[str, list[Any]] = {
        name: []
        for name in (
            "pair_id",
            "seed",
            "split_id",
            "operation_id",
            "progress_index",
            "progress_fraction",
            "frame",
            "active_points",
            "anchor_points",
            "state",
            "tcp_pose",
            "action",
        )
    }
    index_rows = []
    axis_errors = []
    point_counts = []
    centroid_errors = []
    try:
        for pair in pair_rows:
            pair_id = int(pair["pair_id"])
            seed = int(pair["seed"])
            split = str(pair["split"])
            branch = int(pair["replay_branch_frame"])
            if split not in SPLIT_IDS:
                raise ValueError(f"未知 split：{split}")
            for operation_id, operation in enumerate(OPERATIONS):
                trajectory = _trajectory(
                    handles[operation], episodes[operation][seed]
                )
                xyzw = np.asarray(trajectory["obs/pointcloud/xyzw"])
                segmentation = np.asarray(
                    trajectory["obs/pointcloud/segmentation"]
                )
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
                label_diagnostics = {}
                for role in ("active", "anchor"):
                    labels[role], label_diagnostics[role] = _infer_actor_label(
                        xyzw=xyzw,
                        segmentation=segmentation,
                        actor_positions=actor_positions[role],
                        config=config,
                        role=role,
                    )
                    centroid_errors.append(
                        float(
                            label_diagnostics[role][
                                "selected_probe_error_max_m"
                            ]
                        )
                    )
                if labels["active"] == labels["anchor"]:
                    raise ValueError(f"seed={seed} segmentation label 冲突")
                tcp_poses = np.asarray(
                    trajectory["obs/extra/tcp_pose"], dtype=np.float64
                )
                qpos = np.asarray(
                    trajectory["obs/agent/qpos"], dtype=np.float64
                )
                qvel = np.asarray(
                    trajectory["obs/agent/qvel"], dtype=np.float64
                )
                actions = np.asarray(trajectory["actions"], dtype=np.float64)
                frames = _frames(branch, len(actions), config)
                for progress_index, frame in enumerate(frames):
                    points_world = {
                        role: _segmented_points(
                            xyzw[frame], segmentation[frame], labels[role]
                        )
                        for role in ("active", "anchor")
                    }
                    counts = {role: len(value) for role, value in points_world.items()}
                    point_counts.extend(counts.values())
                    if min(counts.values()) < config.minimum_label_points:
                        raise ValueError(
                            f"seed={seed} operation={operation} frame={frame} "
                            f"segmented points 不足：{counts}"
                        )
                    centers = {
                        role: value.mean(axis=0)
                        for role, value in points_world.items()
                    }
                    rotation = _quaternion_wxyz_to_matrix(
                        tcp_poses[frame, 3:7]
                    )
                    sampled = {
                        role: _sample_points(
                            points_world[role],
                            centers[role],
                            rotation,
                            count=(
                                config.active_points
                                if role == "active"
                                else config.anchor_points
                            ),
                            seed=_sample_seed(
                                config.seed,
                                pair_id,
                                operation,
                                progress_index,
                                role,
                            ),
                        )
                        for role in ("active", "anchor")
                    }
                    state = _state(
                        active_center=centers["active"],
                        anchor_center=centers["anchor"],
                        tcp_pose=tcp_poses[frame],
                        qpos=qpos[frame],
                        qvel=qvel[frame],
                    )
                    chunk = _actions(
                        normalized=actions,
                        tcp_poses=tcp_poses,
                        branch=frame,
                        config=config,
                    )
                    privileged_axis = (
                        actor_positions["anchor"][frame]
                        - actor_positions["active"][frame]
                    )
                    privileged_axis[2] = 0.0
                    estimated_axis = rotation @ state[3:6]
                    axis_error = _angle_degrees(
                        estimated_axis, privileged_axis
                    )
                    axis_errors.append(axis_error)
                    progress_fraction = progress_index / (
                        config.progress_samples - 1
                    )
                    scalar_values = {
                        "pair_id": pair_id,
                        "seed": seed,
                        "split_id": SPLIT_IDS[split],
                        "operation_id": operation_id,
                        "progress_index": progress_index,
                        "progress_fraction": progress_fraction,
                        "frame": frame,
                    }
                    for name, value in scalar_values.items():
                        arrays[name].append(value)
                    arrays["active_points"].append(sampled["active"])
                    arrays["anchor_points"].append(sampled["anchor"])
                    arrays["state"].append(state)
                    arrays["tcp_pose"].append(
                        tcp_poses[frame].astype(np.float32)
                    )
                    arrays["action"].append(chunk)
                    index_rows.append(
                        {
                            "row": len(index_rows),
                            "pair_id": pair_id,
                            "seed": seed,
                            "split": split,
                            "operation": operation,
                            "progress_index": progress_index,
                            "progress_fraction": progress_fraction,
                            "frame": frame,
                            "active_label": labels["active"],
                            "anchor_label": labels["anchor"],
                            "active_points_visible": counts["active"],
                            "anchor_points_visible": counts["anchor"],
                            "layout_axis_error_degrees": axis_error,
                        }
                    )
    finally:
        for handle in handles.values():
            handle.close()

    expected_rows = (
        config.expected_pairs * len(OPERATIONS) * config.progress_samples
    )
    if len(index_rows) != expected_rows:
        raise ValueError("progress chunk rows 数量不匹配")
    maximum_axis_error = max(axis_errors)
    if maximum_axis_error > config.maximum_layout_axis_error_degrees:
        raise ValueError(
            f"point-derived layout axis max error={maximum_axis_error:.3f} deg"
        )
    integer_names = {
        "pair_id",
        "seed",
        "split_id",
        "operation_id",
        "progress_index",
        "frame",
    }
    stacked = {
        name: np.asarray(values, dtype=np.int64)
        if name in integer_names
        else np.asarray(values, dtype=np.float32)
        if name == "progress_fraction"
        else np.stack(values)
        for name, values in arrays.items()
    }

    temporary = output_root.with_name(
        f".{output_root.name}.incomplete-{os.getpid()}"
    )
    temporary.mkdir(parents=True)

    def cleanup() -> None:
        shutil.rmtree(temporary, ignore_errors=True)

    atexit.register(cleanup)
    data_path = temporary / "progress_chunks.npz"
    with data_path.open("wb") as stream:
        np.savez_compressed(stream, **stacked)
    index_path = temporary / "index.jsonl"
    with index_path.open("w", encoding="utf-8") as stream:
        for row in index_rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(project_root),
        "config": asdict(config),
        "config_sha256": _sha256(config_path),
        "replay_audit_sha256": _sha256(audit_path),
        "state_features": STATE_FEATURES,
        "audit_only_fields": [
            "operation_id",
            "split_id",
            "progress_index",
            "progress_fraction",
        ],
        "protocol": {
            "sampling": "fixed normalized progress from branch to last H6 start",
            "test_split_used": False,
            "rows_per_trajectory": config.progress_samples,
        },
        "rows": expected_rows,
        "pairs": config.expected_pairs,
        "split_row_counts": {
            split: int(np.sum(stacked["split_id"] == split_id))
            for split, split_id in SPLIT_IDS.items()
        },
        "visible_point_count": {
            "minimum": min(point_counts),
            "median": float(np.median(point_counts)),
        },
        "label_probe_error_max_m": max(centroid_errors),
        "layout_axis_error_degrees": {
            "median": float(np.median(axis_errors)),
            "maximum": maximum_axis_error,
        },
        "files": {},
    }
    report["files"] = {
        "progress_chunks.npz": _sha256(data_path),
        "index.jsonl": _sha256(index_path),
    }
    report_path = temporary / "preprocess_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_root)
    atexit.unregister(cleanup)
    print(
        json.dumps(
            {
                "rows": expected_rows,
                "split_row_counts": report["split_row_counts"],
                "visible_point_count": report["visible_point_count"],
                "layout_axis_error_degrees": report[
                    "layout_axis_error_degrees"
                ],
                "files": report["files"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    resolved_config = arguments.config.resolve()
    run(
        project_root=arguments.project_root.resolve(),
        config_path=resolved_config,
        replay_root=arguments.replay_root.resolve(),
        output_root=arguments.output_root.resolve(),
        config=ProgressChunkConfig.from_json(resolved_config),
    )

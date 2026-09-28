"""审计 ManiSkill 官方 replay 生成的 pointcloud trajectory。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(group: h5py.Group, path: str) -> h5py.Dataset:
    value: Any = group
    for component in path.split("/"):
        if component not in value:
            raise KeyError(f"{group.name} 缺少 {path}")
        value = value[component]
    if not isinstance(value, h5py.Dataset):
        raise TypeError(f"{group.name}/{path} 不是 dataset")
    return value


def _sample_frames(length: int) -> tuple[int, ...]:
    if length <= 0:
        return ()
    return tuple(sorted({0, length // 2, length - 1}))


def run(h5_path: Path, json_path: Path, output_path: Path) -> None:
    if output_path.exists():
        raise FileExistsError(f"输出已存在，拒绝覆盖：{output_path}")
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    episodes = metadata.get("episodes", [])
    episode_successes = [bool(episode.get("success", False)) for episode in episodes]

    trajectories: list[dict[str, Any]] = []
    all_actions = []
    with h5py.File(h5_path, "r") as handle:
        keys = sorted(key for key in handle if key.startswith("traj_"))
        if not keys:
            raise ValueError("HDF5 不含 trajectory groups")
        if len(keys) != len(episodes):
            raise ValueError(
                f"HDF5 trajectories={len(keys)}，metadata episodes={len(episodes)}"
            )
        for key in keys:
            trajectory = handle[key]
            actions = np.asarray(_require(trajectory, "actions"))
            xyzw = _require(trajectory, "obs/pointcloud/xyzw")
            rgb = _require(trajectory, "obs/pointcloud/rgb")
            segmentation = _require(
                trajectory,
                "obs/pointcloud/segmentation",
            )
            tcp_pose = _require(trajectory, "obs/extra/tcp_pose")
            goal_pos = _require(trajectory, "obs/extra/goal_pos")
            success = np.asarray(_require(trajectory, "success"), dtype=bool)
            if actions.ndim != 2 or actions.shape[1] != 7:
                raise ValueError(f"{key} action shape 错误：{actions.shape}")
            if xyzw.shape[-1] != 4 or rgb.shape[-1] != 3:
                raise ValueError(f"{key} pointcloud/rgb shape 错误")
            if segmentation.shape[-1] != 1:
                raise ValueError(f"{key} segmentation shape 错误")
            observation_length = xyzw.shape[0]
            if observation_length not in (len(actions), len(actions) + 1):
                raise ValueError(
                    f"{key} obs/action 时序不一致：{observation_length}/{len(actions)}"
                )
            if not (
                rgb.shape[0]
                == segmentation.shape[0]
                == tcp_pose.shape[0]
                == goal_pos.shape[0]
                == observation_length
            ):
                raise ValueError(f"{key} observation modalities 长度不一致")
            if not np.isfinite(actions).all():
                raise ValueError(f"{key} actions 含 NaN/Inf")

            valid_counts = []
            labels: set[int] = set()
            for frame in _sample_frames(observation_length):
                points = np.asarray(xyzw[frame])
                valid = np.abs(points[..., 3]) > 0.5
                if valid.any() and not np.isfinite(points[..., :3][valid]).all():
                    raise ValueError(f"{key} frame {frame} pointcloud 含 NaN/Inf")
                valid_counts.append(int(valid.sum()))
                frame_labels = np.asarray(segmentation[frame])[..., 0]
                labels.update(int(value) for value in np.unique(frame_labels[valid]))
            all_actions.append(actions)
            trajectories.append(
                {
                    "trajectory": key,
                    "actions": len(actions),
                    "observations": observation_length,
                    "point_count": int(xyzw.shape[-2]),
                    "sampled_valid_points_min": min(valid_counts),
                    "sampled_valid_points_max": max(valid_counts),
                    "sampled_segmentation_labels": sorted(labels),
                    "terminal_success": bool(success[-1]) if success.size else False,
                }
            )

    action_values = np.concatenate(all_actions, axis=0)
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "h5_path": str(h5_path),
        "json_path": str(json_path),
        "h5_sha256": _sha256(h5_path),
        "json_sha256": _sha256(json_path),
        "env_info": metadata.get("env_info", {}),
        "commit_info": metadata.get("commit_info", {}),
        "num_episodes": len(episodes),
        "metadata_success_rate": (
            float(np.mean(episode_successes)) if episode_successes else None
        ),
        "terminal_success_rate": float(
            np.mean([record["terminal_success"] for record in trajectories])
        ),
        "action": {
            "shape_last_dim": int(action_values.shape[1]),
            "min": action_values.min(axis=0).tolist(),
            "max": action_values.max(axis=0).tolist(),
            "finite": bool(np.isfinite(action_values).all()),
        },
        "observation_length_offsets": sorted(
            {
                record["observations"] - record["actions"]
                for record in trajectories
            }
        ),
        "point_count_values": sorted(
            {record["point_count"] for record in trajectories}
        ),
        "sampled_valid_points_min": min(
            record["sampled_valid_points_min"] for record in trajectories
        ),
        "sampled_valid_points_max": max(
            record["sampled_valid_points_max"] for record in trajectories
        ),
        "sampled_segmentation_labels": sorted(
            {
                label
                for record in trajectories
                for label in record["sampled_segmentation_labels"]
            }
        ),
        "trajectories": trajectories,
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
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    run(
        h5_path=arguments.h5.resolve(),
        json_path=arguments.json.resolve(),
        output_path=arguments.output.resolve(),
    )


if __name__ == "__main__":
    main()

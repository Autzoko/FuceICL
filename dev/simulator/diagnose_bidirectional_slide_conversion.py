"""比较成功/失败 BidirectionalSlide controller conversion 的终态。"""

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


def _dataset(group: h5py.Group, path: str) -> h5py.Dataset:
    value: Any = group
    for component in path.split("/"):
        if component not in value:
            raise KeyError(f"{group.name} 缺少 {path}")
        value = value[component]
    if not isinstance(value, h5py.Dataset):
        raise TypeError(f"{group.name}/{path} 不是 dataset")
    return value


def _episodes_by_seed(metadata: dict[str, Any]) -> dict[int, dict[str, Any]]:
    episodes = {
        int(episode["reset_kwargs"]["seed"]): episode
        for episode in metadata.get("episodes", [])
    }
    if len(episodes) != len(metadata.get("episodes", [])):
        raise ValueError("episode reset seeds 不唯一")
    return episodes


def _trajectory(handle: h5py.File, episode_id: int) -> h5py.Group:
    key = f"traj_{episode_id}"
    if key not in handle:
        raise KeyError(f"HDF5 缺少 {key}")
    value = handle[key]
    if not isinstance(value, h5py.Group):
        raise TypeError(f"{key} 不是 group")
    return value


def _goal_diagnostics(trajectory: h5py.Group) -> dict[str, float]:
    cube = np.asarray(
        _dataset(trajectory, "env_states/actors/cube"),
        dtype=np.float64,
    )
    goal = np.asarray(
        _dataset(trajectory, "env_states/actors/goal_region"),
        dtype=np.float64,
    )
    if cube.shape[0] != goal.shape[0] or cube.shape[1] < 10:
        raise ValueError("actor state shape 非法")
    distances = np.linalg.norm(cube[:, :2] - goal[:, :2], axis=1)
    speeds = np.linalg.norm(cube[:, 7:10], axis=1)
    return {
        "terminal_object_goal_xy_distance_m": float(distances[-1]),
        "minimum_object_goal_xy_distance_m": float(distances.min()),
        "terminal_object_speed_mps": float(speeds[-1]),
        "maximum_object_speed_mps": float(speeds.max()),
    }


def _replay_diagnostics(trajectory: h5py.Group) -> dict[str, Any]:
    actions = np.asarray(_dataset(trajectory, "actions"), dtype=np.float64)
    success = np.asarray(_dataset(trajectory, "success"), dtype=bool)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError("replay actions 必须为 [T,7]")
    translation_saturated = np.any(np.abs(actions[:, :3]) >= 1.0 - 1e-6, axis=1)
    rotation_saturated = np.linalg.norm(actions[:, 3:6], axis=1) >= 1.0 - 1e-6
    return {
        "action_steps": len(actions),
        "terminal_success": bool(success.size and success[-1]),
        "translation_saturation_fraction": float(translation_saturated.mean()),
        "rotation_saturation_fraction": float(rotation_saturated.mean()),
        "translation_action_abs_max": float(np.abs(actions[:, :3]).max()),
        "rotation_action_norm_max": float(
            np.linalg.norm(actions[:, 3:6], axis=1).max()
        ),
        **_goal_diagnostics(trajectory),
    }


def run(
    *,
    source_h5: Path,
    source_json: Path,
    replay_h5: Path,
    replay_json: Path,
    output_path: Path,
) -> None:
    if output_path.exists():
        raise FileExistsError(f"输出已存在，拒绝覆盖：{output_path}")
    source_metadata = json.loads(source_json.read_text(encoding="utf-8"))
    replay_metadata = json.loads(replay_json.read_text(encoding="utf-8"))
    source_episodes = _episodes_by_seed(source_metadata)
    replay_episodes = _episodes_by_seed(replay_metadata)
    if source_episodes.keys() != replay_episodes.keys():
        raise ValueError("source/replay episode seeds 不一致")

    rows = []
    with h5py.File(source_h5, "r") as source, h5py.File(replay_h5, "r") as replay:
        for seed, source_episode in source_episodes.items():
            replay_episode = replay_episodes[seed]
            source_episode_id = int(source_episode["episode_id"])
            replay_episode_id = int(replay_episode["episode_id"])
            source_trajectory = _trajectory(source, source_episode_id)
            replay_trajectory = _trajectory(replay, replay_episode_id)
            source_success = np.asarray(
                _dataset(source_trajectory, "success"),
                dtype=bool,
            )
            rows.append(
                {
                    "source_episode_id": source_episode_id,
                    "replay_episode_id": replay_episode_id,
                    "seed": seed,
                    "source": {
                        "action_steps": len(_dataset(source_trajectory, "actions")),
                        "terminal_success": bool(
                            source_success.size and source_success[-1]
                        ),
                        **_goal_diagnostics(source_trajectory),
                    },
                    "replay": _replay_diagnostics(replay_trajectory),
                }
            )
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "diagnostic only; allow_failure saves failed replay and does not "
            "convert it into a successful demonstration"
        ),
        "files": {
            "source_h5_sha256": _sha256(source_h5),
            "source_json_sha256": _sha256(source_json),
            "replay_h5_sha256": _sha256(replay_h5),
            "replay_json_sha256": _sha256(replay_json),
        },
        "episodes": rows,
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
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--replay-h5", type=Path, required=True)
    parser.add_argument("--replay-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(
        source_h5=arguments.source_h5.resolve(),
        source_json=arguments.source_json.resolve(),
        replay_h5=arguments.replay_h5.resolve(),
        replay_json=arguments.replay_json.resolve(),
        output_path=arguments.output.resolve(),
    )

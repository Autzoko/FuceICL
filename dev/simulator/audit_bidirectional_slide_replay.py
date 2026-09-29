"""审计 paired BidirectionalSlide 的 action replay 与 Demo 必要性。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Iterator

import h5py
import numpy as np


@dataclass(frozen=True)
class ReplayAuditConfig:
    """冻结的 replay 审计阈值。"""

    schema_version: str
    expected_pairs: int
    action_horizon: int
    action_difference_tolerance: float
    observation_tolerance: float
    source_qpos_alignment_tolerance_rad: float
    minimum_direction_valid_fraction: float
    maximum_controller_direction_cosine: float
    minimum_controller_direction_separation: float
    maximum_tcp_direction_cosine: float
    minimum_tcp_direction_separation_m: float
    minimum_median_observation_only_mse_lower_bound: float

    @classmethod
    def from_json(cls, path: Path) -> "ReplayAuditConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def __post_init__(self) -> None:
        if self.schema_version != "bidirectional-slide-replay-audit-v1":
            raise ValueError("未知 replay audit schema")
        if self.expected_pairs <= 0 or self.action_horizon <= 0:
            raise ValueError("pair 数量和 action horizon 必须为正")
        if not 0.0 < self.minimum_direction_valid_fraction <= 1.0:
            raise ValueError("direction valid fraction 必须在 (0, 1]")
        positive = (
            self.action_difference_tolerance,
            self.observation_tolerance,
            self.source_qpos_alignment_tolerance_rad,
            self.minimum_controller_direction_separation,
            self.minimum_tcp_direction_separation_m,
            self.minimum_median_observation_only_mse_lower_bound,
        )
        if min(positive) <= 0.0:
            raise ValueError("审计 tolerance/separation/lower bound 必须为正")


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


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _seed(episode: dict[str, Any]) -> int:
    reset_kwargs = episode.get("reset_kwargs", {})
    if "seed" not in reset_kwargs:
        raise KeyError("episode metadata 缺少 reset_kwargs.seed")
    return int(reset_kwargs["seed"])


def _episodes_by_seed(metadata: dict[str, Any]) -> dict[int, dict[str, Any]]:
    episodes = metadata.get("episodes", [])
    result = {_seed(episode): episode for episode in episodes}
    if len(result) != len(episodes):
        raise ValueError("episode reset seed 不唯一")
    return result


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


def _datasets(
    group: h5py.Group,
    prefix: str = "",
) -> Iterator[tuple[str, h5py.Dataset]]:
    for name, value in group.items():
        path = f"{prefix}/{name}" if prefix else name
        if isinstance(value, h5py.Dataset):
            yield path, value
        elif isinstance(value, h5py.Group):
            yield from _datasets(value, path)


def _dataset(group: h5py.Group, path: str) -> h5py.Dataset:
    value: Any = group
    for component in path.split("/"):
        if component not in value:
            raise KeyError(f"{group.name} 缺少 {path}")
        value = value[component]
    if not isinstance(value, h5py.Dataset):
        raise TypeError(f"{group.name}/{path} 不是 dataset")
    return value


def _maximum_absolute_error(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return float("inf")
    if left.size == 0:
        return 0.0
    left_float = left.astype(np.float64)
    right_float = right.astype(np.float64)
    if not np.isfinite(left_float).all() or not np.isfinite(right_float).all():
        raise ValueError("审计数组含 NaN 或 Inf")
    return float(np.max(np.abs(left_float - right_float)))


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return 1.0
    return float(np.dot(left, right) / denominator)


def _terminal_success(trajectory: h5py.Group) -> bool:
    success = np.asarray(_dataset(trajectory, "success"), dtype=bool)
    return bool(success.size and success[-1])


def _branch_index(
    left: np.ndarray,
    right: np.ndarray,
    tolerance: float,
) -> int:
    common = min(len(left), len(right))
    if common <= 0:
        raise ValueError("replay action 为空")
    differences = np.max(np.abs(left[:common] - right[:common]), axis=1)
    indices = np.where(differences > tolerance)[0]
    if indices.size == 0:
        raise ValueError("paired replay actions 没有产生 task-dependent 分叉")
    return int(indices[0])


def _observation_errors(
    left: h5py.Group,
    right: h5py.Group,
    frame: int,
) -> dict[str, float]:
    left_leaves = dict(_datasets(_dataset_group(left, "obs")))
    right_leaves = dict(_datasets(_dataset_group(right, "obs")))
    if left_leaves.keys() != right_leaves.keys():
        missing_left = sorted(right_leaves.keys() - left_leaves.keys())
        missing_right = sorted(left_leaves.keys() - right_leaves.keys())
        raise ValueError(
            "observation leaves 不一致："
            f"left 缺 {missing_left}，right 缺 {missing_right}"
        )
    errors = {}
    for path, left_dataset in left_leaves.items():
        right_dataset = right_leaves[path]
        if len(left_dataset) <= frame or len(right_dataset) <= frame:
            raise IndexError(f"{path} 没有 branch observation frame={frame}")
        errors[path] = _maximum_absolute_error(
            np.asarray(left_dataset[frame]),
            np.asarray(right_dataset[frame]),
        )
    return errors


def _dataset_group(group: h5py.Group, path: str) -> h5py.Group:
    value: Any = group
    for component in path.split("/"):
        if component not in value:
            raise KeyError(f"{group.name} 缺少 {path}")
        value = value[component]
    if not isinstance(value, h5py.Group):
        raise TypeError(f"{group.name}/{path} 不是 group")
    return value


def _fraction(values: list[bool]) -> float:
    return float(np.mean(values)) if values else 0.0


def run(
    *,
    project_root: Path,
    config_path: Path,
    source_root: Path,
    replay_root: Path,
    output_path: Path,
    config: ReplayAuditConfig,
) -> None:
    if output_path.exists():
        raise FileExistsError(f"输出已存在，拒绝覆盖：{output_path}")
    collection_path = source_root / "collection_report.json"
    collection = _load_json(collection_path)
    accepted = collection.get("accepted_pairs", [])
    if len(accepted) != config.expected_pairs:
        raise ValueError(
            f"collection pairs={len(accepted)}，预期 {config.expected_pairs}"
        )

    source_paths = {
        operation: {
            "h5": source_root / operation / "trajectory.h5",
            "json": source_root / operation / "trajectory.json",
        }
        for operation in ("left", "right")
    }
    replay_paths = {
        operation: {
            "h5": replay_root
            / operation
            / "trajectory.pointcloud.pd_ee_delta_pose.physx_cpu.h5",
            "json": replay_root
            / operation
            / "trajectory.pointcloud.pd_ee_delta_pose.physx_cpu.json",
        }
        for operation in ("left", "right")
    }
    for operation in ("left", "right"):
        expected = collection["files"][operation]
        if _sha256(source_paths[operation]["h5"]) != expected["h5_sha256"]:
            raise ValueError(f"{operation} source HDF5 hash 不匹配")
        if _sha256(source_paths[operation]["json"]) != expected["json_sha256"]:
            raise ValueError(f"{operation} source JSON hash 不匹配")

    source_metadata = {
        operation: _load_json(paths["json"])
        for operation, paths in source_paths.items()
    }
    replay_metadata = {
        operation: _load_json(paths["json"])
        for operation, paths in replay_paths.items()
    }
    source_episodes = {
        operation: _episodes_by_seed(metadata)
        for operation, metadata in source_metadata.items()
    }
    replay_episodes = {
        operation: _episodes_by_seed(metadata)
        for operation, metadata in replay_metadata.items()
    }
    expected_seeds = {int(row["seed"]) for row in accepted}
    for operation in ("left", "right"):
        if set(source_episodes[operation]) != expected_seeds:
            raise ValueError(f"{operation} source episode seeds 不完整")
        if set(replay_episodes[operation]) != expected_seeds:
            raise ValueError(f"{operation} replay episode seeds 不完整")

    pairs = []
    source_handles = {
        operation: h5py.File(paths["h5"], "r")
        for operation, paths in source_paths.items()
    }
    replay_handles = {
        operation: h5py.File(paths["h5"], "r")
        for operation, paths in replay_paths.items()
    }
    try:
        for row in accepted:
            seed = int(row["seed"])
            source = {
                operation: _trajectory(
                    source_handles[operation],
                    source_episodes[operation][seed],
                )
                for operation in ("left", "right")
            }
            replay = {
                operation: _trajectory(
                    replay_handles[operation],
                    replay_episodes[operation][seed],
                )
                for operation in ("left", "right")
            }
            source_frame = int(row["source_branch_frame"])
            source_qpos = {
                operation: np.asarray(
                    _dataset(
                        source[operation],
                        "env_states/articulations/panda",
                    )[source_frame, 13:22],
                    dtype=np.float64,
                )
                for operation in ("left", "right")
            }
            source_pair_error = _maximum_absolute_error(
                source_qpos["left"], source_qpos["right"]
            )

            actions = {
                operation: np.asarray(
                    _dataset(replay[operation], "actions"),
                    dtype=np.float64,
                )
                for operation in ("left", "right")
            }
            if any(
                value.ndim != 2 or value.shape[1] != 7
                for value in actions.values()
            ):
                raise ValueError(f"seed={seed} replay action 不是 [T,7]")
            branch = _branch_index(
                actions["left"],
                actions["right"],
                config.action_difference_tolerance,
            )
            horizon = config.action_horizon
            if any(len(value) < branch + horizon for value in actions.values()):
                raise ValueError(f"seed={seed} branch 后不足 H={horizon} actions")
            observation_errors = _observation_errors(
                replay["left"], replay["right"], branch
            )
            if not observation_errors:
                raise ValueError(f"seed={seed} 没有 observation leaves")
            required_leaves = {
                "agent/qpos",
                "agent/qvel",
                "extra/tcp_pose",
                "pointcloud/rgb",
                "pointcloud/segmentation",
                "pointcloud/xyzw",
            }
            missing_leaves = sorted(required_leaves - observation_errors.keys())
            if missing_leaves:
                raise ValueError(
                    f"seed={seed} 缺少必要 observation leaves：{missing_leaves}"
                )
            goal_leaves = sorted(
                path for path in observation_errors if "goal" in path.lower()
            )

            replay_qpos = {
                operation: np.asarray(
                    _dataset(replay[operation], "obs/agent/qpos")[branch],
                    dtype=np.float64,
                )
                for operation in ("left", "right")
            }
            source_alignment = {
                operation: _maximum_absolute_error(
                    replay_qpos[operation], source_qpos[operation]
                )
                for operation in ("left", "right")
            }
            chunks = {
                operation: actions[operation][branch : branch + horizon]
                for operation in ("left", "right")
            }
            controller_direction = {
                operation: chunks[operation][:, :3].sum(axis=0)
                for operation in ("left", "right")
            }
            controller_cosine = _cosine(
                controller_direction["left"],
                controller_direction["right"],
            )
            controller_separation = float(
                np.linalg.norm(
                    controller_direction["left"]
                    - controller_direction["right"]
                )
            )
            controller_x_opposed = bool(
                controller_direction["left"][0]
                * controller_direction["right"][0]
                < 0.0
            )

            tcp = {
                operation: np.asarray(
                    _dataset(replay[operation], "obs/extra/tcp_pose"),
                    dtype=np.float64,
                )
                for operation in ("left", "right")
            }
            if any(len(value) <= branch + horizon for value in tcp.values()):
                raise ValueError(
                    f"seed={seed} branch 后不足 H={horizon} observations"
                )
            tcp_direction = {
                operation: value[branch + horizon, :3] - value[branch, :3]
                for operation, value in tcp.items()
            }
            tcp_cosine = _cosine(tcp_direction["left"], tcp_direction["right"])
            tcp_separation = float(
                np.linalg.norm(tcp_direction["left"] - tcp_direction["right"])
            )
            tcp_x_opposed = bool(
                tcp_direction["left"][0] * tcp_direction["right"][0] < 0.0
            )
            mse_lower_bound = float(
                np.mean((chunks["left"] - chunks["right"]) ** 2) / 4.0
            )
            pairs.append(
                {
                    "pair_id": int(row["pair_id"]),
                    "seed": seed,
                    "split": row["split"],
                    "source_branch_frame": source_frame,
                    "replay_branch_frame": branch,
                    "source_qpos_pair_max_abs_error": source_pair_error,
                    "source_qpos_alignment_max_abs_error": source_alignment,
                    "observation_leaf_max_abs_error": observation_errors,
                    "maximum_observation_error": max(observation_errors.values()),
                    "goal_observation_leaves": goal_leaves,
                    "controller_direction": {
                        key: value.tolist()
                        for key, value in controller_direction.items()
                    },
                    "controller_direction_cosine": controller_cosine,
                    "controller_direction_separation": controller_separation,
                    "controller_x_opposed": controller_x_opposed,
                    "tcp_direction_m": {
                        key: value.tolist() for key, value in tcp_direction.items()
                    },
                    "tcp_direction_cosine": tcp_cosine,
                    "tcp_direction_separation_m": tcp_separation,
                    "tcp_x_opposed": tcp_x_opposed,
                    "observation_only_mse_lower_bound": mse_lower_bound,
                    "terminal_success": {
                        operation: _terminal_success(replay[operation])
                        for operation in ("left", "right")
                    },
                    "action_steps": {
                        operation: len(actions[operation])
                        for operation in ("left", "right")
                    },
                }
            )
    finally:
        for handle in (*source_handles.values(), *replay_handles.values()):
            handle.close()

    controller_valid = [
        row["controller_direction_cosine"]
        <= config.maximum_controller_direction_cosine
        and row["controller_direction_separation"]
        >= config.minimum_controller_direction_separation
        and row["controller_x_opposed"]
        for row in pairs
    ]
    tcp_valid = [
        row["tcp_direction_cosine"] <= config.maximum_tcp_direction_cosine
        and row["tcp_direction_separation_m"]
        >= config.minimum_tcp_direction_separation_m
        and row["tcp_x_opposed"]
        for row in pairs
    ]
    terminal_success = [
        all(row["terminal_success"].values()) for row in pairs
    ]
    lower_bounds = [row["observation_only_mse_lower_bound"] for row in pairs]
    maximum_observation_error = max(
        row["maximum_observation_error"] for row in pairs
    )
    maximum_source_alignment_error = max(
        max(row["source_qpos_alignment_max_abs_error"].values())
        for row in pairs
    )
    goal_leaf_count = sum(len(row["goal_observation_leaves"]) for row in pairs)
    criteria = {
        "P1_complete_successful_pairs": (
            len(pairs) == config.expected_pairs and all(terminal_success)
        ),
        "P2_identical_goal_free_branch_observation": (
            maximum_observation_error <= config.observation_tolerance
            and goal_leaf_count == 0
        ),
        "P3_source_branch_alignment": (
            maximum_source_alignment_error
            <= config.source_qpos_alignment_tolerance_rad
        ),
        "P4_opposing_h6_controller_and_tcp_direction": (
            _fraction(controller_valid) >= config.minimum_direction_valid_fraction
            and _fraction(tcp_valid) >= config.minimum_direction_valid_fraction
        ),
        "P5_positive_observation_only_lower_bound": (
            float(np.median(lower_bounds))
            >= config.minimum_median_observation_only_mse_lower_bound
        ),
    }
    summary = {
        "pairs": len(pairs),
        "terminal_success_fraction": _fraction(terminal_success),
        "maximum_branch_observation_error": maximum_observation_error,
        "goal_observation_leaf_count": goal_leaf_count,
        "maximum_source_qpos_alignment_error_rad": maximum_source_alignment_error,
        "controller_direction_valid_fraction": _fraction(controller_valid),
        "controller_direction_cosine_median": float(
            np.median([row["controller_direction_cosine"] for row in pairs])
        ),
        "controller_direction_separation_minimum": min(
            row["controller_direction_separation"] for row in pairs
        ),
        "tcp_direction_valid_fraction": _fraction(tcp_valid),
        "tcp_direction_cosine_median": float(
            np.median([row["tcp_direction_cosine"] for row in pairs])
        ),
        "tcp_direction_separation_minimum_m": min(
            row["tcp_direction_separation_m"] for row in pairs
        ),
        "observation_only_mse_lower_bound_median": float(np.median(lower_bounds)),
        "observation_only_mse_lower_bound_minimum": min(lower_bounds),
        "criteria": criteria,
        "all_criteria_passed": all(criteria.values()),
    }
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(project_root),
        "protocol": {
            "action_replay_without_env_state_forcing": True,
            "branch_definition": (
                "first paired replay action whose max absolute difference exceeds "
                "action_difference_tolerance"
            ),
            "observation_only_mse_lower_bound": (
                "mean((left_action_chunk-right_action_chunk)^2)/4"
            ),
        },
        "config": asdict(config),
        "config_sha256": _sha256(config_path),
        "collection_report_sha256": _sha256(collection_path),
        "files": {
            operation: {
                kind: {
                    "h5": str(paths["h5"]),
                    "h5_sha256": _sha256(paths["h5"]),
                    "json": str(paths["json"]),
                    "json_sha256": _sha256(paths["json"]),
                }
                for kind, paths in (
                    ("source", source_paths[operation]),
                    ("replay", replay_paths[operation]),
                )
            }
            for operation in ("left", "right")
        },
        "summary": summary,
        "pairs": pairs,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(f"{output_path.suffix}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if not summary["all_criteria_passed"]:
        raise RuntimeError("paired replay 审计未通过全部预注册判据")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(
        project_root=arguments.project_root.resolve(),
        config_path=arguments.config.resolve(),
        source_root=arguments.source_root.resolve(),
        replay_root=arguments.replay_root.resolve(),
        output_path=arguments.output.resolve(),
        config=ReplayAuditConfig.from_json(arguments.config.resolve()),
    )

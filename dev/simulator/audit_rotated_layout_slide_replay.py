"""审计 RotatedLayoutSlide action replay、paired observation 与动作可辨识性。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import h5py
import numpy as np

from dev.simulator.audit_bidirectional_slide_replay import (
    _branch_index,
    _cosine,
    _dataset,
    _episodes_by_seed,
    _fraction,
    _load_json,
    _maximum_absolute_error,
    _observation_errors,
    _sha256,
    _terminal_success,
    _trajectory,
)


OPERATIONS = ("toward", "away")


@dataclass(frozen=True)
class RotatedLayoutReplayAuditConfig:
    """冻结的 rotated-layout replay 审计阈值。"""

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
    def from_json(cls, path: Path) -> "RotatedLayoutReplayAuditConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def __post_init__(self) -> None:
        if self.schema_version != "rotated-layout-slide-replay-audit-v1":
            raise ValueError("未知 rotated-layout replay audit schema")
        if self.expected_pairs <= 0 or self.action_horizon <= 0:
            raise ValueError("pair 数量和 action horizon 必须为正")
        if not 0.0 < self.minimum_direction_valid_fraction <= 1.0:
            raise ValueError("direction valid fraction 必须在 (0,1]")
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


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _paths(root: Path, *, replay: bool) -> dict[str, dict[str, Path]]:
    stem = (
        "trajectory.pointcloud.pd_ee_delta_pose.physx_cpu"
        if replay
        else "trajectory"
    )
    return {
        operation: {
            "h5": root / operation / f"{stem}.h5",
            "json": root / operation / f"{stem}.json",
        }
        for operation in OPERATIONS
    }


def _direction_valid(
    values: dict[str, np.ndarray],
    axis: np.ndarray,
    *,
    maximum_cosine: float,
    minimum_separation: float,
) -> tuple[bool, dict[str, float]]:
    cosine = _cosine(values["toward"], values["away"])
    separation = float(np.linalg.norm(values["toward"] - values["away"]))
    projections = {
        operation: float(np.dot(value, axis))
        for operation, value in values.items()
    }
    # 第一个 task-dependent chunk 是从 neutral 移到物体反侧：toward 沿 -axis。
    sign_valid = projections["toward"] < 0.0 < projections["away"]
    return (
        cosine <= maximum_cosine
        and separation >= minimum_separation
        and sign_valid,
        {
            "cosine": cosine,
            "separation": separation,
            "toward_axis_projection": projections["toward"],
            "away_axis_projection": projections["away"],
        },
    )


def run(
    *,
    project_root: Path,
    config_path: Path,
    source_root: Path,
    replay_root: Path,
    output_path: Path,
    config: RotatedLayoutReplayAuditConfig,
) -> None:
    if output_path.exists():
        raise FileExistsError(f"输出已存在，拒绝覆盖：{output_path}")
    collection_path = source_root / "collection_report.json"
    collection = _load_json(collection_path)
    accepted = collection.get("accepted_pairs", [])
    if len(accepted) != config.expected_pairs:
        raise ValueError("collection pair 数量与 replay 配置不一致")

    source_paths = _paths(source_root, replay=False)
    replay_paths = _paths(replay_root, replay=True)
    for operation in OPERATIONS:
        expected = collection["files"][operation]
        if _sha256(source_paths[operation]["h5"]) != expected["h5_sha256"]:
            raise ValueError(f"{operation} source HDF5 hash 不匹配")
        if _sha256(source_paths[operation]["json"]) != expected["json_sha256"]:
            raise ValueError(f"{operation} source JSON hash 不匹配")

    source_episodes = {
        operation: _episodes_by_seed(_load_json(paths["json"]))
        for operation, paths in source_paths.items()
    }
    replay_episodes = {
        operation: _episodes_by_seed(_load_json(paths["json"]))
        for operation, paths in replay_paths.items()
    }
    expected_seeds = {int(row["seed"]) for row in accepted}
    for operation in OPERATIONS:
        if set(source_episodes[operation]) != expected_seeds:
            raise ValueError(f"{operation} source episode seeds 不完整")
    replay_seed_sets = {
        operation: set(episodes)
        for operation, episodes in replay_episodes.items()
    }
    missing = {
        operation: sorted(expected_seeds - seeds)
        for operation, seeds in replay_seed_sets.items()
    }
    unexpected = {
        operation: sorted(seeds - expected_seeds)
        for operation, seeds in replay_seed_sets.items()
    }
    common_seeds = expected_seeds.intersection(*replay_seed_sets.values())
    if not common_seeds:
        raise ValueError("toward/away replay 没有共同 seed")

    source_handles = {
        operation: h5py.File(paths["h5"], "r")
        for operation, paths in source_paths.items()
    }
    replay_handles = {
        operation: h5py.File(paths["h5"], "r")
        for operation, paths in replay_paths.items()
    }
    pairs = []
    try:
        for row in accepted:
            seed = int(row["seed"])
            if seed not in common_seeds:
                continue
            source = {
                operation: _trajectory(
                    source_handles[operation],
                    source_episodes[operation][seed],
                )
                for operation in OPERATIONS
            }
            replay = {
                operation: _trajectory(
                    replay_handles[operation],
                    replay_episodes[operation][seed],
                )
                for operation in OPERATIONS
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
                for operation in OPERATIONS
            }
            source_pair_error = _maximum_absolute_error(
                source_qpos["toward"], source_qpos["away"]
            )
            actions = {
                operation: np.asarray(
                    _dataset(replay[operation], "actions"),
                    dtype=np.float64,
                )
                for operation in OPERATIONS
            }
            if any(
                value.ndim != 2 or value.shape[1] != 7
                for value in actions.values()
            ):
                raise ValueError(f"seed={seed} replay action 不是 [T,7]")
            branch = _branch_index(
                actions["toward"],
                actions["away"],
                config.action_difference_tolerance,
            )
            horizon = config.action_horizon
            if any(len(value) < branch + horizon for value in actions.values()):
                raise ValueError(f"seed={seed} branch 后不足 H={horizon} actions")
            observation_errors = _observation_errors(
                replay["toward"], replay["away"], branch
            )
            required = {
                "agent/qpos",
                "agent/qvel",
                "extra/tcp_pose",
                "pointcloud/rgb",
                "pointcloud/segmentation",
                "pointcloud/xyzw",
            }
            if missing_leaves := sorted(required - observation_errors.keys()):
                raise ValueError(f"seed={seed} 缺 observation：{missing_leaves}")
            goal_leaves = sorted(
                key for key in observation_errors if "goal" in key.lower()
            )
            replay_qpos = {
                operation: np.asarray(
                    _dataset(replay[operation], "obs/agent/qpos")[branch],
                    dtype=np.float64,
                )
                for operation in OPERATIONS
            }
            source_alignment = {
                operation: _maximum_absolute_error(
                    replay_qpos[operation], source_qpos[operation]
                )
                for operation in OPERATIONS
            }
            chunks = {
                operation: actions[operation][branch : branch + horizon]
                for operation in OPERATIONS
            }
            controller_directions = {
                operation: chunk[:, :3].sum(axis=0)
                for operation, chunk in chunks.items()
            }
            # layout axis 从冻结 yaw 重建；不读取 hidden goal。
            yaw = float(row["axis_yaw_rad"])
            layout_axis = np.asarray([np.cos(yaw), np.sin(yaw), 0.0])
            controller_valid, controller_metrics = _direction_valid(
                controller_directions,
                layout_axis,
                maximum_cosine=config.maximum_controller_direction_cosine,
                minimum_separation=config.minimum_controller_direction_separation,
            )
            tcp = {
                operation: np.asarray(
                    _dataset(replay[operation], "obs/extra/tcp_pose"),
                    dtype=np.float64,
                )
                for operation in OPERATIONS
            }
            if any(len(value) <= branch + horizon for value in tcp.values()):
                raise ValueError(f"seed={seed} branch 后 observation 不足")
            tcp_directions = {
                operation: value[branch + horizon, :3] - value[branch, :3]
                for operation, value in tcp.items()
            }
            tcp_valid, tcp_metrics = _direction_valid(
                tcp_directions,
                layout_axis,
                maximum_cosine=config.maximum_tcp_direction_cosine,
                minimum_separation=config.minimum_tcp_direction_separation_m,
            )
            lower_bound = float(
                np.mean((chunks["toward"] - chunks["away"]) ** 2) / 4.0
            )
            pairs.append(
                {
                    "pair_id": int(row["pair_id"]),
                    "seed": seed,
                    "split": row["split"],
                    "axis_yaw_degrees": float(row["axis_yaw_degrees"]),
                    "source_branch_frame": source_frame,
                    "replay_branch_frame": branch,
                    "source_qpos_pair_max_abs_error": source_pair_error,
                    "source_qpos_alignment_max_abs_error": source_alignment,
                    "maximum_observation_error": max(
                        observation_errors.values()
                    ),
                    "goal_observation_leaves": goal_leaves,
                    "controller_direction": {
                        key: value.tolist()
                        for key, value in controller_directions.items()
                    },
                    "controller_direction_metrics": controller_metrics,
                    "controller_direction_valid": controller_valid,
                    "tcp_direction_m": {
                        key: value.tolist()
                        for key, value in tcp_directions.items()
                    },
                    "tcp_direction_metrics": tcp_metrics,
                    "tcp_direction_valid": tcp_valid,
                    "observation_only_mse_lower_bound": lower_bound,
                    "terminal_success": {
                        operation: _terminal_success(replay[operation])
                        for operation in OPERATIONS
                    },
                    "action_steps": {
                        operation: len(actions[operation])
                        for operation in OPERATIONS
                    },
                }
            )
    finally:
        for handle in (*source_handles.values(), *replay_handles.values()):
            handle.close()

    controller_validity = [
        row["controller_direction_valid"] for row in pairs
    ]
    tcp_validity = [row["tcp_direction_valid"] for row in pairs]
    successes = [all(row["terminal_success"].values()) for row in pairs]
    lower_bounds = [row["observation_only_mse_lower_bound"] for row in pairs]
    maximum_observation_error = max(
        row["maximum_observation_error"] for row in pairs
    )
    maximum_source_alignment = max(
        max(row["source_qpos_alignment_max_abs_error"].values())
        for row in pairs
    )
    replay_complete = all(
        not missing[operation] and not unexpected[operation]
        for operation in OPERATIONS
    )
    criteria = {
        "P1_complete_successful_pairs": (
            replay_complete
            and len(pairs) == config.expected_pairs
            and all(successes)
        ),
        "P2_identical_goal_free_branch_observation": (
            maximum_observation_error <= config.observation_tolerance
            and not any(row["goal_observation_leaves"] for row in pairs)
        ),
        "P3_source_branch_alignment": (
            maximum_source_alignment
            <= config.source_qpos_alignment_tolerance_rad
        ),
        "P4_opposing_h6_controller_and_tcp_direction": (
            _fraction(controller_validity)
            >= config.minimum_direction_valid_fraction
            and _fraction(tcp_validity)
            >= config.minimum_direction_valid_fraction
        ),
        "P5_positive_observation_only_lower_bound": (
            float(np.median(lower_bounds))
            >= config.minimum_median_observation_only_mse_lower_bound
        ),
    }
    summary = {
        "expected_pairs": config.expected_pairs,
        "paired_replay_pairs": len(pairs),
        "missing_replay_seeds": missing,
        "unexpected_replay_seeds": unexpected,
        "terminal_success_fraction": _fraction(successes),
        "maximum_branch_observation_error": maximum_observation_error,
        "maximum_source_qpos_alignment_error_rad": maximum_source_alignment,
        "controller_direction_valid_fraction": _fraction(controller_validity),
        "controller_direction_cosine_median": float(
            np.median(
                [
                    row["controller_direction_metrics"]["cosine"]
                    for row in pairs
                ]
            )
        ),
        "tcp_direction_valid_fraction": _fraction(tcp_validity),
        "tcp_direction_cosine_median": float(
            np.median(
                [row["tcp_direction_metrics"]["cosine"] for row in pairs]
            )
        ),
        "observation_only_mse_lower_bound_median": float(
            np.median(lower_bounds)
        ),
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
            "direction_frame": "visible object-to-anchor layout axis",
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
            for operation in OPERATIONS
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
        raise RuntimeError("rotated-layout replay 未通过全部预注册判据")


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
        config=RotatedLayoutReplayAuditConfig.from_json(
            arguments.config.resolve()
        ),
    )

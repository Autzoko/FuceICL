"""审计双向滑块共同前缀后的观测等价性、动作分离与 expert 成功。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

import gymnasium as gym
import numpy as np
import sapien
import torch

from dev.pointnet.compare_retrievers import _sha256
from dev.simulator import bidirectional_slide_env  # noqa: F401
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)
from mani_skill.utils import common


ENV_IDS = {
    "left": "BidirectionalSlideLeft-v0",
    "right": "BidirectionalSlideRight-v0",
}


@dataclass(frozen=True)
class PairPilotConfig:
    """正式批量采集前冻结的小规模可行性审计。"""

    schema_version: str
    seeds: list[int]
    neutral_height_m: float
    contact_offset_m: float
    state_max_abs_tolerance: float
    observation_max_abs_tolerance: float
    minimum_branch_target_separation_m: float
    require_all_expert_success: bool

    @classmethod
    def from_json(cls, path: Path) -> "PairPilotConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def __post_init__(self) -> None:
        if self.schema_version != "bidirectional-slide-pair-pilot-v1":
            raise ValueError("未知 bidirectional slide pilot schema")
        if not self.seeds or len(self.seeds) != len(set(self.seeds)):
            raise ValueError("pilot seeds 不能为空或重复")
        if min(self.seeds) < 0:
            raise ValueError("pilot seed 不能为负")
        positive = (
            self.neutral_height_m,
            self.contact_offset_m,
            self.state_max_abs_tolerance,
            self.observation_max_abs_tolerance,
            self.minimum_branch_target_separation_m,
        )
        if min(positive) <= 0.0:
            raise ValueError("pilot 距离与 tolerance 必须为正")


@dataclass(frozen=True)
class BranchRecord:
    operation: str
    seed: int
    common_prefix_steps: int
    qpos: np.ndarray
    object_pose: np.ndarray
    tcp_pose: np.ndarray
    observation: dict[str, np.ndarray]
    contact_target: np.ndarray
    goal_position: np.ndarray
    success: bool


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _single(value: Any) -> np.ndarray:
    array = np.asarray(common.to_numpy(value))
    return array[0] if array.ndim > 1 and array.shape[0] == 1 else array


def _flatten_observation(
    value: Any,
    prefix: str = "",
) -> dict[str, np.ndarray]:
    """把 live observation 展平成可逐叶审计的 numpy arrays。"""
    if isinstance(value, Mapping):
        output = {}
        for key, item in sorted(value.items()):
            path = f"{prefix}/{key}" if prefix else str(key)
            output.update(_flatten_observation(item, path))
        return output
    array = np.asarray(common.to_numpy(value))
    if array.ndim > 0 and array.shape[0] == 1:
        array = array[0]
    return {prefix: array}


def _move_or_raise(
    planner: PandaArmMotionPlanningSolver,
    pose: sapien.Pose,
    label: str,
) -> None:
    result = planner.move_to_pose_with_screw(pose)
    if result == -1:
        raise RuntimeError(f"motion planner 无法到达 {label}")


def _run_operation(
    operation: str,
    seed: int,
    config: PairPilotConfig,
) -> BranchRecord:
    environment = gym.make(
        ENV_IDS[operation],
        obs_mode="pointcloud",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        sim_backend="physx_cpu",
    )
    planner = None
    try:
        environment.reset(seed=seed)
        base = environment.unwrapped
        planner = PandaArmMotionPlanningSolver(
            environment,
            debug=False,
            vis=False,
            base_pose=base.agent.robot.pose,
            visualize_target_grasp_pose=False,
            print_env_info=False,
        )
        planner.close_gripper()
        object_position = np.asarray(base.obj.pose.sp.p, dtype=np.float64)
        neutral_position = object_position.copy()
        neutral_position[2] += config.neutral_height_m
        neutral_pose = sapien.Pose(
            p=neutral_position,
            q=np.asarray(base.agent.tcp.pose.sp.q),
        )
        _move_or_raise(planner, neutral_pose, "shared neutral pose")
        branch_observation = _flatten_observation(base.get_obs())
        if any("goal" in key.lower() for key in branch_observation):
            raise RuntimeError("predictor observation 泄漏 goal 字段")
        prefix_steps = int(_single(base.elapsed_steps))
        direction = float(base.operation_direction)
        contact_position = object_position.copy()
        contact_position[0] -= direction * config.contact_offset_m
        contact_pose = sapien.Pose(
            p=contact_position,
            q=np.asarray(base.agent.tcp.pose.sp.q),
        )
        record = BranchRecord(
            operation=operation,
            seed=seed,
            common_prefix_steps=prefix_steps,
            qpos=_single(base.agent.robot.get_qpos()).astype(np.float64),
            object_pose=_single(base.obj.pose.raw_pose).astype(np.float64),
            tcp_pose=_single(base.agent.tcp.pose.raw_pose).astype(np.float64),
            observation=branch_observation,
            contact_target=contact_position,
            goal_position=_single(base.goal_region.pose.p).astype(np.float64),
            success=False,
        )
        _move_or_raise(planner, contact_pose, f"{operation} contact pose")
        goal_position = np.asarray(base.goal_region.pose.sp.p, dtype=np.float64)
        final_position = goal_position.copy()
        final_position[0] -= direction * config.contact_offset_m
        final_pose = sapien.Pose(
            p=final_position,
            q=np.asarray(base.agent.tcp.pose.sp.q),
        )
        _move_or_raise(planner, final_pose, f"{operation} goal pose")
        success = bool(_single(base.evaluate()["success"]))
        return BranchRecord(**{**record.__dict__, "success": success})
    finally:
        if planner is not None:
            planner.close()
        environment.close()


def _maximum_absolute_error(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return float("inf")
    if left.dtype.kind in "OUS" or right.dtype.kind in "OUS":
        return 0.0 if np.array_equal(left, right) else float("inf")
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))


def _compare_pair(
    left: BranchRecord,
    right: BranchRecord,
) -> dict[str, Any]:
    if left.seed != right.seed:
        raise ValueError("pair seed 不一致")
    keys_equal = set(left.observation) == set(right.observation)
    observation_errors = {
        key: _maximum_absolute_error(
            left.observation[key],
            right.observation[key],
        )
        for key in sorted(set(left.observation) & set(right.observation))
    }
    state_errors = {
        "qpos": _maximum_absolute_error(left.qpos, right.qpos),
        "object_pose": _maximum_absolute_error(
            left.object_pose,
            right.object_pose,
        ),
        "tcp_pose": _maximum_absolute_error(left.tcp_pose, right.tcp_pose),
    }
    branch_state = 0.5 * (left.tcp_pose[:3] + right.tcp_pose[:3])
    left_delta = left.contact_target - branch_state
    right_delta = right.contact_target - branch_state
    return {
        "seed": left.seed,
        "common_prefix_steps": {
            "left": left.common_prefix_steps,
            "right": right.common_prefix_steps,
            "equal": left.common_prefix_steps == right.common_prefix_steps,
        },
        "state_max_abs_error": state_errors,
        "observation_keys_equal": keys_equal,
        "observation_leaf_count": len(observation_errors),
        "observation_max_abs_error": max(
            observation_errors.values(),
            default=float("inf"),
        ),
        "observation_leaf_errors": observation_errors,
        "branch_contact_target_separation_m": float(
            np.linalg.norm(left.contact_target - right.contact_target)
        ),
        "branch_target_delta_cosine": float(
            np.dot(left_delta, right_delta)
            / max(np.linalg.norm(left_delta) * np.linalg.norm(right_delta), 1e-12)
        ),
        "goal_separation_m": float(
            np.linalg.norm(left.goal_position - right.goal_position)
        ),
        "success": {"left": left.success, "right": right.success},
    }


def run(
    *,
    project_root: Path,
    config_path: Path,
    output_path: Path,
    config: PairPilotConfig,
) -> None:
    if output_path.exists():
        raise FileExistsError(f"输出已存在，拒绝覆盖：{output_path}")
    pairs = []
    for seed in config.seeds:
        records = {
            operation: _run_operation(operation, seed, config)
            for operation in ("left", "right")
        }
        pairs.append(_compare_pair(records["left"], records["right"]))
        print(json.dumps(pairs[-1], sort_keys=True), flush=True)
    state_maximum = max(
        value
        for pair in pairs
        for value in pair["state_max_abs_error"].values()
    )
    observation_maximum = max(
        pair["observation_max_abs_error"] for pair in pairs
    )
    minimum_separation = min(
        pair["branch_contact_target_separation_m"] for pair in pairs
    )
    criteria = {
        "p1_shared_prefix_length": all(
            pair["common_prefix_steps"]["equal"] for pair in pairs
        ),
        "p2_branch_state_equivalent": (
            state_maximum <= config.state_max_abs_tolerance
        ),
        "p3_predictor_observation_equivalent": (
            all(pair["observation_keys_equal"] for pair in pairs)
            and observation_maximum <= config.observation_max_abs_tolerance
        ),
        "p4_branch_action_target_separated": (
            minimum_separation >= config.minimum_branch_target_separation_m
        ),
        "p5_all_experts_successful": (
            all(
                pair["success"][operation]
                for pair in pairs
                for operation in ("left", "right")
            )
            if config.require_all_expert_success
            else True
        ),
    }
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(project_root),
        "config": asdict(config),
        "config_sha256": _sha256(config_path),
        "environment_ids": ENV_IDS,
        "pairs": pairs,
        "aggregate": {
            "pairs": len(pairs),
            "state_max_abs_error": state_maximum,
            "observation_max_abs_error": observation_maximum,
            "minimum_branch_contact_target_separation_m": minimum_separation,
            "success_rate": float(
                np.mean(
                    [
                        pair["success"][operation]
                        for pair in pairs
                        for operation in ("left", "right")
                    ]
                )
            ),
        },
        "criteria": criteria,
        "pilot_passed": all(criteria.values()),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(f"{output_path.suffix}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_path)
    print(json.dumps(report["aggregate"], indent=2), flush=True)
    print(json.dumps(criteria, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(
        project_root=arguments.project_root.resolve(),
        config_path=arguments.config.resolve(),
        output_path=arguments.output.resolve(),
        config=PairPilotConfig.from_json(arguments.config.resolve()),
    )

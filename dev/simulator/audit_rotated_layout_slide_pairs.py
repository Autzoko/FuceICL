"""审计 RotatedLayoutSlide paired observation 与 frame transport 机制。"""

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

import gymnasium as gym
import numpy as np
import sapien

from dev.pointnet.compare_retrievers import _sha256
from dev.simulator import rotated_layout_slide_env  # noqa: F401
from dev.simulator.audit_bidirectional_slide_pairs import (
    _flatten_observation,
    _maximum_absolute_error,
    _move_or_raise,
    _single,
)
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)


ENV_IDS = {
    "toward": "RotatedLayoutSlideToward-v0",
    "away": "RotatedLayoutSlideAway-v0",
}


@dataclass(frozen=True)
class RotatedLayoutPilotConfig:
    """旋转 layout paired pilot 的冻结协议。"""

    schema_version: str
    seeds: list[int]
    axis_yaw_degrees: list[float]
    neutral_height_m: float
    contact_offset_m: float
    goal_standoff_m: float
    state_max_abs_tolerance: float
    observation_max_abs_tolerance: float
    minimum_branch_target_separation_m: float
    minimum_large_yaw_gap_degrees: float
    maximum_raw_large_gap_cosine: float
    minimum_transport_large_gap_cosine: float
    require_all_expert_success: bool

    @classmethod
    def from_json(cls, path: Path) -> "RotatedLayoutPilotConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def __post_init__(self) -> None:
        if self.schema_version != "rotated-layout-slide-pair-pilot-v1":
            raise ValueError("未知 rotated layout pilot schema")
        if not self.seeds or len(self.seeds) != len(set(self.seeds)):
            raise ValueError("pilot seeds 不能为空或重复")
        if len(self.seeds) != len(self.axis_yaw_degrees):
            raise ValueError("seeds 与 axis yaw 数量不一致")
        if min(self.seeds) < 0:
            raise ValueError("pilot seed 不能为负")
        positive = (
            self.neutral_height_m,
            self.contact_offset_m,
            self.goal_standoff_m,
            self.state_max_abs_tolerance,
            self.observation_max_abs_tolerance,
            self.minimum_branch_target_separation_m,
            self.minimum_large_yaw_gap_degrees,
            self.minimum_transport_large_gap_cosine,
        )
        if min(positive) <= 0.0:
            raise ValueError("pilot 距离/tolerance/cosine 必须为正")
        if not -1.0 <= self.maximum_raw_large_gap_cosine <= 1.0:
            raise ValueError("raw cosine threshold 非法")
        if not -1.0 <= self.minimum_transport_large_gap_cosine <= 1.0:
            raise ValueError("transport cosine threshold 非法")


@dataclass(frozen=True)
class LayoutBranchRecord:
    operation: str
    seed: int
    axis_yaw_rad: float
    layout_axis: np.ndarray
    common_prefix_steps: int
    qpos: np.ndarray
    object_pose: np.ndarray
    anchor_pose: np.ndarray
    tcp_pose: np.ndarray
    observation: dict[str, np.ndarray]
    branch_target: np.ndarray
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


def _run_operation(
    operation: str,
    seed: int,
    axis_yaw_rad: float,
    config: RotatedLayoutPilotConfig,
) -> LayoutBranchRecord:
    environment = gym.make(
        ENV_IDS[operation],
        obs_mode="pointcloud",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        sim_backend="physx_cpu",
    )
    planner = None
    try:
        environment.reset(seed=seed, options={"axis_yaw_rad": axis_yaw_rad})
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
        axis = _single(base.layout_axis).astype(np.float64)
        neutral_position = object_position.copy()
        neutral_position[2] += config.neutral_height_m
        orientation = np.asarray(base.agent.tcp.pose.sp.q)
        _move_or_raise(
            planner,
            sapien.Pose(p=neutral_position, q=orientation),
            "shared neutral pose",
        )
        observation = _flatten_observation(base.get_obs())
        if any("goal" in key.lower() for key in observation):
            raise RuntimeError("predictor observation 泄漏 goal 字段")
        prefix_steps = int(np.asarray(_single(base.elapsed_steps)).reshape(-1)[0])
        direction = float(base.operation_direction)
        branch_position = (
            neutral_position - direction * axis * config.contact_offset_m
        )
        record = LayoutBranchRecord(
            operation=operation,
            seed=seed,
            axis_yaw_rad=axis_yaw_rad,
            layout_axis=axis,
            common_prefix_steps=prefix_steps,
            qpos=_single(base.agent.robot.get_qpos()).astype(np.float64),
            object_pose=_single(base.obj.pose.raw_pose).astype(np.float64),
            anchor_pose=_single(base.anchor.pose.raw_pose).astype(np.float64),
            tcp_pose=_single(base.agent.tcp.pose.raw_pose).astype(np.float64),
            observation=observation,
            branch_target=branch_position,
            goal_position=_single(base.goal_region.pose.p).astype(np.float64),
            success=False,
        )
        _move_or_raise(
            planner,
            sapien.Pose(p=branch_position, q=orientation),
            f"{operation} side-hover pose",
        )
        contact_position = branch_position.copy()
        contact_position[2] = object_position[2]
        _move_or_raise(
            planner,
            sapien.Pose(p=contact_position, q=orientation),
            f"{operation} contact pose",
        )
        goal_position = np.asarray(base.goal_region.pose.sp.p, dtype=np.float64)
        final_position = (
            goal_position - direction * axis * config.goal_standoff_m
        )
        _move_or_raise(
            planner,
            sapien.Pose(p=final_position, q=orientation),
            f"{operation} goal pose",
        )
        success = bool(_single(base.evaluate()["success"]))
        return LayoutBranchRecord(
            **{**record.__dict__, "success": success}
        )
    finally:
        if planner is not None:
            planner.close()
        environment.close()


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / max(denominator, 1e-12))


def _pair_record(
    toward: LayoutBranchRecord,
    away: LayoutBranchRecord,
) -> dict[str, Any]:
    if toward.seed != away.seed:
        raise ValueError("pair seed 不一致")
    observation_keys_equal = set(toward.observation) == set(away.observation)
    observation_errors = {
        key: _maximum_absolute_error(
            toward.observation[key],
            away.observation[key],
        )
        for key in sorted(set(toward.observation) & set(away.observation))
    }
    state_errors = {
        "qpos": _maximum_absolute_error(toward.qpos, away.qpos),
        "object_pose": _maximum_absolute_error(
            toward.object_pose, away.object_pose
        ),
        "anchor_pose": _maximum_absolute_error(
            toward.anchor_pose, away.anchor_pose
        ),
        "tcp_pose": _maximum_absolute_error(toward.tcp_pose, away.tcp_pose),
        "layout_axis": _maximum_absolute_error(
            toward.layout_axis, away.layout_axis
        ),
    }
    branch_state = 0.5 * (toward.tcp_pose[:3] + away.tcp_pose[:3])
    toward_delta = toward.branch_target - branch_state
    away_delta = away.branch_target - branch_state
    return {
        "seed": toward.seed,
        "axis_yaw_degrees": math.degrees(toward.axis_yaw_rad),
        "layout_axis": toward.layout_axis.tolist(),
        "common_prefix_steps": {
            "toward": toward.common_prefix_steps,
            "away": away.common_prefix_steps,
            "equal": toward.common_prefix_steps == away.common_prefix_steps,
        },
        "state_max_abs_error": state_errors,
        "observation_keys_equal": observation_keys_equal,
        "observation_leaf_count": len(observation_errors),
        "observation_max_abs_error": max(
            observation_errors.values(), default=float("inf")
        ),
        "observation_leaf_errors": observation_errors,
        "branch_target_separation_m": float(
            np.linalg.norm(toward.branch_target - away.branch_target)
        ),
        "branch_target_delta_cosine": _cosine(toward_delta, away_delta),
        "goal_separation_m": float(
            np.linalg.norm(toward.goal_position - away.goal_position)
        ),
        "success": {"toward": toward.success, "away": away.success},
    }


def _wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _rotate_z(vector: np.ndarray, angle: float) -> np.ndarray:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    rotation = np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    )
    return rotation @ vector


def _cross_layout_records(
    records: dict[int, dict[str, LayoutBranchRecord]],
) -> list[dict[str, Any]]:
    output = []
    for query_seed, query_operations in records.items():
        for demo_seed, demo_operations in records.items():
            if query_seed == demo_seed:
                continue
            for operation in ENV_IDS:
                query = query_operations[operation]
                demo = demo_operations[operation]
                query_delta = query.branch_target - query.tcp_pose[:3]
                yaw_delta = _wrap_angle(
                    query.axis_yaw_rad - demo.axis_yaw_rad
                )
                # 先在物体中心坐标系表达 Demo 端点，再映射到 query。
                # raw baseline 只对齐平移；transport baseline 同时对齐 layout yaw。
                demo_relative_target = (
                    demo.branch_target - demo.object_pose[:3]
                )
                raw_target = query.object_pose[:3] + demo_relative_target
                transported_target = query.object_pose[:3] + _rotate_z(
                    demo_relative_target,
                    yaw_delta,
                )
                raw_delta = raw_target - query.tcp_pose[:3]
                transported_delta = transported_target - query.tcp_pose[:3]
                output.append(
                    {
                        "query_seed": query_seed,
                        "demo_seed": demo_seed,
                        "operation": operation,
                        "absolute_yaw_gap_degrees": abs(
                            math.degrees(yaw_delta)
                        ),
                        "raw_direction_cosine": _cosine(
                            raw_delta, query_delta
                        ),
                        "transported_direction_cosine": _cosine(
                            transported_delta, query_delta
                        ),
                        "transported_target_error_m": float(
                            np.linalg.norm(
                                transported_target - query.branch_target
                            )
                        ),
                    }
                )
    return output


def run(
    *,
    project_root: Path,
    config_path: Path,
    output_path: Path,
    config: RotatedLayoutPilotConfig,
) -> None:
    if output_path.exists():
        raise FileExistsError(f"输出已存在，拒绝覆盖：{output_path}")
    records: dict[int, dict[str, LayoutBranchRecord]] = {}
    pairs = []
    for seed, yaw_degrees in zip(config.seeds, config.axis_yaw_degrees):
        axis_yaw_rad = math.radians(yaw_degrees)
        records[seed] = {
            operation: _run_operation(
                operation,
                seed,
                axis_yaw_rad,
                config,
            )
            for operation in ENV_IDS
        }
        pairs.append(
            _pair_record(records[seed]["toward"], records[seed]["away"])
        )
        print(json.dumps(pairs[-1], sort_keys=True), flush=True)

    cross_layout = _cross_layout_records(records)
    large_gap = [
        row
        for row in cross_layout
        if row["absolute_yaw_gap_degrees"]
        >= config.minimum_large_yaw_gap_degrees
    ]
    if not large_gap:
        raise RuntimeError("pilot 没有满足阈值的大 yaw-gap contexts")
    state_maximum = max(
        value
        for pair in pairs
        for value in pair["state_max_abs_error"].values()
    )
    observation_maximum = max(
        pair["observation_max_abs_error"] for pair in pairs
    )
    minimum_separation = min(
        pair["branch_target_separation_m"] for pair in pairs
    )
    raw_large_mean = float(
        np.mean([row["raw_direction_cosine"] for row in large_gap])
    )
    transported_large_minimum = min(
        row["transported_direction_cosine"] for row in large_gap
    )
    criteria = {
        "P1_shared_prefix_length": all(
            pair["common_prefix_steps"]["equal"] for pair in pairs
        ),
        "P2_branch_state_equivalent": (
            state_maximum <= config.state_max_abs_tolerance
        ),
        "P3_predictor_observation_equivalent": (
            all(pair["observation_keys_equal"] for pair in pairs)
            and observation_maximum <= config.observation_max_abs_tolerance
        ),
        "P4_branch_action_target_separated": (
            minimum_separation >= config.minimum_branch_target_separation_m
        ),
        "P5_all_experts_successful": (
            all(
                pair["success"][operation]
                for pair in pairs
                for operation in ENV_IDS
            )
            if config.require_all_expert_success
            else True
        ),
        "P6_raw_copy_fails_large_yaw_gap": (
            raw_large_mean <= config.maximum_raw_large_gap_cosine
        ),
        "P7_layout_transport_recovers_direction": (
            transported_large_minimum
            >= config.minimum_transport_large_gap_cosine
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
        "cross_layout": cross_layout,
        "aggregate": {
            "pairs": len(pairs),
            "cross_layout_contexts": len(cross_layout),
            "large_gap_contexts": len(large_gap),
            "state_max_abs_error": state_maximum,
            "observation_max_abs_error": observation_maximum,
            "minimum_branch_target_separation_m": minimum_separation,
            "raw_large_gap_cosine_mean": raw_large_mean,
            "transported_large_gap_cosine_minimum": (
                transported_large_minimum
            ),
            "transported_target_error_maximum_m": max(
                row["transported_target_error_m"] for row in cross_layout
            ),
            "criteria": criteria,
            "all_criteria_passed": all(criteria.values()),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(f"{output_path.suffix}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    print(json.dumps(report["aggregate"], indent=2, sort_keys=True), flush=True)
    if not report["aggregate"]["all_criteria_passed"]:
        raise RuntimeError("RotatedLayoutSlide pilot 未通过全部预注册判据")


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
        config=RotatedLayoutPilotConfig.from_json(arguments.config.resolve()),
    )

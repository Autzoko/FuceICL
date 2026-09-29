"""采集严格配对的 BidirectionalSlide motion-planning trajectories。"""

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

import gymnasium as gym
import numpy as np
import sapien

from dev.pointnet.compare_retrievers import _sha256
from dev.simulator import bidirectional_slide_env  # noqa: F401
from dev.simulator.audit_bidirectional_slide_pairs import (
    ENV_IDS,
    _maximum_absolute_error,
    _move_or_raise,
    _single,
)
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)
from mani_skill.utils.wrappers.record import RecordEpisode


@dataclass(frozen=True)
class PairCollectionConfig:
    """冻结的 paired source trajectory 采集协议。"""

    schema_version: str
    start_seed: int
    target_pairs: int
    maximum_attempts: int
    validation_pairs: int
    test_pairs: int
    split_seed: int
    neutral_height_m: float
    contact_offset_m: float
    left_goal_standoff_m: float
    right_goal_standoff_m: float
    paired_state_tolerance: float
    minimum_branch_target_separation_m: float

    @classmethod
    def from_json(cls, path: Path) -> "PairCollectionConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def __post_init__(self) -> None:
        if self.schema_version != "bidirectional-slide-paired-collection-v1":
            raise ValueError("未知 paired collection schema")
        integers = (
            self.target_pairs,
            self.maximum_attempts,
            self.validation_pairs,
            self.test_pairs,
        )
        if min(integers) <= 0 or self.start_seed < 0 or self.split_seed < 0:
            raise ValueError("paired collection integer 配置非法")
        if self.maximum_attempts < self.target_pairs:
            raise ValueError("maximum_attempts 不能少于 target_pairs")
        if self.validation_pairs + self.test_pairs >= self.target_pairs:
            raise ValueError("train pairs 必须为正")
        positive = (
            self.neutral_height_m,
            self.contact_offset_m,
            self.left_goal_standoff_m,
            self.right_goal_standoff_m,
            self.paired_state_tolerance,
            self.minimum_branch_target_separation_m,
        )
        if min(positive) <= 0.0:
            raise ValueError("paired collection 距离/tolerance 必须为正")


@dataclass(frozen=True)
class ExpertRecord:
    operation: str
    seed: int
    common_prefix_steps: int
    total_steps: int
    qpos: np.ndarray
    object_pose: np.ndarray
    tcp_pose: np.ndarray
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


def _make_recorder(operation: str, output_root: Path) -> RecordEpisode:
    environment = gym.make(
        ENV_IDS[operation],
        obs_mode="none",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        sim_backend="physx_cpu",
    )
    return RecordEpisode(
        environment,
        output_dir=str(output_root / operation),
        trajectory_name="trajectory",
        save_trajectory=True,
        save_video=False,
        save_on_reset=False,
        clean_on_close=True,
        record_reward=True,
        source_type="motionplanning",
        source_desc=(
            "paired hidden-operation expert with a shared neutral prefix"
        ),
    )


def _execute(
    environment: RecordEpisode,
    operation: str,
    seed: int,
    config: PairCollectionConfig,
) -> ExpertRecord:
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
    try:
        planner.close_gripper()
        object_position = np.asarray(base.obj.pose.sp.p, dtype=np.float64)
        neutral_position = object_position.copy()
        neutral_position[2] += config.neutral_height_m
        neutral_pose = sapien.Pose(
            p=neutral_position,
            q=np.asarray(base.agent.tcp.pose.sp.q),
        )
        _move_or_raise(planner, neutral_pose, "shared neutral pose")
        prefix_steps = int(np.asarray(_single(base.elapsed_steps)).reshape(-1)[0])
        direction = float(base.operation_direction)
        branch_position = neutral_position.copy()
        branch_position[0] -= direction * config.contact_offset_m
        record = ExpertRecord(
            operation=operation,
            seed=seed,
            common_prefix_steps=prefix_steps,
            total_steps=0,
            qpos=_single(base.agent.robot.get_qpos()).astype(np.float64),
            object_pose=_single(base.obj.pose.raw_pose).astype(np.float64),
            tcp_pose=_single(base.agent.tcp.pose.raw_pose).astype(np.float64),
            branch_target=branch_position,
            goal_position=_single(base.goal_region.pose.p).astype(np.float64),
            success=False,
        )
        branch_pose = sapien.Pose(
            p=branch_position,
            q=np.asarray(base.agent.tcp.pose.sp.q),
        )
        _move_or_raise(planner, branch_pose, f"{operation} side-hover pose")
        contact_position = branch_position.copy()
        contact_position[2] = object_position[2]
        contact_pose = sapien.Pose(
            p=contact_position,
            q=np.asarray(base.agent.tcp.pose.sp.q),
        )
        _move_or_raise(planner, contact_pose, f"{operation} contact pose")
        goal_position = np.asarray(base.goal_region.pose.sp.p, dtype=np.float64)
        standoff = (
            config.right_goal_standoff_m
            if operation == "right"
            else config.left_goal_standoff_m
        )
        final_position = goal_position.copy()
        final_position[0] -= direction * standoff
        final_pose = sapien.Pose(
            p=final_position,
            q=np.asarray(base.agent.tcp.pose.sp.q),
        )
        _move_or_raise(planner, final_pose, f"{operation} goal pose")
        success = bool(_single(base.evaluate()["success"]))
        total_steps = int(np.asarray(_single(base.elapsed_steps)).reshape(-1)[0])
        return ExpertRecord(
            **{
                **record.__dict__,
                "total_steps": total_steps,
                "success": success,
            }
        )
    finally:
        planner.close()


def _pair_audit(
    left: ExpertRecord,
    right: ExpertRecord,
    config: PairCollectionConfig,
) -> dict[str, Any]:
    state_errors = {
        "qpos": _maximum_absolute_error(left.qpos, right.qpos),
        "object_pose": _maximum_absolute_error(
            left.object_pose,
            right.object_pose,
        ),
        "tcp_pose": _maximum_absolute_error(left.tcp_pose, right.tcp_pose),
    }
    maximum = max(state_errors.values())
    separation = float(np.linalg.norm(left.branch_target - right.branch_target))
    center = 0.5 * (left.tcp_pose[:3] + right.tcp_pose[:3])
    left_delta = left.branch_target - center
    right_delta = right.branch_target - center
    cosine = float(
        np.dot(left_delta, right_delta)
        / max(np.linalg.norm(left_delta) * np.linalg.norm(right_delta), 1e-12)
    )
    passed = (
        left.common_prefix_steps == right.common_prefix_steps
        and maximum <= config.paired_state_tolerance
        and separation >= config.minimum_branch_target_separation_m
    )
    return {
        "state_max_abs_error": state_errors,
        "maximum_state_error": maximum,
        "common_prefix_steps": left.common_prefix_steps,
        "branch_target_separation_m": separation,
        "branch_target_delta_cosine": cosine,
        "paired_audit_passed": passed,
    }


def _assign_splits(
    seeds: list[int],
    config: PairCollectionConfig,
) -> dict[int, str]:
    ranked = sorted(
        seeds,
        key=lambda seed: hashlib.sha256(
            f"{config.split_seed}:{seed}".encode()
        ).hexdigest(),
    )
    test = set(ranked[: config.test_pairs])
    validation = set(
        ranked[config.test_pairs : config.test_pairs + config.validation_pairs]
    )
    return {
        seed: "test" if seed in test else "val" if seed in validation else "train"
        for seed in seeds
    }


def run(
    *,
    project_root: Path,
    config_path: Path,
    output_root: Path,
    config: PairCollectionConfig,
) -> None:
    if output_root.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output_root}")
    temporary = output_root.with_name(f".{output_root.name}.incomplete-{os.getpid()}")
    temporary.mkdir(parents=True)

    def cleanup() -> None:
        shutil.rmtree(temporary, ignore_errors=True)

    atexit.register(cleanup)
    recorders = {
        operation: _make_recorder(operation, temporary)
        for operation in ("left", "right")
    }
    attempts = []
    accepted = []
    try:
        for offset in range(config.maximum_attempts):
            seed = config.start_seed + offset
            records: dict[str, ExpertRecord] = {}
            errors = {}
            for operation in ("left", "right"):
                try:
                    records[operation] = _execute(
                        recorders[operation],
                        operation,
                        seed,
                        config,
                    )
                # pilot 已验证路径；正式采集仍完整记录 planner failure。
                except Exception as error:
                    errors[operation] = f"{type(error).__name__}: {error}"
            pair_audit = None
            if not errors and len(records) == 2:
                pair_audit = _pair_audit(
                    records["left"],
                    records["right"],
                    config,
                )
                if not pair_audit["paired_audit_passed"]:
                    raise RuntimeError(f"seed={seed} paired state 审计失败")
            save = bool(
                not errors
                and pair_audit is not None
                and records["left"].success
                and records["right"].success
            )
            for recorder in recorders.values():
                recorder.flush_trajectory(save=save)
            attempt = {
                "seed": seed,
                "accepted": save,
                "errors": errors,
                "success": {
                    operation: records[operation].success
                    if operation in records
                    else False
                    for operation in ("left", "right")
                },
                "pair_audit": pair_audit,
            }
            attempts.append(attempt)
            print(json.dumps(attempt, sort_keys=True), flush=True)
            if save:
                accepted.append(
                    {
                        "pair_id": len(accepted),
                        "seed": seed,
                        "source_branch_frame": records[
                            "left"
                        ].common_prefix_steps,
                        "source_steps": {
                            operation: records[operation].total_steps
                            for operation in ("left", "right")
                        },
                        "pair_audit": pair_audit,
                    }
                )
            if len(accepted) == config.target_pairs:
                break
    finally:
        for recorder in recorders.values():
            recorder.close()
    if len(accepted) != config.target_pairs:
        raise RuntimeError(
            f"只采集 {len(accepted)}/{config.target_pairs} successful pairs"
        )
    splits = _assign_splits([int(row["seed"]) for row in accepted], config)
    for row in accepted:
        row["split"] = splits[int(row["seed"])]
    split_counts = {
        split: sum(row["split"] == split for row in accepted)
        for split in ("train", "val", "test")
    }
    files = {}
    for operation in ("left", "right"):
        root = temporary / operation
        h5_path = root / "trajectory.h5"
        json_path = root / "trajectory.json"
        if not h5_path.is_file() or not json_path.is_file():
            raise FileNotFoundError(f"{operation} source trajectory 文件缺失")
        files[operation] = {
            "h5": str(Path(operation) / h5_path.name),
            "h5_sha256": _sha256(h5_path),
            "json": str(Path(operation) / json_path.name),
            "json_sha256": _sha256(json_path),
        }
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(project_root),
        "config": asdict(config),
        "config_sha256": _sha256(config_path),
        "files": files,
        "attempts": attempts,
        "accepted_pairs": accepted,
        "summary": {
            "attempted_seeds": len(attempts),
            "accepted_pairs": len(accepted),
            "acceptance_rate": len(accepted) / len(attempts),
            "split_counts": split_counts,
            "maximum_paired_state_error": max(
                row["pair_audit"]["maximum_state_error"] for row in accepted
            ),
            "minimum_branch_target_separation_m": min(
                row["pair_audit"]["branch_target_separation_m"]
                for row in accepted
            ),
        },
    }
    (temporary / "collection_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_root)
    atexit.unregister(cleanup)
    print(json.dumps(report["summary"], indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(
        project_root=arguments.project_root.resolve(),
        config_path=arguments.config.resolve(),
        output_root=arguments.output_root.resolve(),
        config=PairCollectionConfig.from_json(arguments.config.resolve()),
    )

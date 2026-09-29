"""在真实 ManiSkill controller 上确认冻结 H6 Demo-conditioned predictor。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import gymnasium as gym
import h5py
import numpy as np
import torch

from dev.pointnet.compare_retrievers import _sha256
from dev.simulator import rotated_layout_slide_env  # noqa: F401
from dev.simulator.audit_bidirectional_slide_pairs import _single
from dev.simulator.audit_rotated_layout_slide_pairs import ENV_IDS
from dev.simulator.evaluate_maniskill_closed_loop import _scalar_bool
from dev.simulator.evaluate_rotated_layout_slide_demo_policy_confirmation import (
    _cross_transport,
    _load_data,
    _load_policy,
)
from dev.simulator.evaluate_rotated_layout_slide_transport import (
    OPERATIONS,
    SPLITS,
    _bootstrap_difference,
)
from dev.simulator.maniskill_action_bridge import canonical_to_controller
from dev.simulator.preprocess_bidirectional_slide_predictor import (
    _metadata_by_seed,
    _trajectory,
)


POLICIES = (
    "expert_replay",
    "point_transport",
    "demo_anchored_policy",
    "wrong_operation_policy",
    "no_demo",
)
OPERATION_NAMES = dict(OPERATIONS)
OPERATION_KEYS = tuple(OPERATION_NAMES.values())


@dataclass(frozen=True)
class ActionExecutionConfig:
    """冻结的 H6 controller 执行协议与门槛。"""

    schema_version: str
    seed: int
    expected_pairs: int
    action_horizon: int
    bootstrap_resamples: int
    position_limit_m: float
    rotation_scale_rad: float
    maximum_branch_alignment_error_m: float
    maximum_expert_endpoint_error_m: float
    minimum_axis_projection_m: float
    minimum_policy_direction_accuracy: float
    maximum_wrong_operation_direction_accuracy: float
    minimum_endpoint_error_improvement: float
    maximum_no_demo_displacement_m: float

    @classmethod
    def from_json(cls, path: Path) -> "ActionExecutionConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def __post_init__(self) -> None:
        if self.schema_version != "rotated-layout-slide-action-execution-v1":
            raise ValueError("未知 action execution schema")
        integers = (
            self.expected_pairs,
            self.action_horizon,
            self.bootstrap_resamples,
        )
        if min(integers) <= 0 or self.seed < 0:
            raise ValueError("action execution integer 配置非法")
        if self.action_horizon != 6:
            raise ValueError("v1 固定 H=6")
        positive = (
            self.position_limit_m,
            abs(self.rotation_scale_rad),
            self.maximum_branch_alignment_error_m,
            self.maximum_expert_endpoint_error_m,
            self.minimum_axis_projection_m,
            self.maximum_no_demo_displacement_m,
        )
        if min(positive) <= 0.0 or self.rotation_scale_rad == 0.0:
            raise ValueError("action execution scale/tolerance 必须为正")
        probabilities = (
            self.minimum_policy_direction_accuracy,
            self.maximum_wrong_operation_direction_accuracy,
            self.minimum_endpoint_error_improvement,
        )
        if any(not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("action execution rate 门槛必须位于 [0,1]")


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _stats(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("统计输入必须是一维有限非空数组")
    return {
        "count": len(array),
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": float(array.max()),
    }


def _nearest_demo(
    query_state: np.ndarray,
    bank_data: dict[str, np.ndarray],
    bank_indices: np.ndarray,
    operation_id: int,
    state_std: np.ndarray,
) -> tuple[int, float]:
    """在 typed operation bucket 内按冻结状态尺度选择最近 Demo。"""
    candidates = bank_indices[
        bank_data["operation_id"][bank_indices] == operation_id
    ]
    if not len(candidates):
        raise ValueError(f"operation={operation_id} 没有候选 Demo")
    normalized = (bank_data["state"][candidates] - query_state) / state_std
    distances = np.mean(normalized.astype(np.float64) ** 2, axis=1)
    nearest = int(np.argmin(distances))
    return int(candidates[nearest]), float(distances[nearest])


def _load_replay(
    replay_root: Path,
) -> tuple[
    dict[str, h5py.File],
    dict[str, dict[int, dict[str, Any]]],
    dict[str, Any],
]:
    audit_path = replay_root / "replay_audit_report.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("summary", {}).get("all_criteria_passed", False):
        raise ValueError("query replay audit 未通过")
    handles: dict[str, h5py.File] = {}
    episodes: dict[str, dict[int, dict[str, Any]]] = {}
    for operation in OPERATION_KEYS:
        stem = "trajectory.pointcloud.pd_ee_delta_pose.physx_cpu"
        h5_path = replay_root / operation / f"{stem}.h5"
        json_path = replay_root / operation / f"{stem}.json"
        expected = audit["files"][operation]["replay"]
        if _sha256(h5_path) != expected["h5_sha256"]:
            raise ValueError(f"{operation} replay HDF5 hash 不匹配")
        if _sha256(json_path) != expected["json_sha256"]:
            raise ValueError(f"{operation} replay JSON hash 不匹配")
        handles[operation] = h5py.File(h5_path, "r")
        episodes[operation] = _metadata_by_seed(json_path)
    return handles, episodes, audit


@torch.inference_mode()
def _predictions(
    *,
    model: torch.nn.Module,
    query_data: dict[str, np.ndarray],
    bank_data: dict[str, np.ndarray],
    query_index: int,
    bank_indices: np.ndarray,
    state_std: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any], float]:
    operation_id = int(query_data["operation_id"][query_index])
    correct_demo, correct_distance = _nearest_demo(
        query_data["state"][query_index],
        bank_data,
        bank_indices,
        operation_id,
        state_std,
    )
    wrong_demo, wrong_distance = _nearest_demo(
        query_data["state"][query_index],
        bank_data,
        bank_indices,
        1 - operation_id,
        state_std,
    )
    correct_anchor = _cross_transport(
        query_data, bank_data, query_index, correct_demo
    )
    wrong_anchor = _cross_transport(
        query_data, bank_data, query_index, wrong_demo
    )

    def infer(demo_index: int, anchor: np.ndarray, mask: float) -> np.ndarray:
        prediction = model(
            torch.from_numpy(query_data["state"][query_index][None]).float(),
            torch.from_numpy(bank_data["state"][demo_index][None]).float(),
            torch.from_numpy(anchor[None]).float(),
            torch.tensor([mask], dtype=torch.float32),
        )
        return prediction[0].cpu().numpy()

    started = time.perf_counter_ns()
    correct_policy = infer(correct_demo, correct_anchor, 1.0)
    wrong_policy = infer(wrong_demo, wrong_anchor, 1.0)
    no_demo = infer(correct_demo, correct_anchor, 0.0)
    latency_ms = (time.perf_counter_ns() - started) / 1e6
    output = {
        "point_transport": correct_anchor,
        "demo_anchored_policy": correct_policy,
        "wrong_operation_policy": wrong_policy,
        "no_demo": no_demo,
    }
    provenance = {
        "correct_demo_row": correct_demo,
        "correct_geometry_distance": correct_distance,
        "wrong_demo_row": wrong_demo,
        "wrong_geometry_distance": wrong_distance,
    }
    return output, provenance, latency_ms


def _environment(operation: str) -> gym.Env:
    environment = gym.make(
        ENV_IDS[operation],
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        sim_backend="physx_cpu",
        render_backend="gpu",
        num_envs=1,
        reconfiguration_freq=1,
    )
    if environment.action_space.shape != (7,):
        environment.close()
        raise ValueError(f"未预期的 action space：{environment.action_space}")
    return environment


def _pose(base: Any, actor: str) -> np.ndarray:
    value = (
        base.agent.tcp.pose.raw_pose
        if actor == "tcp"
        else base.obj.pose.raw_pose
    )
    return _single(value).astype(np.float64)


def _execute(
    *,
    environment: gym.Env,
    seed: int,
    yaw: float,
    prefix: np.ndarray,
    actions: np.ndarray,
    expert_controller: bool,
    config: ActionExecutionConfig,
) -> dict[str, Any]:
    environment.reset(seed=seed, options={"axis_yaw_rad": yaw})
    terminated_early = False
    truncated_early = False
    for action in prefix:
        _, _, terminated, truncated, _ = environment.step(action)
        terminated_early = terminated_early or _scalar_bool(terminated)
        truncated_early = truncated_early or _scalar_bool(truncated)
    base = environment.unwrapped
    branch_tcp = _pose(base, "tcp")
    branch_object = _pose(base, "object")
    translation_clips = 0
    rotation_clips = 0
    executed_actions = []
    for step, action in enumerate(actions):
        if expert_controller:
            controller = np.asarray(action, dtype=np.float32)
        else:
            converted = canonical_to_controller(
                action,
                _pose(base, "tcp"),
                position_limit_m=config.position_limit_m,
                rotation_scale_rad=config.rotation_scale_rad,
            )
            controller = converted.value
            translation_clips += int(converted.translation_clipped)
            rotation_clips += int(converted.rotation_clipped)
        _, _, terminated, truncated, _ = environment.step(controller)
        if step + 1 < len(actions):
            terminated_early = terminated_early or _scalar_bool(terminated)
            truncated_early = truncated_early or _scalar_bool(truncated)
        executed_actions.append(controller.tolist())
    final_tcp = _pose(base, "tcp")
    final_object = _pose(base, "object")
    return {
        "branch_tcp_pose": branch_tcp.tolist(),
        "branch_object_pose": branch_object.tolist(),
        "final_tcp_pose": final_tcp.tolist(),
        "final_object_pose": final_object.tolist(),
        "tcp_displacement_m": (final_tcp[:3] - branch_tcp[:3]).tolist(),
        "object_displacement_m": (
            final_object[:3] - branch_object[:3]
        ).tolist(),
        "translation_clips": translation_clips,
        "rotation_clips": rotation_clips,
        "executed_controller_actions": executed_actions,
        "terminated_early": terminated_early,
        "truncated_early": truncated_early,
        "final_success": _scalar_bool(base.evaluate()["success"]),
    }


def _summaries(
    rows: list[dict[str, Any]], config: ActionExecutionConfig
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for policy in POLICIES:
        selected = [row for row in rows if row["policy"] == policy]
        output[policy] = {
            "queries": len(selected),
            "direction_accuracy": float(
                np.mean([row["direction_valid"] for row in selected])
            ),
            "signed_axis_projection_m": _stats(
                [row["signed_axis_projection_m"] for row in selected]
            ),
            "tcp_displacement_m": _stats(
                [row["tcp_displacement_norm_m"] for row in selected]
            ),
            "object_displacement_m": _stats(
                [row["object_displacement_norm_m"] for row in selected]
            ),
            "branch_alignment_error_m": _stats(
                [row["branch_alignment_error_m"] for row in selected]
            ),
            "endpoint_error_to_expert_m": (
                _stats(
                    [row["endpoint_error_to_expert_m"] for row in selected]
                )
                if policy != "expert_replay"
                else None
            ),
            "translation_clip_rate": sum(
                row["translation_clips"] for row in selected
            )
            / (len(selected) * config.action_horizon),
            "rotation_clip_rate": sum(
                row["rotation_clips"] for row in selected
            )
            / (len(selected) * config.action_horizon),
            "early_termination_or_truncation_rate": float(
                np.mean(
                    [
                        row["terminated_early"] or row["truncated_early"]
                        for row in selected
                    ]
                )
            ),
            "success_rate_after_h6": float(
                np.mean([row["final_success"] for row in selected])
            ),
        }
    return output


def _pair_means(rows: list[dict[str, Any]], policy: str) -> np.ndarray:
    selected = [row for row in rows if row["policy"] == policy]
    pair_ids = sorted({int(row["pair_id"]) for row in selected})
    return np.asarray(
        [
            np.mean(
                [
                    row["endpoint_error_to_expert_m"]
                    for row in selected
                    if int(row["pair_id"]) == pair_id
                ]
            )
            for pair_id in pair_ids
        ],
        dtype=np.float64,
    )


def run(
    *,
    project_root: Path,
    config_path: Path,
    bank_root: Path,
    query_root: Path,
    training_root: Path,
    replay_root: Path,
    output_path: Path,
    config: ActionExecutionConfig,
) -> None:
    if output_path.exists():
        raise FileExistsError(f"输出已存在，拒绝覆盖：{output_path}")
    torch.set_num_threads(1)
    bank_data, bank_report = _load_data(bank_root)
    query_data, query_report = _load_data(query_root)
    if int(query_report["pairs"]) != config.expected_pairs:
        raise ValueError("query predictor pair 数与 execution 配置不一致")
    if len(query_data["pair_id"]) != 2 * config.expected_pairs:
        raise ValueError("query predictor rows 数量不匹配")
    if set(query_data["seed"]) & set(bank_data["seed"]):
        raise ValueError("query seeds 与 Demo bank 重叠")

    training_report_path = training_root / "report.json"
    checkpoint_path = training_root / "checkpoint.pt"
    training_report = json.loads(
        training_report_path.read_text(encoding="utf-8")
    )
    model, checkpoint = _load_policy(checkpoint_path, training_report)
    if _sha256(bank_root / "branch_samples.npz") != checkpoint["data_sha256"]:
        raise ValueError("checkpoint 与 Demo bank 不匹配")
    if model.config.action_horizon != config.action_horizon:
        raise ValueError("checkpoint action horizon 不匹配")
    state_std = model.state_std.detach().cpu().numpy()
    bank_indices = np.where(bank_data["split_id"] == SPLITS["train"])[0]
    if not len(bank_indices):
        raise ValueError("Demo bank train split 为空")

    handles, episodes, replay_audit = _load_replay(replay_root)
    audit_path = replay_root / "replay_audit_report.json"
    if query_report["replay_audit_sha256"] != _sha256(audit_path):
        raise ValueError("query predictor data 与 replay audit 不匹配")
    audit_pairs = {
        int(row["seed"]): row for row in replay_audit.get("pairs", [])
    }
    if len(audit_pairs) != config.expected_pairs:
        raise ValueError("replay audit pair 数量不匹配")

    environments = {
        operation: _environment(operation) for operation in OPERATION_KEYS
    }
    rows: list[dict[str, Any]] = []
    prediction_latencies = []
    try:
        for query_index in range(len(query_data["pair_id"])):
            operation_id = int(query_data["operation_id"][query_index])
            operation = OPERATION_NAMES[operation_id]
            seed = int(query_data["seed"][query_index])
            pair_id = int(query_data["pair_id"][query_index])
            yaw = float(query_data["axis_yaw_rad"][query_index])
            branch = int(query_data["branch_frame"][query_index])
            audit_row = audit_pairs[seed]
            if branch != int(audit_row["replay_branch_frame"]):
                raise ValueError(f"seed={seed} branch frame 不匹配")
            trajectory = _trajectory(handles[operation], episodes[operation][seed])
            controller_actions = np.asarray(
                trajectory["actions"], dtype=np.float32
            )
            recorded_tcp = np.asarray(
                trajectory["obs/extra/tcp_pose"], dtype=np.float64
            )
            if len(controller_actions) < branch + config.action_horizon:
                raise ValueError(f"seed={seed} replay action 长度不足")
            prefix = controller_actions[:branch]
            expert_chunk = controller_actions[
                branch : branch + config.action_horizon
            ]
            predictions, retrieval, latency = _predictions(
                model=model,
                query_data=query_data,
                bank_data=bank_data,
                query_index=query_index,
                bank_indices=bank_indices,
                state_std=state_std,
            )
            prediction_latencies.append(latency)
            action_by_policy = {
                "expert_replay": expert_chunk,
                **predictions,
            }
            expert_final: np.ndarray | None = None
            pending: list[dict[str, Any]] = []
            axis = np.asarray([math.cos(yaw), math.sin(yaw), 0.0])
            expected_sign = -1.0 if operation == "toward" else 1.0
            for policy in POLICIES:
                rollout = _execute(
                    environment=environments[operation],
                    seed=seed,
                    yaw=yaw,
                    prefix=prefix,
                    actions=action_by_policy[policy],
                    expert_controller=policy == "expert_replay",
                    config=config,
                )
                branch_tcp = np.asarray(rollout["branch_tcp_pose"])
                final_tcp = np.asarray(rollout["final_tcp_pose"])
                tcp_displacement = np.asarray(rollout["tcp_displacement_m"])
                object_displacement = np.asarray(
                    rollout["object_displacement_m"]
                )
                signed_projection = float(
                    expected_sign * np.dot(tcp_displacement, axis)
                )
                row = {
                    "query_row": query_index,
                    "pair_id": pair_id,
                    "seed": seed,
                    "operation": operation,
                    "policy": policy,
                    "axis_yaw_degrees": math.degrees(yaw),
                    "branch_frame": branch,
                    "branch_alignment_error_m": float(
                        np.linalg.norm(
                            branch_tcp[:3]
                            - query_data["tcp_pose"][query_index, :3]
                        )
                    ),
                    "signed_axis_projection_m": signed_projection,
                    "direction_valid": (
                        signed_projection >= config.minimum_axis_projection_m
                    ),
                    "tcp_displacement_norm_m": float(
                        np.linalg.norm(tcp_displacement)
                    ),
                    "object_displacement_norm_m": float(
                        np.linalg.norm(object_displacement)
                    ),
                    "action_mse_to_expert": (
                        None
                        if policy == "expert_replay"
                        else float(
                            np.mean(
                                (
                                    action_by_policy[policy]
                                    - query_data["action"][query_index]
                                )
                                ** 2
                            )
                        )
                    ),
                    "retrieval": retrieval if policy != "expert_replay" else None,
                    **rollout,
                }
                if policy == "expert_replay":
                    expert_final = final_tcp
                    row["recorded_endpoint_error_m"] = float(
                        np.linalg.norm(
                            final_tcp[:3]
                            - recorded_tcp[
                                branch + config.action_horizon, :3
                            ]
                        )
                    )
                    row["endpoint_error_to_expert_m"] = None
                else:
                    pending.append(row)
                rows.append(row)
            if expert_final is None:
                raise RuntimeError("expert reference 未执行")
            for row in pending:
                row["recorded_endpoint_error_m"] = None
                row["endpoint_error_to_expert_m"] = float(
                    np.linalg.norm(
                        np.asarray(row["final_tcp_pose"])[:3]
                        - expert_final[:3]
                    )
                )
    finally:
        for environment in environments.values():
            environment.close()
        for handle in handles.values():
            handle.close()

    summaries = _summaries(rows, config)
    point_pair = _pair_means(rows, "point_transport")
    policy_pair = _pair_means(rows, "demo_anchored_policy")
    comparison = _bootstrap_difference(
        policy_pair,
        point_pair,
        seed=config.seed,
        resamples=config.bootstrap_resamples,
    )
    point_mean = float(point_pair.mean())
    policy_mean = float(policy_pair.mean())
    relative_improvement = (point_mean - policy_mean) / max(point_mean, 1e-12)
    expert_rows = [row for row in rows if row["policy"] == "expert_replay"]
    maximum_expert_endpoint_error = max(
        float(row["recorded_endpoint_error_m"]) for row in expert_rows
    )
    maximum_branch_alignment = max(
        float(row["branch_alignment_error_m"]) for row in rows
    )
    any_early_stop = any(
        row["terminated_early"] or row["truncated_early"] for row in rows
    )
    learned_policies = (
        "point_transport",
        "demo_anchored_policy",
        "wrong_operation_policy",
        "no_demo",
    )
    no_clipping = all(
        summaries[policy]["translation_clip_rate"] == 0.0
        and summaries[policy]["rotation_clip_rate"] == 0.0
        for policy in learned_policies
    )
    criteria = {
        "E1_controller_execution_valid": (
            maximum_branch_alignment
            <= config.maximum_branch_alignment_error_m
            and maximum_expert_endpoint_error
            <= config.maximum_expert_endpoint_error_m
            and not any_early_stop
        ),
        "E2_demo_causality": (
            summaries["demo_anchored_policy"]["direction_accuracy"]
            >= config.minimum_policy_direction_accuracy
            and summaries["wrong_operation_policy"]["direction_accuracy"]
            <= config.maximum_wrong_operation_direction_accuracy
        ),
        "E3_residual_endpoint_improvement": (
            relative_improvement
            >= config.minimum_endpoint_error_improvement
            and comparison["ci95_high"] < 0.0
        ),
        "E4_structure_and_safety": (
            no_clipping
            and summaries["no_demo"]["tcp_displacement_m"]["maximum"]
            <= config.maximum_no_demo_displacement_m
        ),
    }
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(project_root),
        "config": asdict(config),
        "config_sha256": _sha256(config_path),
        "protocol": {
            "task_key": "typed relation_effect selects operation bucket",
            "geometry_retrieval": "nearest 21D normalized state within bucket",
            "execution": "reset plus controller-prefix replay; no env-state forcing",
            "confirmation_pairs": config.expected_pairs,
            "policies": list(POLICIES),
        },
        "provenance": {
            "bank_data_sha256": _sha256(bank_root / "branch_samples.npz"),
            "bank_report_sha256": _sha256(bank_root / "preprocess_report.json"),
            "query_data_sha256": _sha256(query_root / "branch_samples.npz"),
            "query_report_sha256": _sha256(query_root / "preprocess_report.json"),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "training_report_sha256": _sha256(training_report_path),
            "replay_audit_sha256": _sha256(audit_path),
            "query_replay_hash_matches": True,
            "seed_overlap": False,
        },
        "prediction_latency_ms_for_three_policy_forwards": _stats(
            prediction_latencies
        ),
        "summaries": summaries,
        "policy_minus_point_endpoint_error_pair_bootstrap": comparison,
        "endpoint_error_relative_improvement": relative_improvement,
        "maximum_branch_alignment_error_m": maximum_branch_alignment,
        "maximum_expert_endpoint_error_m": maximum_expert_endpoint_error,
        "criteria": criteria,
        "all_criteria_passed": all(criteria.values()),
        "rows": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(f"{output_path.suffix}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    print(
        json.dumps(
            {
                "summaries": summaries,
                "endpoint_comparison": comparison,
                "endpoint_error_relative_improvement": relative_improvement,
                "criteria": criteria,
                "all_criteria_passed": all(criteria.values()),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    if not all(criteria.values()):
        raise RuntimeError("H6 action execution 未通过全部预注册判据")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--query-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    resolved_config = arguments.config.resolve()
    run(
        project_root=arguments.project_root.resolve(),
        config_path=resolved_config,
        bank_root=arguments.bank_root.resolve(),
        query_root=arguments.query_root.resolve(),
        training_root=arguments.training_root.resolve(),
        replay_root=arguments.replay_root.resolve(),
        output_path=arguments.output.resolve(),
        config=ActionExecutionConfig.from_json(resolved_config),
    )

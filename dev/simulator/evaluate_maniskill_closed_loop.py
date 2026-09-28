"""在未见 PickCube seeds 上评估 retrieval-conditioned receding horizon。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Any, Mapping, Sequence

import gymnasium as gym
import mani_skill  # noqa: F401  # 导入时注册 ManiSkill environments。
import numpy as np
import sapien
import torch

from dev.pointnet.compare_retrievers import _sha256
from dev.predictor.jacobian_transport_model import (
    JacobianTransportConfig,
    LocalJacobianActionTransport,
)
from dev.predictor.train_action_chunks import _denormalize_actions
from dev.simulator.evaluate_maniskill_demo_prior import (
    _action_protocol,
    _load_split,
)
from dev.simulator.maniskill_action_bridge import (
    audit_round_trip,
    canonical_to_controller,
)
from dev.simulator.phase_transport import (
    PHASE_NAMES,
    fit_phase_weights,
    phase_ids,
    phase_matched_nearest,
    transport_actions,
)
from dev.simulator.preprocess_maniskill_chunks import (
    _active_points,
    _canonical_geometry,
    _gripper_open,
)


POLICIES = (
    "zero_pose_hold_gripper",
    "geometry_rank1_copy",
    "local_jacobian_transport",
    "factorized_transport",
)
CHUNK_POLICIES = frozenset(
    (
        "geometry_rank1_copy_h6",
        "phase_matched_copy_h6",
        "phase_factorized_transport_h6",
    )
)
SUPPORTED_POLICIES = frozenset((*POLICIES, *CHUNK_POLICIES))


@dataclass(frozen=True)
class ClosedLoopConfig:
    """预注册的 PickCube closed-loop pilot 配置。"""

    schema_version: str
    seeds: list[int]
    policies: list[str]
    max_episode_steps: int
    minimum_label_points: int
    position_limit_m: float
    rotation_scale_rad: float
    bootstrap_resamples: int
    stop_on_truncation: bool = True
    phase_ridge_lambda: float = 0.001

    @classmethod
    def from_json(cls, path: Path) -> "ClosedLoopConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def __post_init__(self) -> None:
        if not self.schema_version.strip():
            raise ValueError("schema_version 不能为空")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds 必须非空且不能重复")
        if (
            not self.policies
            or len(set(self.policies)) != len(self.policies)
            or not set(self.policies).issubset(SUPPORTED_POLICIES)
        ):
            raise ValueError(
                "policies 必须非空、唯一且属于支持集合"
            )
        if min(
            self.max_episode_steps,
            self.minimum_label_points,
            self.bootstrap_resamples,
        ) <= 0:
            raise ValueError("step、point 与 bootstrap 数必须为正")
        if self.position_limit_m <= 0 or self.rotation_scale_rad == 0:
            raise ValueError("controller scale 非法")
        if self.phase_ridge_lambda <= 0:
            raise ValueError("phase ridge lambda 必须为正")


@dataclass(frozen=True)
class ObservationContext:
    geometry: torch.Tensor | None
    tcp_pose: np.ndarray
    goal_position: np.ndarray
    qpos: np.ndarray
    active_center: np.ndarray | None
    visible_points: int


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _single(value: Any) -> np.ndarray:
    array = _to_numpy(value)
    return array[0] if array.ndim > 1 else array


def _scalar_bool(value: Any) -> bool:
    array = _to_numpy(value).reshape(-1)
    return bool(array[0]) if len(array) else False


def _read_source_seeds(path: Path) -> set[int]:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    return {int(episode["episode_seed"]) for episode in metadata["episodes"]}


def _active_label(records: Sequence[Mapping[str, Any]]) -> int:
    labels = {int(record["active_label"]) for record in records}
    if len(labels) != 1:
        raise ValueError(f"train manifest active labels 不唯一：{sorted(labels)}")
    return labels.pop()


def _observation_context(
    observation: Mapping[str, Any],
    *,
    previous_tcp_pose: np.ndarray | None,
    active_label: int,
    minimum_label_points: int,
) -> ObservationContext:
    pointcloud = observation["pointcloud"]
    xyzw = _single(pointcloud["xyzw"])
    segmentation = _single(pointcloud["segmentation"])
    tcp_pose = _single(observation["extra"]["tcp_pose"]).astype(np.float64)
    goal_position = _single(observation["extra"]["goal_pos"]).astype(np.float64)
    qpos = _single(observation["agent"]["qpos"]).astype(np.float64)
    points = _active_points(xyzw, segmentation, active_label)
    center = points.mean(axis=0) if len(points) else None
    if len(points) < minimum_label_points:
        return ObservationContext(
            None,
            tcp_pose,
            goal_position,
            qpos,
            center,
            len(points),
        )
    previous = tcp_pose if previous_tcp_pose is None else previous_tcp_pose
    geometry = _canonical_geometry(
        active_points=points,
        active_center=center,
        goal_position=goal_position,
        tcp_poses=np.stack((previous, tcp_pose)),
        qpos=np.stack((qpos, qpos)),
        frame=1,
    )
    return ObservationContext(
        torch.from_numpy(geometry).float(),
        tcp_pose,
        goal_position,
        qpos,
        center,
        len(points),
    )


def _load_model(
    checkpoint_path: Path,
    *,
    data_summary_sha256: str,
    device: torch.device,
) -> tuple[LocalJacobianActionTransport, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if payload.get("data_summary_sha256") != data_summary_sha256:
        raise ValueError("LJAT checkpoint 与 ManiSkill chunks 不匹配")
    model = LocalJacobianActionTransport(
        JacobianTransportConfig(**payload["model_config"])
    )
    model.load_state_dict(payload["model"])
    return model.to(device).eval(), payload


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _audit_runtime_controller(
    environment: gym.Env,
    config: ClosedLoopConfig,
) -> dict[str, Any]:
    """验证 action bridge 的常量与实际 ManiSkill controller 完全一致。"""
    controller = environment.unwrapped.agent.controller
    if "arm" not in controller.controllers or "gripper" not in controller.controllers:
        raise ValueError("Panda combined controller 缺少 arm/gripper")
    arm = controller.controllers["arm"]
    arm_config = arm.config
    position_lower = np.broadcast_to(arm_config.pos_lower, 3).astype(float)
    position_upper = np.broadcast_to(arm_config.pos_upper, 3).astype(float)
    rotation_lower = np.broadcast_to(arm_config.rot_lower, 3).astype(float)
    rotation_upper = np.broadcast_to(arm_config.rot_upper, 3).astype(float)
    if not np.allclose(position_lower, -config.position_limit_m):
        raise ValueError("runtime controller pos_lower 与 bridge 配置不一致")
    if not np.allclose(position_upper, config.position_limit_m):
        raise ValueError("runtime controller pos_upper 与 bridge 配置不一致")
    if not np.allclose(rotation_lower, config.rotation_scale_rad):
        raise ValueError("runtime controller rot_lower 与 bridge 配置不一致")
    if not np.allclose(rotation_upper, -config.rotation_scale_rad):
        raise ValueError("runtime controller rot_upper 与 bridge 配置不一致")
    expected_frame = "root_translation:root_aligned_body_rotation"
    if arm_config.frame != expected_frame or not arm_config.use_delta:
        raise ValueError("runtime controller frame/use_delta 与 bridge 假设不一致")
    if not arm_config.normalize_action:
        raise ValueError("runtime arm controller 未启用 action normalization")
    return {
        "combined_controller": type(controller).__name__,
        "arm_controller": type(arm).__name__,
        "frame": arm_config.frame,
        "use_delta": bool(arm_config.use_delta),
        "normalize_action": bool(arm_config.normalize_action),
        "position_lower": position_lower.tolist(),
        "position_upper": position_upper.tolist(),
        "rotation_lower": rotation_lower.tolist(),
        "rotation_upper": rotation_upper.tolist(),
        "action_mapping": {
            key: list(value) for key, value in controller.action_mapping.items()
        },
    }


@torch.inference_mode()
def _policy_action(
    *,
    policy: str,
    context: ObservationContext,
    train_records: Sequence[Mapping[str, Any]],
    train_geometry: torch.Tensor,
    train_actions: torch.Tensor,
    geometry_std: torch.Tensor,
    model: LocalJacobianActionTransport | None,
    pose_scales: torch.Tensor,
    config: ClosedLoopConfig,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    _sync(device)
    started = time.perf_counter()
    refusal = context.geometry is None
    selected_index: int | None = None
    distance: float | None = None
    if refusal or policy == "zero_pose_hold_gripper":
        canonical = np.zeros(7, dtype=np.float32)
        canonical[6] = _gripper_open(context.qpos)
    else:
        query = context.geometry.to(device)
        distances = torch.linalg.vector_norm(
            (train_geometry - query[None, :]) / geometry_std,
            dim=1,
        ) / math.sqrt(train_geometry.shape[1])
        selected_index = int(distances.argmin())
        distance = float(distances[selected_index].cpu())
        demo_geometry = train_geometry[selected_index : selected_index + 1]
        demo_actions = train_actions[selected_index : selected_index + 1]
        if policy == "geometry_rank1_copy":
            normalized = demo_actions
        else:
            if model is None:
                raise ValueError(f"policy={policy} 需要 LJAT checkpoint")
            demo_mask = torch.ones(
                demo_actions.shape[:2], dtype=torch.bool, device=device
            )
            transported = model(
                query[None, :],
                demo_geometry,
                demo_actions,
                demo_mask,
            )
            if policy == "local_jacobian_transport":
                normalized = transported
            elif policy == "factorized_transport":
                normalized = transported.clone()
                normalized[..., 3:6] = demo_actions[..., 3:6]
            else:
                raise ValueError(f"未知 policy：{policy}")
        canonical = _denormalize_actions(
            normalized, pose_scales
        )[0, 0].cpu().numpy()
    converted = canonical_to_controller(
        canonical,
        context.tcp_pose,
        position_limit_m=config.position_limit_m,
        rotation_scale_rad=config.rotation_scale_rad,
    )
    _sync(device)
    latency_ms = (time.perf_counter() - started) * 1000.0
    return converted.value, {
        "refusal": refusal,
        "visible_points": context.visible_points,
        "selected_index": selected_index,
        "selected_chunk_id": (
            None
            if selected_index is None
            else str(train_records[selected_index]["chunk_id"])
        ),
        "retrieval_distance": distance,
        "canonical_first_action": canonical.tolist(),
        "controller_action": converted.value.tolist(),
        "translation_clipped": converted.translation_clipped,
        "rotation_clipped": converted.rotation_clipped,
        "unscaled_translation_norm_m": converted.unscaled_translation_norm_m,
        "unscaled_rotation_norm_rad": converted.unscaled_rotation_norm_rad,
        "replanned": True,
        "plan_step": 0,
        "plan_length": 1,
        "policy_latency_ms": latency_ms,
    }


@torch.inference_mode()
def _geometry_copy_chunk(
    *,
    context: ObservationContext,
    train_geometry: torch.Tensor,
    train_actions: torch.Tensor,
    geometry_std: torch.Tensor,
    pose_scales: torch.Tensor,
    execution_horizon: int,
    device: torch.device,
) -> tuple[np.ndarray, int | None, float | None, bool]:
    """检索一次并返回待连续执行的物理 action tokens。"""
    if execution_horizon <= 0 or execution_horizon > train_actions.shape[1]:
        raise ValueError("execution horizon 超出保存的 action chunk")
    if context.geometry is None:
        canonical = np.zeros((1, 7), dtype=np.float32)
        canonical[0, 6] = _gripper_open(context.qpos)
        return canonical, None, None, True
    query = context.geometry.to(device)
    distances = torch.linalg.vector_norm(
        (train_geometry - query[None, :]) / geometry_std,
        dim=1,
    ) / math.sqrt(train_geometry.shape[1])
    selected_index = int(distances.argmin())
    normalized = train_actions[
        selected_index : selected_index + 1, :execution_horizon
    ]
    canonical = _denormalize_actions(normalized, pose_scales)[0].cpu().numpy()
    return canonical, selected_index, float(distances[selected_index].cpu()), False


@torch.inference_mode()
def _phase_matched_chunk(
    *,
    context: ObservationContext,
    train_geometry: torch.Tensor,
    train_actions: torch.Tensor,
    train_phase: torch.Tensor,
    geometry_std: torch.Tensor,
    pose_scales: torch.Tensor,
    phase_weights: torch.Tensor,
    execution_horizon: int,
    apply_transport: bool,
    device: torch.device,
) -> tuple[
    np.ndarray,
    int | None,
    float | None,
    bool,
    int | None,
    dict[str, float | int] | None,
]:
    """按当前 gripper phase 检索，并可选地迁移 Demo translation。"""
    if execution_horizon <= 0 or execution_horizon > train_actions.shape[1]:
        raise ValueError("execution horizon 超出保存的 action chunk")
    if context.geometry is None:
        canonical = np.zeros((1, 7), dtype=np.float32)
        canonical[0, 6] = _gripper_open(context.qpos)
        return canonical, None, None, True, None, None
    query = context.geometry.to(device)
    query_phase = phase_ids(query[None, :])
    distances = torch.linalg.vector_norm(
        (train_geometry - query[None, :]) / geometry_std,
        dim=1,
    ) / math.sqrt(train_geometry.shape[1])
    selected_index = int(
        phase_matched_nearest(
            distances=distances[None, :],
            query_phase=query_phase,
            candidate_phase=train_phase,
        )[0]
    )
    normalized = train_actions[
        selected_index : selected_index + 1, :execution_horizon
    ]
    diagnostics: dict[str, float | int] | None = None
    if apply_transport:
        normalized, diagnostics = transport_actions(
            query_geometry=query[None, :],
            demo_geometry=train_geometry[selected_index : selected_index + 1],
            demo_actions=normalized,
            phase=query_phase,
            weights=phase_weights,
            position_scale_m=float(pose_scales[0]),
        )
    canonical = _denormalize_actions(normalized, pose_scales)[0].cpu().numpy()
    return (
        canonical,
        selected_index,
        float(distances[selected_index].cpu()),
        False,
        int(query_phase[0]),
        diagnostics,
    )


def _initial_signature(
    observation: Mapping[str, Any], context: ObservationContext
) -> np.ndarray:
    goal = _single(observation["extra"]["goal_pos"])
    state = np.concatenate((context.tcp_pose, goal, context.qpos))
    if context.active_center is not None:
        state = np.concatenate((state, context.active_center))
    return state.astype(np.float64)


def _stats(values: Sequence[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(array),
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def _summarize_policy(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    steps = [step for record in records for step in record["step_records"]]
    successes = [bool(record["success"]) for record in records]
    initial_successes = [
        bool(record.get("initial_success", False)) for record in records
    ]
    success_steps = [
        int(record["success_step"])
        for record in records
        if record["success_step"] is not None
    ]
    grasp_steps = [
        next(
            (
                int(step["step"])
                for step in record["step_records"]
                if bool(step.get("is_grasped", False))
            ),
            None,
        )
        for record in records
    ]
    placed_steps = [
        next(
            (
                int(step["step"])
                for step in record["step_records"]
                if bool(step.get("is_obj_placed", False))
            ),
            None,
        )
        for record in records
    ]
    grasped_records = [
        record
        for record, first_grasp in zip(records, grasp_steps)
        if first_grasp is not None
    ]
    post_grasp_steps = []
    for record in grasped_records:
        first_index = next(
            index
            for index, step in enumerate(record["step_records"])
            if bool(step.get("is_grasped", False))
        )
        post_grasp_steps.extend(record["step_records"][first_index:])
    transport_clipped = sum(
        int(step.get("transport_translation_components_clipped", 0))
        for step in steps
    )
    transport_total = sum(
        int(step.get("transport_translation_components_total", 0))
        for step in steps
    )
    return {
        "episodes": len(records),
        "successes": sum(successes),
        "success_rate": float(np.mean(successes)),
        "initial_successes": sum(initial_successes),
        "non_initial_successes": sum(
            success and not initial
            for success, initial in zip(successes, initial_successes)
        ),
        "success_step": _stats(success_steps),
        "episodes_ever_grasped": sum(step is not None for step in grasp_steps),
        "episodes_finally_grasped": sum(
            bool(record["step_records"][-1].get("is_grasped", False))
            for record in records
            if record["step_records"]
        ),
        "episodes_lost_grasp_after_acquisition": sum(
            not bool(record["step_records"][-1].get("is_grasped", False))
            for record in grasped_records
            if record["step_records"]
        ),
        "first_grasp_step": _stats(
            [step for step in grasp_steps if step is not None]
        ),
        "episodes_ever_object_placed": sum(
            step is not None for step in placed_steps
        ),
        "first_object_placed_step": _stats(
            [step for step in placed_steps if step is not None]
        ),
        "episodes_ever_placed_and_static": sum(
            any(
                bool(step.get("is_obj_placed", False))
                and bool(step.get("is_robot_static", False))
                for step in record["step_records"]
            )
            for record in records
        ),
        "max_reward": _stats(
            [
                max(float(step["reward"]) for step in record["step_records"])
                for record in records
                if record["step_records"]
            ]
        ),
        "final_reward": _stats(
            [
                float(record["step_records"][-1]["reward"])
                for record in records
                if record["step_records"]
            ]
        ),
        "minimum_observed_object_to_goal_m": _stats(
            [
                min(
                    float(step["observed_object_to_goal_m"])
                    for step in record["step_records"]
                    if step["observed_object_to_goal_m"] is not None
                )
                for record in records
                if any(
                    step["observed_object_to_goal_m"] is not None
                    for step in record["step_records"]
                )
            ]
        ),
        "executed_steps": len(steps),
        "replans": sum(bool(step.get("replanned", True)) for step in steps),
        "refusals": sum(bool(step["refusal"]) for step in steps),
        "refusal_rate": float(np.mean([bool(step["refusal"]) for step in steps])),
        "translation_clip_rate": float(
            np.mean([bool(step["translation_clipped"]) for step in steps])
        ),
        "rotation_clip_rate": float(
            np.mean([bool(step["rotation_clipped"]) for step in steps])
        ),
        "transport_translation_component_clip_rate": (
            transport_clipped / transport_total if transport_total else None
        ),
        "post_grasp_open_command_rate": (
            float(
                np.mean(
                    [
                        float(step["canonical_first_action"][6]) > 0.0
                        for step in post_grasp_steps
                    ]
                )
            )
            if post_grasp_steps
            else None
        ),
        "policy_latency_ms": _stats(
            [float(step["policy_latency_ms"]) for step in steps]
        ),
        "preprocessing_latency_ms": _stats(
            [float(step["preprocessing_latency_ms"]) for step in steps]
        ),
        "end_to_end_policy_latency_ms": _stats(
            [float(step["end_to_end_policy_latency_ms"]) for step in steps]
        ),
        "environment_step_latency_ms": _stats(
            [float(step["environment_step_latency_ms"]) for step in steps]
        ),
        "retrieval_distance": _stats(
            [
                float(step["retrieval_distance"])
                for step in steps
                if step["retrieval_distance"] is not None
            ]
        ),
        "visible_points": _stats(
            [float(step["visible_points"]) for step in steps]
        ),
        "observed_tcp_to_object_m": _stats(
            [
                float(step["observed_tcp_to_object_m"])
                for step in steps
                if step["observed_tcp_to_object_m"] is not None
            ]
        ),
        "observed_object_to_goal_m": _stats(
            [
                float(step["observed_object_to_goal_m"])
                for step in steps
                if step["observed_object_to_goal_m"] is not None
            ]
        ),
    }


def _paired_bootstrap(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    seed: int,
    resamples: int,
) -> dict[str, float | int]:
    if reference.shape != candidate.shape or reference.ndim != 1:
        raise ValueError("paired bootstrap 输入 shape 不一致")
    rng = np.random.default_rng(seed)
    deltas = candidate - reference
    indices = rng.integers(0, len(deltas), size=(resamples, len(deltas)))
    sampled = deltas[indices].mean(axis=1)
    return {
        "candidate_minus_reference": float(deltas.mean()),
        "ci95_low": float(np.quantile(sampled, 0.025)),
        "ci95_high": float(np.quantile(sampled, 0.975)),
        "candidate_wins": int(np.sum(deltas > 0)),
        "reference_wins": int(np.sum(deltas < 0)),
        "ties": int(np.sum(deltas == 0)),
        "num_seed_pairs": len(deltas),
        "num_resamples": resamples,
    }


def _success_vector(
    records: Sequence[Mapping[str, Any]], seeds: Sequence[int]
) -> np.ndarray:
    by_seed = {int(record["seed"]): float(record["success"]) for record in records}
    if set(by_seed) != set(seeds):
        raise ValueError("policy rollout seeds 不完整")
    return np.asarray([by_seed[seed] for seed in seeds], dtype=np.float64)


def run(
    *,
    project_root: Path,
    data_root: Path,
    source_metadata_path: Path,
    checkpoint_path: Path | None,
    config_path: Path,
    output_path: Path,
    config: ClosedLoopConfig,
    device: torch.device,
) -> None:
    if output_path.exists():
        raise FileExistsError(f"输出已存在，拒绝覆盖：{output_path}")
    source_seeds = _read_source_seeds(source_metadata_path)
    overlap = source_seeds & set(config.seeds)
    if overlap:
        raise ValueError(f"closed-loop seeds 与 replay 重叠：{sorted(overlap)}")
    data_hash = _sha256(data_root / "summary.json")
    (
        pose_scales,
        _,
        _,
        action_representation,
    ) = _action_protocol(data_root)
    train_records, train_geometry_cpu, train_actions_cpu = _load_split(
        data_root, "train", pose_scales
    )
    active_label = _active_label(train_records)
    computed_mean = train_geometry_cpu.mean(dim=0)
    computed_std = train_geometry_cpu.std(dim=0, unbiased=False).clamp_min(1e-4)
    requires_ljat = bool(
        {"local_jacobian_transport", "factorized_transport"}
        & set(config.policies)
    )
    model: LocalJacobianActionTransport | None = None
    checkpoint: dict[str, Any] | None = None
    if checkpoint_path is not None:
        model, checkpoint = _load_model(
            checkpoint_path,
            data_summary_sha256=data_hash,
            device=device,
        )
        checkpoint_scales = checkpoint.get("action_pose_scales")
        if checkpoint_scales is not None and not torch.allclose(
            torch.as_tensor(checkpoint_scales).float(), pose_scales.float()
        ):
            raise ValueError("checkpoint action pose scales 与数据表示不一致")
        checkpoint_representation = checkpoint.get("action_representation")
        if (
            checkpoint_representation is not None
            and checkpoint_representation != action_representation
        ):
            raise ValueError("checkpoint action representation 与数据不一致")
        geometry_mean = checkpoint["geometry_mean"].float()
        geometry_std = checkpoint["geometry_std"].float().clamp_min(1e-4)
        if not torch.allclose(geometry_mean, computed_mean):
            raise ValueError("checkpoint geometry mean 与 train bank 不一致")
        if not torch.allclose(geometry_std, computed_std):
            raise ValueError("checkpoint geometry std 与 train bank 不一致")
    else:
        if requires_ljat:
            raise ValueError("LJAT policies 必须提供 checkpoint")
        geometry_std = computed_std

    phase_policy_requested = bool(
        {"phase_matched_copy_h6", "phase_factorized_transport_h6"}
        & set(config.policies)
    )
    train_phase_cpu = phase_ids(train_geometry_cpu)
    phase_weights_cpu = torch.zeros(
        2, 3, train_actions_cpu.shape[1] * 3, dtype=torch.float32
    )
    phase_fit: dict[str, Any] | None = None
    if phase_policy_requested:
        standardized_train = (train_geometry_cpu - computed_mean) / computed_std
        train_distances = torch.cdist(
            standardized_train, standardized_train
        ) / math.sqrt(train_geometry_cpu.shape[1])
        train_episodes = torch.tensor(
            [int(record["episode"]) for record in train_records]
        )
        phase_demo_indices = phase_matched_nearest(
            distances=train_distances,
            query_phase=train_phase_cpu,
            candidate_phase=train_phase_cpu,
            query_episodes=train_episodes,
            candidate_episodes=train_episodes,
        )
        phase_weights_cpu, phase_sample_counts = fit_phase_weights(
            query_geometry=train_geometry_cpu,
            demo_geometry=train_geometry_cpu[phase_demo_indices],
            target_actions=train_actions_cpu,
            demo_actions=train_actions_cpu[phase_demo_indices],
            phase=train_phase_cpu,
            position_scale_m=float(pose_scales[0]),
            ridge_lambda=config.phase_ridge_lambda,
        )
        phase_fit = {
            "model_parameters": int(phase_weights_cpu.numel()),
            "ridge_lambda": config.phase_ridge_lambda,
            "sample_counts": phase_sample_counts,
            "weights": phase_weights_cpu.tolist(),
            "weight_frobenius_norm": {
                PHASE_NAMES[index]: float(torch.linalg.matrix_norm(value))
                for index, value in enumerate(phase_weights_cpu)
            },
        }
    train_geometry = train_geometry_cpu.to(device)
    train_actions = train_actions_cpu.to(device)
    train_phase = train_phase_cpu.to(device)
    phase_weights = phase_weights_cpu.to(device)
    geometry_std = geometry_std.to(device)
    round_trip = audit_round_trip(
        position_limit_m=config.position_limit_m,
        rotation_scale_rad=config.rotation_scale_rad,
    )

    environment = gym.make(
        "PickCube-v1",
        obs_mode="pointcloud",
        control_mode="pd_ee_delta_pose",
        sim_backend="physx_cpu",
        render_backend="gpu",
        num_envs=1,
        reconfiguration_freq=1,
    )
    rollouts: dict[str, list[dict[str, Any]]] = {
        policy: [] for policy in config.policies
    }
    initial_references: dict[int, np.ndarray] = {}
    initial_signature_max_abs_difference = 0.0
    try:
        if environment.action_space.shape != (7,):
            raise ValueError(f"未预期的 action space：{environment.action_space}")
        runtime_controller = _audit_runtime_controller(environment, config)
        for policy in config.policies:
            for seed in config.seeds:
                observation, reset_info = environment.reset(seed=seed)
                previous_tcp_pose: np.ndarray | None = None
                buffered_actions: list[np.ndarray] = []
                buffered_index: int | None = None
                buffered_distance: float | None = None
                buffered_plan_length = 0
                buffered_plan_step = 0
                buffered_phase: int | None = None
                buffered_transport_diagnostics: dict[str, float | int] | None = None
                step_records = []
                success = _scalar_bool(reset_info.get("success", False))
                initial_success = success
                success_step: int | None = 0 if success else None
                for step in range(config.max_episode_steps):
                    _sync(device)
                    preprocessing_started = time.perf_counter()
                    context = _observation_context(
                        observation,
                        previous_tcp_pose=previous_tcp_pose,
                        active_label=active_label,
                        minimum_label_points=config.minimum_label_points,
                    )
                    _sync(device)
                    preprocessing_latency_ms = (
                        time.perf_counter() - preprocessing_started
                    ) * 1000.0
                    if step == 0:
                        signature = _initial_signature(observation, context)
                        reference = initial_references.setdefault(seed, signature)
                        if signature.shape != reference.shape:
                            raise ValueError("相同 seed 的 initial signature shape 不一致")
                        difference = float(np.max(np.abs(signature - reference)))
                        initial_signature_max_abs_difference = max(
                            initial_signature_max_abs_difference, difference
                        )
                        if difference > 5e-3:
                            raise ValueError(
                                f"seed={seed} 跨 policy 初始状态差异 {difference:.4g}"
                            )
                    if policy in CHUNK_POLICIES:
                        _sync(device)
                        policy_started = time.perf_counter()
                        replanned = not buffered_actions
                        refusal = False
                        if replanned:
                            buffered_phase = None
                            buffered_transport_diagnostics = None
                            if policy == "geometry_rank1_copy_h6":
                                (
                                    plan,
                                    buffered_index,
                                    buffered_distance,
                                    refusal,
                                ) = _geometry_copy_chunk(
                                    context=context,
                                    train_geometry=train_geometry,
                                    train_actions=train_actions,
                                    geometry_std=geometry_std,
                                    pose_scales=pose_scales,
                                    execution_horizon=train_actions.shape[1],
                                    device=device,
                                )
                            else:
                                (
                                    plan,
                                    buffered_index,
                                    buffered_distance,
                                    refusal,
                                    buffered_phase,
                                    buffered_transport_diagnostics,
                                ) = _phase_matched_chunk(
                                    context=context,
                                    train_geometry=train_geometry,
                                    train_actions=train_actions,
                                    train_phase=train_phase,
                                    geometry_std=geometry_std,
                                    pose_scales=pose_scales,
                                    phase_weights=phase_weights,
                                    execution_horizon=train_actions.shape[1],
                                    apply_transport=(
                                        policy
                                        == "phase_factorized_transport_h6"
                                    ),
                                    device=device,
                                )
                            buffered_actions = [token for token in plan]
                            buffered_plan_length = len(buffered_actions)
                            buffered_plan_step = 0
                        canonical = buffered_actions.pop(0)
                        converted = canonical_to_controller(
                            canonical,
                            context.tcp_pose,
                            position_limit_m=config.position_limit_m,
                            rotation_scale_rad=config.rotation_scale_rad,
                        )
                        _sync(device)
                        action = converted.value
                        action_record = {
                            "refusal": refusal,
                            "visible_points": context.visible_points,
                            "selected_index": buffered_index,
                            "selected_chunk_id": (
                                None
                                if buffered_index is None
                                else str(
                                    train_records[buffered_index]["chunk_id"]
                                )
                            ),
                            "retrieval_distance": buffered_distance,
                            "retrieval_phase": (
                                None
                                if buffered_phase is None
                                else PHASE_NAMES[buffered_phase]
                            ),
                            "canonical_first_action": canonical.tolist(),
                            "controller_action": action.tolist(),
                            "translation_clipped": (
                                converted.translation_clipped
                            ),
                            "rotation_clipped": converted.rotation_clipped,
                            "unscaled_translation_norm_m": (
                                converted.unscaled_translation_norm_m
                            ),
                            "unscaled_rotation_norm_rad": (
                                converted.unscaled_rotation_norm_rad
                            ),
                            "replanned": replanned,
                            "plan_step": buffered_plan_step,
                            "plan_length": buffered_plan_length,
                            "transport_translation_components_clipped": (
                                int(
                                    buffered_transport_diagnostics[
                                        "translation_components_clipped"
                                    ]
                                )
                                if replanned
                                and buffered_transport_diagnostics is not None
                                else 0
                            ),
                            "transport_translation_components_total": (
                                int(
                                    buffered_transport_diagnostics[
                                        "translation_components_total"
                                    ]
                                )
                                if replanned
                                and buffered_transport_diagnostics is not None
                                else 0
                            ),
                            "policy_latency_ms": (
                                time.perf_counter() - policy_started
                            )
                            * 1000.0,
                        }
                        buffered_plan_step += 1
                    else:
                        action, action_record = _policy_action(
                            policy=policy,
                            context=context,
                            train_records=train_records,
                            train_geometry=train_geometry,
                            train_actions=train_actions,
                            geometry_std=geometry_std,
                            model=model,
                            pose_scales=pose_scales,
                            config=config,
                            device=device,
                        )
                    action_record["preprocessing_latency_ms"] = (
                        preprocessing_latency_ms
                    )
                    action_record["observed_tcp_to_object_m"] = (
                        None
                        if context.active_center is None
                        else float(
                            np.linalg.norm(
                                context.active_center - context.tcp_pose[:3]
                            )
                        )
                    )
                    action_record["observed_object_to_goal_m"] = (
                        None
                        if context.active_center is None
                        else float(
                            np.linalg.norm(
                                context.active_center - context.goal_position
                            )
                        )
                    )
                    action_record["end_to_end_policy_latency_ms"] = (
                        preprocessing_latency_ms
                        + float(action_record["policy_latency_ms"])
                    )
                    step_started = time.perf_counter()
                    next_observation, reward, terminated, truncated, info = (
                        environment.step(action)
                    )
                    _sync(device)
                    environment_latency_ms = (
                        time.perf_counter() - step_started
                    ) * 1000.0
                    step_success = _scalar_bool(info.get("success", False))
                    action_record.update(
                        {
                            "step": step + 1,
                            "reward": float(_to_numpy(reward).reshape(-1)[0]),
                            "success": step_success,
                            "terminated": _scalar_bool(terminated),
                            "truncated": _scalar_bool(truncated),
                            "is_grasped": _scalar_bool(
                                info.get("is_grasped", False)
                            ),
                            "is_obj_placed": _scalar_bool(
                                info.get("is_obj_placed", False)
                            ),
                            "is_robot_static": _scalar_bool(
                                info.get("is_robot_static", False)
                            ),
                            "environment_step_latency_ms": environment_latency_ms,
                        }
                    )
                    step_records.append(action_record)
                    previous_tcp_pose = context.tcp_pose
                    observation = next_observation
                    if step_success:
                        success = True
                        success_step = step + 1
                        break
                    if _scalar_bool(terminated) or (
                        config.stop_on_truncation and _scalar_bool(truncated)
                    ):
                        break
                rollouts[policy].append(
                    {
                        "seed": seed,
                        "success": success,
                        "initial_success": initial_success,
                        "success_step": success_step,
                        "steps": len(step_records),
                        "step_records": step_records,
                    }
                )
                print(
                    json.dumps(
                        {
                            "policy": policy,
                            "seed": seed,
                            "success": success,
                            "success_step": success_step,
                            "steps": len(step_records),
                        }
                    ),
                    flush=True,
                )
    finally:
        environment.close()

    summaries = {
        policy: _summarize_policy(rollouts[policy]) for policy in config.policies
    }
    successes = {
        policy: _success_vector(rollouts[policy], config.seeds)
        for policy in config.policies
    }
    comparisons = {}
    pairs = (
        ("zero_pose_hold_gripper", "geometry_rank1_copy"),
        ("geometry_rank1_copy", "local_jacobian_transport"),
        ("local_jacobian_transport", "factorized_transport"),
        ("geometry_rank1_copy", "factorized_transport"),
        ("geometry_rank1_copy", "geometry_rank1_copy_h6"),
        ("geometry_rank1_copy_h6", "phase_matched_copy_h6"),
        ("phase_matched_copy_h6", "phase_factorized_transport_h6"),
        ("geometry_rank1_copy_h6", "phase_factorized_transport_h6"),
    )
    for offset, (reference, candidate) in enumerate(pairs):
        if reference not in successes or candidate not in successes:
            continue
        comparisons[f"{candidate}_minus_{reference}"] = _paired_bootstrap(
            successes[reference],
            successes[candidate],
            seed=config.seeds[0] + offset,
            resamples=config.bootstrap_resamples,
        )

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "task": "PickCube-v1",
            "observation": "current pointcloud segmentation + TCP/goal/qpos",
            "retrieval": (
                "policy-specific standardized 17D geometry top-1; phase "
                "policies constrain candidates by current gripper state"
            ),
            "execution": (
                "policy-specific H=1 receding horizon or open-loop H=6; "
                "controller conversion always uses the live TCP pose"
            ),
            "text_stage": "not applicable in fixed single-task closed-loop pilot",
            "success": (
                "first environment info.success within "
                f"{config.max_episode_steps} steps"
            ),
            "truncation": (
                "stop" if config.stop_on_truncation else "record but continue"
            ),
            "refusal": "visible active points < threshold; zero/hold action",
            "phase_diagnostics": (
                "environment is_grasped/is_obj_placed/is_robot_static are "
                "logged only after action execution and never used by policy"
            ),
            "seed_split": "all rollout seeds disjoint from 32 replay episodes",
        },
        "config": asdict(config),
        "action_protocol": {
            "representation": action_representation,
            "pose_scales": pose_scales.tolist(),
        },
        "device": str(device),
        "cuda_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "hostname": platform.node(),
        "versions": {
            "mani_skill": mani_skill.__version__,
            "gymnasium": gym.__version__,
            "numpy": np.__version__,
            "sapien": sapien.__version__,
            "torch": torch.__version__,
        },
        "git_commit": _git_commit(project_root),
        "data_summary_sha256": data_hash,
        "train_manifest_sha256": _sha256(data_root / "manifest-train.jsonl"),
        "source_metadata_sha256": _sha256(source_metadata_path),
        "checkpoint_sha256": (
            None if checkpoint_path is None else _sha256(checkpoint_path)
        ),
        "config_sha256": _sha256(config_path),
        "phase_transport_fit": phase_fit,
        "active_label_from_train_manifest": active_label,
        "seed_overlap_with_replay": [],
        "runtime_controller": runtime_controller,
        "controller_round_trip": round_trip,
        "initial_signature_max_abs_difference": (
            initial_signature_max_abs_difference
        ),
        "policy_summaries": summaries,
        "paired_seed_bootstrap": comparisons,
        "rollouts": rollouts,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(f"{output_path.suffix}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    print(
        json.dumps(
            {
                "policy_summaries": summaries,
                "paired_seed_bootstrap": comparisons,
                "output": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-metadata", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA，但当前节点没有可用 GPU")
    run(
        project_root=arguments.project_root.resolve(),
        data_root=arguments.data_root.resolve(),
        source_metadata_path=arguments.source_metadata.resolve(),
        checkpoint_path=(
            None
            if arguments.checkpoint is None
            else arguments.checkpoint.resolve()
        ),
        config_path=arguments.config.resolve(),
        output_path=arguments.output.resolve(),
        config=ClosedLoopConfig.from_json(arguments.config.resolve()),
        device=device,
    )


if __name__ == "__main__":
    main()

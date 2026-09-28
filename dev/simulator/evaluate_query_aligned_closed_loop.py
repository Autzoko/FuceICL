"""在 ManiSkill 单任务上评估冻结 QA-LRDT/BCSG action chunks。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import time
from typing import Any, Mapping, Sequence

import gymnasium as gym
import mani_skill  # noqa: F401  # 导入时注册 ManiSkill environments。
import numpy as np
import sapien
import torch

from dev.pointnet.compare_retrievers import _sha256
from dev.predictor.benefit_calibrated_shrinkage import (
    BenefitCalibratedGate,
    BenefitGateConfig,
    apply_shrinkage,
    frozen_transport_features,
)
from dev.predictor.low_rank_demo_transport import (
    LowRankDemoActionTransport,
    LowRankTransportConfig,
)
from dev.predictor.query_aligned_transport import (
    QueryAlignedLowRankDemoTransport,
    QueryAlignedTransportConfig,
)
from dev.predictor.train_action_chunks import _denormalize_actions
from dev.simulator.evaluate_maniskill_closed_loop import (
    _active_label,
    _audit_runtime_controller,
    _git_commit,
    _paired_bootstrap,
    _read_source_seeds,
    _scalar_bool,
    _single,
    _stats,
    _success_vector,
    _summarize_policy,
    _sync,
    _to_numpy,
)
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
    _canonical_geometry,
    _gripper_open,
    _segmented_points,
)


POLICIES = (
    "phase_matched_copy_h6",
    "phase_factorized_transport_h6",
    "fixed_low_rank_transport_h6",
    "query_aligned_transport_h6",
    "bcsg_h6",
)


@dataclass(frozen=True)
class ClosedLoopConfig:
    """冻结的三任务 closed-loop 配置。"""

    schema_version: str
    seed_generation_seed: int
    seeds: list[int]
    prior_evaluation_seeds: list[int]
    policies: list[str]
    max_episode_steps: int
    execution_horizon: int
    minimum_label_points: int
    position_limit_m: float
    rotation_scale_rad: float
    phase_ridge_lambda: float
    bootstrap_resamples: int
    stop_on_truncation: bool

    @classmethod
    def from_json(cls, path: Path) -> "ClosedLoopConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def __post_init__(self) -> None:
        if not self.schema_version.strip():
            raise ValueError("schema_version 不能为空")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("closed-loop seeds 必须非空且唯一")
        if set(self.seeds) & set(self.prior_evaluation_seeds):
            raise ValueError("closed-loop seeds 与先前评估重叠")
        if tuple(self.policies) != POLICIES:
            raise ValueError("本实验要求冻结的五个策略及顺序")
        positive = (
            self.max_episode_steps,
            self.execution_horizon,
            self.minimum_label_points,
            self.position_limit_m,
            abs(self.rotation_scale_rad),
            self.phase_ridge_lambda,
            self.bootstrap_resamples,
        )
        if min(positive) <= 0:
            raise ValueError("closed-loop 正数配置非法")


@dataclass(frozen=True)
class RuntimeContext:
    """由当前观测构造的、与离线预处理一致的策略输入。"""

    geometry: torch.Tensor | None
    tcp_pose: np.ndarray
    goal_position: np.ndarray | None
    qpos: np.ndarray
    active_center: np.ndarray | None
    active_visible_points: int
    target_visible_points: int | None


@dataclass(frozen=True)
class FrozenModels:
    """冻结 checkpoint 中部署所需的三个模型。"""

    fixed_transport: LowRankDemoActionTransport
    transport: QueryAlignedLowRankDemoTransport
    gate: BenefitCalibratedGate
    checkpoint: dict[str, Any]


def _unique_optional_label(
    records: Sequence[Mapping[str, Any]],
    key: str,
) -> int | None:
    values = {
        int(record[key])
        for record in records
        if record.get(key) is not None
    }
    if len(values) > 1:
        raise ValueError(f"manifest {key} 不唯一：{sorted(values)}")
    return next(iter(values)) if values else None


def _runtime_context(
    observation: Mapping[str, Any],
    *,
    previous_tcp_pose: np.ndarray | None,
    active_label: int,
    target_label: int | None,
    minimum_label_points: int,
) -> RuntimeContext:
    pointcloud = observation["pointcloud"]
    xyzw = _single(pointcloud["xyzw"])
    segmentation = _single(pointcloud["segmentation"])
    tcp_pose = _single(observation["extra"]["tcp_pose"]).astype(np.float64)
    qpos = _single(observation["agent"]["qpos"]).astype(np.float64)
    active_points = _segmented_points(xyzw, segmentation, active_label)
    active_center = active_points.mean(axis=0) if len(active_points) else None
    target_points = None
    if target_label is None:
        goal_position = _single(observation["extra"]["goal_pos"]).astype(
            np.float64
        )
        target_visible_points = None
    else:
        target_points = _segmented_points(xyzw, segmentation, target_label)
        goal_position = target_points.mean(axis=0) if len(target_points) else None
        target_visible_points = len(target_points)
    visible = len(active_points) >= minimum_label_points
    if target_points is not None:
        visible &= len(target_points) >= minimum_label_points
    if not visible or goal_position is None or active_center is None:
        return RuntimeContext(
            geometry=None,
            tcp_pose=tcp_pose,
            goal_position=goal_position,
            qpos=qpos,
            active_center=active_center,
            active_visible_points=len(active_points),
            target_visible_points=target_visible_points,
        )
    previous = tcp_pose if previous_tcp_pose is None else previous_tcp_pose
    geometry = _canonical_geometry(
        active_points=active_points,
        active_center=active_center,
        goal_position=goal_position,
        tcp_poses=np.stack((previous, tcp_pose)),
        qpos=np.stack((qpos, qpos)),
        frame=1,
    )
    return RuntimeContext(
        geometry=torch.from_numpy(geometry).float(),
        tcp_pose=tcp_pose,
        goal_position=goal_position,
        qpos=qpos,
        active_center=active_center,
        active_visible_points=len(active_points),
        target_visible_points=target_visible_points,
    )


def _initial_signature(context: RuntimeContext) -> np.ndarray:
    geometry = (
        np.zeros(17, dtype=np.float64)
        if context.geometry is None
        else context.geometry.numpy().astype(np.float64)
    )
    goal = (
        np.zeros(3, dtype=np.float64)
        if context.goal_position is None
        else context.goal_position
    )
    return np.concatenate(
        (
            context.tcp_pose,
            goal,
            context.qpos,
            geometry,
            np.asarray(
                [
                    context.active_visible_points,
                    context.target_visible_points or 0,
                ],
                dtype=np.float64,
            ),
        )
    )


def _load_models(
    checkpoint_path: Path,
    *,
    task_id: str,
    data_summary_sha256: str,
    pose_scales: torch.Tensor,
    action_representation: str,
    device: torch.device,
) -> FrozenModels:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    expected_hash = checkpoint.get("data_summary_sha256", {}).get(task_id)
    if expected_hash != data_summary_sha256:
        raise ValueError("checkpoint 与当前 task data summary 不匹配")
    if not torch.equal(
        torch.as_tensor(checkpoint["action_pose_scales"]).float(),
        pose_scales.float(),
    ):
        raise ValueError("checkpoint action scales 不匹配")
    if checkpoint.get("action_representation") != action_representation:
        raise ValueError("checkpoint action representation 不匹配")
    fixed_transport = LowRankDemoActionTransport(
        LowRankTransportConfig(
            **checkpoint["model_configs"]["fixed_low_rank_transport"]
        ),
        geometry_mean=checkpoint["geometry_mean"],
        geometry_std=checkpoint["geometry_std"],
    )
    fixed_transport.load_state_dict(
        checkpoint["models"]["fixed_low_rank_transport"]
    )
    transport = QueryAlignedLowRankDemoTransport(
        QueryAlignedTransportConfig(
            **checkpoint["model_configs"]["query_aligned_transport"]
        ),
        geometry_mean=checkpoint["geometry_mean"],
        geometry_std=checkpoint["geometry_std"],
    )
    transport.load_state_dict(
        checkpoint["models"]["query_aligned_transport"]
    )
    gate = BenefitCalibratedGate(
        BenefitGateConfig(**checkpoint["gate_config"])
    )
    gate.load_state_dict(checkpoint["gate"])
    return FrozenModels(
        fixed_transport=fixed_transport.to(device).eval(),
        transport=transport.to(device).eval(),
        gate=gate.to(device).eval(),
        checkpoint=checkpoint,
    )


def _bank_indices(
    records: Sequence[Mapping[str, Any]],
    expected_chunk_ids: Sequence[str],
) -> torch.Tensor:
    by_id = {str(record["chunk_id"]): index for index, record in enumerate(records)}
    if len(by_id) != len(records):
        raise ValueError("train manifest chunk_id 不唯一")
    missing = [chunk_id for chunk_id in expected_chunk_ids if chunk_id not in by_id]
    if missing:
        raise ValueError(f"checkpoint bank chunks 缺失：{missing[:3]}")
    indices = torch.tensor(
        [by_id[chunk_id] for chunk_id in expected_chunk_ids],
        dtype=torch.long,
    )
    if len(set(indices.tolist())) != len(indices):
        raise ValueError("checkpoint bank chunk IDs 重复")
    return indices


def _fit_phase_baseline(
    *,
    records: Sequence[Mapping[str, Any]],
    geometry: torch.Tensor,
    actions: torch.Tensor,
    position_scale_m: float,
    ridge_lambda: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    mean = geometry.mean(dim=0)
    std = geometry.std(dim=0, unbiased=False).clamp_min(1e-4)
    standardized = (geometry - mean) / std
    distances = torch.cdist(standardized, standardized) / math.sqrt(
        geometry.shape[1]
    )
    phases = phase_ids(geometry)
    episodes = torch.tensor([int(record["episode"]) for record in records])
    demo_indices = phase_matched_nearest(
        distances=distances,
        query_phase=phases,
        candidate_phase=phases,
        query_episodes=episodes,
        candidate_episodes=episodes,
    )
    weights, counts = fit_phase_weights(
        query_geometry=geometry,
        demo_geometry=geometry[demo_indices],
        target_actions=actions,
        demo_actions=actions[demo_indices],
        phase=phases,
        position_scale_m=position_scale_m,
        ridge_lambda=ridge_lambda,
    )
    return weights, {
        "model_parameters": int(weights.numel()),
        "ridge_lambda": ridge_lambda,
        "sample_counts": counts,
        "weight_frobenius_norm": {
            PHASE_NAMES[index]: float(torch.linalg.matrix_norm(value))
            for index, value in enumerate(weights)
        },
    }


@torch.inference_mode()
def _structural_audit(
    models: FrozenModels,
    geometry: torch.Tensor,
    actions: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    demo_geometry = geometry[:4].to(device)
    demo_actions = actions[:4].to(device)
    mask = torch.ones(demo_actions.shape[:2], dtype=torch.bool, device=device)
    identity, features = frozen_transport_features(
        models.transport,
        demo_geometry,
        demo_geometry,
        demo_actions,
        mask,
    )
    transported = models.transport(
        geometry[4:8].to(device),
        demo_geometry,
        demo_actions,
        mask,
    )
    zero_mask = torch.zeros_like(mask)
    zero_actions = torch.zeros_like(demo_actions)
    no_demo, no_demo_features = frozen_transport_features(
        models.transport,
        demo_geometry,
        torch.zeros_like(demo_geometry),
        zero_actions,
        zero_mask,
    )
    no_demo_bcsg = apply_shrinkage(
        zero_actions,
        no_demo,
        models.gate(no_demo_features),
        zero_mask,
    )
    fixed_identity = models.fixed_transport(
        demo_geometry,
        demo_geometry,
        demo_actions,
        mask,
    )
    fixed_no_demo = models.fixed_transport(
        demo_geometry,
        torch.zeros_like(demo_geometry),
        zero_actions,
        zero_mask,
    )
    return {
        "fixed_identity_exact": bool(torch.equal(fixed_identity, demo_actions)),
        "fixed_no_demo_exact": bool(torch.equal(fixed_no_demo, zero_actions)),
        "identity_exact": bool(torch.equal(identity, demo_actions)),
        "no_demo_qa_exact": bool(torch.equal(no_demo, zero_actions)),
        "no_demo_bcsg_exact": bool(torch.equal(no_demo_bcsg, zero_actions)),
        "gate_zero_exact": bool(
            torch.equal(
                apply_shrinkage(
                    demo_actions,
                    transported,
                    torch.zeros(len(demo_actions), device=device),
                    mask,
                ),
                demo_actions,
            )
        ),
        "gate_one_exact": bool(
            torch.equal(
                apply_shrinkage(
                    demo_actions,
                    transported,
                    torch.ones(len(demo_actions), device=device),
                    mask,
                ),
                transported,
            )
        ),
        "gate_feature_dim": int(features.shape[1]),
    }


@torch.inference_mode()
def _plan_chunk(
    *,
    policy: str,
    context: RuntimeContext,
    bank_geometry: torch.Tensor,
    bank_actions: torch.Tensor,
    bank_phase: torch.Tensor,
    retrieval_mean: torch.Tensor,
    retrieval_std: torch.Tensor,
    phase_weights: torch.Tensor,
    models: FrozenModels,
    pose_scales: torch.Tensor,
    execution_horizon: int,
    device: torch.device,
) -> tuple[np.ndarray, int | None, float | None, bool, dict[str, Any]]:
    if context.geometry is None:
        hold = np.zeros((1, 7), dtype=np.float32)
        hold[0, 6] = _gripper_open(context.qpos)
        return hold, None, None, True, {}
    query = context.geometry.to(device)
    distances = torch.linalg.vector_norm(
        ((bank_geometry - retrieval_mean) / retrieval_std)
        - ((query - retrieval_mean) / retrieval_std)[None],
        dim=1,
    ) / math.sqrt(bank_geometry.shape[1])
    query_phase = phase_ids(query[None])
    selected = int(
        phase_matched_nearest(
            distances=distances[None],
            query_phase=query_phase,
            candidate_phase=bank_phase,
        )[0]
    )
    demo_geometry = bank_geometry[selected : selected + 1]
    demo_actions = bank_actions[
        selected : selected + 1,
        :execution_horizon,
    ]
    diagnostics: dict[str, Any] = {}
    if policy == "phase_matched_copy_h6":
        prediction = demo_actions
    elif policy == "phase_factorized_transport_h6":
        prediction, diagnostics = transport_actions(
            query_geometry=query[None],
            demo_geometry=demo_geometry,
            demo_actions=demo_actions,
            phase=query_phase,
            weights=phase_weights,
            position_scale_m=float(pose_scales[0]),
        )
    elif policy == "fixed_low_rank_transport_h6":
        mask = torch.ones(
            demo_actions.shape[:2],
            dtype=torch.bool,
            device=device,
        )
        _sync(device)
        predictor_started = time.perf_counter()
        prediction = models.fixed_transport(
            query[None],
            demo_geometry,
            demo_actions,
            mask,
        )
        _sync(device)
        diagnostics = {
            "predictor_latency_ms": (
                time.perf_counter() - predictor_started
            )
            * 1000.0,
            "normalized_translation_residual_l2_mean": float(
                torch.linalg.vector_norm(
                    prediction[..., :3] - demo_actions[..., :3],
                    dim=-1,
                ).mean()
            ),
            "translation_components_outside_normalized_range": int(
                (prediction[..., :3].abs() > 1.0).sum()
            ),
        }
    elif policy in {"query_aligned_transport_h6", "bcsg_h6"}:
        mask = torch.ones(
            demo_actions.shape[:2],
            dtype=torch.bool,
            device=device,
        )
        _sync(device)
        predictor_started = time.perf_counter()
        if policy == "query_aligned_transport_h6":
            transported = models.transport(
                query[None],
                demo_geometry,
                demo_actions,
                mask,
            )
            prediction = transported
            gate_value = None
        else:
            transported, features = frozen_transport_features(
                models.transport,
                query[None],
                demo_geometry,
                demo_actions,
                mask,
            )
            gate = models.gate(features)
            prediction = apply_shrinkage(
                demo_actions,
                transported,
                gate,
                mask,
            )
            gate_value = float(gate[0])
        _sync(device)
        diagnostics = {
            "predictor_latency_ms": (
                time.perf_counter() - predictor_started
            )
            * 1000.0,
            "predicted_gate": gate_value,
            "normalized_translation_residual_l2_mean": float(
                torch.linalg.vector_norm(
                    transported[..., :3] - demo_actions[..., :3],
                    dim=-1,
                ).mean()
            ),
            "translation_components_outside_normalized_range": int(
                (prediction[..., :3].abs() > 1.0).sum()
            ),
        }
    else:
        raise ValueError(f"未知 closed-loop policy：{policy}")
    canonical = _denormalize_actions(prediction, pose_scales)[0].cpu().numpy()
    return canonical, selected, float(distances[selected]), False, diagnostics


def _policy_diagnostics(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    replans = [
        step
        for record in records
        for step in record["step_records"]
        if bool(step["replanned"])
    ]
    predictor_latencies = [
        float(step["predictor_latency_ms"])
        for step in replans
        if step.get("predictor_latency_ms") is not None
    ]
    gates = [
        float(step["predicted_gate"])
        for step in replans
        if step.get("predicted_gate") is not None
    ]
    residuals = [
        float(step["normalized_translation_residual_l2_mean"])
        for step in replans
        if step.get("normalized_translation_residual_l2_mean") is not None
    ]
    return {
        "predictor_latency_ms": _stats(predictor_latencies),
        "predicted_gate": _stats(gates),
        "normalized_translation_residual_l2_mean": _stats(residuals),
        "translation_components_outside_normalized_range": sum(
            int(step.get("translation_components_outside_normalized_range", 0))
            for step in replans
        ),
    }


def run(
    *,
    project_root: Path,
    task_id: str,
    data_root: Path,
    source_metadata_path: Path,
    checkpoint_path: Path,
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
    summary = json.loads((data_root / "summary.json").read_text(encoding="utf-8"))
    stored_task = str(summary.get("config", {}).get("task_id", ""))
    if stored_task != task_id:
        raise ValueError(f"data task={stored_task} 与 env={task_id} 不一致")
    pose_scales, _, _, action_representation = _action_protocol(data_root)
    records, full_geometry, full_actions = _load_split(
        data_root,
        "train",
        pose_scales,
    )
    models = _load_models(
        checkpoint_path,
        task_id=task_id,
        data_summary_sha256=data_hash,
        pose_scales=pose_scales,
        action_representation=action_representation,
        device=device,
    )
    expected_bank = models.checkpoint.get("bank_chunk_ids", {}).get(task_id)
    if not expected_bank:
        raise ValueError("checkpoint 缺少当前任务 Retriever bank")
    selection = _bank_indices(records, expected_bank)
    bank_records = [records[index] for index in selection.tolist()]
    bank_geometry_cpu = full_geometry[selection]
    bank_actions_cpu = full_actions[selection]
    if config.execution_horizon != bank_actions_cpu.shape[1]:
        raise ValueError("closed-loop horizon 与 checkpoint action chunks 不一致")
    active_label = _active_label(bank_records)
    target_label = _unique_optional_label(bank_records, "target_label")
    target_source = str(
        summary.get("config", {}).get(
            "target_position_source",
            "observation_goal_pos",
        )
    )
    if (target_source == "segmented_actor_centroid") != (target_label is not None):
        raise ValueError("target source 与 manifest target label 不一致")
    retrieval_mean_cpu = bank_geometry_cpu.mean(dim=0)
    retrieval_std_cpu = bank_geometry_cpu.std(
        dim=0,
        unbiased=False,
    ).clamp_min(1e-4)
    phase_weights_cpu, phase_fit = _fit_phase_baseline(
        records=bank_records,
        geometry=bank_geometry_cpu,
        actions=bank_actions_cpu,
        position_scale_m=float(pose_scales[0]),
        ridge_lambda=config.phase_ridge_lambda,
    )
    structural = _structural_audit(
        models,
        bank_geometry_cpu,
        bank_actions_cpu,
        device,
    )
    if not all(value for key, value in structural.items() if key.endswith("exact")):
        raise RuntimeError("checkpoint structural audit 失败")

    bank_geometry = bank_geometry_cpu.to(device)
    bank_actions = bank_actions_cpu.to(device)
    bank_phase = phase_ids(bank_geometry_cpu).to(device)
    retrieval_mean = retrieval_mean_cpu.to(device)
    retrieval_std = retrieval_std_cpu.to(device)
    phase_weights = phase_weights_cpu.to(device)
    environment = gym.make(
        task_id,
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
    initial_signatures: dict[int, np.ndarray] = {}
    initial_signature_max_abs_difference = 0.0
    try:
        if environment.action_space.shape != (7,):
            raise ValueError(f"未预期的 action space：{environment.action_space}")
        runtime_controller = _audit_runtime_controller(environment, config)
        for policy in config.policies:
            for seed in config.seeds:
                observation, reset_info = environment.reset(seed=seed)
                previous_tcp_pose: np.ndarray | None = None
                initial_context = _runtime_context(
                    observation,
                    previous_tcp_pose=None,
                    active_label=active_label,
                    target_label=target_label,
                    minimum_label_points=config.minimum_label_points,
                )
                signature = _initial_signature(initial_context)
                if seed not in initial_signatures:
                    initial_signatures[seed] = signature
                else:
                    difference = float(
                        np.max(np.abs(signature - initial_signatures[seed]))
                    )
                    initial_signature_max_abs_difference = max(
                        initial_signature_max_abs_difference,
                        difference,
                    )
                buffered_actions: list[np.ndarray] = []
                buffered_index: int | None = None
                buffered_distance: float | None = None
                buffered_refusal = False
                buffered_diagnostics: dict[str, Any] = {}
                buffered_plan_length = 0
                buffered_plan_step = 0
                step_records = []
                success = _scalar_bool(reset_info.get("success", False))
                initial_success = success
                success_step: int | None = 0 if success else None
                for step in range(config.max_episode_steps):
                    _sync(device)
                    preprocessing_started = time.perf_counter()
                    context = _runtime_context(
                        observation,
                        previous_tcp_pose=previous_tcp_pose,
                        active_label=active_label,
                        target_label=target_label,
                        minimum_label_points=config.minimum_label_points,
                    )
                    preprocessing_latency_ms = (
                        time.perf_counter() - preprocessing_started
                    ) * 1000.0
                    _sync(device)
                    policy_started = time.perf_counter()
                    replanned = not buffered_actions
                    if replanned:
                        (
                            plan,
                            buffered_index,
                            buffered_distance,
                            buffered_refusal,
                            buffered_diagnostics,
                        ) = _plan_chunk(
                            policy=policy,
                            context=context,
                            bank_geometry=bank_geometry,
                            bank_actions=bank_actions,
                            bank_phase=bank_phase,
                            retrieval_mean=retrieval_mean,
                            retrieval_std=retrieval_std,
                            phase_weights=phase_weights,
                            models=models,
                            pose_scales=pose_scales,
                            execution_horizon=config.execution_horizon,
                            device=device,
                        )
                        buffered_actions = [token for token in plan]
                        buffered_plan_length = len(buffered_actions)
                        buffered_plan_step = 0
                    canonical = buffered_actions.pop(0).copy()
                    converted = canonical_to_controller(
                        canonical,
                        context.tcp_pose,
                        position_limit_m=config.position_limit_m,
                        rotation_scale_rad=config.rotation_scale_rad,
                    )
                    _sync(device)
                    policy_latency_ms = (
                        time.perf_counter() - policy_started
                    ) * 1000.0
                    action_record = {
                        "refusal": buffered_refusal,
                        "visible_points": context.active_visible_points,
                        "target_visible_points": context.target_visible_points,
                        "selected_index": buffered_index,
                        "selected_chunk_id": (
                            None
                            if buffered_index is None
                            else str(bank_records[buffered_index]["chunk_id"])
                        ),
                        "retrieval_distance": buffered_distance,
                        "canonical_first_action": canonical.tolist(),
                        "controller_action": converted.value.tolist(),
                        "translation_clipped": converted.translation_clipped,
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
                        "grammar_projected": False,
                        "transport_translation_components_clipped": (
                            int(
                                buffered_diagnostics.get(
                                    "translation_components_clipped",
                                    0,
                                )
                            )
                            if replanned
                            else 0
                        ),
                        "transport_translation_components_total": (
                            int(
                                buffered_diagnostics.get(
                                    "translation_components_total",
                                    0,
                                )
                            )
                            if replanned
                            else 0
                        ),
                        "predictor_latency_ms": (
                            buffered_diagnostics.get("predictor_latency_ms")
                            if replanned
                            else None
                        ),
                        "predicted_gate": (
                            buffered_diagnostics.get("predicted_gate")
                            if replanned
                            else None
                        ),
                        "normalized_translation_residual_l2_mean": (
                            buffered_diagnostics.get(
                                "normalized_translation_residual_l2_mean"
                            )
                            if replanned
                            else None
                        ),
                        "translation_components_outside_normalized_range": (
                            int(
                                buffered_diagnostics.get(
                                    "translation_components_outside_normalized_range",
                                    0,
                                )
                            )
                            if replanned
                            else 0
                        ),
                        "policy_latency_ms": policy_latency_ms,
                        "preprocessing_latency_ms": preprocessing_latency_ms,
                        "end_to_end_policy_latency_ms": (
                            policy_latency_ms + preprocessing_latency_ms
                        ),
                        "observed_tcp_to_object_m": (
                            None
                            if context.active_center is None
                            else float(
                                np.linalg.norm(
                                    context.active_center - context.tcp_pose[:3]
                                )
                            )
                        ),
                        "observed_object_to_goal_m": (
                            None
                            if context.active_center is None
                            or context.goal_position is None
                            else float(
                                np.linalg.norm(
                                    context.active_center - context.goal_position
                                )
                            )
                        ),
                    }
                    buffered_plan_step += 1
                    step_started = time.perf_counter()
                    next_observation, reward, terminated, truncated, info = (
                        environment.step(converted.value)
                    )
                    _sync(device)
                    action_record.update(
                        {
                            "step": step + 1,
                            "reward": float(_to_numpy(reward).reshape(-1)[0]),
                            "success": _scalar_bool(info.get("success", False)),
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
                            "environment_step_latency_ms": (
                                time.perf_counter() - step_started
                            )
                            * 1000.0,
                        }
                    )
                    step_records.append(action_record)
                    previous_tcp_pose = context.tcp_pose
                    observation = next_observation
                    if action_record["success"]:
                        success = True
                        success_step = step + 1
                        break
                    if action_record["terminated"] or (
                        config.stop_on_truncation and action_record["truncated"]
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
                            "task": task_id,
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
        policy: _summarize_policy(rollouts[policy])
        for policy in config.policies
    }
    diagnostics = {
        policy: _policy_diagnostics(rollouts[policy])
        for policy in config.policies
    }
    success_vectors = {
        policy: _success_vector(rollouts[policy], config.seeds)
        for policy in config.policies
    }
    pairs = (
        ("phase_matched_copy_h6", "phase_factorized_transport_h6"),
        ("phase_matched_copy_h6", "fixed_low_rank_transport_h6"),
        ("phase_matched_copy_h6", "query_aligned_transport_h6"),
        ("phase_matched_copy_h6", "bcsg_h6"),
        ("fixed_low_rank_transport_h6", "bcsg_h6"),
        ("query_aligned_transport_h6", "bcsg_h6"),
        ("phase_factorized_transport_h6", "bcsg_h6"),
    )
    comparisons = {
        f"{candidate}_minus_{reference}": _paired_bootstrap(
            success_vectors[reference],
            success_vectors[candidate],
            seed=config.seed_generation_seed + offset,
            resamples=config.bootstrap_resamples,
        )
        for offset, (reference, candidate) in enumerate(pairs)
    }
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "task": task_id,
            "observation": (
                "current segmented pointcloud + TCP/goal/qpos canonical 17D"
            ),
            "retrieval": (
                "checkpoint operator bank; same phase; bank-standardized 17D top-1"
            ),
            "execution": "H=6 open-loop chunk with observation/retrieval every 6 steps",
            "predictor": "strict Demo-residual QA-LRDT with optional BCSG",
            "seed_unit": "paired policy reset on identical task seed",
        },
        "task": task_id,
        "config": asdict(config),
        "config_sha256": _sha256(config_path),
        "git_commit": _git_commit(project_root),
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
        "data_summary_sha256": data_hash,
        "train_manifest_sha256": _sha256(data_root / "manifest-train.jsonl"),
        "source_metadata_sha256": _sha256(source_metadata_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_git_commit": models.checkpoint.get("git_commit"),
        "bank": {
            "chunks": len(bank_records),
            "chunk_ids_match_checkpoint": True,
            "active_label": active_label,
            "target_label": target_label,
            "target_position_source": target_source,
        },
        "action_protocol": {
            "representation": action_representation,
            "pose_scales": pose_scales.tolist(),
        },
        "structural_audit": structural,
        "phase_transport_fit": phase_fit,
        "runtime_controller": runtime_controller,
        "controller_round_trip": audit_round_trip(
            position_limit_m=config.position_limit_m,
            rotation_scale_rad=config.rotation_scale_rad,
        ),
        "seed_overlap_with_replay": [],
        "initial_signature_max_abs_difference": (
            initial_signature_max_abs_difference
        ),
        "summaries": summaries,
        "policy_diagnostics": diagnostics,
        "paired_bootstrap": comparisons,
        "rollouts": rollouts,
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
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-metadata", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
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
        task_id=arguments.task_id,
        data_root=arguments.data_root.resolve(),
        source_metadata_path=arguments.source_metadata.resolve(),
        checkpoint_path=arguments.checkpoint.resolve(),
        config_path=arguments.config.resolve(),
        output_path=arguments.output.resolve(),
        config=ClosedLoopConfig.from_json(arguments.config.resolve()),
        device=device,
    )


if __name__ == "__main__":
    main()

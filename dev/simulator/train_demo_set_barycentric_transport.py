"""训练并审计轻量 Demo-Set Barycentric Transport（DSBT）。"""

from __future__ import annotations

import argparse
import atexit
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dev.pointnet.compare_retrievers import _sha256
from dev.predictor.benefit_calibrated_shrinkage import apply_shrinkage
from dev.predictor.demo_set_barycentric_transport import (
    DemoSetBarycentricConfig,
    DemoSetBarycentricTransport,
    mix_step_barycentric,
)
from dev.predictor.retriever_anchored_demo_mixture import (
    mix_demo_hypotheses,
    retrieval_prior,
)
from dev.predictor.train_action_chunks import _physical_metrics
from dev.simulator.evaluate_maniskill_demo_prior import _bootstrap_comparison
from dev.simulator.evaluate_radm_external_confirmation import _load_radm
from dev.simulator.train_causal_history_gate import (
    EXPECTED_TASKS,
    _checkpoint_bank_indices,
    _load_frozen_models,
)
from dev.simulator.train_maniskill_low_rank_transport import (
    TaskData,
    _load_task,
    _phase,
    _selection_hash,
)
from dev.simulator.train_maniskill_query_aligned_bcsg import _qa_diagnostics
from dev.simulator.train_retriever_anchored_demo_mixture import (
    PreparedSplit,
    _git_commit,
    _indices_for_episodes,
    _oracle_best,
    _oracle_convex,
    _predict as _predict_radm,
    _prepare_split,
    _seed_everything,
    _selection_digest,
    _stats,
    _sync,
)


EXPECTED_VARIANTS = {
    "set_step": {
        "use_set_context": True,
        "per_step_weights": True,
    },
    "independent_step": {
        "use_set_context": False,
        "per_step_weights": True,
    },
    "set_global": {
        "use_set_context": True,
        "per_step_weights": False,
    },
}


@dataclass(frozen=True)
class DSBTTrainConfig:
    """结果揭盲前冻结的 DSBT 可行性协议。"""

    schema_version: str
    seed: int
    source_checkpoint_sha256: str
    radm_checkpoint_sha256: str
    operator_train_episodes: list[int]
    mixture_train_episodes: list[int]
    validation_episodes: list[int]
    candidate_count: int
    action_horizon: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    token_dim: int
    scorer_hidden_dim: int
    maximum_odds_distortion: float
    gradient_clip_norm: float
    bootstrap_resamples: int
    latency_iterations: int
    primary_variant: str
    variants: dict[str, dict[str, bool]]
    maximum_new_model_parameters: int
    maximum_total_parameters: int
    maximum_latency_p95_ms: float
    minimum_improved_tasks: int

    @classmethod
    def from_json(cls, path: Path) -> "DSBTTrainConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def __post_init__(self) -> None:
        if self.schema_version != "maniskill-demo-set-barycentric-transport-v1":
            raise ValueError("未知 DSBT schema")
        positive = (
            self.candidate_count,
            self.action_horizon,
            self.epochs,
            self.batch_size,
            self.learning_rate,
            self.token_dim,
            self.scorer_hidden_dim,
            self.maximum_odds_distortion,
            self.gradient_clip_norm,
            self.bootstrap_resamples,
            self.latency_iterations,
            self.maximum_new_model_parameters,
            self.maximum_total_parameters,
            self.maximum_latency_p95_ms,
            self.minimum_improved_tasks,
        )
        if min(positive) <= 0 or self.weight_decay < 0.0:
            raise ValueError("DSBT 训练参数非法")
        if self.candidate_count != 4 or self.action_horizon != 6:
            raise ValueError("DSBT v1 固定 K=4、H=6")
        if self.maximum_odds_distortion != 4.0:
            raise ValueError("DSBT v1 固定 odds distortion=4")
        if self.primary_variant != "set_step" or self.variants != EXPECTED_VARIANTS:
            raise ValueError("DSBT 主模型或消融协议不匹配")
        if any(
            len(value) != 64
            for value in (
                self.source_checkpoint_sha256,
                self.radm_checkpoint_sha256,
            )
        ):
            raise ValueError("DSBT checkpoint SHA256 非法")
        groups = [
            set(self.operator_train_episodes),
            set(self.mixture_train_episodes),
            set(self.validation_episodes),
        ]
        originals = [
            self.operator_train_episodes,
            self.mixture_train_episodes,
            self.validation_episodes,
        ]
        if any(
            not group or len(group) != len(original)
            for group, original in zip(groups, originals)
        ):
            raise ValueError("DSBT episode split 不能为空或重复")
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise ValueError("DSBT episode splits 必须互斥")


@dataclass(frozen=True)
class Normalizers:
    geometry_delta_mean: torch.Tensor
    geometry_delta_std: torch.Tensor
    action_delta_mean: torch.Tensor
    action_delta_std: torch.Tensor
    distance_mean: torch.Tensor
    distance_std: torch.Tensor


def _normalizers(prepared: Sequence[PreparedSplit]) -> Normalizers:
    geometry_delta = torch.cat(
        [
            item.query_geometry[:, None] - item.demo_geometry
            for item in prepared
        ]
    )
    action_delta_values = []
    for item in prepared:
        prior_prediction = mix_demo_hypotheses(
            item.hypotheses,
            item.prior,
            item.mask,
        )
        action_delta_values.append(
            item.hypotheses[..., :3] - prior_prediction[:, None, :, :3]
        )
    action_delta = torch.cat(action_delta_values)
    distances = torch.cat([item.distances for item in prepared])
    return Normalizers(
        geometry_delta_mean=geometry_delta.flatten(0, 1).mean(dim=0),
        geometry_delta_std=geometry_delta.flatten(0, 1)
        .std(dim=0, unbiased=False)
        .clamp_min(1e-5),
        action_delta_mean=action_delta.flatten(0, 1).mean(dim=0),
        action_delta_std=action_delta.flatten(0, 1)
        .std(dim=0, unbiased=False)
        .clamp_min(1e-5),
        distance_mean=distances.mean(),
        distance_std=distances.std(unbiased=False).clamp_min(1e-5),
    )


def _make_model(
    *,
    config: DSBTTrainConfig,
    variant: Mapping[str, bool],
    geometry_dim: int,
    normalizers: Normalizers,
    device: torch.device,
) -> DemoSetBarycentricTransport:
    return DemoSetBarycentricTransport(
        DemoSetBarycentricConfig(
            geometry_dim=geometry_dim,
            action_horizon=config.action_horizon,
            token_dim=config.token_dim,
            scorer_hidden_dim=config.scorer_hidden_dim,
            maximum_odds_distortion=config.maximum_odds_distortion,
            use_set_context=variant["use_set_context"],
            per_step_weights=variant["per_step_weights"],
        ),
        geometry_delta_mean=normalizers.geometry_delta_mean,
        geometry_delta_std=normalizers.geometry_delta_std,
        action_delta_mean=normalizers.action_delta_mean,
        action_delta_std=normalizers.action_delta_std,
        distance_mean=normalizers.distance_mean,
        distance_std=normalizers.distance_std,
    ).to(device)


def _train_variant(
    *,
    name: str,
    prepared: Sequence[PreparedSplit],
    config: DSBTTrainConfig,
    normalizers: Normalizers,
    device: torch.device,
    log_path: Path,
) -> DemoSetBarycentricTransport:
    _seed_everything(config.seed)
    model = _make_model(
        config=config,
        variant=config.variants[name],
        geometry_dim=prepared[0].query_geometry.shape[1],
        normalizers=normalizers,
        device=device,
    )
    dataset = TensorDataset(
        torch.cat([item.query_geometry for item in prepared]),
        torch.cat([item.demo_geometry for item in prepared]),
        torch.cat([item.hypotheses for item in prepared]),
        torch.cat([item.distances for item in prepared]),
        torch.cat([item.prior for item in prepared]),
        torch.cat([item.mask for item in prepared]),
        torch.cat([item.target for item in prepared]),
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    for epoch in range(config.epochs):
        model.train()
        total = 0.0
        examples = 0
        for values in loader:
            query, demo, hypotheses, distances, prior, mask, target = [
                value.to(device) for value in values
            ]
            optimizer.zero_grad(set_to_none=True)
            weights = model(query, demo, hypotheses, distances, prior, mask)
            prediction = mix_step_barycentric(hypotheses, weights, mask)
            loss = (prediction[..., :3] - target[..., :3]).square().mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            total += float(loss.detach()) * len(query)
            examples += len(query)
        if epoch == 0 or (epoch + 1) % 10 == 0:
            row = {
                "variant": name,
                "epoch": epoch + 1,
                "translation_mse": total / examples,
            }
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
    return model.eval()


@torch.inference_mode()
def _predict_variant(
    model: DemoSetBarycentricTransport,
    prepared: PreparedSplit,
    device: torch.device,
    *,
    hypotheses: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    candidate_actions = prepared.hypotheses if hypotheses is None else hypotheses
    weights = model(
        prepared.query_geometry.to(device),
        prepared.demo_geometry.to(device),
        candidate_actions.to(device),
        prepared.distances.to(device),
        prepared.prior.to(device),
        prepared.mask.to(device),
    )
    prediction = mix_step_barycentric(
        candidate_actions.to(device),
        weights,
        prepared.mask.to(device),
    )
    return prediction.cpu(), weights.cpu()


def _same_phase_action_shuffle(
    prepared: PreparedSplit,
) -> tuple[torch.Tensor, torch.Tensor]:
    """同 task/phase 循环置换整组 action hypotheses，作为动作干预。"""
    mapping = torch.arange(len(prepared.target))
    phases = _phase(prepared.query_geometry)
    for phase in (False, True):
        indices = torch.flatnonzero(phases == phase)
        if len(indices) < 2:
            raise ValueError("同 phase shuffled intervention 候选不足")
        shift = max(1, len(indices) // 2)
        mapping[indices] = torch.roll(indices, shifts=shift)
    if bool((mapping == torch.arange(len(mapping))).any()):
        raise RuntimeError("shuffled intervention 包含固定点")
    return prepared.hypotheses[mapping], mapping


@torch.inference_mode()
def _structural_audit(
    *,
    model: DemoSetBarycentricTransport,
    prepared: Sequence[PreparedSplit],
    device: torch.device,
) -> dict[str, float | bool]:
    maximum_odds = 0.0
    convex_violation = 0.0
    deviation_bound_violation = 0.0
    discrete_error = 0.0
    permutation_error = 0.0
    for item in prepared:
        prediction, weights = _predict_variant(model, item, device)
        prior_step = item.prior[..., None]
        odds = (weights[:, :, None] / weights[:, None, :]) / (
            prior_step[:, :, None] / prior_step[:, None, :]
        )
        maximum_odds = max(maximum_odds, float(odds.max()))
        candidate = item.hypotheses[..., :3]
        predicted = prediction[..., :3]
        convex_violation = max(
            convex_violation,
            float((candidate.amin(dim=1) - predicted).clamp_min(0).max()),
            float((predicted - candidate.amax(dim=1)).clamp_min(0).max()),
        )
        prior_prediction = mix_demo_hypotheses(
            item.hypotheses,
            item.prior,
            item.mask,
        )
        pairwise = torch.linalg.vector_norm(
            candidate[:, :, None] - candidate[:, None, :],
            dim=-1,
        )
        diameter = pairwise.amax(dim=(1, 2))
        total_variation = 0.5 * (weights - prior_step).abs().sum(dim=1)
        deviation = torch.linalg.vector_norm(
            predicted - prior_prediction[..., :3],
            dim=-1,
        )
        deviation_bound_violation = max(
            deviation_bound_violation,
            float((deviation - diameter * total_variation).clamp_min(0).max()),
        )
        discrete_error = max(
            discrete_error,
            float(
                (
                    prediction[..., 3:]
                    - item.hypotheses[:, 0, :, 3:]
                ).abs().max()
            ),
        )
        permutation = torch.tensor([2, 0, 3, 1])
        permuted_weights = model(
            item.query_geometry.to(device),
            item.demo_geometry[:, permutation].to(device),
            item.hypotheses[:, permutation].to(device),
            item.distances[:, permutation].to(device),
            item.prior[:, permutation].to(device),
            item.mask[:, permutation].to(device),
        )
        permuted = mix_step_barycentric(
            item.hypotheses[:, permutation].to(device),
            permuted_weights,
            item.mask[:, permutation].to(device),
        ).cpu()
        permutation_error = max(
            permutation_error,
            float((prediction[..., :3] - permuted[..., :3]).abs().max()),
        )

    first = prepared[0]
    candidates = first.hypotheses.shape[1]
    empty_mask = torch.zeros((1, candidates), dtype=torch.bool, device=device)
    empty_weights = model(
        torch.zeros((1, first.query_geometry.shape[1]), device=device),
        torch.zeros((1, candidates, first.demo_geometry.shape[2]), device=device),
        torch.zeros(
            (1, candidates, first.hypotheses.shape[2], 7),
            device=device,
        ),
        torch.zeros((1, candidates), device=device),
        torch.zeros((1, candidates), device=device),
        empty_mask,
    )
    empty_output = mix_step_barycentric(
        torch.zeros(
            (1, candidates, first.hypotheses.shape[2], 7),
            device=device,
        ),
        empty_weights,
        empty_mask,
    )
    return {
        "maximum_odds_distortion": maximum_odds,
        "translation_convex_hull_max_violation": convex_violation,
        "deviation_bound_max_violation": deviation_bound_violation,
        "discrete_prior_max_abs_error": discrete_error,
        "translation_permutation_max_abs_error": permutation_error,
        "no_demo_weights_exact": bool(
            torch.equal(empty_weights, torch.zeros_like(empty_weights))
        ),
        "no_demo_action_exact": bool(
            torch.equal(empty_output, torch.zeros_like(empty_output))
        ),
    }


@torch.inference_mode()
def _latency(
    *,
    model: DemoSetBarycentricTransport,
    transport: nn.Module,
    gate: nn.Module,
    sample: PreparedSplit,
    iterations: int,
    device: torch.device,
) -> dict[str, float | int]:
    candidates = sample.hypotheses.shape[1]
    query = sample.query_geometry[:1].repeat(candidates, 1).to(device)
    query_batch = sample.query_geometry[:1].to(device)
    demo_geometry = sample.demo_geometry[:1].reshape(candidates, -1).to(device)
    demo_geometry_batch = sample.demo_geometry[:1].to(device)
    demo_actions = sample.demo_actions[:1].reshape(
        candidates,
        sample.demo_actions.shape[2],
        7,
    ).to(device)
    distances = sample.distances[:1].to(device)
    candidate_mask = torch.ones(
        (1, candidates),
        dtype=torch.bool,
        device=device,
    )
    values = []
    for iteration in range(iterations + 20):
        _sync(device)
        started = time.perf_counter()
        transported, features = _qa_diagnostics(
            transport,
            query,
            demo_geometry,
            demo_actions,
            device,
        )
        gates = gate(features.to(device))
        action_mask = torch.ones(
            demo_actions.shape[:2],
            dtype=torch.bool,
            device=device,
        )
        hypotheses = apply_shrinkage(
            demo_actions,
            transported.to(device),
            gates,
            action_mask,
        )[None]
        prior = retrieval_prior(distances, candidate_mask)
        weights = model(
            query_batch,
            demo_geometry_batch,
            hypotheses,
            distances,
            prior,
            candidate_mask,
        )
        mix_step_barycentric(hypotheses, weights, candidate_mask)
        _sync(device)
        if iteration >= 20:
            values.append((time.perf_counter() - started) * 1000.0)
    return {"iterations": len(values), **_stats(values)}


def _protocol_options(task: TaskData) -> dict[str, Any]:
    return {
        "pose_scales": task.pose_scales,
        "translation_threshold_m": task.translation_threshold_m,
        "rotation_threshold_rad": task.rotation_threshold_rad,
    }


def run(
    *,
    project_root: Path,
    data_roots: Sequence[Path],
    source_checkpoint_path: Path,
    radm_checkpoint_path: Path,
    config_path: Path,
    output_root: Path,
    config: DSBTTrainConfig,
    device: torch.device,
) -> None:
    if output_root.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output_root}")
    if _sha256(source_checkpoint_path) != config.source_checkpoint_sha256:
        raise ValueError("DSBT source checkpoint SHA256 不匹配")
    if _sha256(radm_checkpoint_path) != config.radm_checkpoint_sha256:
        raise ValueError("DSBT RADM checkpoint SHA256 不匹配")
    temporary = output_root.with_name(f".{output_root.name}.incomplete-{os.getpid()}")
    temporary.mkdir(parents=True)

    def cleanup() -> None:
        shutil.rmtree(temporary, ignore_errors=True)

    atexit.register(cleanup)
    _seed_everything(config.seed)
    checkpoint = torch.load(source_checkpoint_path, map_location="cpu")
    transport, gate = _load_frozen_models(checkpoint, device)
    radm = _load_radm(
        radm_checkpoint_path,
        config.source_checkpoint_sha256,
        device,
    )
    tasks = [_load_task(root) for root in data_roots]
    task_by_id = {task.task_id: task for task in tasks}
    if len(tasks) != 3 or set(task_by_id) != EXPECTED_TASKS:
        raise ValueError("DSBT 必须提供三个唯一 ManiSkill tasks")

    train_prepared = []
    validation_prepared = []
    data_audit = {}
    first_task = tasks[0]
    for task in tasks:
        if task.action_representation != checkpoint["action_representation"]:
            raise ValueError(f"{task.task_id} action representation 不匹配")
        if not torch.equal(
            task.pose_scales.float(),
            torch.as_tensor(checkpoint["action_pose_scales"]).float(),
        ):
            raise ValueError(f"{task.task_id} action scales 不匹配")
        if task.train_actions.shape[1] != config.action_horizon:
            raise ValueError(f"{task.task_id} action horizon 不匹配")
        if task.train_actions.shape[1:] != first_task.train_actions.shape[1:]:
            raise ValueError("DSBT task action shape 不一致")
        if (
            task.translation_threshold_m != first_task.translation_threshold_m
            or task.rotation_threshold_rad != first_task.rotation_threshold_rad
        ):
            raise ValueError("DSBT task evaluation thresholds 不一致")
        observed_train = {int(row["episode"]) for row in task.train_records}
        observed_validation = {int(row["episode"]) for row in task.val_records}
        expected_train = set(config.operator_train_episodes) | set(
            config.mixture_train_episodes
        )
        if observed_train != expected_train or observed_validation != set(
            config.validation_episodes
        ):
            raise ValueError(f"{task.task_id} split 与 DSBT 冻结协议不一致")
        summary_hash = _sha256(task.root / "summary.json")
        if checkpoint["data_summary_sha256"][task.task_id] != summary_hash:
            raise ValueError(f"{task.task_id} summary 与 checkpoint 不匹配")
        bank_indices = _checkpoint_bank_indices(
            task.train_records,
            checkpoint["bank_chunk_ids"][task.task_id],
        )
        train_indices = _indices_for_episodes(
            task.train_records,
            set(config.mixture_train_episodes),
        )
        bank_records = [task.train_records[index] for index in bank_indices.tolist()]
        bank_geometry = task.train_geometry[bank_indices]
        bank_actions = task.train_actions[bank_indices]
        train_prepared.append(
            _prepare_split(
                task_id=task.task_id,
                records=[
                    task.train_records[index] for index in train_indices.tolist()
                ],
                query_geometry=task.train_geometry[train_indices],
                target=task.train_actions[train_indices],
                bank_records=bank_records,
                bank_geometry=bank_geometry,
                bank_actions=bank_actions,
                transport=transport,
                gate=gate,
                candidate_count=config.candidate_count,
                device=device,
            )
        )
        validation_prepared.append(
            _prepare_split(
                task_id=task.task_id,
                records=task.val_records,
                query_geometry=task.val_geometry,
                target=task.val_actions,
                bank_records=bank_records,
                bank_geometry=bank_geometry,
                bank_actions=bank_actions,
                transport=transport,
                gate=gate,
                candidate_count=config.candidate_count,
                device=device,
            )
        )
        data_audit[task.task_id] = {
            "summary_sha256": summary_hash,
            "bank_chunks": len(bank_indices),
            "bank_selection_sha256": _selection_hash(bank_indices),
            "mixture_train_queries": len(train_indices),
            "validation_queries": len(task.val_records),
        }

    normalizers = _normalizers(train_prepared)
    models = {
        name: _train_variant(
            name=name,
            prepared=train_prepared,
            config=config,
            normalizers=normalizers,
            device=device,
            log_path=temporary / "training_metrics.jsonl",
        )
        for name in config.variants
    }
    primary = models[config.primary_variant]
    per_task = {}
    aggregate_target = []
    aggregate_groups = []
    aggregate_predictions: dict[str, list[torch.Tensor]] = {}
    aggregate_weights = []
    aggregate_prior = []
    aggregate_candidates = []
    artifact: dict[str, list[np.ndarray]] = {}
    for offset, prepared in enumerate(validation_prepared):
        task = task_by_id[prepared.task_id]
        radm_predictions, _ = _predict_radm(radm, prepared, device)
        predictions = {
            "rank1_bcsg": radm_predictions["rank1_bcsg"],
            "uniform_mixture": radm_predictions["uniform_mixture"],
            "retriever_prior_mixture": radm_predictions[
                "retriever_prior_mixture"
            ],
            "radm": radm_predictions["radm"],
            "oracle_best_in_4": _oracle_best(
                prepared.hypotheses,
                prepared.target,
            ),
            "oracle_convex": _oracle_convex(
                prepared.hypotheses,
                prepared.target,
            ),
        }
        variant_weights = {}
        for name, model in models.items():
            prediction, weights = _predict_variant(model, prepared, device)
            predictions[f"dsbt_{name}"] = prediction
            variant_weights[name] = weights
        shuffled_hypotheses, shuffle_mapping = _same_phase_action_shuffle(prepared)
        shuffled_prediction, _ = _predict_variant(
            primary,
            prepared,
            device,
            hypotheses=shuffled_hypotheses,
        )
        predictions["dsbt_set_step_shuffled_actions"] = shuffled_prediction
        action_mask = torch.ones(prepared.target.shape[:2], dtype=torch.bool)
        options = _protocol_options(task)
        metrics = {
            name: _physical_metrics(value, prepared.target, action_mask, **options)
            for name, value in predictions.items()
        }
        groups = [
            f"{prepared.task_id}:episode-{int(row['episode'])}"
            for row in prepared.records
        ]
        primary_name = f"dsbt_{config.primary_variant}"
        comparisons = {
            f"{primary_name}_minus_{reference}": _bootstrap_comparison(
                reference=predictions[reference],
                candidate=predictions[primary_name],
                target=prepared.target,
                mask=action_mask,
                group_ids=groups,
                seed=config.seed + offset * 100 + index,
                resamples=config.bootstrap_resamples,
                **options,
            )
            for index, reference in enumerate(
                (
                    "retriever_prior_mixture",
                    "radm",
                    "dsbt_independent_step",
                    "dsbt_set_global",
                )
            )
        }
        comparisons["shuffled_actions_minus_dsbt_set_step"] = (
            _bootstrap_comparison(
                reference=predictions[primary_name],
                candidate=predictions["dsbt_set_step_shuffled_actions"],
                target=prepared.target,
                mask=action_mask,
                group_ids=groups,
                seed=config.seed + offset * 100 + 10,
                resamples=config.bootstrap_resamples,
                **options,
            )
        )
        per_task[prepared.task_id] = {
            "queries": len(prepared.target),
            "episodes": len(set(groups)),
            "candidate_selection_sha256": _selection_digest(
                prepared.candidate_indices
            ),
            "shuffle_mapping_sha256": _selection_digest(
                shuffle_mapping[:, None]
            ),
            "metrics": metrics,
            "paired_episode_bootstrap": comparisons,
        }
        aggregate_target.append(prepared.target)
        aggregate_groups.extend(groups)
        aggregate_weights.append(variant_weights[config.primary_variant])
        aggregate_prior.append(prepared.prior)
        aggregate_candidates.append(prepared.hypotheses)
        for name, value in predictions.items():
            aggregate_predictions.setdefault(name, []).append(value)
        rows = {
            "task": np.asarray([prepared.task_id] * len(prepared.target)),
            "group_id": np.asarray(groups),
            "target": prepared.target.numpy(),
            "prior": prepared.prior.numpy(),
            "primary_weights": variant_weights[config.primary_variant].numpy(),
            "candidate_indices": prepared.candidate_indices.numpy(),
            **{name: value.numpy() for name, value in predictions.items()},
        }
        for name, value in rows.items():
            artifact.setdefault(name, []).append(value)

    target = torch.cat(aggregate_target)
    predictions = {
        name: torch.cat(values) for name, values in aggregate_predictions.items()
    }
    action_mask = torch.ones(target.shape[:2], dtype=torch.bool)
    options = _protocol_options(first_task)
    metrics = {
        name: _physical_metrics(value, target, action_mask, **options)
        for name, value in predictions.items()
    }
    primary_name = f"dsbt_{config.primary_variant}"
    comparisons = {
        f"{primary_name}_minus_{reference}": _bootstrap_comparison(
            reference=predictions[reference],
            candidate=predictions[primary_name],
            target=target,
            mask=action_mask,
            group_ids=aggregate_groups,
            seed=config.seed + 1000 + index,
            resamples=config.bootstrap_resamples,
            **options,
        )
        for index, reference in enumerate(
            (
                "retriever_prior_mixture",
                "radm",
                "dsbt_independent_step",
                "dsbt_set_global",
            )
        )
    }
    comparisons["shuffled_actions_minus_dsbt_set_step"] = _bootstrap_comparison(
        reference=predictions[primary_name],
        candidate=predictions["dsbt_set_step_shuffled_actions"],
        target=target,
        mask=action_mask,
        group_ids=aggregate_groups,
        seed=config.seed + 1010,
        resamples=config.bootstrap_resamples,
        **options,
    )
    structure = _structural_audit(
        model=primary,
        prepared=validation_prepared,
        device=device,
    )
    latency = _latency(
        model=primary,
        transport=transport,
        gate=gate,
        sample=validation_prepared[0],
        iterations=config.latency_iterations,
        device=device,
    )
    parameters = {
        "qa_lrdat": sum(value.numel() for value in transport.parameters()),
        "bcsg": sum(value.numel() for value in gate.parameters()),
        "radm_reference": sum(value.numel() for value in radm.parameters()),
        **{
            f"dsbt_{name}": sum(value.numel() for value in model.parameters())
            for name, model in models.items()
        },
    }
    primary_parameters = parameters[primary_name]
    total_primary_parameters = (
        parameters["qa_lrdat"] + parameters["bcsg"] + primary_parameters
    )
    primary_prior = comparisons[
        "dsbt_set_step_minus_retriever_prior_mixture"
    ]["translation_l2_m"]
    primary_radm = comparisons["dsbt_set_step_minus_radm"]["translation_l2_m"]
    shuffled = comparisons["shuffled_actions_minus_dsbt_set_step"][
        "translation_l2_m"
    ]
    improved_tasks = sum(
        values["metrics"][primary_name]["translation_l2_m"]
        < values["metrics"]["radm"]["translation_l2_m"]
        for values in per_task.values()
    )
    structural_passed = (
        structure["maximum_odds_distortion"] <= 4.0 + 1e-6
        and structure["translation_convex_hull_max_violation"] <= 1e-6
        and structure["deviation_bound_max_violation"] <= 1e-6
        and structure["discrete_prior_max_abs_error"] == 0.0
        and structure["translation_permutation_max_abs_error"] <= 1e-6
        and structure["no_demo_weights_exact"]
        and structure["no_demo_action_exact"]
    )
    criteria = {
        "d1_dsbt_significantly_better_than_fixed_prior": (
            primary_prior["ci95_high"] < 0.0
        ),
        "d2_dsbt_significantly_better_than_radm": (
            primary_radm["ci95_high"] < 0.0
        ),
        "d2_at_least_two_tasks_better_than_radm": (
            improved_tasks >= config.minimum_improved_tasks
        ),
        "d3_shuffled_demo_actions_significantly_worse": (
            shuffled["ci95_low"] > 0.0
        ),
        "d4_structural_guarantees": structural_passed,
        "d5_new_model_parameter_budget": (
            primary_parameters < config.maximum_new_model_parameters
        ),
        "d5_total_parameter_budget": (
            total_primary_parameters < config.maximum_total_parameters
        ),
        "d5_latency_p95_budget": (
            latency["p95"] < config.maximum_latency_p95_ms
        ),
    }

    checkpoint_output = {
        "schema_version": 1,
        "git_commit": _git_commit(project_root),
        "source_checkpoint_sha256": _sha256(source_checkpoint_path),
        "config_sha256": _sha256(config_path),
        "primary_variant": config.primary_variant,
        "variants": {
            name: {
                "config": asdict(model.config),
                "model": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
            }
            for name, model in models.items()
        },
    }
    checkpoint_path = temporary / "demo_set_barycentric_transport.pt"
    torch.save(checkpoint_output, checkpoint_path)
    artifact_path = temporary / "predictions.npz"
    with artifact_path.open("wb") as stream:
        np.savez_compressed(
            stream,
            **{name: np.concatenate(values) for name, values in artifact.items()},
        )
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(project_root),
        "protocol": (
            "K=4 frozen QA-LRDT/BCSG hypotheses; prior-anchored "
            "DeepSets context with per-action-step barycentric weights"
        ),
        "config": asdict(config),
        "config_sha256": _sha256(config_path),
        "source_checkpoint_sha256": _sha256(source_checkpoint_path),
        "radm_checkpoint_sha256": _sha256(radm_checkpoint_path),
        "device": str(device),
        "data": data_audit,
        "parameters": {
            **parameters,
            "total_primary": total_primary_parameters,
        },
        "latency_ms": latency,
        "structural": structure,
        "per_task": per_task,
        "aggregate": {
            "queries": len(target),
            "episodes": len(set(aggregate_groups)),
            "metrics": metrics,
            "paired_episode_bootstrap": comparisons,
            "primary_weight_rank_mean": torch.cat(aggregate_weights)
            .mean(dim=(0, 2))
            .tolist(),
            "retrieval_prior_rank_mean": torch.cat(aggregate_prior)
            .mean(dim=0)
            .tolist(),
            "candidate_action_span": {
                "mean": float(
                    torch.linalg.vector_norm(
                        torch.cat(aggregate_candidates)[..., :3][:, :, None]
                        - torch.cat(aggregate_candidates)[..., :3][:, None, :],
                        dim=-1,
                    ).amax(dim=(1, 2, 3)).mean()
                )
            },
        },
        "criteria": criteria,
        "feasibility_passed": all(criteria.values()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "predictions_sha256": _sha256(artifact_path),
    }
    (temporary / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_root)
    atexit.unregister(cleanup)
    print(json.dumps(report["aggregate"], indent=2), flush=True)
    print(json.dumps(report["structural"], indent=2), flush=True)
    print(json.dumps(criteria, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, action="append", required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--radm-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(
        project_root=arguments.project_root.resolve(),
        data_roots=[path.resolve() for path in arguments.data_root],
        source_checkpoint_path=arguments.source_checkpoint.resolve(),
        radm_checkpoint_path=arguments.radm_checkpoint.resolve(),
        config_path=arguments.config.resolve(),
        output_root=arguments.output_root.resolve(),
        config=DSBTTrainConfig.from_json(arguments.config.resolve()),
        device=torch.device(arguments.device),
    )

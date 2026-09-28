"""训练并评估受 Retriever prior 约束的多 Demo action mixture。"""

from __future__ import annotations

import argparse
import atexit
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dev.pointnet.compare_retrievers import _sha256
from dev.predictor.benefit_calibrated_shrinkage import apply_shrinkage
from dev.predictor.retriever_anchored_demo_mixture import (
    CANDIDATE_FEATURE_DIM,
    RetrieverAnchoredDemoMixture,
    RetrieverAnchoredMixtureConfig,
    mix_demo_hypotheses,
    retrieval_prior,
)
from dev.predictor.train_action_chunks import _physical_metrics
from dev.simulator.evaluate_maniskill_demo_prior import _bootstrap_comparison
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


@dataclass(frozen=True)
class RADMTrainConfig:
    """冻结的多 Demo mixture 可行性协议。"""

    schema_version: str
    seed: int
    source_checkpoint_sha256: str
    operator_train_episodes: list[int]
    mixture_train_episodes: list[int]
    validation_episodes: list[int]
    candidate_count: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    hidden_dim: int
    maximum_odds_distortion: float
    gradient_clip_norm: float
    bootstrap_resamples: int
    latency_iterations: int

    @classmethod
    def from_json(cls, path: Path) -> "RADMTrainConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def __post_init__(self) -> None:
        if self.schema_version != "maniskill-retriever-anchored-demo-mixture-v1":
            raise ValueError("未知 RADM schema")
        positive = (
            self.candidate_count,
            self.epochs,
            self.batch_size,
            self.learning_rate,
            self.hidden_dim,
            self.maximum_odds_distortion,
            self.gradient_clip_norm,
            self.bootstrap_resamples,
            self.latency_iterations,
        )
        if min(positive) <= 0 or self.weight_decay < 0:
            raise ValueError("RADM 训练参数非法")
        if self.candidate_count != 4 or self.maximum_odds_distortion != 4.0:
            raise ValueError("RADM v1 固定 K=4、odds distortion=4")
        if len(self.source_checkpoint_sha256) != 64:
            raise ValueError("RADM source checkpoint SHA256 非法")
        groups = tuple(
            set(values)
            for values in (
                self.operator_train_episodes,
                self.mixture_train_episodes,
                self.validation_episodes,
            )
        )
        originals = (
            self.operator_train_episodes,
            self.mixture_train_episodes,
            self.validation_episodes,
        )
        if any(not values for values in groups) or any(
            len(values) != len(original)
            for values, original in zip(groups, originals)
        ):
            raise ValueError("RADM episode split 不能为空或重复")
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise ValueError("RADM episode split 必须互斥")


@dataclass(frozen=True)
class PreparedSplit:
    """冻结候选前向后供 mixture 训练/评估的数据。"""

    task_id: str
    records: list[dict[str, Any]]
    target: torch.Tensor
    hypotheses: torch.Tensor
    features: torch.Tensor
    distances: torch.Tensor
    prior: torch.Tensor
    mask: torch.Tensor
    candidate_indices: torch.Tensor
    query_geometry: torch.Tensor
    demo_geometry: torch.Tensor
    demo_actions: torch.Tensor


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _sync(device: torch.device) -> None:
    """仅在 CUDA 计时时同步，避免引入仿真环境依赖。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _stats(values: Sequence[float]) -> dict[str, float | int]:
    """返回固定延迟统计；调用方保证输入非空。"""
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        raise ValueError("延迟统计不能为空")
    return {
        "count": len(array),
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def _indices_for_episodes(
    records: Sequence[Mapping[str, Any]],
    episodes: set[int],
) -> torch.Tensor:
    indices = [
        index
        for index, record in enumerate(records)
        if int(record["episode"]) in episodes
    ]
    if not indices:
        raise ValueError(f"episodes={sorted(episodes)} 没有 chunks")
    return torch.tensor(indices, dtype=torch.long)


def _diverse_top_k(
    *,
    candidate_records: Sequence[Mapping[str, Any]],
    candidate_geometry: torch.Tensor,
    query_records: Sequence[Mapping[str, Any]],
    query_geometry: torch.Tensor,
    candidate_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """同 phase 检索，并强制候选来自不同 episodes。"""
    mean = candidate_geometry.mean(dim=0)
    std = candidate_geometry.std(dim=0, unbiased=False).clamp_min(1e-4)
    distances = torch.cdist(
        (query_geometry - mean) / std,
        (candidate_geometry - mean) / std,
    ) / math.sqrt(candidate_geometry.shape[1])
    valid = _phase(query_geometry)[:, None] == _phase(candidate_geometry)[None]
    masked = distances.masked_fill(~valid, torch.inf)
    candidate_episodes = [int(record["episode"]) for record in candidate_records]
    selected = []
    selected_distances = []
    for row in range(len(query_records)):
        episode_seen = set()
        indices = []
        values = []
        for index in torch.argsort(masked[row]).tolist():
            value = float(masked[row, index])
            if not math.isfinite(value):
                break
            episode = candidate_episodes[index]
            if episode in episode_seen:
                continue
            episode_seen.add(episode)
            indices.append(index)
            values.append(value)
            if len(indices) == candidate_count:
                break
        if len(indices) != candidate_count:
            raise ValueError(
                "至少一个 query 没有 K 个跨 episode 同 phase Demo"
            )
        selected.append(indices)
        selected_distances.append(values)
    return (
        torch.tensor(selected, dtype=torch.long),
        torch.tensor(selected_distances, dtype=torch.float32),
    )


def _candidate_disagreement(hypotheses: torch.Tensor) -> torch.Tensor:
    translation = hypotheses[..., :3]
    pairwise = torch.linalg.vector_norm(
        translation[:, :, None] - translation[:, None, :],
        dim=-1,
    ).mean(dim=-1)
    count = hypotheses.shape[1]
    return pairwise.sum(dim=2) / max(count - 1, 1)


def _prepare_split(
    *,
    task_id: str,
    records: list[dict[str, Any]],
    query_geometry: torch.Tensor,
    target: torch.Tensor,
    bank_records: list[dict[str, Any]],
    bank_geometry: torch.Tensor,
    bank_actions: torch.Tensor,
    transport: nn.Module,
    gate: nn.Module,
    candidate_count: int,
    device: torch.device,
) -> PreparedSplit:
    indices, distances = _diverse_top_k(
        candidate_records=bank_records,
        candidate_geometry=bank_geometry,
        query_records=records,
        query_geometry=query_geometry,
        candidate_count=candidate_count,
    )
    demo_geometry = bank_geometry[indices]
    demo_actions = bank_actions[indices]
    batch, candidates = indices.shape
    flat_query = query_geometry[:, None].expand(-1, candidates, -1).reshape(
        batch * candidates,
        -1,
    )
    flat_geometry = demo_geometry.reshape(batch * candidates, -1)
    flat_actions = demo_actions.reshape(
        batch * candidates,
        demo_actions.shape[2],
        demo_actions.shape[3],
    )
    transported, base = _qa_diagnostics(
        transport,
        flat_query,
        flat_geometry,
        flat_actions,
        device,
    )
    with torch.inference_mode():
        gates = gate(base.to(device)).cpu()
    mask = torch.ones(flat_actions.shape[:2], dtype=torch.bool)
    hypotheses = apply_shrinkage(
        flat_actions,
        transported,
        gates,
        mask,
    ).reshape(batch, candidates, flat_actions.shape[1], flat_actions.shape[2])
    base = base.reshape(batch, candidates, -1)
    disagreement = _candidate_disagreement(hypotheses)
    features = torch.cat(
        (base, distances[..., None], disagreement[..., None]),
        dim=-1,
    )
    if features.shape[-1] != CANDIDATE_FEATURE_DIM:
        raise RuntimeError("RADM candidate feature dimension 错误")
    candidate_mask = torch.ones_like(distances, dtype=torch.bool)
    prior = retrieval_prior(distances, candidate_mask)
    return PreparedSplit(
        task_id=task_id,
        records=records,
        target=target,
        hypotheses=hypotheses,
        features=features,
        distances=distances,
        prior=prior,
        mask=candidate_mask,
        candidate_indices=indices,
        query_geometry=query_geometry,
        demo_geometry=demo_geometry,
        demo_actions=demo_actions,
    )


def _train_model(
    *,
    prepared: Sequence[PreparedSplit],
    config: RADMTrainConfig,
    device: torch.device,
    log_path: Path,
) -> RetrieverAnchoredDemoMixture:
    features = torch.cat([item.features for item in prepared])
    prior = torch.cat([item.prior for item in prepared])
    masks = torch.cat([item.mask for item in prepared])
    hypotheses = torch.cat([item.hypotheses for item in prepared])
    targets = torch.cat([item.target for item in prepared])
    feature_mean = features.reshape(-1, features.shape[-1]).mean(dim=0)
    feature_std = features.reshape(-1, features.shape[-1]).std(
        dim=0,
        unbiased=False,
    ).clamp_min(1e-5)
    model = RetrieverAnchoredDemoMixture(
        RetrieverAnchoredMixtureConfig(
            input_dim=features.shape[-1],
            hidden_dim=config.hidden_dim,
            maximum_odds_distortion=config.maximum_odds_distortion,
        ),
        feature_mean=feature_mean,
        feature_std=feature_std,
    ).to(device)
    loader = DataLoader(
        TensorDataset(features, prior, masks, hypotheses, targets),
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
        total = 0.0
        examples = 0
        model.train()
        for feature, prior_value, mask, candidate, target in loader:
            feature = feature.to(device)
            prior_value = prior_value.to(device)
            mask = mask.to(device)
            candidate = candidate.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            posterior = model(feature, prior_value, mask)
            prediction = mix_demo_hypotheses(candidate, posterior, mask)
            loss = (prediction[..., :3] - target[..., :3]).square().mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            total += float(loss.detach()) * len(feature)
            examples += len(feature)
        if epoch == 0 or (epoch + 1) % 10 == 0:
            row = {
                "epoch": epoch + 1,
                "translation_mse": total / examples,
            }
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
    return model.eval()


def _oracle_best(
    hypotheses: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    loss = (hypotheses[..., :3] - target[:, None, :, :3]).square().mean(
        dim=(2, 3)
    )
    index = loss.argmin(dim=1)
    return hypotheses[torch.arange(len(hypotheses)), index]


def _oracle_convex(
    hypotheses: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """枚举 K=4 active sets，求 analysis-only convex-hull projection。"""
    source = hypotheses[..., :3].flatten(2).double()
    target_flat = target[..., :3].flatten(1).double()
    batch, candidates, width = source.shape
    best_loss = torch.full((batch,), torch.inf, dtype=torch.float64)
    best = source[:, 0].clone()
    for bits in range(1, 1 << candidates):
        subset = [index for index in range(candidates) if bits & (1 << index)]
        values = source[:, subset]
        count = len(subset)
        gram = values @ values.transpose(1, 2)
        rhs = (values * target_flat[:, None]).sum(dim=2)
        system = torch.zeros((batch, count + 1, count + 1), dtype=torch.float64)
        system[:, :count, :count] = gram
        system[:, :count, count] = 1.0
        system[:, count, :count] = 1.0
        objective = torch.cat(
            (rhs, torch.ones((batch, 1), dtype=torch.float64)),
            dim=1,
        )
        solution = torch.linalg.lstsq(system, objective[..., None]).solution[
            :, :count, 0
        ]
        feasible = (solution >= -1e-7).all(dim=1)
        weights = solution.clamp_min(0.0)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        prediction = (values * weights[..., None]).sum(dim=1)
        loss = (prediction - target_flat).square().mean(dim=1)
        update = feasible & (loss < best_loss)
        best_loss = torch.where(update, loss, best_loss)
        best = torch.where(update[:, None], prediction, best)
    output = hypotheses[:, 0].clone()
    output[..., :3] = best.float().reshape(batch, hypotheses.shape[2], 3)
    return output


@torch.inference_mode()
def _predict(
    model: RetrieverAnchoredDemoMixture,
    prepared: PreparedSplit,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    features = prepared.features.to(device)
    prior = prepared.prior.to(device)
    mask = prepared.mask.to(device)
    hypotheses = prepared.hypotheses.to(device)
    posterior = model(features, prior, mask)
    uniform = prepared.mask.float()
    uniform = uniform / uniform.sum(dim=1, keepdim=True)
    predictions = {
        "rank1_bcsg": prepared.hypotheses[:, 0],
        "uniform_mixture": mix_demo_hypotheses(
            prepared.hypotheses,
            uniform,
            prepared.mask,
        ),
        "retriever_prior_mixture": mix_demo_hypotheses(
            prepared.hypotheses,
            prepared.prior,
            prepared.mask,
        ),
        "radm": mix_demo_hypotheses(hypotheses, posterior, mask).cpu(),
        "oracle_best_in_4": _oracle_best(
            prepared.hypotheses,
            prepared.target,
        ),
        "oracle_convex": _oracle_convex(
            prepared.hypotheses,
            prepared.target,
        ),
    }
    return predictions, posterior.cpu()


def _selection_digest(values: torch.Tensor) -> str:
    array = values.numpy().astype("<i8", copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


@torch.inference_mode()
def _latency(
    *,
    model: RetrieverAnchoredDemoMixture,
    transport: nn.Module,
    gate: nn.Module,
    sample: PreparedSplit,
    iterations: int,
    device: torch.device,
) -> dict[str, float | int]:
    query = sample.query_geometry[:1].repeat(4, 1).to(device)
    demo_geometry = sample.demo_geometry[:1].reshape(4, -1).to(device)
    demo_actions = sample.demo_actions[:1].reshape(
        4,
        sample.demo_actions.shape[2],
        7,
    ).to(device)
    distances = sample.distances[:1].to(device)
    mask = torch.ones((1, 4), dtype=torch.bool, device=device)
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
        base = features.to(device)[None]
        disagreement = _candidate_disagreement(hypotheses)
        candidate_features = torch.cat(
            (base, distances[..., None], disagreement[..., None]),
            dim=-1,
        )
        prior = retrieval_prior(distances, mask)
        posterior = model(candidate_features, prior, mask)
        mix_demo_hypotheses(hypotheses, posterior, mask)
        _sync(device)
        elapsed = (time.perf_counter() - started) * 1000.0
        if iteration >= 20:
            values.append(elapsed)
    return {"iterations": len(values), **_stats(values)}


def run(
    *,
    project_root: Path,
    data_roots: Sequence[Path],
    checkpoint_path: Path,
    config_path: Path,
    output_root: Path,
    config: RADMTrainConfig,
    device: torch.device,
) -> None:
    if output_root.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output_root}")
    if _sha256(checkpoint_path) != config.source_checkpoint_sha256:
        raise ValueError("RADM source checkpoint SHA256 不匹配")
    temporary = output_root.with_name(f".{output_root.name}.incomplete-{os.getpid()}")
    temporary.mkdir(parents=True)

    def cleanup() -> None:
        shutil.rmtree(temporary, ignore_errors=True)

    atexit.register(cleanup)
    _seed_everything(config.seed)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    transport, gate = _load_frozen_models(checkpoint, device)
    tasks = [_load_task(root) for root in data_roots]
    if {task.task_id for task in tasks} != EXPECTED_TASKS or len(tasks) != 3:
        raise ValueError("RADM 必须提供三个唯一 ManiSkill tasks")
    first_task = tasks[0]
    for task in tasks:
        if task.action_representation != checkpoint["action_representation"]:
            raise ValueError(f"{task.task_id} action representation 不匹配")
        if not torch.equal(
            task.pose_scales.float(),
            torch.as_tensor(checkpoint["action_pose_scales"]).float(),
        ):
            raise ValueError(f"{task.task_id} action scales 与 checkpoint 不匹配")
        if task.train_actions.shape[1:] != first_task.train_actions.shape[1:]:
            raise ValueError("RADM task action chunk shape 不一致")
        if (
            task.translation_threshold_m != first_task.translation_threshold_m
            or task.rotation_threshold_rad != first_task.rotation_threshold_rad
        ):
            raise ValueError("RADM task evaluation thresholds 不一致")
    train_prepared = []
    val_prepared = []
    bank_audit = {}
    for task in tasks:
        observed_train = {int(row["episode"]) for row in task.train_records}
        observed_val = {int(row["episode"]) for row in task.val_records}
        expected_train = set(config.operator_train_episodes) | set(
            config.mixture_train_episodes
        )
        if observed_train != expected_train or observed_val != set(
            config.validation_episodes
        ):
            raise ValueError(f"{task.task_id} split 与 RADM 冻结协议不一致")
        summary_hash = _sha256(task.root / "summary.json")
        if checkpoint["data_summary_sha256"][task.task_id] != summary_hash:
            raise ValueError(f"{task.task_id} bank summary 与 checkpoint 不匹配")
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
        train_records = [
            task.train_records[index] for index in train_indices.tolist()
        ]
        train_prepared.append(
            _prepare_split(
                task_id=task.task_id,
                records=train_records,
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
        val_prepared.append(
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
        bank_audit[task.task_id] = {
            "chunks": len(bank_indices),
            "selection_sha256": _selection_hash(bank_indices),
            "summary_sha256": summary_hash,
        }
    model = _train_model(
        prepared=train_prepared,
        config=config,
        device=device,
        log_path=temporary / "training_metrics.jsonl",
    )
    per_task = {}
    aggregate_target = []
    aggregate_groups = []
    aggregate_predictions: dict[str, list[torch.Tensor]] = {}
    aggregate_posterior = []
    artifact: dict[str, list[np.ndarray]] = {}
    task_by_id = {task.task_id: task for task in tasks}
    for offset, prepared in enumerate(val_prepared):
        task = task_by_id[prepared.task_id]
        predictions, posterior = _predict(model, prepared, device)
        action_mask = torch.ones(prepared.target.shape[:2], dtype=torch.bool)
        options = {
            "pose_scales": task.pose_scales,
            "translation_threshold_m": task.translation_threshold_m,
            "rotation_threshold_rad": task.rotation_threshold_rad,
        }
        metrics = {
            name: _physical_metrics(value, prepared.target, action_mask, **options)
            for name, value in predictions.items()
        }
        groups = [
            f"{prepared.task_id}:{row['episode']}" for row in prepared.records
        ]
        comparisons = {
            f"radm_minus_{reference}": _bootstrap_comparison(
                reference=predictions[reference],
                candidate=predictions["radm"],
                target=prepared.target,
                mask=action_mask,
                group_ids=groups,
                seed=config.seed + offset * 100 + index,
                resamples=config.bootstrap_resamples,
                **options,
            )
            for index, reference in enumerate(
                ("rank1_bcsg", "retriever_prior_mixture", "uniform_mixture")
            )
        }
        odds = (posterior[:, :, None] / posterior[:, None, :]) / (
            prepared.prior[:, :, None] / prepared.prior[:, None, :]
        )
        recovery_denominator = (
            metrics["rank1_bcsg"]["translation_l2_m"]
            - metrics["oracle_best_in_4"]["translation_l2_m"]
        )
        recovery = (
            metrics["rank1_bcsg"]["translation_l2_m"]
            - metrics["radm"]["translation_l2_m"]
        ) / max(recovery_denominator, 1e-12)
        per_task[prepared.task_id] = {
            "queries": len(prepared.target),
            "candidate_selection_sha256": _selection_digest(
                prepared.candidate_indices
            ),
            "metrics": metrics,
            "paired_episode_bootstrap": comparisons,
            "oracle_gap_recovery": recovery,
            "posterior": {
                "rank_mean": posterior.mean(dim=0).tolist(),
                "maximum_odds_distortion": float(odds.max()),
            },
        }
        aggregate_target.append(prepared.target)
        aggregate_groups.extend(groups)
        aggregate_posterior.append(posterior)
        for name, value in predictions.items():
            aggregate_predictions.setdefault(name, []).append(value)
        rows = {
            "task": np.asarray([prepared.task_id] * len(prepared.target)),
            "group_id": np.asarray(groups),
            "target": prepared.target.numpy(),
            "posterior": posterior.numpy(),
            "prior": prepared.prior.numpy(),
            **{name: value.numpy() for name, value in predictions.items()},
        }
        for name, value in rows.items():
            artifact.setdefault(name, []).append(value)
    target = torch.cat(aggregate_target)
    predictions = {
        name: torch.cat(values) for name, values in aggregate_predictions.items()
    }
    mask = torch.ones(target.shape[:2], dtype=torch.bool)
    options = {
        "pose_scales": first_task.pose_scales,
        "translation_threshold_m": first_task.translation_threshold_m,
        "rotation_threshold_rad": first_task.rotation_threshold_rad,
    }
    metrics = {
        name: _physical_metrics(value, target, mask, **options)
        for name, value in predictions.items()
    }
    comparisons = {
        f"radm_minus_{reference}": _bootstrap_comparison(
            reference=predictions[reference],
            candidate=predictions["radm"],
            target=target,
            mask=mask,
            group_ids=aggregate_groups,
            seed=config.seed + 1000 + index,
            resamples=config.bootstrap_resamples,
            **options,
        )
        for index, reference in enumerate(
            ("rank1_bcsg", "retriever_prior_mixture", "uniform_mixture")
        )
    }
    posterior = torch.cat(aggregate_posterior)
    prior = torch.cat([item.prior for item in val_prepared])
    odds = (posterior[:, :, None] / posterior[:, None, :]) / (
        prior[:, :, None] / prior[:, None, :]
    )
    rank1_translation = metrics["rank1_bcsg"]["translation_l2_m"]
    oracle_translation = metrics["oracle_best_in_4"]["translation_l2_m"]
    radm_translation = metrics["radm"]["translation_l2_m"]
    recovery = (rank1_translation - radm_translation) / max(
        rank1_translation - oracle_translation,
        1e-12,
    )
    sample = val_prepared[0]
    with torch.inference_mode():
        no_demo_features = torch.zeros(
            (1, config.candidate_count, CANDIDATE_FEATURE_DIM),
            device=device,
        )
        no_demo_prior = torch.zeros((1, config.candidate_count), device=device)
        no_demo_mask = torch.zeros(
            (1, config.candidate_count),
            dtype=torch.bool,
            device=device,
        )
        no_demo_weights = model(
            no_demo_features,
            no_demo_prior,
            no_demo_mask,
        )
        no_demo_output = mix_demo_hypotheses(
            torch.zeros(
                (1, config.candidate_count, target.shape[1], target.shape[2]),
                device=device,
            ),
            no_demo_weights,
            no_demo_mask,
        )
    radm_translation_values = predictions["radm"][..., :3]
    candidate_values = torch.cat(
        [item.hypotheses for item in val_prepared]
    )[..., :3]
    action_span_diameter = torch.linalg.vector_norm(
        candidate_values[:, :, None] - candidate_values[:, None, :],
        dim=-1,
    ).amax(dim=(1, 2, 3))
    convex_violation = max(
        float(
            (candidate_values.amin(dim=1) - radm_translation_values)
            .clamp_min(0)
            .max()
        ),
        float(
            (radm_translation_values - candidate_values.amax(dim=1))
            .clamp_min(0)
            .max()
        ),
    )
    discrete_error = float(
        (
            predictions["radm"][..., 3:]
            - predictions["rank1_bcsg"][..., 3:]
        ).abs().max()
    )
    latency = _latency(
        model=model,
        transport=transport,
        gate=gate,
        sample=sample,
        iterations=config.latency_iterations,
        device=device,
    )
    parameter_counts = {
        "qa_lrdat": sum(value.numel() for value in transport.parameters()),
        "bcsg": sum(value.numel() for value in gate.parameters()),
        "radm": sum(value.numel() for value in model.parameters()),
    }
    radm_rank1 = comparisons["radm_minus_rank1_bcsg"]["translation_l2_m"]
    radm_prior = comparisons[
        "radm_minus_retriever_prior_mixture"
    ]["translation_l2_m"]
    improved_tasks = sum(
        values["metrics"]["radm"]["translation_l2_m"]
        < values["metrics"]["rank1_bcsg"]["translation_l2_m"]
        for values in per_task.values()
    )
    criteria = {
        "r1_radm_translation_better_than_rank1_bcsg": (
            radm_rank1["ci95_high"] < 0.0
        ),
        "r2_radm_translation_better_than_retriever_prior": (
            radm_prior["ci95_high"] < 0.0
        ),
        "r2_at_least_two_tasks_better_than_rank1": improved_tasks >= 2,
        "r3_recovers_at_least_20_percent_oracle_gap": recovery >= 0.20,
        "r4_odds_bound": float(odds.max()) <= 4.0 + 1e-6,
        "r4_translation_convex_hull": convex_violation <= 1e-6,
        "r4_no_demo_exact": float(no_demo_output.abs().max()) == 0.0,
        "r4_discrete_prior_exact": discrete_error == 0.0,
        "r5_parameter_budget": sum(parameter_counts.values()) < 250_000,
        "r5_latency_p95_below_8ms": latency["p95"] < 8.0,
    }
    checkpoint_output = {
        "schema_version": 1,
        "git_commit": _git_commit(project_root),
        "source_checkpoint_sha256": _sha256(checkpoint_path),
        "config": asdict(model.config),
        "model": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
    }
    checkpoint_output_path = temporary / "retriever_anchored_demo_mixture.pt"
    torch.save(checkpoint_output, checkpoint_output_path)
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
            "K=4 frozen QA-LRDT/BCSG hypotheses; odds-anchored convex "
            "translation mixture"
        ),
        "config": asdict(config),
        "config_sha256": _sha256(config_path),
        "source_checkpoint_sha256": _sha256(checkpoint_path),
        "data": bank_audit,
        "device": str(device),
        "parameters": parameter_counts,
        "per_task": per_task,
        "aggregate": {
            "queries": len(target),
            "episodes": len(set(aggregate_groups)),
            "metrics": metrics,
            "paired_episode_bootstrap": comparisons,
            "oracle_gap_recovery": recovery,
            "posterior_rank_mean": posterior.mean(dim=0).tolist(),
            "maximum_odds_distortion": float(odds.max()),
            "normalized_action_span_diameter": {
                "mean": float(action_span_diameter.mean()),
                "p95": float(torch.quantile(action_span_diameter, 0.95)),
                "max": float(action_span_diameter.max()),
            },
        },
        "structural": {
            "translation_convex_hull_max_violation": convex_violation,
            "discrete_prior_max_abs_error": discrete_error,
            "no_demo_max_abs_output": float(no_demo_output.abs().max()),
        },
        "latency": latency,
        "criteria": criteria,
        "checkpoint_sha256": _sha256(checkpoint_output_path),
        "predictions_sha256": _sha256(artifact_path),
    }
    (temporary / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_root)
    atexit.unregister(cleanup)
    print(json.dumps(report["aggregate"], indent=2), flush=True)
    print(json.dumps(criteria, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(
        project_root=arguments.project_root.resolve(),
        data_roots=[path.resolve() for path in arguments.data_root],
        checkpoint_path=arguments.checkpoint.resolve(),
        config_path=arguments.config.resolve(),
        output_root=arguments.output_root.resolve(),
        config=RADMTrainConfig.from_json(arguments.config.resolve()),
        device=torch.device(arguments.device),
    )

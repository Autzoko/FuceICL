"""在可执行 ManiSkill chunks 上外部验证 QA-LRDT 与 BCSG。"""

from __future__ import annotations

import argparse
import atexit
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
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
from dev.predictor.benefit_calibrated_shrinkage import (
    BenefitCalibratedGate,
    BenefitGateConfig,
    apply_shrinkage,
    constant_optimal_shrinkage,
    frozen_transport_features,
    optimal_shrinkage_target,
)
from dev.predictor.low_rank_demo_transport import (
    LowRankDemoActionTransport,
    LowRankTransportConfig,
)
from dev.predictor.query_aligned_transport import (
    QueryAlignedLowRankDemoTransport,
    QueryAlignedTransportConfig,
)
from dev.predictor.train_action_chunks import (
    _masked_sample_loss,
    _physical_metrics,
)
from dev.simulator.evaluate_maniskill_demo_prior import _bootstrap_comparison
from dev.simulator.train_maniskill_low_rank_transport import (
    TaskData,
    _load_task,
    _nearest_demo_indices,
    _selection_hash,
)


MODEL_NAMES = ("fixed_low_rank_transport", "query_aligned_transport")


@dataclass(frozen=True)
class TrainConfig:
    """跨环境复现实验的冻结配置。"""

    schema_version: str
    seed: int
    operator_train_episodes: list[int]
    gate_calibration_episodes: list[int]
    validation_episodes: list[int]
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    rank: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    feedforward_dim: int
    dropout: float
    residual_limit: float
    gate_epochs: int
    gate_batch_size: int
    gate_learning_rate: float
    gate_weight_decay: float
    gate_hidden_dim: int
    bootstrap_resamples: int
    latency_iterations: int

    @classmethod
    def from_json(cls, path: Path) -> "TrainConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def __post_init__(self) -> None:
        positive = (
            self.epochs,
            self.batch_size,
            self.learning_rate,
            self.gradient_clip_norm,
            self.rank,
            self.hidden_dim,
            self.num_layers,
            self.num_heads,
            self.feedforward_dim,
            self.residual_limit,
            self.gate_epochs,
            self.gate_batch_size,
            self.gate_learning_rate,
            self.gate_hidden_dim,
            self.bootstrap_resamples,
            self.latency_iterations,
        )
        if not self.schema_version.strip() or min(positive) <= 0:
            raise ValueError("schema 与训练正数参数必须有效")
        if min(self.weight_decay, self.gate_weight_decay) < 0:
            raise ValueError("weight decay 不能为负")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim 必须能被 num_heads 整除")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout 必须位于 [0,1)")
        operator = set(self.operator_train_episodes)
        calibration = set(self.gate_calibration_episodes)
        validation = set(self.validation_episodes)
        if (
            not operator
            or not calibration
            or not validation
            or len(operator) != len(self.operator_train_episodes)
            or len(calibration) != len(self.gate_calibration_episodes)
            or len(validation) != len(self.validation_episodes)
            or operator & calibration
            or operator & validation
            or calibration & validation
        ):
            raise ValueError(
                "operator/calibration/validation episodes 必须非空、唯一且互斥"
            )


@dataclass(frozen=True)
class TaskProtocol:
    """单任务冻结 bank、calibration 与 final validation selection。"""

    task: TaskData
    bank_indices: torch.Tensor
    calibration_indices: torch.Tensor
    train_demo_indices: torch.Tensor
    calibration_demo_indices: torch.Tensor
    validation_demo_indices: torch.Tensor
    train_distances: torch.Tensor
    calibration_distances: torch.Tensor
    validation_distances: torch.Tensor


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


def _record_indices(
    records: Sequence[Mapping[str, Any]],
    episodes: set[int],
) -> torch.Tensor:
    indices = [
        index
        for index, record in enumerate(records)
        if int(record["episode"]) in episodes
    ]
    if not indices:
        raise ValueError(f"episodes={sorted(episodes)} 没有 records")
    return torch.tensor(indices, dtype=torch.long)


def _slice_records(
    records: Sequence[dict[str, Any]],
    indices: torch.Tensor,
) -> list[dict[str, Any]]:
    return [records[index] for index in indices.tolist()]


def _build_protocol(task: TaskData, config: TrainConfig) -> TaskProtocol:
    operator_episodes = set(config.operator_train_episodes)
    calibration_episodes = set(config.gate_calibration_episodes)
    observed_train = {int(record["episode"]) for record in task.train_records}
    observed_validation = {
        int(record["episode"]) for record in task.val_records
    }
    expected = operator_episodes | calibration_episodes
    if observed_train != expected:
        raise ValueError(
            f"{task.task_id} train episodes={sorted(observed_train)} "
            f"与冻结协议={sorted(expected)} 不一致"
        )
    if observed_validation != set(config.validation_episodes):
        raise ValueError(
            f"{task.task_id} validation episodes={sorted(observed_validation)} "
            f"与冻结协议={sorted(config.validation_episodes)} 不一致"
        )
    bank_indices = _record_indices(task.train_records, operator_episodes)
    calibration_indices = _record_indices(
        task.train_records,
        calibration_episodes,
    )
    bank_records = _slice_records(task.train_records, bank_indices)
    calibration_records = _slice_records(task.train_records, calibration_indices)
    bank_geometry = task.train_geometry[bank_indices]
    train_demo, train_distance = _nearest_demo_indices(
        candidate_records=bank_records,
        candidate_geometry=bank_geometry,
        query_records=bank_records,
        query_geometry=bank_geometry,
        exclude_same_episode=True,
    )
    calibration_demo, calibration_distance = _nearest_demo_indices(
        candidate_records=bank_records,
        candidate_geometry=bank_geometry,
        query_records=calibration_records,
        query_geometry=task.train_geometry[calibration_indices],
        exclude_same_episode=False,
    )
    validation_demo, validation_distance = _nearest_demo_indices(
        candidate_records=bank_records,
        candidate_geometry=bank_geometry,
        query_records=task.val_records,
        query_geometry=task.val_geometry,
        exclude_same_episode=False,
    )
    return TaskProtocol(
        task=task,
        bank_indices=bank_indices,
        calibration_indices=calibration_indices,
        train_demo_indices=train_demo,
        calibration_demo_indices=calibration_demo,
        validation_demo_indices=validation_demo,
        train_distances=train_distance,
        calibration_distances=calibration_distance,
        validation_distances=validation_distance,
    )


def _build_model(
    name: str,
    *,
    horizon: int,
    geometry_dim: int,
    geometry_mean: torch.Tensor,
    geometry_std: torch.Tensor,
    config: TrainConfig,
) -> nn.Module:
    shared = {
        "horizon": horizon,
        "geometry_dim": geometry_dim,
        "transported_action_dim": 3,
        "rank": config.rank,
        "hidden_dim": config.hidden_dim,
        "num_layers": config.num_layers,
        "num_heads": config.num_heads,
        "feedforward_dim": config.feedforward_dim,
        "dropout": config.dropout,
        "residual_limit": config.residual_limit,
    }
    if name == "fixed_low_rank_transport":
        return LowRankDemoActionTransport(
            LowRankTransportConfig(**shared),
            geometry_mean=geometry_mean,
            geometry_std=geometry_std,
        )
    if name == "query_aligned_transport":
        return QueryAlignedLowRankDemoTransport(
            QueryAlignedTransportConfig(**shared),
            geometry_mean=geometry_mean,
            geometry_std=geometry_std,
        )
    raise ValueError(f"未知模型：{name}")


def _train_transport(
    *,
    name: str,
    dataset: TensorDataset,
    geometry_mean: torch.Tensor,
    geometry_std: torch.Tensor,
    horizon: int,
    config: TrainConfig,
    device: torch.device,
    log_path: Path,
) -> nn.Module:
    _seed_everything(config.seed)
    model = _build_model(
        name,
        horizon=horizon,
        geometry_dim=int(geometry_mean.numel()),
        geometry_mean=geometry_mean,
        geometry_std=geometry_std,
        config=config,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
    )
    for epoch in range(config.epochs):
        generator = torch.Generator().manual_seed(config.seed + epoch)
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
        )
        total = 0.0
        examples = 0
        model.train()
        for query, target, demo_geometry, demo_actions in loader:
            query = query.to(device)
            target = target.to(device)
            demo_geometry = demo_geometry.to(device)
            demo_actions = demo_actions.to(device)
            mask = torch.ones(
                demo_actions.shape[:2],
                dtype=torch.bool,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            prediction = model(query, demo_geometry, demo_actions, mask)
            loss = _masked_sample_loss(prediction, target, mask).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            total += float(loss.detach()) * len(query)
            examples += len(query)
        scheduler.step()
        if epoch == 0 or (epoch + 1) % 10 == 0:
            row = {
                "model": name,
                "epoch": epoch + 1,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "action_loss": total / examples,
            }
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
    return model.eval()


@torch.inference_mode()
def _predict(
    model: nn.Module,
    query: torch.Tensor,
    demo_geometry: torch.Tensor,
    demo_actions: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
) -> torch.Tensor:
    predictions = []
    for start in range(0, len(query), batch_size):
        stop = start + batch_size
        current_actions = demo_actions[start:stop].to(device)
        mask = torch.ones(
            current_actions.shape[:2],
            dtype=torch.bool,
            device=device,
        )
        predictions.append(
            model(
                query[start:stop].to(device),
                demo_geometry[start:stop].to(device),
                current_actions,
                mask,
            ).cpu()
        )
    return torch.cat(predictions)


@torch.inference_mode()
def _qa_diagnostics(
    model: QueryAlignedLowRankDemoTransport,
    query: torch.Tensor,
    demo_geometry: torch.Tensor,
    demo_actions: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    predictions = []
    features = []
    for start in range(0, len(query), batch_size):
        stop = start + batch_size
        current_actions = demo_actions[start:stop].to(device)
        mask = torch.ones(
            current_actions.shape[:2],
            dtype=torch.bool,
            device=device,
        )
        prediction, feature = frozen_transport_features(
            model,
            query[start:stop].to(device),
            demo_geometry[start:stop].to(device),
            current_actions,
            mask,
        )
        predictions.append(prediction.cpu())
        features.append(feature.cpu())
    return torch.cat(predictions), torch.cat(features)


def _train_gate(
    *,
    features: torch.Tensor,
    targets: torch.Tensor,
    energy: torch.Tensor,
    config: TrainConfig,
    device: torch.device,
    log_path: Path,
) -> BenefitCalibratedGate:
    _seed_everything(config.seed)
    gate = BenefitCalibratedGate(
        BenefitGateConfig(hidden_dim=config.gate_hidden_dim),
        feature_mean=features.mean(dim=0),
        feature_std=features.std(dim=0, unbiased=False).clamp_min(1e-5),
    ).to(device)
    optimizer = torch.optim.AdamW(
        gate.parameters(),
        lr=config.gate_learning_rate,
        weight_decay=config.gate_weight_decay,
    )
    weights = energy / energy.mean().clamp_min(1e-8)
    dataset = TensorDataset(features, targets, weights)
    for epoch in range(config.gate_epochs):
        generator = torch.Generator().manual_seed(config.seed + epoch)
        loader = DataLoader(
            dataset,
            batch_size=config.gate_batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
        )
        total = 0.0
        examples = 0
        gate.train()
        for feature, target, weight in loader:
            feature = feature.to(device)
            target = target.to(device)
            weight = weight.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = gate(feature)
            loss = (weight * (prediction - target).square()).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(gate.parameters(), config.gradient_clip_norm)
            optimizer.step()
            total += float(loss.detach()) * len(feature)
            examples += len(feature)
        if epoch == 0 or (epoch + 1) % 10 == 0:
            row = {
                "model": "benefit_calibrated_gate",
                "epoch": epoch + 1,
                "weighted_gate_mse": total / examples,
            }
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
    return gate.eval()


def _wrong_task_demo(
    *,
    protocol: TaskProtocol,
    protocols: Sequence[TaskProtocol],
) -> tuple[torch.Tensor, torch.Tensor]:
    candidates = [
        item
        for item in protocols
        if item.task.task_id != protocol.task.task_id
    ]
    records = []
    geometry = []
    actions = []
    for item in candidates:
        records.extend(_slice_records(item.task.train_records, item.bank_indices))
        geometry.append(item.task.train_geometry[item.bank_indices])
        actions.append(item.task.train_actions[item.bank_indices])
    candidate_geometry = torch.cat(geometry)
    candidate_actions = torch.cat(actions)
    indices, _ = _nearest_demo_indices(
        candidate_records=records,
        candidate_geometry=candidate_geometry,
        query_records=protocol.task.val_records,
        query_geometry=protocol.task.val_geometry,
        exclude_same_episode=False,
    )
    return candidate_geometry[indices], candidate_actions[indices]


def _bootstrap_pair(
    *,
    reference: torch.Tensor,
    candidate: torch.Tensor,
    target: torch.Tensor,
    task: TaskData,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    mask = torch.ones(target.shape[:2], dtype=torch.bool)
    groups = [f"{task.task_id}:{row['episode']}" for row in task.val_records]
    return _bootstrap_comparison(
        reference=reference,
        candidate=candidate,
        target=target,
        mask=mask,
        group_ids=groups,
        seed=seed,
        resamples=resamples,
        pose_scales=task.pose_scales,
        translation_threshold_m=task.translation_threshold_m,
        rotation_threshold_rad=task.rotation_threshold_rad,
    )


@torch.inference_mode()
def _latency(
    *,
    qa: QueryAlignedLowRankDemoTransport,
    gate: BenefitCalibratedGate,
    query: torch.Tensor,
    demo_geometry: torch.Tensor,
    demo_actions: torch.Tensor,
    iterations: int,
    device: torch.device,
) -> dict[str, float | int]:
    query = query[:1].to(device)
    demo_geometry = demo_geometry[:1].to(device)
    demo_actions = demo_actions[:1].to(device)
    mask = torch.ones(demo_actions.shape[:2], dtype=torch.bool, device=device)

    def predict() -> torch.Tensor:
        transported, features = frozen_transport_features(
            qa,
            query,
            demo_geometry,
            demo_actions,
            mask,
        )
        return apply_shrinkage(demo_actions, transported, gate(features), mask)

    for _ in range(20):
        predict()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        predict()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        samples.append(1000.0 * (time.perf_counter() - started))
    values = np.asarray(samples)
    return {
        "iterations": iterations,
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.quantile(values, 0.95)),
        "max_ms": float(values.max()),
    }


def run(
    *,
    project_root: Path,
    data_roots: Sequence[Path],
    output_root: Path,
    config_path: Path,
    config: TrainConfig,
    device: torch.device,
) -> None:
    if output_root.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output_root}")
    temporary = output_root.with_name(f".{output_root.name}.incomplete-{os.getpid()}")
    temporary.mkdir(parents=True)

    def cleanup_incomplete() -> None:
        shutil.rmtree(temporary, ignore_errors=True)

    atexit.register(cleanup_incomplete)
    _seed_everything(config.seed)
    tasks = [_load_task(root) for root in data_roots]
    if len(tasks) != 3 or len({task.task_id for task in tasks}) != 3:
        raise ValueError("外部验证固定要求三个不同 ManiSkill tasks")
    first = tasks[0]
    for task in tasks[1:]:
        if task.action_representation != first.action_representation:
            raise ValueError("任务 action representation 不一致")
        if not torch.equal(task.pose_scales, first.pose_scales):
            raise ValueError("任务 action scales 不一致")
        if task.train_actions.shape[1:] != first.train_actions.shape[1:]:
            raise ValueError("任务 action chunk shape 不一致")
        if (
            task.translation_threshold_m != first.translation_threshold_m
            or task.rotation_threshold_rad != first.rotation_threshold_rad
        ):
            raise ValueError("任务 action evaluation thresholds 不一致")
    protocols = [_build_protocol(task, config) for task in tasks]

    train_query = torch.cat(
        [item.task.train_geometry[item.bank_indices] for item in protocols]
    )
    train_target = torch.cat(
        [item.task.train_actions[item.bank_indices] for item in protocols]
    )
    train_demo_geometry = torch.cat(
        [
            item.task.train_geometry[item.bank_indices][item.train_demo_indices]
            for item in protocols
        ]
    )
    train_demo_actions = torch.cat(
        [
            item.task.train_actions[item.bank_indices][item.train_demo_indices]
            for item in protocols
        ]
    )
    geometry_mean = train_query.mean(dim=0)
    geometry_std = train_query.std(dim=0, unbiased=False).clamp_min(1e-4)
    train_dataset = TensorDataset(
        train_query,
        train_target,
        train_demo_geometry,
        train_demo_actions,
    )
    models = {
        name: _train_transport(
            name=name,
            dataset=train_dataset,
            geometry_mean=geometry_mean,
            geometry_std=geometry_std,
            horizon=int(train_target.shape[1]),
            config=config,
            device=device,
            log_path=temporary / "training_metrics.jsonl",
        )
        for name in MODEL_NAMES
    }
    qa = models["query_aligned_transport"]
    if not isinstance(qa, QueryAlignedLowRankDemoTransport):
        raise TypeError("QA model 类型错误")

    calibration_query = []
    calibration_target = []
    calibration_demo_geometry = []
    calibration_demo_actions = []
    for item in protocols:
        bank_geometry = item.task.train_geometry[item.bank_indices]
        bank_actions = item.task.train_actions[item.bank_indices]
        calibration_query.append(
            item.task.train_geometry[item.calibration_indices]
        )
        calibration_target.append(
            item.task.train_actions[item.calibration_indices]
        )
        calibration_demo_geometry.append(
            bank_geometry[item.calibration_demo_indices]
        )
        calibration_demo_actions.append(
            bank_actions[item.calibration_demo_indices]
        )
    calibration_query_tensor = torch.cat(calibration_query)
    calibration_target_tensor = torch.cat(calibration_target)
    calibration_demo_geometry_tensor = torch.cat(calibration_demo_geometry)
    calibration_demo_action_tensor = torch.cat(calibration_demo_actions)
    calibration_transport, calibration_features = _qa_diagnostics(
        qa,
        calibration_query_tensor,
        calibration_demo_geometry_tensor,
        calibration_demo_action_tensor,
        device,
    )
    calibration_mask = torch.ones(
        calibration_target_tensor.shape[:2],
        dtype=torch.bool,
    )
    optimal_gate, residual_energy = optimal_shrinkage_target(
        calibration_demo_action_tensor,
        calibration_transport,
        calibration_target_tensor,
        calibration_mask,
    )
    constant_gate = constant_optimal_shrinkage(
        calibration_demo_action_tensor,
        calibration_transport,
        calibration_target_tensor,
        calibration_mask,
    )
    gate = _train_gate(
        features=calibration_features,
        targets=optimal_gate,
        energy=residual_energy,
        config=config,
        device=device,
        log_path=temporary / "training_metrics.jsonl",
    )

    per_task: dict[str, Any] = {}
    artifact: dict[str, list[np.ndarray]] = {
        "task": [],
        "group_id": [],
        "target": [],
        "demo_action_copy": [],
        "fixed_low_rank_transport": [],
        "query_aligned_transport": [],
        "constant_shrinkage": [],
        "bcsg": [],
        "predicted_gate": [],
    }
    aggregate_predictions: dict[str, list[torch.Tensor]] = {
        name: []
        for name in (
            "demo_action_copy",
            "fixed_low_rank_transport",
            "query_aligned_transport",
            "constant_shrinkage",
            "bcsg",
        )
    }
    aggregate_targets = []
    aggregate_groups = []
    identity_error = {name: 0.0 for name in MODEL_NAMES}
    no_demo_error = {name: 0.0 for name in MODEL_NAMES}
    for task_offset, item in enumerate(protocols):
        task = item.task
        bank_geometry = task.train_geometry[item.bank_indices]
        bank_actions = task.train_actions[item.bank_indices]
        demo_geometry = bank_geometry[item.validation_demo_indices]
        demo_actions = bank_actions[item.validation_demo_indices]
        fixed = _predict(
            models["fixed_low_rank_transport"],
            task.val_geometry,
            demo_geometry,
            demo_actions,
            device,
        )
        transported, features = _qa_diagnostics(
            qa,
            task.val_geometry,
            demo_geometry,
            demo_actions,
            device,
        )
        with torch.inference_mode():
            predicted_gate = gate(features.to(device)).cpu()
        constant_gates = torch.full_like(predicted_gate, float(constant_gate))
        mask = torch.ones(task.val_actions.shape[:2], dtype=torch.bool)
        constant_prediction = apply_shrinkage(
            demo_actions,
            transported,
            constant_gates,
            mask,
        )
        bcsg = apply_shrinkage(
            demo_actions,
            transported,
            predicted_gate,
            mask,
        )
        wrong_geometry, wrong_actions = _wrong_task_demo(
            protocol=item,
            protocols=protocols,
        )
        wrong_task = _predict(
            qa,
            task.val_geometry,
            wrong_geometry,
            wrong_actions,
            device,
        )
        shuffled = _predict(
            qa,
            task.val_geometry,
            demo_geometry,
            torch.roll(demo_actions, shifts=1, dims=0),
            device,
        )
        predictions = {
            "demo_action_copy": demo_actions,
            "fixed_low_rank_transport": fixed,
            "query_aligned_transport": transported,
            "constant_shrinkage": constant_prediction,
            "bcsg": bcsg,
        }
        options = {
            "pose_scales": task.pose_scales,
            "translation_threshold_m": task.translation_threshold_m,
            "rotation_threshold_rad": task.rotation_threshold_rad,
        }
        metrics = {
            name: _physical_metrics(value, task.val_actions, mask, **options)
            for name, value in predictions.items()
        }
        metrics["wrong_task_qa"] = _physical_metrics(
            wrong_task,
            task.val_actions,
            mask,
            **options,
        )
        metrics["shuffled_action_qa"] = _physical_metrics(
            shuffled,
            task.val_actions,
            mask,
            **options,
        )
        comparisons = {
            "qa_minus_copy": _bootstrap_pair(
                reference=demo_actions,
                candidate=transported,
                target=task.val_actions,
                task=task,
                seed=config.seed + task_offset * 100,
                resamples=config.bootstrap_resamples,
            ),
            "bcsg_minus_copy": _bootstrap_pair(
                reference=demo_actions,
                candidate=bcsg,
                target=task.val_actions,
                task=task,
                seed=config.seed + task_offset * 100 + 10,
                resamples=config.bootstrap_resamples,
            ),
            "bcsg_minus_qa": _bootstrap_pair(
                reference=transported,
                candidate=bcsg,
                target=task.val_actions,
                task=task,
                seed=config.seed + task_offset * 100 + 20,
                resamples=config.bootstrap_resamples,
            ),
            "bcsg_minus_constant": _bootstrap_pair(
                reference=constant_prediction,
                candidate=bcsg,
                target=task.val_actions,
                task=task,
                seed=config.seed + task_offset * 100 + 30,
                resamples=config.bootstrap_resamples,
            ),
        }
        per_task[task.task_id] = {
            "bank_queries": len(item.bank_indices),
            "calibration_queries": len(item.calibration_indices),
            "validation_queries": len(task.val_records),
            "bank_indices_sha256": _selection_hash(item.bank_indices),
            "validation_selection_sha256": _selection_hash(
                item.validation_demo_indices
            ),
            "validation_distance_mean": float(item.validation_distances.mean()),
            "validation_distance_p95": float(
                torch.quantile(item.validation_distances, 0.95)
            ),
            "predicted_gate_mean": float(predicted_gate.mean()),
            "predicted_gate_std": float(predicted_gate.std(unbiased=False)),
            "metrics": metrics,
            "paired_episode_bootstrap": comparisons,
        }
        for name, value in predictions.items():
            aggregate_predictions[name].append(value)
            artifact[name].append(value.numpy())
        aggregate_targets.append(task.val_actions)
        aggregate_groups.extend(
            [f"{task.task_id}:{row['episode']}" for row in task.val_records]
        )
        artifact["task"].append(
            np.asarray([task.task_id] * len(task.val_records))
        )
        artifact["group_id"].append(
            np.asarray(
                [f"{task.task_id}:{row['episode']}" for row in task.val_records]
            )
        )
        artifact["target"].append(task.val_actions.numpy())
        artifact["predicted_gate"].append(predicted_gate.numpy())

        for name, model in models.items():
            identity = _predict(
                model,
                demo_geometry,
                demo_geometry,
                demo_actions,
                device,
            )
            no_demo_actions = torch.zeros_like(demo_actions)
            no_demo_geometry = torch.zeros_like(demo_geometry)
            no_demo_mask = torch.zeros_like(mask)
            with torch.inference_mode():
                no_demo = model(
                    task.val_geometry.to(device),
                    no_demo_geometry.to(device),
                    no_demo_actions.to(device),
                    no_demo_mask.to(device),
                ).cpu()
            identity_error[name] = max(
                identity_error[name],
                float((identity - demo_actions).abs().max()),
            )
            no_demo_error[name] = max(
                no_demo_error[name],
                float(no_demo.abs().max()),
            )

    combined_target = torch.cat(aggregate_targets)
    combined_mask = torch.ones(combined_target.shape[:2], dtype=torch.bool)
    combined_predictions = {
        name: torch.cat(values) for name, values in aggregate_predictions.items()
    }
    if not all(torch.equal(task.pose_scales, first.pose_scales) for task in tasks):
        raise RuntimeError("aggregate action scales 前置检查失效")
    aggregate_options = {
        "pose_scales": first.pose_scales,
        "translation_threshold_m": first.translation_threshold_m,
        "rotation_threshold_rad": first.rotation_threshold_rad,
    }
    aggregate_metrics = {
        name: _physical_metrics(
            value,
            combined_target,
            combined_mask,
            **aggregate_options,
        )
        for name, value in combined_predictions.items()
    }
    aggregate_bootstrap = {
        name: _bootstrap_comparison(
            reference=combined_predictions[reference],
            candidate=combined_predictions[candidate],
            target=combined_target,
            mask=combined_mask,
            group_ids=aggregate_groups,
            seed=config.seed + offset * 1000,
            resamples=config.bootstrap_resamples,
            **aggregate_options,
        )
        for offset, (name, reference, candidate) in enumerate(
            (
                ("qa_minus_copy", "demo_action_copy", "query_aligned_transport"),
                ("bcsg_minus_copy", "demo_action_copy", "bcsg"),
                ("bcsg_minus_qa", "query_aligned_transport", "bcsg"),
                ("bcsg_minus_constant", "constant_shrinkage", "bcsg"),
            )
        )
    }
    bcsg_copy = aggregate_bootstrap["bcsg_minus_copy"]
    bcsg_qa = aggregate_bootstrap["bcsg_minus_qa"]
    improved_tasks = sum(
        task_report["metrics"]["bcsg"]["translation_l2_m"]
        < task_report["metrics"]["demo_action_copy"]["translation_l2_m"]
        for task_report in per_task.values()
    )
    no_task_chunk_degradation = all(
        report["paired_episode_bootstrap"]["bcsg_minus_copy"]
        ["chunk_threshold_accuracy"]["ci95_high"]
        >= 0.0
        for report in per_task.values()
    )
    first_protocol = protocols[0]
    first_bank_geometry = first.train_geometry[first_protocol.bank_indices]
    first_bank_actions = first.train_actions[first_protocol.bank_indices]
    first_demo_geometry = first_bank_geometry[
        first_protocol.validation_demo_indices
    ]
    first_demo_actions = first_bank_actions[first_protocol.validation_demo_indices]
    first_qa_prediction = _predict(
        qa,
        first.val_geometry,
        first_demo_geometry,
        first_demo_actions,
        device,
    )
    no_demo_actions = torch.zeros_like(first_demo_actions)
    no_demo_geometry = torch.zeros_like(first_demo_geometry)
    no_demo_mask = torch.zeros(
        first_demo_actions.shape[:2],
        dtype=torch.bool,
    )
    no_demo_prediction, no_demo_features = _qa_diagnostics(
        qa,
        first.val_geometry,
        no_demo_geometry,
        no_demo_actions,
        device,
    )
    with torch.inference_mode():
        no_demo_gate = gate(no_demo_features.to(device)).cpu()
    bcsg_no_demo = apply_shrinkage(
        no_demo_actions,
        no_demo_prediction,
        no_demo_gate,
        no_demo_mask,
    )
    latency = _latency(
        qa=qa,
        gate=gate,
        query=first.val_geometry,
        demo_geometry=first_demo_geometry,
        demo_actions=first_demo_actions,
        iterations=config.latency_iterations,
        device=device,
    )
    structural = {
        "identity_max_abs_error": identity_error,
        "no_demo_max_abs_output": no_demo_error,
        "bcsg_no_demo_max_abs_output": float(bcsg_no_demo.abs().max()),
        "gate_zero_exact": bool(
            torch.equal(
                apply_shrinkage(
                    first_demo_actions,
                    first_qa_prediction,
                    torch.zeros(len(first_demo_actions)),
                    torch.ones(first_demo_actions.shape[:2], dtype=torch.bool),
                ),
                first_demo_actions,
            )
        ),
        "gate_one_exact": bool(
            torch.equal(
                apply_shrinkage(
                    first_demo_actions,
                    first_qa_prediction,
                    torch.ones(len(first_demo_actions)),
                    torch.ones(first_demo_actions.shape[:2], dtype=torch.bool),
                ),
                first_qa_prediction,
            )
        ),
        "rotation_gripper_max_abs_error": float(
            (
                combined_predictions["bcsg"][..., 3:]
                - combined_predictions["demo_action_copy"][..., 3:]
            ).abs().max()
        ),
        "query_only_action_head": False,
    }
    parameters = {
        name: sum(parameter.numel() for parameter in model.parameters())
        for name, model in models.items()
    }
    parameters["benefit_calibrated_gate"] = sum(
        parameter.numel() for parameter in gate.parameters()
    )
    parameters["qa_plus_gate"] = (
        parameters["query_aligned_transport"]
        + parameters["benefit_calibrated_gate"]
    )
    criteria = {
        "e1_bcsg_translation_better_than_copy": (
            bcsg_copy["translation_l2_m"]["ci95_high"] < 0.0
        ),
        "e1_at_least_two_tasks_improve": improved_tasks >= 2,
        "e2_bcsg_translation_better_than_qa": (
            bcsg_qa["translation_l2_m"]["ci95_high"] < 0.0
        ),
        "e2_bcsg_point_better_than_constant": (
            aggregate_bootstrap["bcsg_minus_constant"]["translation_l2_m"]
            ["candidate_minus_reference"]
            < 0.0
        ),
        "e3_no_task_chunk_significantly_worse": no_task_chunk_degradation,
        "e4_structural": (
            max(identity_error.values()) == 0.0
            and max(no_demo_error.values()) == 0.0
            and structural["bcsg_no_demo_max_abs_output"] == 0.0
            and structural["gate_zero_exact"]
            and structural["gate_one_exact"]
            and structural["rotation_gripper_max_abs_error"] == 0.0
        ),
        "e4_parameter_budget": parameters["qa_plus_gate"] < 250_000,
        "e4_latency_budget": latency["p95_ms"] < 5.0,
    }

    checkpoint = {
        "models": {
            name: {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            }
            for name, model in models.items()
        },
        "model_configs": {
            name: asdict(model.config) for name, model in models.items()
        },
        "gate": {
            key: value.detach().cpu() for key, value in gate.state_dict().items()
        },
        "gate_config": asdict(gate.config),
        "constant_gate": float(constant_gate),
        "train_config": asdict(config),
        "geometry_mean": geometry_mean,
        "geometry_std": geometry_std,
        "action_pose_scales": first.pose_scales,
        "action_representation": first.action_representation,
        "data_summary_sha256": {
            task.task_id: _sha256(task.root / "summary.json") for task in tasks
        },
        "bank_chunk_ids": {
            item.task.task_id: [
                str(item.task.train_records[index]["chunk_id"])
                for index in item.bank_indices.tolist()
            ]
            for item in protocols
        },
        "git_commit": _git_commit(project_root),
    }
    checkpoint_path = temporary / "query_aligned_bcsg.pt"
    checkpoint_temporary = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(checkpoint, checkpoint_temporary)
    os.replace(checkpoint_temporary, checkpoint_path)
    artifact_path = temporary / "validation_predictions.npz"
    artifact_temporary = artifact_path.with_suffix(".npz.tmp")
    with artifact_temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            **{
                key: np.concatenate(values)
                for key, values in artifact.items()
            },
        )
    os.replace(artifact_temporary, artifact_path)
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "split": "20 operator-train / 4 gate-calibration / 8 validation episodes",
            "retrieval": "per-task phase-matched standardized 17D rank-1",
            "predictor": "translation-only fixed LR-DAT vs QA-LRDT + BCSG",
            "checkpoint_selection": "fixed final epoch; validation unused",
            "bootstrap_unit": "task + validation episode",
        },
        "git_commit": _git_commit(project_root),
        "device": str(device),
        "cuda_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "config": asdict(config),
        "config_sha256": _sha256(config_path),
        "data_summary_sha256": checkpoint["data_summary_sha256"],
        "train_queries": len(train_dataset),
        "calibration_queries": len(calibration_target_tensor),
        "validation_queries": len(combined_target),
        "constant_gate": float(constant_gate),
        "model_parameters": parameters,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "prediction_artifact_sha256": _sha256(artifact_path),
        "calibration": {
            "optimal_gate_mean": float(optimal_gate.mean()),
            "optimal_gate_zero_fraction": float((optimal_gate == 0).float().mean()),
            "optimal_gate_one_fraction": float((optimal_gate == 1).float().mean()),
        },
        "structural_audit": structural,
        "latency": latency,
        "aggregate_metrics": aggregate_metrics,
        "aggregate_paired_bootstrap": aggregate_bootstrap,
        "tasks": per_task,
        "preregistered_criteria": criteria,
    }
    (temporary / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (temporary / "TRAINING_COMPLETE").write_text(
        "maniskill_query_aligned_bcsg_v1\n",
        encoding="utf-8",
    )
    temporary.rename(output_root)
    atexit.unregister(cleanup_incomplete)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA，但当前节点没有可用 GPU")
    run(
        project_root=arguments.project_root.resolve(),
        data_roots=[path.resolve() for path in arguments.data_root],
        output_root=arguments.output_root.resolve(),
        config_path=arguments.config.resolve(),
        config=TrainConfig.from_json(arguments.config.resolve()),
        device=device,
    )


if __name__ == "__main__":
    main()

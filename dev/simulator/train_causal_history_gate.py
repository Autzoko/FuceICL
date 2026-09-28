"""冻结 QA-LRDT，仅训练并评估最小 causal-history benefit gate。"""

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
    frozen_transport_features,
    optimal_shrinkage_target,
)
from dev.predictor.causal_history_gate import (
    CausalHistoryBenefitGate,
    CausalHistoryGateConfig,
    causal_history_features,
    first_order_history,
    shuffled_history_within_episode,
)
from dev.predictor.query_aligned_transport import (
    QueryAlignedLowRankDemoTransport,
    QueryAlignedTransportConfig,
)
from dev.predictor.train_action_chunks import _physical_metrics
from dev.simulator.evaluate_maniskill_demo_prior import _bootstrap_comparison
from dev.simulator.train_maniskill_low_rank_transport import (
    TaskData,
    _load_task,
    _nearest_demo_indices,
    _selection_hash,
)
from dev.simulator.train_maniskill_query_aligned_bcsg import _qa_diagnostics


EXPECTED_TASKS = frozenset(("PickCube-v1", "PushCube-v1", "PullCube-v1"))
VARIANTS = ("first_order", "two_lag", "shuffled_two_lag")


@dataclass(frozen=True)
class HistoryGateTrainConfig:
    """冻结的 CHBG mechanism-feasibility 协议。"""

    schema_version: str
    seed: int
    operator_train_episodes: list[int]
    gate_train_episodes: list[int]
    validation_episodes: list[int]
    gate_epochs: int
    gate_batch_size: int
    gate_learning_rate: float
    gate_weight_decay: float
    gate_hidden_dim: int
    gradient_clip_norm: float
    bootstrap_resamples: int
    latency_iterations: int

    @classmethod
    def from_json(cls, path: Path) -> "HistoryGateTrainConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def __post_init__(self) -> None:
        positive = (
            self.gate_epochs,
            self.gate_batch_size,
            self.gate_learning_rate,
            self.gate_hidden_dim,
            self.gradient_clip_norm,
            self.bootstrap_resamples,
            self.latency_iterations,
        )
        if self.schema_version != "maniskill-causal-history-gate-v1":
            raise ValueError("未知 CHBG schema")
        if min(positive) <= 0 or self.gate_weight_decay < 0:
            raise ValueError("CHBG 训练超参数非法")
        groups = tuple(
            set(values)
            for values in (
                self.operator_train_episodes,
                self.gate_train_episodes,
                self.validation_episodes,
            )
        )
        if any(not values for values in groups):
            raise ValueError("CHBG episode splits 不能为空")
        if any(
            len(values) != len(original)
            for values, original in zip(
                groups,
                (
                    self.operator_train_episodes,
                    self.gate_train_episodes,
                    self.validation_episodes,
                ),
            )
        ):
            raise ValueError("CHBG episode splits 不能包含重复值")
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise ValueError("CHBG episode splits 必须互斥")


@dataclass(frozen=True)
class TaskProtocol:
    task: TaskData
    bank_indices: torch.Tensor
    train_indices: torch.Tensor
    train_demo_indices: torch.Tensor
    val_demo_indices: torch.Tensor
    train_history: torch.Tensor
    val_history: torch.Tensor


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


def _checkpoint_bank_indices(
    records: Sequence[Mapping[str, Any]],
    chunk_ids: Sequence[str],
) -> torch.Tensor:
    by_id = {str(record["chunk_id"]): index for index, record in enumerate(records)}
    if len(by_id) != len(records):
        raise ValueError("manifest chunk_id 不唯一")
    missing = [value for value in chunk_ids if value not in by_id]
    if missing:
        raise ValueError(f"checkpoint bank chunks 缺失：{missing[:3]}")
    return torch.tensor([by_id[value] for value in chunk_ids], dtype=torch.long)


def _load_frozen_models(
    checkpoint: Mapping[str, Any],
    device: torch.device,
) -> tuple[QueryAlignedLowRankDemoTransport, BenefitCalibratedGate]:
    transport = QueryAlignedLowRankDemoTransport(
        QueryAlignedTransportConfig(
            **checkpoint["model_configs"]["query_aligned_transport"]
        ),
        geometry_mean=checkpoint["geometry_mean"],
        geometry_std=checkpoint["geometry_std"],
    )
    transport.load_state_dict(checkpoint["models"]["query_aligned_transport"])
    gate = BenefitCalibratedGate(
        BenefitGateConfig(**checkpoint["gate_config"])
    )
    gate.load_state_dict(checkpoint["gate"])
    return transport.to(device).eval(), gate.to(device).eval()


def _build_protocol(
    task: TaskData,
    checkpoint: Mapping[str, Any],
    config: HistoryGateTrainConfig,
) -> TaskProtocol:
    train_episodes = set(config.gate_train_episodes)
    expected_train = set(config.operator_train_episodes) | train_episodes
    observed_train = {int(record["episode"]) for record in task.train_records}
    observed_val = {int(record["episode"]) for record in task.val_records}
    if observed_train != expected_train:
        raise ValueError(f"{task.task_id} train split 与冻结协议不一致")
    if observed_val != set(config.validation_episodes):
        raise ValueError(f"{task.task_id} validation split 与冻结协议不一致")
    bank_indices = _checkpoint_bank_indices(
        task.train_records,
        checkpoint["bank_chunk_ids"][task.task_id],
    )
    train_indices = _indices_for_episodes(task.train_records, train_episodes)
    bank_records = [task.train_records[index] for index in bank_indices.tolist()]
    train_records = [task.train_records[index] for index in train_indices.tolist()]
    bank_geometry = task.train_geometry[bank_indices]
    train_demo, _ = _nearest_demo_indices(
        candidate_records=bank_records,
        candidate_geometry=bank_geometry,
        query_records=train_records,
        query_geometry=task.train_geometry[train_indices],
        exclude_same_episode=False,
    )
    val_demo, _ = _nearest_demo_indices(
        candidate_records=bank_records,
        candidate_geometry=bank_geometry,
        query_records=task.val_records,
        query_geometry=task.val_geometry,
        exclude_same_episode=False,
    )
    full_train_history = causal_history_features(
        task.train_records,
        task.train_geometry,
    )
    val_history = causal_history_features(task.val_records, task.val_geometry)
    return TaskProtocol(
        task=task,
        bank_indices=bank_indices,
        train_indices=train_indices,
        train_demo_indices=train_demo,
        val_demo_indices=val_demo,
        train_history=full_train_history[train_indices],
        val_history=val_history,
    )


def _variant_history(
    variant: str,
    records: Sequence[Mapping[str, Any]],
    history: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    if variant == "first_order":
        return first_order_history(history)
    if variant == "two_lag":
        return history
    if variant == "shuffled_two_lag":
        return shuffled_history_within_episode(records, history, seed=seed)
    raise ValueError(f"未知 history variant：{variant}")


def _train_gate(
    *,
    features: torch.Tensor,
    targets: torch.Tensor,
    energy: torch.Tensor,
    config: HistoryGateTrainConfig,
    device: torch.device,
    variant: str,
    log_path: Path,
) -> CausalHistoryBenefitGate:
    _seed_everything(config.seed)
    model_config = CausalHistoryGateConfig(
        input_dim=features.shape[1],
        hidden_dim=config.gate_hidden_dim,
    )
    gate = CausalHistoryBenefitGate(
        model_config,
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
                "model": variant,
                "epoch": epoch + 1,
                "weighted_gate_mse": total / examples,
            }
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
    return gate.eval()


def _weighted_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    energy: torch.Tensor,
) -> float:
    return float(
        (energy * (prediction - target).square()).sum()
        / energy.sum().clamp_min(1e-8)
    )


@torch.inference_mode()
def _latency(
    *,
    transport: QueryAlignedLowRankDemoTransport,
    gate: CausalHistoryBenefitGate,
    query: torch.Tensor,
    demo_geometry: torch.Tensor,
    demo_actions: torch.Tensor,
    history: torch.Tensor,
    device: torch.device,
    iterations: int,
) -> dict[str, float | int]:
    query = query[:1].to(device)
    demo_geometry = demo_geometry[:1].to(device)
    demo_actions = demo_actions[:1].to(device)
    history = history[:1].to(device)
    mask = torch.ones(demo_actions.shape[:2], dtype=torch.bool, device=device)

    def predict() -> torch.Tensor:
        transported, base = frozen_transport_features(
            transport,
            query,
            demo_geometry,
            demo_actions,
            mask,
        )
        value = gate(torch.cat((base, history), dim=1))
        return apply_shrinkage(demo_actions, transported, value, mask)

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
        "maximum_ms": float(values.max()),
    }


def run(
    *,
    project_root: Path,
    data_roots: Sequence[Path],
    checkpoint_path: Path,
    config_path: Path,
    output_root: Path,
    config: HistoryGateTrainConfig,
    device: torch.device,
) -> None:
    if output_root.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output_root}")
    temporary = output_root.with_name(f".{output_root.name}.incomplete-{os.getpid()}")
    temporary.mkdir(parents=True)

    def cleanup() -> None:
        shutil.rmtree(temporary, ignore_errors=True)

    atexit.register(cleanup)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    tasks = [_load_task(root) for root in data_roots]
    if len(tasks) != 3 or {task.task_id for task in tasks} != EXPECTED_TASKS:
        raise ValueError("CHBG 必须恰好覆盖冻结三任务")
    for task in tasks:
        expected_hash = checkpoint["data_summary_sha256"].get(task.task_id)
        if expected_hash != _sha256(task.root / "summary.json"):
            raise ValueError(f"{task.task_id} data 与 checkpoint 不匹配")
        if task.action_representation != checkpoint["action_representation"]:
            raise ValueError(f"{task.task_id} action representation 不匹配")
        if not torch.equal(
            task.pose_scales.float(),
            torch.as_tensor(checkpoint["action_pose_scales"]).float(),
        ):
            raise ValueError(f"{task.task_id} action scales 不匹配")
    transport, baseline_gate = _load_frozen_models(checkpoint, device)
    protocols = [
        _build_protocol(task, checkpoint, config) for task in tasks
    ]

    calibration_base = []
    calibration_history: dict[str, list[torch.Tensor]] = {
        variant: [] for variant in VARIANTS
    }
    calibration_target = []
    calibration_energy = []
    for offset, item in enumerate(protocols):
        task = item.task
        bank_geometry = task.train_geometry[item.bank_indices]
        bank_actions = task.train_actions[item.bank_indices]
        demo_geometry = bank_geometry[item.train_demo_indices]
        demo_actions = bank_actions[item.train_demo_indices]
        target = task.train_actions[item.train_indices]
        transported, base = _qa_diagnostics(
            transport,
            task.train_geometry[item.train_indices],
            demo_geometry,
            demo_actions,
            device,
        )
        mask = torch.ones(target.shape[:2], dtype=torch.bool)
        optimal, energy = optimal_shrinkage_target(
            demo_actions,
            transported,
            target,
            mask,
        )
        records = [task.train_records[index] for index in item.train_indices.tolist()]
        calibration_base.append(base)
        for variant in VARIANTS:
            calibration_history[variant].append(
                _variant_history(
                    variant,
                    records,
                    item.train_history,
                    seed=config.seed + offset,
                )
            )
        calibration_target.append(optimal)
        calibration_energy.append(energy)
    base_tensor = torch.cat(calibration_base)
    target_tensor = torch.cat(calibration_target)
    energy_tensor = torch.cat(calibration_energy)
    training_features = {
        variant: torch.cat((base_tensor, torch.cat(values)), dim=1)
        for variant, values in calibration_history.items()
    }
    gates = {
        variant: _train_gate(
            features=training_features[variant],
            targets=target_tensor,
            energy=energy_tensor,
            config=config,
            device=device,
            variant=variant,
            log_path=temporary / "training_metrics.jsonl",
        )
        for variant in VARIANTS
    }

    per_task: dict[str, Any] = {}
    aggregate_target = []
    aggregate_predictions: dict[str, list[torch.Tensor]] = {
        name: [] for name in ("copy", "bcsg", *VARIANTS)
    }
    aggregate_groups: list[str] = []
    aggregate_gate_mse: dict[str, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {
        name: [] for name in ("bcsg", *VARIANTS)
    }
    latency_inputs = None
    for offset, item in enumerate(protocols):
        task = item.task
        bank_geometry = task.train_geometry[item.bank_indices]
        bank_actions = task.train_actions[item.bank_indices]
        demo_geometry = bank_geometry[item.val_demo_indices]
        demo_actions = bank_actions[item.val_demo_indices]
        transported, base = _qa_diagnostics(
            transport,
            task.val_geometry,
            demo_geometry,
            demo_actions,
            device,
        )
        mask = torch.ones(task.val_actions.shape[:2], dtype=torch.bool)
        optimal, energy = optimal_shrinkage_target(
            demo_actions,
            transported,
            task.val_actions,
            mask,
        )
        with torch.inference_mode():
            baseline_values = baseline_gate(base.to(device)).cpu()
        history_values = {}
        predictions = {
            "copy": demo_actions,
            "bcsg": apply_shrinkage(
                demo_actions,
                transported,
                baseline_values,
                mask,
            ),
        }
        for variant, gate in gates.items():
            history = _variant_history(
                variant,
                task.val_records,
                item.val_history,
                seed=config.seed + 100 + offset,
            )
            features = torch.cat((base, history), dim=1)
            with torch.inference_mode():
                values = gate(features.to(device)).cpu()
            history_values[variant] = values
            predictions[variant] = apply_shrinkage(
                demo_actions,
                transported,
                values,
                mask,
            )
        options = {
            "pose_scales": task.pose_scales,
            "translation_threshold_m": task.translation_threshold_m,
            "rotation_threshold_rad": task.rotation_threshold_rad,
        }
        metrics = {
            name: _physical_metrics(value, task.val_actions, mask, **options)
            for name, value in predictions.items()
        }
        groups = [f"{task.task_id}:{row['episode']}" for row in task.val_records]
        comparisons = {
            f"{name}_minus_bcsg": _bootstrap_comparison(
                reference=predictions["bcsg"],
                candidate=predictions[name],
                target=task.val_actions,
                mask=mask,
                group_ids=groups,
                seed=config.seed + offset * 100 + index,
                resamples=config.bootstrap_resamples,
                **options,
            )
            for index, name in enumerate(VARIANTS)
        }
        gate_mse = {
            "bcsg": _weighted_mse(baseline_values, optimal, energy),
            **{
                name: _weighted_mse(values, optimal, energy)
                for name, values in history_values.items()
            },
        }
        per_task[task.task_id] = {
            "bank_chunks": len(item.bank_indices),
            "gate_train_chunks": len(item.train_indices),
            "validation_chunks": len(task.val_records),
            "bank_selection_sha256": _selection_hash(item.bank_indices),
            "weighted_gate_mse": gate_mse,
            "metrics": metrics,
            "paired_episode_bootstrap": comparisons,
        }
        aggregate_target.append(task.val_actions)
        aggregate_groups.extend(groups)
        for name, value in predictions.items():
            aggregate_predictions[name].append(value)
        aggregate_gate_mse["bcsg"].append((baseline_values, optimal, energy))
        for name, values in history_values.items():
            aggregate_gate_mse[name].append((values, optimal, energy))
        if latency_inputs is None:
            latency_inputs = (
                task.val_geometry,
                demo_geometry,
                demo_actions,
                item.val_history,
            )

    target = torch.cat(aggregate_target)
    predictions = {
        name: torch.cat(values) for name, values in aggregate_predictions.items()
    }
    mask = torch.ones(target.shape[:2], dtype=torch.bool)
    first = tasks[0]
    options = {
        "pose_scales": first.pose_scales,
        "translation_threshold_m": first.translation_threshold_m,
        "rotation_threshold_rad": first.rotation_threshold_rad,
    }
    metrics = {
        name: _physical_metrics(value, target, mask, **options)
        for name, value in predictions.items()
    }
    comparisons = {
        f"{name}_minus_bcsg": _bootstrap_comparison(
            reference=predictions["bcsg"],
            candidate=predictions[name],
            target=target,
            mask=mask,
            group_ids=aggregate_groups,
            seed=config.seed + 1000 + index,
            resamples=config.bootstrap_resamples,
            **options,
        )
        for index, name in enumerate(VARIANTS)
    }
    comparisons["two_lag_minus_shuffled"] = _bootstrap_comparison(
        reference=predictions["shuffled_two_lag"],
        candidate=predictions["two_lag"],
        target=target,
        mask=mask,
        group_ids=aggregate_groups,
        seed=config.seed + 2000,
        resamples=config.bootstrap_resamples,
        **options,
    )
    gate_mse = {}
    for name, values in aggregate_gate_mse.items():
        prediction = torch.cat([value[0] for value in values])
        optimal = torch.cat([value[1] for value in values])
        energy = torch.cat([value[2] for value in values])
        gate_mse[name] = _weighted_mse(prediction, optimal, energy)
    if latency_inputs is None:
        raise RuntimeError("缺少 CHBG latency 输入")
    latency = _latency(
        transport=transport,
        gate=gates["two_lag"],
        query=latency_inputs[0],
        demo_geometry=latency_inputs[1],
        demo_actions=latency_inputs[2],
        history=latency_inputs[3],
        device=device,
        iterations=config.latency_iterations,
    )
    no_demo_actions = torch.zeros_like(latency_inputs[2][:1]).to(device)
    no_demo_geometry = torch.zeros_like(latency_inputs[1][:1]).to(device)
    no_demo_mask = torch.zeros(
        no_demo_actions.shape[:2], dtype=torch.bool, device=device
    )
    with torch.inference_mode():
        no_demo, no_demo_base = frozen_transport_features(
            transport,
            latency_inputs[0][:1].to(device),
            no_demo_geometry,
            no_demo_actions,
            no_demo_mask,
        )
        no_demo_gate = gates["two_lag"](
            torch.cat(
                (no_demo_base, torch.zeros_like(latency_inputs[3][:1]).to(device)),
                dim=1,
            )
        )
        no_demo_output = apply_shrinkage(
            no_demo_actions,
            no_demo,
            no_demo_gate,
            no_demo_mask,
        )
        regular_mask = torch.ones_like(no_demo_mask)
        regular_transport, _ = frozen_transport_features(
            transport,
            latency_inputs[0][:1].to(device),
            latency_inputs[1][:1].to(device),
            latency_inputs[2][:1].to(device),
            regular_mask,
        )
        gate_zero_output = apply_shrinkage(
            latency_inputs[2][:1].to(device),
            regular_transport,
            torch.zeros(1, device=device),
            regular_mask,
        )
        gate_zero_error = float(
            (
                gate_zero_output - latency_inputs[2][:1].to(device)
            ).abs().max()
        )
    parameter_counts = {
        "qa_lrdat": sum(value.numel() for value in transport.parameters()),
        "frozen_bcsg": sum(value.numel() for value in baseline_gate.parameters()),
        **{
            name: sum(value.numel() for value in gate.parameters())
            for name, gate in gates.items()
        },
    }
    two_lag_delta = comparisons["two_lag_minus_bcsg"]["translation_l2_m"]
    shuffled_delta = comparisons["two_lag_minus_shuffled"]["translation_l2_m"]
    criteria = {
        "h1_gate_mse_improves_at_least_10_percent": (
            gate_mse["two_lag"] <= 0.9 * gate_mse["bcsg"]
        ),
        "h2_translation_ci_better_than_bcsg": two_lag_delta["ci95_high"] < 0.0,
        "h3_gate_mse_better_than_shuffled": (
            gate_mse["two_lag"] < gate_mse["shuffled_two_lag"]
        ),
        "h3_translation_better_than_shuffled": (
            shuffled_delta["candidate_minus_reference"] < 0.0
        ),
        "h4_parameter_budget": (
            parameter_counts["qa_lrdat"] + parameter_counts["two_lag"] < 250_000
        ),
        "h4_no_demo_exact": float(no_demo_output.abs().max()) == 0.0,
        "h4_gate_zero_exact_copy": gate_zero_error == 0.0,
        "h4_latency_p95_below_5ms": latency["p95_ms"] < 5.0,
    }
    checkpoint_output = {
        "schema_version": 1,
        "git_commit": _git_commit(project_root),
        "source_checkpoint_sha256": _sha256(checkpoint_path),
        "models": {
            name: {key: value.detach().cpu() for key, value in gate.state_dict().items()}
            for name, gate in gates.items()
        },
        "model_configs": {
            name: asdict(gate.config) for name, gate in gates.items()
        },
    }
    checkpoint_output_path = temporary / "causal_history_gates.pt"
    torch.save(checkpoint_output, checkpoint_output_path)
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(project_root),
        "protocol": (
            "frozen QA-LRDT; 4 gate-train / 8 validation episodes per task; "
            "causal two-lag geometry differences"
        ),
        "config": asdict(config),
        "config_sha256": _sha256(config_path),
        "source_checkpoint_sha256": _sha256(checkpoint_path),
        "data_summary_sha256": {
            task.task_id: _sha256(task.root / "summary.json") for task in tasks
        },
        "device": str(device),
        "parameters": parameter_counts,
        "calibration": {
            "queries": len(target_tensor),
            "optimal_gate_mean": float(target_tensor.mean()),
            "residual_energy_mean": float(energy_tensor.mean()),
        },
        "per_task": per_task,
        "aggregate": {
            "queries": len(target),
            "weighted_gate_mse": gate_mse,
            "metrics": metrics,
            "paired_episode_bootstrap": comparisons,
        },
        "latency": latency,
        "structural": {
            "two_lag_no_demo_max_abs_output": float(no_demo_output.abs().max()),
            "gate_zero_max_abs_error_from_copy": gate_zero_error,
        },
        "criteria": criteria,
        "checkpoint_sha256": _sha256(checkpoint_output_path),
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
    configuration = HistoryGateTrainConfig.from_json(arguments.config.resolve())
    run(
        project_root=arguments.project_root.resolve(),
        data_roots=[path.resolve() for path in arguments.data_root],
        checkpoint_path=arguments.checkpoint.resolve(),
        config_path=arguments.config.resolve(),
        output_root=arguments.output_root.resolve(),
        config=configuration,
        device=torch.device(arguments.device),
    )

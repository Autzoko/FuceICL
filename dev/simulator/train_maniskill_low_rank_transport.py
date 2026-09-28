"""在多任务 ManiSkill executable chunks 上训练并审计 LR-DAT。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dev.pointnet.compare_retrievers import _sha256
from dev.predictor.low_rank_demo_transport import (
    LowRankDemoActionTransport,
    LowRankTransportConfig,
)
from dev.predictor.train_action_chunks import (
    _masked_sample_loss,
    _physical_metrics,
)
from dev.simulator.evaluate_maniskill_demo_prior import (
    _action_protocol,
    _bootstrap_comparison,
    _load_split,
)


@dataclass(frozen=True)
class TrainConfig:
    """结果揭盲前冻结的多任务 LR-DAT 训练配置。"""

    schema_version: str
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    stability_weight: float
    rank: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    feedforward_dim: int
    dropout: float
    residual_limit: float
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
            self.bootstrap_resamples,
            self.latency_iterations,
        )
        if not self.schema_version.strip() or min(positive) <= 0:
            raise ValueError("schema version 与训练正数参数必须有效")
        if self.weight_decay < 0 or self.stability_weight < 0:
            raise ValueError("regularization weight 不能为负")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout 必须位于 [0,1)")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim 必须能被 num_heads 整除")


@dataclass(frozen=True)
class TaskData:
    task_id: str
    root: Path
    pose_scales: torch.Tensor
    translation_threshold_m: float
    rotation_threshold_rad: float
    action_representation: str
    train_records: list[dict[str, Any]]
    train_geometry: torch.Tensor
    train_actions: torch.Tensor
    val_records: list[dict[str, Any]]
    val_geometry: torch.Tensor
    val_actions: torch.Tensor


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


def _load_task(root: Path) -> TaskData:
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    task_id = str(summary.get("config", {}).get("task_id", ""))
    if not task_id:
        raise ValueError(f"{root} summary 缺少 task_id")
    scales, translation, rotation, representation = _action_protocol(root)
    train_records, train_geometry, train_actions = _load_split(
        root,
        "train",
        scales,
    )
    val_records, val_geometry, val_actions = _load_split(root, "val", scales)
    return TaskData(
        task_id=task_id,
        root=root,
        pose_scales=scales,
        translation_threshold_m=translation,
        rotation_threshold_rad=rotation,
        action_representation=representation,
        train_records=train_records,
        train_geometry=train_geometry,
        train_actions=train_actions,
        val_records=val_records,
        val_geometry=val_geometry,
        val_actions=val_actions,
    )


def _phase(geometry: torch.Tensor) -> torch.Tensor:
    """只用当前可观测 gripper opening 区分离散接触阶段。"""
    return geometry[:, 15] >= 0.5


def _nearest_demo_indices(
    *,
    candidate_records: Sequence[dict[str, Any]],
    candidate_geometry: torch.Tensor,
    query_records: Sequence[dict[str, Any]],
    query_geometry: torch.Tensor,
    exclude_same_episode: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    mean = candidate_geometry.mean(dim=0)
    std = candidate_geometry.std(dim=0, unbiased=False).clamp_min(1e-4)
    scale = float(np.sqrt(candidate_geometry.shape[1]))
    distances = torch.cdist(
        (query_geometry - mean) / std,
        (candidate_geometry - mean) / std,
    ) / scale
    valid = _phase(query_geometry)[:, None] == _phase(candidate_geometry)[None, :]
    if exclude_same_episode:
        query_episodes = torch.tensor(
            [int(record["episode"]) for record in query_records]
        )
        candidate_episodes = torch.tensor(
            [int(record["episode"]) for record in candidate_records]
        )
        valid &= query_episodes[:, None] != candidate_episodes[None, :]
    if not bool(valid.any(dim=1).all()):
        raise ValueError("至少一个 query 没有跨 episode、同 phase Demo")
    masked = distances.masked_fill(~valid, torch.inf)
    selected = masked.argmin(dim=1)
    return selected, masked[torch.arange(len(query_geometry)), selected]


def _selection_hash(indices: torch.Tensor) -> str:
    values = indices.detach().cpu().numpy().astype("<i8", copy=False)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _wrong_task_demo(
    query: TaskData,
    tasks: Sequence[TaskData],
    mean: torch.Tensor,
    std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    candidates = [task for task in tasks if task.task_id != query.task_id]
    geometry = torch.cat([task.train_geometry for task in candidates])
    actions = torch.cat([task.train_actions for task in candidates])
    distances = torch.cdist(
        (query.val_geometry - mean) / std,
        (geometry - mean) / std,
    ) / float(np.sqrt(geometry.shape[1]))
    valid = _phase(query.val_geometry)[:, None] == _phase(geometry)[None, :]
    if not bool(valid.any(dim=1).all()):
        raise ValueError(f"{query.task_id} 缺少 wrong-task phase-matched Demo")
    indices = distances.masked_fill(~valid, torch.inf).argmin(dim=1)
    return geometry[indices], actions[indices]


def _build_model(
    config: TrainConfig,
    *,
    horizon: int,
    geometry_dim: int,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> LowRankDemoActionTransport:
    model_config = LowRankTransportConfig(
        horizon=horizon,
        geometry_dim=geometry_dim,
        rank=config.rank,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        feedforward_dim=config.feedforward_dim,
        dropout=config.dropout,
        residual_limit=config.residual_limit,
    )
    return LowRankDemoActionTransport(
        model_config,
        geometry_mean=mean,
        geometry_std=std,
    )


def _train(
    *,
    model: LowRankDemoActionTransport,
    dataset: TensorDataset,
    config: TrainConfig,
    device: torch.device,
    log_path: Path,
) -> LowRankDemoActionTransport:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
    )
    model.to(device)
    for epoch in range(config.epochs):
        generator = torch.Generator().manual_seed(config.seed + epoch)
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
        )
        totals = {"loss": 0.0, "prediction": 0.0, "stability": 0.0}
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
            prediction_loss = _masked_sample_loss(
                prediction,
                target,
                mask,
            ).mean()
            bounds = model.operator_frobenius_upper_bound(
                demo_geometry,
                demo_actions,
                mask,
            )
            stability_loss = bounds.square().mean()
            loss = prediction_loss + config.stability_weight * stability_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            batch_size = len(query)
            totals["loss"] += float(loss.detach()) * batch_size
            totals["prediction"] += float(prediction_loss.detach()) * batch_size
            totals["stability"] += float(stability_loss.detach()) * batch_size
            examples += batch_size
        scheduler.step()
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == config.epochs:
            row = {
                "epoch": epoch + 1,
                "learning_rate": optimizer.param_groups[0]["lr"],
                **{name: value / examples for name, value in totals.items()},
            }
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
    return model.eval()


@torch.inference_mode()
def _predict(
    model: LowRankDemoActionTransport,
    query: torch.Tensor,
    demo_geometry: torch.Tensor,
    demo_actions: torch.Tensor,
    device: torch.device,
    *,
    valid_demo: bool = True,
) -> torch.Tensor:
    mask = torch.full(
        demo_actions.shape[:2],
        valid_demo,
        dtype=torch.bool,
        device=device,
    )
    return model(
        query.to(device),
        demo_geometry.to(device),
        demo_actions.to(device),
        mask,
    ).cpu()


@torch.inference_mode()
def _latency(
    model: LowRankDemoActionTransport,
    task: TaskData,
    demo_indices: torch.Tensor,
    device: torch.device,
    iterations: int,
) -> dict[str, float | int]:
    query = task.val_geometry[:1].to(device)
    demo_geometry = task.train_geometry[demo_indices[:1]].to(device)
    demo_actions = task.train_actions[demo_indices[:1]].to(device)
    mask = torch.ones((1, demo_actions.shape[1]), dtype=torch.bool, device=device)
    for _ in range(20):
        model(query, demo_geometry, demo_actions, mask)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        model(query, demo_geometry, demo_actions, mask)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        samples.append(1000.0 * (time.perf_counter() - start))
    values = np.asarray(samples)
    return {
        "iterations": iterations,
        "batch_size": 1,
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
    held_out_task: str | None,
) -> None:
    if output_root.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output_root}")
    temporary = output_root.with_name(f".{output_root.name}.incomplete-{os.getpid()}")
    temporary.mkdir(parents=True)
    _seed_everything(config.seed)

    tasks = [_load_task(root) for root in data_roots]
    task_ids = [task.task_id for task in tasks]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("data roots 含重复 task_id")
    if len(tasks) < 2:
        raise ValueError("多任务实验至少需要两个 task roots")
    if held_out_task is not None and held_out_task not in task_ids:
        raise ValueError(f"未知 held-out task：{held_out_task}")
    first = tasks[0]
    for task in tasks[1:]:
        if task.action_representation != first.action_representation:
            raise ValueError("各任务 action representation 不一致")
        if not torch.equal(task.pose_scales, first.pose_scales):
            raise ValueError("各任务 action scales 不一致")
        if task.train_actions.shape[1:] != first.train_actions.shape[1:]:
            raise ValueError("各任务 action chunk shape 不一致")

    source_tasks = [task for task in tasks if task.task_id != held_out_task]
    train_queries = []
    train_targets = []
    train_demo_geometry = []
    train_demo_actions = []
    train_selection: dict[str, dict[str, Any]] = {}
    val_selection: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for task in tasks:
        train_indices, train_distances = _nearest_demo_indices(
            candidate_records=task.train_records,
            candidate_geometry=task.train_geometry,
            query_records=task.train_records,
            query_geometry=task.train_geometry,
            exclude_same_episode=True,
        )
        val_indices, val_distances = _nearest_demo_indices(
            candidate_records=task.train_records,
            candidate_geometry=task.train_geometry,
            query_records=task.val_records,
            query_geometry=task.val_geometry,
            exclude_same_episode=False,
        )
        val_selection[task.task_id] = (val_indices, val_distances)
        train_selection[task.task_id] = {
            "indices_sha256": _selection_hash(train_indices),
            "distance_mean": float(train_distances.mean()),
            "distance_p95": float(torch.quantile(train_distances, 0.95)),
        }
        if task.task_id != held_out_task:
            train_queries.append(task.train_geometry)
            train_targets.append(task.train_actions)
            train_demo_geometry.append(task.train_geometry[train_indices])
            train_demo_actions.append(task.train_actions[train_indices])

    query_tensor = torch.cat(train_queries)
    target_tensor = torch.cat(train_targets)
    demo_geometry_tensor = torch.cat(train_demo_geometry)
    demo_action_tensor = torch.cat(train_demo_actions)
    geometry_mean = query_tensor.mean(dim=0)
    geometry_std = query_tensor.std(dim=0, unbiased=False).clamp_min(1e-4)
    dataset = TensorDataset(
        query_tensor,
        target_tensor,
        demo_geometry_tensor,
        demo_action_tensor,
    )
    model = _build_model(
        config,
        horizon=int(target_tensor.shape[1]),
        geometry_dim=int(query_tensor.shape[1]),
        mean=geometry_mean,
        std=geometry_std,
    )
    model = _train(
        model=model,
        dataset=dataset,
        config=config,
        device=device,
        log_path=temporary / "training_metrics.jsonl",
    )

    per_task: dict[str, Any] = {}
    identity_error = 0.0
    no_demo_error = 0.0
    for task in tasks:
        indices, distances = val_selection[task.task_id]
        demo_geometry = task.train_geometry[indices]
        demo_actions = task.train_actions[indices]
        mask = torch.ones(task.val_actions.shape[:2], dtype=torch.bool)
        prediction = _predict(
            model,
            task.val_geometry,
            demo_geometry,
            demo_actions,
            device,
        )
        no_demo = _predict(
            model,
            task.val_geometry,
            torch.zeros_like(task.val_geometry),
            torch.zeros_like(task.val_actions),
            device,
            valid_demo=False,
        )
        wrong_geometry, wrong_actions = _wrong_task_demo(
            task,
            tasks,
            geometry_mean,
            geometry_std,
        )
        wrong_task = _predict(
            model,
            task.val_geometry,
            wrong_geometry,
            wrong_actions,
            device,
        )
        shuffled_actions = torch.roll(demo_actions, shifts=1, dims=0)
        shuffled = _predict(
            model,
            task.val_geometry,
            demo_geometry,
            shuffled_actions,
            device,
        )
        identity = _predict(
            model,
            demo_geometry,
            demo_geometry,
            demo_actions,
            device,
        )
        identity_error = max(
            identity_error,
            float((identity - demo_actions).abs().max()),
        )
        no_demo_error = max(no_demo_error, float(no_demo.abs().max()))
        options = {
            "pose_scales": task.pose_scales,
            "translation_threshold_m": task.translation_threshold_m,
            "rotation_threshold_rad": task.rotation_threshold_rad,
        }
        predictions = {
            "retrieved_demo_copy": demo_actions,
            "low_rank_demo_transport": prediction,
            "wrong_task_demo_transport": wrong_task,
            "shuffled_action_transport": shuffled,
        }
        metrics = {
            name: _physical_metrics(value, task.val_actions, mask, **options)
            for name, value in predictions.items()
        }
        groups = [f"{task.task_id}:{row['episode']}" for row in task.val_records]
        per_task[task.task_id] = {
            "train_queries": len(task.train_actions),
            "validation_queries": len(task.val_actions),
            "validation_episodes": len({row["episode"] for row in task.val_records}),
            "validation_demo_selection_sha256": _selection_hash(indices),
            "validation_demo_distance_mean": float(distances.mean()),
            "validation_demo_distance_p95": float(torch.quantile(distances, 0.95)),
            "metrics": metrics,
            "paired_episode_bootstrap": {
                "transport_minus_copy": _bootstrap_comparison(
                    reference=demo_actions,
                    candidate=prediction,
                    target=task.val_actions,
                    mask=mask,
                    group_ids=groups,
                    seed=config.seed,
                    resamples=config.bootstrap_resamples,
                    **options,
                )
            },
        }

    checkpoint = {
        "model": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "model_config": asdict(model.config),
        "train_config": asdict(config),
        "held_out_task": held_out_task,
        "source_tasks": [task.task_id for task in source_tasks],
        "geometry_mean": geometry_mean,
        "geometry_std": geometry_std,
        "action_pose_scales": first.pose_scales,
        "action_representation": first.action_representation,
        "data_summary_sha256": {
            task.task_id: _sha256(task.root / "summary.json") for task in tasks
        },
        "git_commit": _git_commit(project_root),
    }
    checkpoint_tmp = temporary / "low_rank_demo_transport.pt.tmp"
    checkpoint_path = temporary / "low_rank_demo_transport.pt"
    torch.save(checkpoint, checkpoint_tmp)
    os.replace(checkpoint_tmp, checkpoint_path)

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "predictor": "rank-4 Demo-conditioned local action transport",
            "input": "17D query/Demo canonical geometry + Demo Hx7 action",
            "retrieval": "per-task standardized geometry, phase matched, cross episode",
            "output": "Demo action plus continuous 6D low-rank residual; Demo gripper unchanged",
            "checkpoint_selection": "fixed final epoch; validation unused",
            "counterfactuals": "wrong-task Demo and same-task shuffled Demo action",
            "bootstrap_unit": "validation episode",
        },
        "device": str(device),
        "cuda_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "config": asdict(config),
        "held_out_task": held_out_task,
        "source_tasks": [task.task_id for task in source_tasks],
        "git_commit": _git_commit(project_root),
        "config_sha256": _sha256(config_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "data_summary_sha256": checkpoint["data_summary_sha256"],
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "train_queries": len(dataset),
        "train_selection": train_selection,
        "structural_audit": {
            "identity_max_abs_error": identity_error,
            "no_demo_max_abs_output": no_demo_error,
            "query_only_action_head": False,
            "transport_rank": config.rank,
        },
        "latency": _latency(
            model,
            tasks[0],
            val_selection[tasks[0].task_id][0],
            device,
            config.latency_iterations,
        ),
        "tasks": per_task,
    }
    report_path = temporary / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.joinpath("TRAINING_COMPLETE").write_text(
        "maniskill_low_rank_demo_transport_v1\n",
        encoding="utf-8",
    )
    temporary.rename(output_root)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--held-out-task")
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
        held_out_task=arguments.held_out_task,
    )


if __name__ == "__main__":
    main()

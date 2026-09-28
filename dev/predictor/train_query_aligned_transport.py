"""在 RLBench task-heldout folds 上训练并审计 QA-LRDT。"""

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
from torch.utils.data import DataLoader

from dev.pointnet.compare_retrievers import (
    ComparisonConfig,
    _read_jsonl,
    _sha256,
    encode_contexts,
    load_model as load_retriever,
)
from dev.pointnet.dataset import PointNetContextStore
from dev.pointnet.evaluate_end_to_end_retriever import (
    _base_candidate_mask,
    _load_text_scores,
    _text_candidate_masks,
)
from dev.predictor.action_chunk_data import ActionChunkStore
from dev.predictor.action_chunk_dataset import ActionChunkPairDataset
from dev.predictor.canonical_geometry import (
    geometry_statistics,
    load_canonical_geometries,
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
    POSE_SCALES,
    _condition_tensors,
    _load_action_tensors,
    _masked_sample_loss,
    _physical_metrics,
    _selection_indices,
    evaluate_predictors,
)
from dev.simulator.evaluate_maniskill_demo_prior import _bootstrap_comparison


MODEL_NAMES = ("fixed_low_rank_transport", "query_aligned_transport")


@dataclass(frozen=True)
class TrainConfig:
    """三个 fold 共用、结果揭盲前冻结的配置。"""

    schema_version: str
    seed: int
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
    retriever_batch_size: int
    retriever_num_workers: int
    text_group_budget: int
    text_score_weight: float
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
            self.retriever_batch_size,
            self.text_group_budget,
            self.bootstrap_resamples,
            self.latency_iterations,
        )
        if not self.schema_version.strip() or min(positive) <= 0:
            raise ValueError("schema_version 与训练正数参数必须有效")
        if self.weight_decay < 0 or self.retriever_num_workers < 0:
            raise ValueError("weight decay/worker 不能为负")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim 必须能被 num_heads 整除")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout 必须位于 [0,1)")
        if not 0.0 <= self.text_score_weight <= 1.0:
            raise ValueError("text_score_weight 必须位于 [0,1]")


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


def _filter_pair_rows(
    pair_rows: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    source_tasks: set[str],
) -> list[Mapping[str, Any]]:
    task_by_id = {str(row["chunk_id"]): str(row["task"]) for row in records}
    filtered = [
        row
        for row in pair_rows
        if task_by_id.get(str(row["query_id"])) in source_tasks
    ]
    if not filtered:
        raise ValueError("source tasks 没有训练 pair rows")
    return filtered


def _build_model(
    name: str,
    *,
    horizon: int,
    geometry_mean: torch.Tensor,
    geometry_std: torch.Tensor,
    config: TrainConfig,
) -> nn.Module:
    shared = {
        "horizon": horizon,
        "rank": config.rank,
        "hidden_dim": config.hidden_dim,
        "num_layers": config.num_layers,
        "num_heads": config.num_heads,
        "feedforward_dim": config.feedforward_dim,
        "dropout": config.dropout,
        "residual_limit": config.residual_limit,
        "transported_action_dim": 3,
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


def _train_model(
    *,
    name: str,
    dataset: ActionChunkPairDataset,
    geometry_mean: torch.Tensor,
    geometry_std: torch.Tensor,
    config: TrainConfig,
    device: torch.device,
    log_path: Path,
) -> nn.Module:
    _seed_everything(config.seed)
    model = _build_model(
        name,
        horizon=int(dataset.actions.shape[1]),
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
        dataset.set_epoch(epoch)
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
        for batch in loader:
            values = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                values["query_context"],
                values["demo_context"],
                values["demo_actions"],
                values["demo_mask"],
            )
            loss = _masked_sample_loss(
                prediction,
                values["target_actions"],
                values["target_mask"],
            ).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            batch_size = len(values["target_actions"])
            total += float(loss.detach()) * batch_size
            examples += batch_size
        scheduler.step()
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == config.epochs:
            record = {
                "model": name,
                "epoch": epoch + 1,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "action_loss": total / examples,
            }
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
    return model.eval()


def _subset_selections(
    *,
    query_indices: Sequence[int],
    selections: Mapping[str, Sequence[tuple[int, int]]],
    records: Sequence[Mapping[str, Any]],
    tasks: set[str],
) -> tuple[list[int], dict[str, list[tuple[int, int]]]]:
    positions = [
        position
        for position, query_index in enumerate(query_indices)
        if str(records[query_index]["task"]) in tasks
    ]
    if not positions:
        raise ValueError(f"受控 eval subset 没有 tasks={sorted(tasks)}")
    return (
        [query_indices[position] for position in positions],
        {
            name: [values[position] for position in positions]
            for name, values in selections.items()
        },
    )


def _primary_selection_indices(
    *,
    records: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    action_masks: torch.Tensor,
    retrieval_scores: torch.Tensor,
    text_mask: torch.Tensor,
) -> tuple[list[int], dict[str, list[tuple[int, int]]]]:
    """选择主指标 query，不依赖 hard-negative 完整性。"""
    id_to_index = {
        str(record["chunk_id"]): index
        for index, record in enumerate(records)
    }
    pairs = {str(row["query_id"]): row for row in pair_rows}
    full = action_masks.all(dim=1)
    eligible: list[int] = []
    selections = {"oracle": [], "retrieved": []}
    for query_index, record in enumerate(records):
        if not bool(full[query_index]):
            continue
        pair = pairs.get(str(record["chunk_id"]))
        if pair is None:
            continue
        positives = [
            id_to_index[identifier]
            for identifier in pair["positive_ids"]
            if identifier in id_to_index
            and bool(full[id_to_index[identifier]])
        ]
        if not positives:
            continue
        candidate_mask = (
            _base_candidate_mask(records, query_index)
            & text_mask[query_index]
            & full
        )
        ranked = torch.argsort(
            retrieval_scores[query_index].masked_fill(
                ~candidate_mask,
                float("-inf"),
            ),
            descending=True,
            stable=True,
        )
        ranked = [
            index for index in ranked.tolist() if bool(candidate_mask[index])
        ]
        if not ranked:
            continue
        eligible.append(query_index)
        selections["oracle"].append((positives[0], positives[0]))
        selections["retrieved"].append((ranked[0], ranked[0]))
    if not eligible:
        raise ValueError("没有满足主指标定义的 validation queries")
    return eligible, selections


@torch.inference_mode()
def _evaluate_primary(
    *,
    models: Mapping[str, nn.Module],
    geometries: torch.Tensor,
    actions: torch.Tensor,
    action_masks: torch.Tensor,
    query_indices: Sequence[int],
    selections: Mapping[str, Sequence[tuple[int, int]]],
    device: torch.device,
) -> dict[str, Any]:
    """只评估预注册主对照，hard negatives 由独立子集审计。"""
    queries = geometries[list(query_indices)]
    targets = actions[list(query_indices)]
    target_masks = action_masks[list(query_indices)]
    all_models: dict[str, nn.Module | None] = {
        "demo_action_copy": None,
        **models,
    }
    report: dict[str, Any] = {}
    for name, model in all_models.items():
        report[name] = {}
        for condition in ("retrieved", "oracle"):
            demo_geometry, demo_actions, demo_mask = _condition_tensors(
                selections[condition],
                geometries,
                actions,
                action_masks,
            )
            prediction = (
                demo_actions * demo_mask.float().unsqueeze(-1)
                if model is None
                else model(
                    queries.to(device),
                    demo_geometry.to(device),
                    demo_actions.to(device),
                    demo_mask.to(device),
                ).cpu()
            )
            report[name][condition] = {
                **_physical_metrics(prediction, targets, target_masks),
                "valid_demo_step_fraction": float(demo_mask.float().mean()),
            }
    return report


@torch.inference_mode()
def _bootstrap_models(
    *,
    models: Mapping[str, nn.Module],
    geometries: torch.Tensor,
    actions: torch.Tensor,
    action_masks: torch.Tensor,
    records: Sequence[Mapping[str, Any]],
    query_indices: Sequence[int],
    selections: Mapping[str, Sequence[tuple[int, int]]],
    config: TrainConfig,
    device: torch.device,
) -> dict[str, Any]:
    queries = geometries[list(query_indices)]
    targets = actions[list(query_indices)]
    target_masks = action_masks[list(query_indices)]
    groups = [
        (
            f"{records[index]['task']}:"
            f"{records[index]['variation']}:"
            f"{records[index]['episode']}"
        )
        for index in query_indices
    ]
    options = {
        "pose_scales": POSE_SCALES,
        "translation_threshold_m": 0.05,
        "rotation_threshold_rad": 0.25,
    }
    report = {}
    for condition_offset, condition in enumerate(("retrieved", "oracle")):
        demo_geometry, demo_actions, demo_mask = _condition_tensors(
            selections[condition],
            geometries,
            actions,
            action_masks,
        )
        copy = demo_actions * demo_mask.float().unsqueeze(-1)
        predictions = {
            name: model(
                queries.to(device),
                demo_geometry.to(device),
                demo_actions.to(device),
                demo_mask.to(device),
            ).cpu()
            for name, model in models.items()
        }
        comparisons = {}
        for model_offset, (name, prediction) in enumerate(predictions.items()):
            comparisons[f"{name}_minus_copy"] = _bootstrap_comparison(
                reference=copy,
                candidate=prediction,
                target=targets,
                mask=target_masks,
                group_ids=groups,
                seed=config.seed + condition_offset * 100 + model_offset * 10,
                resamples=config.bootstrap_resamples,
                **options,
            )
        comparisons["query_aligned_minus_fixed"] = _bootstrap_comparison(
            reference=predictions["fixed_low_rank_transport"],
            candidate=predictions["query_aligned_transport"],
            target=targets,
            mask=target_masks,
            group_ids=groups,
            seed=config.seed + condition_offset * 100 + 50,
            resamples=config.bootstrap_resamples,
            **options,
        )
        report[condition] = comparisons
    return report


@torch.inference_mode()
def _write_prediction_artifact(
    *,
    output_path: Path,
    fold_name: str,
    held_out_tasks: set[str],
    models: Mapping[str, nn.Module],
    geometries: torch.Tensor,
    actions: torch.Tensor,
    action_masks: torch.Tensor,
    records: Sequence[Mapping[str, Any]],
    query_indices: Sequence[int],
    selections: Mapping[str, Sequence[tuple[int, int]]],
    device: torch.device,
) -> dict[str, Any]:
    """保存跨 fold 统计所需的最小逐 query 预测。"""
    index_tensor = torch.as_tensor(query_indices, dtype=torch.long)
    queries = geometries[index_tensor]
    targets = actions[index_tensor]
    target_masks = action_masks[index_tensor]
    payload: dict[str, np.ndarray] = {
        "schema_version": np.asarray([1], dtype=np.int64),
        "fold_name": np.asarray([fold_name]),
        "held_out_tasks": np.asarray(sorted(held_out_tasks)),
        "query_indices": index_tensor.numpy(),
        "chunk_ids": np.asarray(
            [str(records[index]["chunk_id"]) for index in query_indices]
        ),
        "tasks": np.asarray(
            [str(records[index]["task"]) for index in query_indices]
        ),
        "group_ids": np.asarray(
            [
                (
                    f"{records[index]['task']}:"
                    f"{records[index]['variation']}:"
                    f"{records[index]['episode']}"
                )
                for index in query_indices
            ]
        ),
        "target_actions": targets.numpy(),
        "target_masks": target_masks.numpy(),
    }
    condition_metrics: dict[str, Any] = {}
    for condition in ("retrieved", "oracle"):
        demo_geometry, demo_actions, demo_mask = _condition_tensors(
            selections[condition],
            geometries,
            actions,
            action_masks,
        )
        predictions = {
            "demo_action_copy": demo_actions * demo_mask.float().unsqueeze(-1),
            **{
                name: model(
                    queries.to(device),
                    demo_geometry.to(device),
                    demo_actions.to(device),
                    demo_mask.to(device),
                ).cpu()
                for name, model in models.items()
            },
        }
        condition_metrics[condition] = {}
        for name, prediction in predictions.items():
            payload[f"{condition}__{name}"] = prediction.numpy()
            condition_metrics[condition][name] = _physical_metrics(
                prediction,
                targets,
                target_masks,
            )

    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    os.replace(temporary, output_path)
    return {
        "schema_version": 1,
        "sha256": _sha256(output_path),
        "query_count": len(query_indices),
        "conditions": ["retrieved", "oracle"],
        "metrics": condition_metrics,
    }


@torch.inference_mode()
def _structural_audit(
    *,
    models: Mapping[str, nn.Module],
    geometries: torch.Tensor,
    actions: torch.Tensor,
    action_masks: torch.Tensor,
    query_indices: Sequence[int],
    selections: Mapping[str, Sequence[tuple[int, int]]],
    device: torch.device,
) -> dict[str, dict[str, float | bool]]:
    sample_count = min(len(query_indices), 64)
    positions = list(range(sample_count))
    demos, demo_actions, demo_mask = _condition_tensors(
        [selections["retrieved"][position] for position in positions],
        geometries,
        actions,
        action_masks,
    )
    no_mask = torch.zeros_like(demo_mask)
    report = {}
    for name, model in models.items():
        identity = model(
            demos.to(device),
            demos.to(device),
            demo_actions.to(device),
            demo_mask.to(device),
        ).cpu()
        no_demo = model(
            geometries[list(query_indices[:sample_count])].to(device),
            demos.to(device),
            demo_actions.to(device),
            no_mask.to(device),
        ).cpu()
        regular = model(
            geometries[list(query_indices[:sample_count])].to(device),
            demos.to(device),
            demo_actions.to(device),
            demo_mask.to(device),
        ).cpu()
        report[name] = {
            "identity_max_abs_error": float((identity - demo_actions).abs().max()),
            "no_demo_max_abs_output": float(no_demo.abs().max()),
            "rotation_gripper_max_abs_error": float(
                (regular[..., 3:] - demo_actions[..., 3:]).abs().max()
            ),
            "query_only_action_head": False,
        }
    return report


@torch.inference_mode()
def _latency(
    *,
    model: nn.Module,
    geometry: torch.Tensor,
    actions: torch.Tensor,
    mask: torch.Tensor,
    iterations: int,
    device: torch.device,
) -> dict[str, float | int]:
    query = geometry[:1].to(device)
    demo = geometry[1:2].to(device)
    demo_actions = actions[1:2].to(device)
    demo_mask = mask[1:2].to(device)
    for _ in range(20):
        model(query, demo, demo_actions, demo_mask)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        model(query, demo, demo_actions, demo_mask)
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
    context_root: Path,
    action_root: Path,
    retriever_checkpoint: Path,
    text_scores_path: Path,
    output_root: Path,
    config_path: Path,
    config: TrainConfig,
    fold_name: str,
    held_out_tasks: set[str],
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
    context_hash = _sha256(context_root / "summary.json")
    action_hash = _sha256(action_root / "summary.json")
    retriever_payload = torch.load(retriever_checkpoint, map_location="cpu")
    if retriever_payload.get("dataset_summary_sha256") != context_hash:
        raise ValueError("Retriever checkpoint 与 context 数据不匹配")

    context_stores = {
        split: PointNetContextStore(context_root, split, cache_size=4)
        for split in ("train", "val")
    }
    action_stores = {
        split: ActionChunkStore(action_root, split, cache_size=4)
        for split in ("train", "val")
    }
    task_sets = {
        split: {str(row["task"]) for row in context_stores[split].records}
        for split in context_stores
    }
    if not held_out_tasks or not held_out_tasks.issubset(task_sets["train"]):
        raise ValueError("held-out tasks 必须是非空 train task 子集")
    if not held_out_tasks.issubset(task_sets["val"]):
        raise ValueError("held-out tasks 未被 validation 完整覆盖")
    source_tasks = task_sets["train"] - held_out_tasks

    geometries = {
        split: load_canonical_geometries(context_stores[split])
        for split in context_stores
    }
    source_mask = torch.tensor(
        [
            str(row["task"]) in source_tasks
            for row in context_stores["train"].records
        ],
        dtype=torch.bool,
    )
    geometry_mean, geometry_std = geometry_statistics(
        geometries["train"][source_mask]
    )
    loaded = {
        split: _load_action_tensors(context_stores[split], action_stores[split])
        for split in context_stores
    }
    actions = {split: loaded[split][0] for split in loaded}
    action_masks = {split: loaded[split][1] for split in loaded}
    pair_rows = {
        split: _read_jsonl(context_root / f"pairs-{split}.jsonl")
        for split in context_stores
    }
    source_pair_rows = _filter_pair_rows(
        pair_rows["train"],
        context_stores["train"].records,
        source_tasks,
    )
    train_dataset = ActionChunkPairDataset(
        embeddings=geometries["train"],
        actions=actions["train"],
        action_masks=action_masks["train"],
        records=context_stores["train"].records,
        pair_rows=source_pair_rows,
        seed=config.seed,
    )
    models = {
        name: _train_model(
            name=name,
            dataset=train_dataset,
            geometry_mean=geometry_mean,
            geometry_std=geometry_std,
            config=config,
            device=device,
            log_path=temporary / "training_metrics.jsonl",
        )
        for name in MODEL_NAMES
    }
    checkpoint_hashes = {}
    for name, model in models.items():
        checkpoint = {
            "model": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
            "model_name": name,
            "model_config": asdict(model.config),
            "train_config": asdict(config),
            "fold_name": fold_name,
            "held_out_tasks": sorted(held_out_tasks),
            "source_tasks": sorted(source_tasks),
            "geometry_mean": geometry_mean,
            "geometry_std": geometry_std,
            "context_summary_sha256": context_hash,
            "action_summary_sha256": action_hash,
            "git_commit": _git_commit(project_root),
        }
        checkpoint_path = temporary / f"{name}.pt"
        torch.save(checkpoint, checkpoint_path.with_suffix(".pt.tmp"))
        os.replace(checkpoint_path.with_suffix(".pt.tmp"), checkpoint_path)
        checkpoint_hashes[name] = _sha256(checkpoint_path)

    retriever = load_retriever(retriever_checkpoint, device)
    val_embeddings = encode_contexts(
        retriever,
        context_stores["val"],
        ComparisonConfig(
            batch_size=config.retriever_batch_size,
            num_workers=config.retriever_num_workers,
        ),
        device,
    )
    text_scores, chunk_groups, text_metadata = _load_text_scores(
        text_scores_path,
        context_stores["val"].records,
    )
    if text_metadata.get("manifest_sha256") != _sha256(
        context_root / "manifest-val.jsonl"
    ):
        raise ValueError("文本分数与 val manifest 不匹配")
    text_mask = _text_candidate_masks(
        text_scores,
        chunk_groups,
        (config.text_group_budget,),
    )[config.text_group_budget]
    pointnet_scores = (val_embeddings @ val_embeddings.T + 1.0) / 2.0
    normalized_text = ((text_scores + 1.0) / 2.0).clamp(0.0, 1.0)
    retrieval_scores = (
        config.text_score_weight * normalized_text
        + (1.0 - config.text_score_weight) * pointnet_scores
    )
    primary_query_indices, primary_selections = _primary_selection_indices(
        records=context_stores["val"].records,
        pair_rows=pair_rows["val"],
        action_masks=action_masks["val"],
        retrieval_scores=retrieval_scores,
        text_mask=text_mask,
    )
    control_query_indices, control_selections = _selection_indices(
        records=context_stores["val"].records,
        pair_rows=pair_rows["val"],
        action_masks=action_masks["val"],
        retrieval_scores=retrieval_scores,
        text_mask=text_mask,
        seed=config.seed,
    )
    held_query_indices, held_selections = _subset_selections(
        query_indices=primary_query_indices,
        selections=primary_selections,
        records=context_stores["val"].records,
        tasks=held_out_tasks,
    )
    source_query_indices, source_selections = _subset_selections(
        query_indices=primary_query_indices,
        selections=primary_selections,
        records=context_stores["val"].records,
        tasks=source_tasks,
    )
    observed_held_tasks = {
        str(context_stores["val"].records[index]["task"])
        for index in held_query_indices
    }
    if observed_held_tasks != held_out_tasks:
        missing = sorted(held_out_tasks - observed_held_tasks)
        raise ValueError(f"主指标未覆盖全部 held-out tasks：{missing}")
    held_control_indices, held_control_selections = _subset_selections(
        query_indices=control_query_indices,
        selections=control_selections,
        records=context_stores["val"].records,
        tasks=held_out_tasks,
    )
    source_control_indices, source_control_selections = _subset_selections(
        query_indices=control_query_indices,
        selections=control_selections,
        records=context_stores["val"].records,
        tasks=source_tasks,
    )
    held_metrics = _evaluate_primary(
        models=models,
        geometries=geometries["val"],
        actions=actions["val"],
        action_masks=action_masks["val"],
        query_indices=held_query_indices,
        selections=held_selections,
        device=device,
    )
    source_metrics = _evaluate_primary(
        models=models,
        geometries=geometries["val"],
        actions=actions["val"],
        action_masks=action_masks["val"],
        query_indices=source_query_indices,
        selections=source_selections,
        device=device,
    )
    controlled_sensitivity = {
        "held_out": evaluate_predictors(
            models=models,
            embeddings=geometries["val"],
            actions=actions["val"],
            action_masks=action_masks["val"],
            query_indices=held_control_indices,
            selections=held_control_selections,
            device=device,
        ),
        "source": evaluate_predictors(
            models=models,
            embeddings=geometries["val"],
            actions=actions["val"],
            action_masks=action_masks["val"],
            query_indices=source_control_indices,
            selections=source_control_selections,
            device=device,
        ),
    }
    per_task = {}
    for task in sorted(held_out_tasks):
        task_indices, task_selections = _subset_selections(
            query_indices=held_query_indices,
            selections=held_selections,
            records=context_stores["val"].records,
            tasks={task},
        )
        per_task[task] = {
            "eval_queries": len(task_indices),
            "metrics": _evaluate_primary(
                models=models,
                geometries=geometries["val"],
                actions=actions["val"],
                action_masks=action_masks["val"],
                query_indices=task_indices,
                selections=task_selections,
                device=device,
            ),
        }
    prediction_artifact = _write_prediction_artifact(
        output_path=temporary / "held_out_predictions.npz",
        fold_name=fold_name,
        held_out_tasks=held_out_tasks,
        models=models,
        geometries=geometries["val"],
        actions=actions["val"],
        action_masks=action_masks["val"],
        records=context_stores["val"].records,
        query_indices=held_query_indices,
        selections=held_selections,
        device=device,
    )
    report = {
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "split": "task-heldout model training; held-out task Demo at inference",
            "retrieved_demo": "frozen text top-10 + PointNet reranking",
            "models": "translation-only fixed LR-DAT vs query-aligned LR-DAT",
            "checkpoint_selection": "fixed final epoch; validation unused",
            "bootstrap_unit": "task + variation + episode",
            "primary_eligibility": (
                "full action horizon + positive Demo + retrievable candidate"
            ),
            "hard_negative_eligibility": (
                "separate diagnostic subset requiring every control type"
            ),
        },
        "fold_name": fold_name,
        "held_out_tasks": sorted(held_out_tasks),
        "source_tasks": sorted(source_tasks),
        "device": str(device),
        "cuda_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "config": asdict(config),
        "config_sha256": _sha256(config_path),
        "git_commit": _git_commit(project_root),
        "context_summary_sha256": context_hash,
        "action_summary_sha256": action_hash,
        "retriever_checkpoint_sha256": _sha256(retriever_checkpoint),
        "text_scores_sha256": _sha256(text_scores_path),
        "train_pairs": len(train_dataset),
        "held_out_eval_queries": len(held_query_indices),
        "source_eval_queries": len(source_query_indices),
        "held_out_control_queries": len(held_control_indices),
        "source_control_queries": len(source_control_indices),
        "model_parameters": {
            name: sum(parameter.numel() for parameter in model.parameters())
            for name, model in models.items()
        },
        "checkpoint_sha256": checkpoint_hashes,
        "held_out_prediction_artifact": prediction_artifact,
        "geometry_mean": geometry_mean.tolist(),
        "geometry_std": geometry_std.tolist(),
        "structural_audit": _structural_audit(
            models=models,
            geometries=geometries["val"],
            actions=actions["val"],
            action_masks=action_masks["val"],
            query_indices=held_query_indices,
            selections=held_selections,
            device=device,
        ),
        "latency": {
            name: _latency(
                model=model,
                geometry=geometries["val"],
                actions=actions["val"],
                mask=action_masks["val"],
                iterations=config.latency_iterations,
                device=device,
            )
            for name, model in models.items()
        },
        "held_out_metrics": held_metrics,
        "source_metrics": source_metrics,
        "controlled_sensitivity": controlled_sensitivity,
        "held_out_paired_bootstrap": _bootstrap_models(
            models=models,
            geometries=geometries["val"],
            actions=actions["val"],
            action_masks=action_masks["val"],
            records=context_stores["val"].records,
            query_indices=held_query_indices,
            selections=held_selections,
            config=config,
            device=device,
        ),
        "held_out_tasks_metrics": per_task,
    }
    report_path = temporary / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.joinpath("TRAINING_COMPLETE").write_text(
        "rlbench_query_aligned_transport_task_heldout_v2\n",
        encoding="utf-8",
    )
    temporary.rename(output_root)
    atexit.unregister(cleanup_incomplete)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--context-root", type=Path, required=True)
    parser.add_argument("--action-root", type=Path, required=True)
    parser.add_argument("--retriever-checkpoint", type=Path, required=True)
    parser.add_argument("--text-scores", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold-name", required=True)
    parser.add_argument("--held-out-task", action="append", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA，但当前节点没有可用 GPU")
    run(
        project_root=arguments.project_root.resolve(),
        context_root=arguments.context_root.resolve(),
        action_root=arguments.action_root.resolve(),
        retriever_checkpoint=arguments.retriever_checkpoint.resolve(),
        text_scores_path=arguments.text_scores.resolve(),
        output_root=arguments.output_root.resolve(),
        config_path=arguments.config.resolve(),
        config=TrainConfig.from_json(arguments.config.resolve()),
        fold_name=arguments.fold_name,
        held_out_tasks=set(arguments.held_out_task),
        device=device,
    )


if __name__ == "__main__":
    main()

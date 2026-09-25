"""训练 Tiny PointNet++ 几何 Siamese Retriever，并评估 Recall@K。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dev.pointnet.dataset import (
    PointNetContextDataset,
    PointNetContextStore,
    PointNetPairDataset,
)
from dev.pointnet.retriever_model import GeometricSiameseRetriever


FUTURE_SCALES = torch.tensor(
    [0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.08, 0.1, 0.1, 0.1],
    dtype=torch.float32,
)


@dataclass(frozen=True)
class TrainConfig:
    seed: int
    epochs: int
    batch_size: int
    num_workers: int
    shard_cache_size: int
    learning_rate: float
    weight_decay: float
    triplet_margin: float
    alignment_weight: float
    phase_weight: float
    future_weight: float
    gradient_clip_norm: float
    point_jitter_std_m: float
    point_jitter_clip_m: float
    point_dropout_probability: float
    amp: bool
    eval_batch_size: int
    recall_k: tuple[int, ...]

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainConfig":
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        values["recall_k"] = tuple(values["recall_k"])
        return cls(**values)

    def __post_init__(self) -> None:
        positive_integers = (
            self.epochs,
            self.batch_size,
            self.shard_cache_size,
            self.eval_batch_size,
            *self.recall_k,
        )
        if min(positive_integers) <= 0 or self.num_workers < 0:
            raise ValueError("epoch/batch/cache/Recall@K 必须为正，worker 不能为负")
        if min(
            self.learning_rate,
            self.triplet_margin,
            self.gradient_clip_norm,
        ) <= 0:
            raise ValueError("learning rate、margin 和 gradient clip 必须为正")
        if not 0.0 <= self.point_dropout_probability < 1.0:
            raise ValueError("point dropout 必须位于 [0, 1)")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit(root: Path) -> str:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return process.stdout.strip() if process.returncode == 0 else "unknown"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _move_context(
    context: Mapping[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in context.items()
    }


def _concatenate_contexts(
    contexts: Sequence[Mapping[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    keys = ("active_points", "target_points", "state", "target_valid")
    return {key: torch.cat([context[key] for context in contexts]) for key in keys}


def _augment_points(
    points: torch.Tensor,
    *,
    jitter_std: float,
    jitter_clip: float,
    dropout_probability: float,
    valid_samples: torch.Tensor | None = None,
) -> torch.Tensor:
    augmented = points.clone()
    noise = torch.randn_like(augmented).mul_(jitter_std).clamp_(
        -jitter_clip, jitter_clip
    )
    if valid_samples is not None:
        noise = noise * valid_samples[:, None, None].to(noise.dtype)
    augmented.add_(noise)
    if dropout_probability > 0:
        dropped = torch.rand(
            augmented.shape[:2], device=augmented.device
        ) < dropout_probability
        if valid_samples is not None:
            dropped &= valid_samples[:, None]
        replacement = augmented[:, :1].expand_as(augmented)
        augmented = torch.where(dropped[..., None], replacement, augmented)
    return augmented


def _augment_context(
    context: Mapping[str, torch.Tensor],
    config: TrainConfig,
) -> dict[str, torch.Tensor]:
    result = dict(context)
    result["active_points"] = _augment_points(
        context["active_points"],
        jitter_std=config.point_jitter_std_m,
        jitter_clip=config.point_jitter_clip_m,
        dropout_probability=config.point_dropout_probability,
    )
    result["target_points"] = _augment_points(
        context["target_points"],
        jitter_std=config.point_jitter_std_m,
        jitter_clip=config.point_jitter_clip_m,
        dropout_probability=config.point_dropout_probability,
        valid_samples=context["target_valid"].bool(),
    )
    return result


def _losses(
    outputs: Mapping[str, torch.Tensor],
    anchor: Mapping[str, torch.Tensor],
    batch_size: int,
    config: TrainConfig,
) -> dict[str, torch.Tensor]:
    embeddings = outputs["embedding"]
    anchor_embedding, positive_embedding, negative_embedding = embeddings.split(
        batch_size
    )
    positive_similarity = (anchor_embedding * positive_embedding).sum(dim=-1)
    negative_similarity = (anchor_embedding * negative_embedding).sum(dim=-1)
    triplet = torch.relu(
        config.triplet_margin + negative_similarity - positive_similarity
    ).mean()
    alignment = (1.0 - positive_similarity).mean()
    phase = nn.functional.cross_entropy(
        outputs["phase_logits"][:batch_size],
        anchor["phase_id"],
    )
    future_prediction = outputs["future_prediction"][:batch_size]
    scales = FUTURE_SCALES.to(future_prediction.device)
    future_target = anchor["future_target"] / scales
    element_loss = nn.functional.smooth_l1_loss(
        future_prediction,
        future_target,
        reduction="none",
    )
    future_mask = torch.ones_like(element_loss)
    future_mask[:, 7:] = anchor["effect_valid"].float().reshape(-1, 1)
    future = (element_loss * future_mask).sum() / future_mask.sum().clamp_min(1.0)
    total = (
        triplet
        + config.alignment_weight * alignment
        + config.phase_weight * phase
        + config.future_weight * future
    )
    return {
        "total": total,
        "triplet": triplet,
        "alignment": alignment,
        "phase": phase,
        "future": future,
        "positive_similarity": positive_similarity.mean(),
        "negative_similarity": negative_similarity.mean(),
        "triplet_accuracy": (
            positive_similarity > negative_similarity
        ).float().mean(),
    }


def _mean_metrics(sums: dict[str, float], examples: int) -> dict[str, float]:
    return {name: value / max(examples, 1) for name, value in sums.items()}


def run_pair_epoch(
    *,
    model: GeometricSiameseRetriever,
    loader: DataLoader,
    config: TrainConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    sums: dict[str, float] = {}
    examples = 0
    for batch in loader:
        anchor = _move_context(batch["anchor"], device)
        positive = _move_context(batch["positive"], device)
        negative = _move_context(batch["negative"], device)
        batch_size = anchor["state"].shape[0]
        if training:
            anchor = _augment_context(anchor, config)
            positive = _augment_context(positive, config)
            negative = _augment_context(negative, config)
            optimizer.zero_grad(set_to_none=True)
        joined = _concatenate_contexts((anchor, positive, negative))
        with torch.set_grad_enabled(training):
            with torch.cuda.amp.autocast(
                enabled=config.amp and device.type == "cuda"
            ):
                outputs = model(joined)
                losses = _losses(outputs, anchor, batch_size, config)
            if training:
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    model.parameters(), config.gradient_clip_norm
                )
                scaler.step(optimizer)
                scaler.update()
        for name, value in losses.items():
            sums[name] = sums.get(name, 0.0) + float(value.detach()) * batch_size
        examples += batch_size
    return _mean_metrics(sums, examples)


@torch.no_grad()
def evaluate_recall(
    *,
    model: GeometricSiameseRetriever,
    store: PointNetContextStore,
    config: TrainConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    loader = DataLoader(
        PointNetContextDataset(store),
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    embeddings = torch.empty(
        (len(store.records), model.config.embedding_dim), dtype=torch.float32
    )
    for batch in loader:
        indices = batch["index"].long()
        context = _move_context(batch["context"], device)
        outputs = model(context)
        embeddings[indices] = outputs["embedding"].float().cpu()

    identifiers = [row["chunk_id"] for row in store.records]
    id_to_index = {identifier: index for index, identifier in enumerate(identifiers)}
    pair_rows = {
        row["query_id"]: row
        for row in _read_pair_rows(store.root / f"pairs-{store.split}.jsonl")
    }
    similarity = embeddings @ embeddings.T
    metrics: dict[str, float] = {}
    for pool_name in ("global", "same_task"):
        recalls = {value: 0 for value in config.recall_k}
        reciprocal_rank = 0.0
        queries = 0
        for query_index, record in enumerate(store.records):
            pair = pair_rows[record["chunk_id"]]
            positive_indices = {
                id_to_index[identifier]
                for identifier in pair["positive_ids"]
                if identifier in id_to_index
            }
            if not positive_indices:
                continue
            candidate_indices = []
            for index, candidate in enumerate(store.records):
                same_episode = (
                    candidate["task"] == record["task"]
                    and candidate["episode"] == record["episode"]
                )
                if index == query_index or same_episode:
                    continue
                if pool_name == "same_task" and candidate["task"] != record["task"]:
                    continue
                candidate_indices.append(index)
            ranked = sorted(
                candidate_indices,
                key=lambda index: float(similarity[query_index, index]),
                reverse=True,
            )
            positive_ranks = [
                rank
                for rank, index in enumerate(ranked, start=1)
                if index in positive_indices
            ]
            if not positive_ranks:
                continue
            best_rank = min(positive_ranks)
            for value in config.recall_k:
                recalls[value] += int(best_rank <= value)
            reciprocal_rank += 1.0 / best_rank
            queries += 1
        for value, hits in recalls.items():
            metrics[f"{pool_name}_recall@{value}"] = hits / max(queries, 1)
        metrics[f"{pool_name}_mrr"] = reciprocal_rank / max(queries, 1)
        metrics[f"{pool_name}_queries"] = float(queries)
    return metrics


def _read_pair_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _save_checkpoint(
    path: Path,
    *,
    model: GeometricSiameseRetriever,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    metrics: Mapping[str, Any],
    train_config: TrainConfig,
    dataset_hash: str,
    git_commit: str,
) -> None:
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "model_config": asdict(model.config),
        "train_config": asdict(train_config),
        "metrics": dict(metrics),
        "dataset_summary_sha256": dataset_hash,
        "git_commit": git_commit,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train(
    *,
    project_root: Path,
    data_root: Path,
    output_root: Path,
    config: TrainConfig,
    device: torch.device,
) -> None:
    if output_root.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output_root}")
    if not data_root.joinpath("PREPROCESS_COMPLETE").is_file():
        raise FileNotFoundError("数据缺少 PREPROCESS_COMPLETE，拒绝训练未完成产物")
    output_root.mkdir(parents=True)
    _seed_everything(config.seed)

    train_dataset = PointNetPairDataset(
        data_root,
        "train",
        seed=config.seed,
        cache_size=config.shard_cache_size,
    )
    val_dataset = PointNetPairDataset(
        data_root,
        "val",
        seed=config.seed + 1,
        cache_size=config.shard_cache_size,
    )
    state_mean, state_std = train_dataset.store.state_statistics()
    model = GeometricSiameseRetriever(
        state_mean=torch.from_numpy(state_mean),
        state_std=torch.from_numpy(state_std),
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
    scaler = torch.cuda.amp.GradScaler(
        enabled=config.amp and device.type == "cuda"
    )
    metadata = {
        "device": str(device),
        "cuda_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "train_pairs": len(train_dataset),
        "val_pairs": len(val_dataset),
        "state_mean": state_mean.tolist(),
        "state_std": state_std.tolist(),
        "train_config": asdict(config),
    }
    output_root.joinpath("run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    dataset_hash = _sha256(data_root / "summary.json")
    git_commit = _git_commit(project_root)
    best_score = float("-inf")
    log_path = output_root / "metrics.jsonl"

    for epoch in range(config.epochs):
        train_dataset.set_epoch(epoch)
        val_dataset.set_epoch(0)
        generator = torch.Generator().manual_seed(config.seed + epoch)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
        )
        train_metrics = run_pair_epoch(
            model=model,
            loader=train_loader,
            config=config,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
        )
        val_metrics = run_pair_epoch(
            model=model,
            loader=val_loader,
            config=config,
            device=device,
            optimizer=None,
            scaler=scaler,
        )
        recall_metrics = evaluate_recall(
            model=model,
            store=val_dataset.store,
            config=config,
            device=device,
        )
        scheduler.step()
        record = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "val": val_metrics,
            "retrieval": recall_metrics,
        }
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)
        score = recall_metrics.get("same_task_recall@4", float("-inf"))
        if score > best_score:
            best_score = score
            _save_checkpoint(
                output_root / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                metrics=record,
                train_config=config,
                dataset_hash=dataset_hash,
                git_commit=git_commit,
            )
        _save_checkpoint(
            output_root / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch + 1,
            metrics=record,
            train_config=config,
            dataset_hash=dataset_hash,
            git_commit=git_commit,
        )

    output_root.joinpath("TRAINING_COMPLETE").write_text(
        f"best_same_task_recall@4={best_score:.8f}\n",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA，但当前节点没有可用 GPU")
    config = TrainConfig.from_json(args.config)
    train(
        project_root=args.project_root,
        data_root=args.data_root,
        output_root=args.output_root,
        config=config,
        device=device,
    )


if __name__ == "__main__":
    main()

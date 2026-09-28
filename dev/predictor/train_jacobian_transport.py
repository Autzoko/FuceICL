"""训练并评估 EEF-canonical Local Jacobian Action Transport。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import random
import subprocess
from typing import Any

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
    _load_text_scores,
    _text_candidate_masks,
)
from dev.predictor.action_chunk_data import ActionChunkStore
from dev.predictor.action_chunk_dataset import ActionChunkPairDataset
from dev.predictor.canonical_geometry import (
    FEATURE_NAMES,
    geometry_statistics,
    load_canonical_geometries,
)
from dev.predictor.jacobian_transport_model import (
    JacobianTransportConfig,
    LocalJacobianActionTransport,
)
from dev.predictor.train_action_chunks import (
    _load_action_tensors,
    _masked_sample_loss,
    _selection_indices,
    evaluate_predictors,
)


@dataclass(frozen=True)
class TrainConfig:
    """Jacobian transport pilot 的固定优化配置。"""

    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
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

    @classmethod
    def from_json(cls, path: Path) -> "TrainConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def __post_init__(self) -> None:
        positive = (
            self.epochs,
            self.batch_size,
            self.learning_rate,
            self.gradient_clip_norm,
            self.hidden_dim,
            self.num_layers,
            self.num_heads,
            self.feedforward_dim,
            self.retriever_batch_size,
            self.text_group_budget,
        )
        if min(positive) <= 0:
            raise ValueError("训练尺寸、学习率和模型容量必须为正")
        if self.weight_decay < 0 or self.retriever_num_workers < 0:
            raise ValueError("weight decay/worker 不能为负")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout 必须位于 [0, 1)")
        if not 0.0 <= self.text_score_weight <= 1.0:
            raise ValueError("text_score_weight 必须位于 [0, 1]")


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


def _model_config(
    config: TrainConfig,
    horizon: int,
) -> JacobianTransportConfig:
    return JacobianTransportConfig(
        horizon=horizon,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        feedforward_dim=config.feedforward_dim,
        dropout=config.dropout,
        residual_limit=config.residual_limit,
    )


def train_model(
    *,
    dataset: ActionChunkPairDataset,
    geometry_mean: torch.Tensor,
    geometry_std: torch.Tensor,
    config: TrainConfig,
    device: torch.device,
    log_path: Path,
) -> LocalJacobianActionTransport:
    """训练单一假设模型，不用 validation 选择 checkpoint。"""
    _seed_everything(config.seed)
    model = LocalJacobianActionTransport(
        _model_config(config, int(dataset.actions.shape[1])),
        geometry_mean=geometry_mean,
        geometry_std=geometry_std,
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
        model.train()
        total_loss = 0.0
        examples = 0
        for batch in loader:
            values = {key: value.to(device) for key, value in batch.items()}
            batch_size = len(values["target_actions"])
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
            total_loss += float(loss.detach()) * batch_size
            examples += batch_size
        scheduler.step()
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == config.epochs:
            record = {
                "epoch": epoch + 1,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "action": total_loss / examples,
            }
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
    return model.eval()


def run(
    *,
    project_root: Path,
    context_root: Path,
    action_root: Path,
    retriever_checkpoint: Path,
    text_scores_path: Path,
    output_root: Path,
    config: TrainConfig,
    device: torch.device,
) -> None:
    if output_root.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output_root}")
    temporary = output_root.with_name(f".{output_root.name}.incomplete-{os.getpid()}")
    temporary.mkdir(parents=True)
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
    geometries = {
        split: load_canonical_geometries(context_stores[split])
        for split in ("train", "val")
    }
    geometry_mean, geometry_std = geometry_statistics(geometries["train"])
    loaded = {
        split: _load_action_tensors(context_stores[split], action_stores[split])
        for split in ("train", "val")
    }
    actions = {split: loaded[split][0] for split in loaded}
    action_masks = {split: loaded[split][1] for split in loaded}
    pair_rows = {
        split: _read_jsonl(context_root / f"pairs-{split}.jsonl")
        for split in ("train", "val")
    }
    train_dataset = ActionChunkPairDataset(
        embeddings=geometries["train"],
        actions=actions["train"],
        action_masks=action_masks["train"],
        records=context_stores["train"].records,
        pair_rows=pair_rows["train"],
        seed=config.seed,
    )
    model = train_model(
        dataset=train_dataset,
        geometry_mean=geometry_mean,
        geometry_std=geometry_std,
        config=config,
        device=device,
        log_path=temporary / "training_metrics.jsonl",
    )
    checkpoint = {
        "model": model.state_dict(),
        "model_config": asdict(model.config),
        "train_config": asdict(config),
        "geometry_feature_names": FEATURE_NAMES,
        "context_summary_sha256": context_hash,
        "action_summary_sha256": action_hash,
        "retriever_checkpoint_sha256": _sha256(retriever_checkpoint),
        "git_commit": _git_commit(project_root),
    }
    checkpoint_path = temporary / "local_jacobian_transport.pt"
    torch.save(checkpoint, checkpoint_path.with_suffix(".pt.tmp"))
    os.replace(checkpoint_path.with_suffix(".pt.tmp"), checkpoint_path)

    retriever = load_retriever(retriever_checkpoint, device)
    embedding_config = ComparisonConfig(
        batch_size=config.retriever_batch_size,
        num_workers=config.retriever_num_workers,
    )
    val_embeddings = encode_contexts(
        retriever,
        context_stores["val"],
        embedding_config,
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
    query_indices, selections = _selection_indices(
        records=context_stores["val"].records,
        pair_rows=pair_rows["val"],
        action_masks=action_masks["val"],
        retrieval_scores=retrieval_scores,
        text_mask=text_mask,
        seed=config.seed,
    )
    metrics = evaluate_predictors(
        models={"local_jacobian_transport": model},
        embeddings=geometries["val"],
        actions=actions["val"],
        action_masks=action_masks["val"],
        query_indices=query_indices,
        selections=selections,
        device=device,
    )
    report: dict[str, Any] = {
        "protocol": {
            "split": "same 488-query controlled E3 subset",
            "hypothesis": "first-order transport from canonical geometry delta",
            "retrieved_demo": "text top-10 + PointNet reranking",
            "checkpoint_selection": "fixed final epoch",
        },
        "device": str(device),
        "cuda_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "config": asdict(config),
        "git_commit": _git_commit(project_root),
        "context_summary_sha256": context_hash,
        "action_summary_sha256": action_hash,
        "retriever_checkpoint_sha256": _sha256(retriever_checkpoint),
        "text_scores_sha256": _sha256(text_scores_path),
        "train_pairs": len(train_dataset),
        "eval_queries": len(query_indices),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "geometry_feature_names": FEATURE_NAMES,
        "geometry_mean": geometry_mean.tolist(),
        "geometry_std": geometry_std.tolist(),
        "metrics": metrics,
    }
    temporary.joinpath("report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.joinpath("TRAINING_COMPLETE").write_text(
        "local_jacobian_action_transport_pilot_v1\n",
        encoding="utf-8",
    )
    temporary.rename(output_root)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--context-root", type=Path, required=True)
    parser.add_argument("--action-root", type=Path, required=True)
    parser.add_argument("--retriever-checkpoint", type=Path, required=True)
    parser.add_argument("--text-scores", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA，但当前节点没有可用 GPU")
    run(
        project_root=args.project_root,
        context_root=args.context_root,
        action_root=args.action_root,
        retriever_checkpoint=args.retriever_checkpoint,
        text_scores_path=args.text_scores,
        output_root=args.output_root,
        config=TrainConfig.from_json(args.config),
        device=device,
    )


if __name__ == "__main__":
    main()

"""训练并评估冻结 Retriever/LJAT 上的轻量 Demo utility selector。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import subprocess
from typing import Any

import numpy as np
import torch
from torch import nn

from dev.pointnet.compare_retrievers import (
    ComparisonConfig,
    _sha256,
    encode_contexts,
    load_model as load_retriever,
)
from dev.pointnet.dataset import PointNetContextStore
from dev.pointnet.evaluate_end_to_end_retriever import (
    _load_text_scores,
    _text_candidate_masks,
)
from dev.pointnet.evaluate_selective_retrieval import _episode_key
from dev.predictor.action_chunk_data import ActionChunkStore
from dev.predictor.canonical_geometry import load_canonical_geometries
from dev.predictor.demo_utility_selector import (
    DemoUtilitySelector,
    UtilitySelectorConfig,
)
from dev.predictor.evaluate_demo_action_consensus import (
    ConsensusConfig,
    _aggregate_hypotheses,
    _paired_bootstrap_report,
    _predict_hypotheses,
    _rank_full_horizon_demos,
)
from dev.predictor.evaluate_jacobian_significance import _load_transport
from dev.predictor.train_action_chunks import (
    _load_action_tensors,
    _physical_metrics,
)
from dev.predictor.train_jacobian_transport import TrainConfig


@dataclass(frozen=True)
class UtilityTrainConfig:
    """预注册的固定训练配置。"""

    seed: int = 20260928
    candidates: int = 4
    hidden_dim: int = 64
    epochs: int = 80
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    bootstrap_resamples: int = 5000

    def __post_init__(self) -> None:
        positive = (
            self.candidates,
            self.hidden_dim,
            self.epochs,
            self.batch_size,
            self.learning_rate,
            self.gradient_clip_norm,
            self.bootstrap_resamples,
        )
        if min(positive) <= 0 or self.weight_decay < 0:
            raise ValueError("训练尺寸、学习率、梯度门槛必须为正")


@dataclass
class PreparedSplit:
    """已冻结候选和 LJAT hypotheses 的内存数据。"""

    features: torch.Tensor
    best_candidates: torch.Tensor
    candidate_errors: torch.Tensor
    hypotheses: torch.Tensor
    targets: torch.Tensor
    target_masks: torch.Tensor
    group_ids: list[str]
    query_count: int
    episode_count: int


def _git_commit(project_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
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


def _build_features(
    *,
    model: nn.Module,
    geometries: torch.Tensor,
    hypotheses: torch.Tensor,
    query_indices: list[int],
    candidate_indices: torch.Tensor,
    combined_scores: torch.Tensor,
    pointnet_scores: torch.Tensor,
    normalized_text: torch.Tensor,
) -> torch.Tensor:
    """只使用线上可获得的 query/Demo、action prior 和 Retriever 分数。"""
    query = geometries[query_indices]
    demo = geometries[candidate_indices]
    feature_mask = model.geometry_feature_mask.cpu()
    geometry_mean = model.geometry_mean.cpu()
    geometry_std = model.geometry_std.cpu()
    normalized_query = (query - geometry_mean) / geometry_std * feature_mask
    normalized_demo = (demo - geometry_mean) / geometry_std * feature_mask
    geometry_delta = normalized_query[:, None, :] - normalized_demo

    flattened_actions = hypotheses.flatten(start_dim=2)
    distances = torch.cdist(flattened_actions, flattened_actions, p=1)
    distances = distances / flattened_actions.shape[-1]
    centrality = distances.sum(dim=2, keepdim=True) / max(
        hypotheses.shape[1] - 1, 1
    )
    queries = torch.tensor(query_indices, dtype=torch.long)[:, None]
    score_features = torch.stack(
        (
            normalized_text[queries, candidate_indices],
            pointnet_scores[queries, candidate_indices],
            combined_scores[queries, candidate_indices],
        ),
        dim=-1,
    )
    return torch.cat(
        (
            normalized_query[:, None, :].expand_as(normalized_demo),
            geometry_delta,
            flattened_actions,
            score_features,
            centrality,
        ),
        dim=-1,
    )


@torch.inference_mode()
def _prepare_split(
    *,
    split: str,
    context_root: Path,
    action_root: Path,
    text_scores_path: Path,
    retriever: nn.Module,
    transport: nn.Module,
    train_config: TrainConfig,
    utility_config: UtilityTrainConfig,
    device: torch.device,
) -> PreparedSplit:
    store = PointNetContextStore(context_root, split, cache_size=4)
    action_store = ActionChunkStore(action_root, split, cache_size=4)
    geometries = load_canonical_geometries(store)
    actions, action_masks = _load_action_tensors(store, action_store)
    embeddings = encode_contexts(
        retriever,
        store,
        ComparisonConfig(
            batch_size=train_config.retriever_batch_size,
            num_workers=train_config.retriever_num_workers,
        ),
        device,
    )
    text_scores, chunk_groups, metadata = _load_text_scores(
        text_scores_path, store.records
    )
    if split == "val" and metadata.get("manifest_sha256") != _sha256(
        context_root / "manifest-val.jsonl"
    ):
        raise ValueError("文本分数与 val manifest 不匹配")
    text_mask = _text_candidate_masks(
        text_scores,
        chunk_groups,
        (train_config.text_group_budget,),
    )[train_config.text_group_budget]
    pointnet_scores = (embeddings @ embeddings.T + 1.0) / 2.0
    normalized_text = ((text_scores + 1.0) / 2.0).clamp(0.0, 1.0)
    combined_scores = (
        train_config.text_score_weight * normalized_text
        + (1.0 - train_config.text_score_weight) * pointnet_scores
    )
    query_indices, candidate_indices, _ = _rank_full_horizon_demos(
        records=store.records,
        action_masks=action_masks,
        retrieval_scores=combined_scores,
        text_mask=text_mask,
        max_k=utility_config.candidates,
    )
    hypotheses = _predict_hypotheses(
        model=transport,
        geometries=geometries,
        actions=actions,
        action_masks=action_masks,
        query_indices=query_indices,
        candidate_indices=candidate_indices,
        device=device,
    )
    targets = actions[query_indices]
    target_masks = action_masks[query_indices]
    candidate_errors = (hypotheses - targets[:, None]).abs().mean(dim=(2, 3))
    features = _build_features(
        model=transport,
        geometries=geometries,
        hypotheses=hypotheses,
        query_indices=query_indices,
        candidate_indices=candidate_indices,
        combined_scores=combined_scores,
        pointnet_scores=pointnet_scores,
        normalized_text=normalized_text,
    )
    group_ids = [_episode_key(store.records[index]) for index in query_indices]
    return PreparedSplit(
        features=features,
        best_candidates=candidate_errors.argmin(dim=1),
        candidate_errors=candidate_errors,
        hypotheses=hypotheses,
        targets=targets,
        target_masks=target_masks,
        group_ids=group_ids,
        query_count=len(query_indices),
        episode_count=len(set(group_ids)),
    )


def _train_selector(
    *,
    prepared: PreparedSplit,
    config: UtilityTrainConfig,
    device: torch.device,
    log_path: Path,
) -> DemoUtilitySelector:
    _seed_everything(config.seed)
    flat = prepared.features.flatten(end_dim=1)
    feature_mean = flat.mean(dim=0)
    feature_std = flat.std(dim=0).clamp_min(1e-5)
    model = DemoUtilitySelector(
        UtilitySelectorConfig(
            input_dim=prepared.features.shape[-1],
            hidden_dim=config.hidden_dim,
        ),
        feature_mean=feature_mean,
        feature_std=feature_std,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    sample_count = len(prepared.features)
    for epoch in range(config.epochs):
        generator = torch.Generator().manual_seed(config.seed + epoch)
        permutation = torch.randperm(sample_count, generator=generator)
        model.train()
        loss_total = 0.0
        correct = 0
        for start in range(0, sample_count, config.batch_size):
            indices = permutation[start : start + config.batch_size]
            features = prepared.features[indices].to(device)
            targets = prepared.best_candidates[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            scores = model(features)
            loss = nn.functional.cross_entropy(scores, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            loss_total += float(loss.detach()) * len(indices)
            correct += int((scores.argmax(dim=1) == targets).sum())
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == config.epochs:
            record = {
                "epoch": epoch + 1,
                "loss": loss_total / sample_count,
                "best_candidate_accuracy": correct / sample_count,
            }
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
    return model.eval()


@torch.inference_mode()
def _evaluate_selector(
    *,
    model: DemoUtilitySelector,
    prepared: PreparedSplit,
    config: UtilityTrainConfig,
    device: torch.device,
    with_bootstrap: bool,
) -> dict[str, Any]:
    scores = model(prepared.features.to(device)).cpu()
    selected = scores.argmax(dim=1)
    batch = torch.arange(len(selected))
    selector_prediction = prepared.hypotheses[batch, selected]
    baselines, _ = _aggregate_hypotheses(
        prepared.hypotheses,
        prepared.targets,
        (1, 2, config.candidates),
    )
    predictions = {**baselines, "utility_selector": selector_prediction}
    metrics = {
        name: _physical_metrics(
            prediction, prepared.targets, prepared.target_masks
        )
        for name, prediction in predictions.items()
    }
    bootstrap = {}
    if with_bootstrap:
        consensus_config = ConsensusConfig(
            k_values=(1, 2, config.candidates),
            bootstrap_resamples=config.bootstrap_resamples,
            seed=config.seed,
        )
        bootstrap = _paired_bootstrap_report(
            predictions=predictions,
            targets=prepared.targets,
            target_masks=prepared.target_masks,
            group_ids=prepared.group_ids,
            config=consensus_config,
        )
    selected_error = prepared.candidate_errors[batch, selected]
    rank1_error = prepared.candidate_errors[:, 0]
    oracle_error = prepared.candidate_errors.min(dim=1).values
    available_gap = float(rank1_error.mean() - oracle_error.mean())
    recovered_gap = float(rank1_error.mean() - selected_error.mean())
    return {
        "metrics": metrics,
        "paired_episode_bootstrap_vs_ljat_rank1": bootstrap,
        "selection": {
            "best_candidate_accuracy": float(
                (selected == prepared.best_candidates).float().mean()
            ),
            "rank_distribution": {
                str(rank + 1): int((selected == rank).sum())
                for rank in range(config.candidates)
            },
            "rank1_agreement": float((selected == 0).float().mean()),
            "mean_selected_nmae": float(selected_error.mean()),
            "mean_rank1_nmae": float(rank1_error.mean()),
            "mean_oracle_nmae": float(oracle_error.mean()),
            "available_rank1_to_oracle_gap": available_gap,
            "recovered_gap": recovered_gap,
            "fraction_oracle_gap_recovered": (
                recovered_gap / available_gap if available_gap > 0 else None
            ),
        },
    }


def run(
    *,
    project_root: Path,
    context_root: Path,
    action_root: Path,
    retriever_checkpoint: Path,
    text_scores_path: Path,
    transport_checkpoint: Path,
    output_root: Path,
    train_config: TrainConfig,
    utility_config: UtilityTrainConfig,
    device: torch.device,
) -> None:
    if output_root.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output_root}")
    temporary = output_root.with_name(f".{output_root.name}.incomplete-{os.getpid()}")
    temporary.mkdir(parents=True)
    context_hash = _sha256(context_root / "summary.json")
    action_hash = _sha256(action_root / "summary.json")
    retriever = load_retriever(retriever_checkpoint, device)
    transport, transport_payload = _load_transport(
        transport_checkpoint,
        context_hash=context_hash,
        action_hash=action_hash,
        device=device,
    )
    if transport_payload.get("train_config") != asdict(train_config):
        raise ValueError("评估配置与 LJAT checkpoint 的训练配置不一致")
    prepared = {
        split: _prepare_split(
            split=split,
            context_root=context_root,
            action_root=action_root,
            text_scores_path=text_scores_path,
            retriever=retriever,
            transport=transport,
            train_config=train_config,
            utility_config=utility_config,
            device=device,
        )
        for split in ("train", "val")
    }
    selector = _train_selector(
        prepared=prepared["train"],
        config=utility_config,
        device=device,
        log_path=temporary / "training_metrics.jsonl",
    )
    evaluations = {
        split: _evaluate_selector(
            model=selector,
            prepared=prepared[split],
            config=utility_config,
            device=device,
            with_bootstrap=split == "val",
        )
        for split in ("train", "val")
    }
    checkpoint = {
        "model": selector.state_dict(),
        "model_config": asdict(selector.config),
        "utility_train_config": asdict(utility_config),
        "ljat_train_config": asdict(train_config),
        "context_summary_sha256": context_hash,
        "action_summary_sha256": action_hash,
        "retriever_checkpoint_sha256": _sha256(retriever_checkpoint),
        "text_scores_sha256": _sha256(text_scores_path),
        "transport_checkpoint_sha256": _sha256(transport_checkpoint),
        "git_commit": _git_commit(project_root),
    }
    checkpoint_path = temporary / "demo_utility_selector.pt"
    torch.save(checkpoint, checkpoint_path.with_suffix(".pt.tmp"))
    os.replace(checkpoint_path.with_suffix(".pt.tmp"), checkpoint_path)
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "candidates": "frozen text top-10 + PointNet top-4; one per episode",
            "predictor": "frozen LJAT",
            "label": "minimum train-split action NMAE candidate index",
            "validation_future_used_for_training": False,
            "checkpoint_selection": "fixed final epoch",
        },
        "device": str(device),
        "config": {
            "ljat_train": asdict(train_config),
            "utility_train": asdict(utility_config),
            "model": asdict(selector.config),
        },
        "model_parameters": sum(
            parameter.numel() for parameter in selector.parameters()
        ),
        "git_commit": _git_commit(project_root),
        "transport_git_commit": transport_payload.get("git_commit", "unknown"),
        "context_summary_sha256": context_hash,
        "action_summary_sha256": action_hash,
        "retriever_checkpoint_sha256": _sha256(retriever_checkpoint),
        "text_scores_sha256": _sha256(text_scores_path),
        "transport_checkpoint_sha256": _sha256(transport_checkpoint),
        "split_summary": {
            split: {
                "queries": values.query_count,
                "episode_groups": values.episode_count,
            }
            for split, values in prepared.items()
        },
        "evaluations": evaluations,
    }
    temporary.joinpath("report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.joinpath("TRAINING_COMPLETE").write_text(
        "demo_utility_selector_v1\n", encoding="utf-8"
    )
    temporary.rename(output_root)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--context-root", type=Path, required=True)
    parser.add_argument("--action-root", type=Path, required=True)
    parser.add_argument("--retriever-checkpoint", type=Path, required=True)
    parser.add_argument("--text-scores", type=Path, required=True)
    parser.add_argument("--transport-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
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
        transport_checkpoint=arguments.transport_checkpoint.resolve(),
        output_root=arguments.output_root.resolve(),
        train_config=TrainConfig.from_json(arguments.config.resolve()),
        utility_config=UtilityTrainConfig(),
        device=device,
    )


if __name__ == "__main__":
    main()

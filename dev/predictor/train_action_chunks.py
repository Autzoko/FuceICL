"""训练并评估轻量 Demo-conditioned H-step action-chunk Predictor。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
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
from dev.predictor.action_chunk_model import (
    ChunkPredictorConfig,
    build_chunk_predictor,
)


POSE_SCALES = torch.tensor(
    [0.10, 0.10, 0.10, 0.50, 0.50, 0.50], dtype=torch.float32
)
MODEL_NAMES = (
    "query_only",
    "demo_concat",
    "demo_action_prior_no_dependency",
    "demo_action_prior",
)
SELECTION_NAMES = (
    "oracle",
    "retrieved",
    "random_same_task",
    "wrong_phase",
    "wrong_layout",
    "wrong_task",
    "no_demo",
    "shuffled_action",
)


@dataclass(frozen=True)
class TrainConfig:
    """固定 action-chunk pilot 的容量与优化配置。"""

    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    dependency_weight: float
    dependency_margin: float
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
            self.dependency_margin,
            self.gradient_clip_norm,
            self.hidden_dim,
            self.num_layers,
            self.num_heads,
            self.feedforward_dim,
            self.retriever_batch_size,
            self.text_group_budget,
        )
        if min(positive) <= 0:
            raise ValueError(
                "训练尺寸、学习率、margin 和模型容量必须为正"
            )
        if min(self.weight_decay, self.dependency_weight) < 0:
            raise ValueError("weight decay 和 dependency weight 不能为负")
        if self.retriever_num_workers < 0:
            raise ValueError("retriever_num_workers 不能为负")
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


def _pose_scale_tensor(
    actions: torch.Tensor,
    pose_scales: torch.Tensor | Sequence[float] | None,
) -> torch.Tensor:
    values = POSE_SCALES if pose_scales is None else torch.as_tensor(pose_scales)
    if values.shape != (6,) or not bool(torch.all(values > 0)):
        raise ValueError("action pose scales 必须是 6 个正数")
    return values.to(device=actions.device, dtype=actions.dtype)


def _normalize_actions(
    actions: torch.Tensor,
    pose_scales: torch.Tensor | Sequence[float] | None = None,
) -> torch.Tensor:
    normalized = actions.clone().float()
    normalized[..., :6] /= _pose_scale_tensor(normalized, pose_scales)
    normalized[..., 6] = 2.0 * normalized[..., 6] - 1.0
    return normalized


def _denormalize_actions(
    actions: torch.Tensor,
    pose_scales: torch.Tensor | Sequence[float] | None = None,
) -> torch.Tensor:
    physical = actions.clone().float()
    physical[..., :6] *= _pose_scale_tensor(physical, pose_scales)
    physical[..., 6] = ((physical[..., 6] + 1.0) * 0.5).clamp(0.0, 1.0)
    return physical


def _load_action_tensors(
    context_store: PointNetContextStore,
    action_store: ActionChunkStore,
) -> tuple[torch.Tensor, torch.Tensor]:
    context_ids = [str(record["chunk_id"]) for record in context_store.records]
    action_ids = [str(record["chunk_id"]) for record in action_store.records]
    if context_ids != action_ids:
        raise ValueError("context 与 action manifest ID/顺序不一致")
    values = [action_store.get(identifier) for identifier in context_ids]
    actions = torch.from_numpy(np.stack([value["actions"] for value in values]))
    masks = torch.from_numpy(np.stack([value["valid_mask"] for value in values]))
    return _normalize_actions(actions), masks.bool()


def _model_config(
    train_config: TrainConfig,
    horizon: int,
) -> ChunkPredictorConfig:
    return ChunkPredictorConfig(
        horizon=horizon,
        hidden_dim=train_config.hidden_dim,
        num_layers=train_config.num_layers,
        num_heads=train_config.num_heads,
        feedforward_dim=train_config.feedforward_dim,
        dropout=train_config.dropout,
        residual_limit=train_config.residual_limit,
    )


def _masked_sample_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    losses = nn.functional.smooth_l1_loss(
        prediction,
        target,
        reduction="none",
    ).mean(dim=-1)
    weights = mask.float()
    return (losses * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def train_predictor(
    *,
    name: str,
    dataset: ActionChunkPairDataset,
    config: TrainConfig,
    device: torch.device,
    log_path: Path,
) -> nn.Module:
    """使用固定末轮 checkpoint，避免在 pilot val 上隐式调参。"""
    _seed_everything(config.seed)
    horizon = int(dataset.actions.shape[1])
    architecture = (
        "demo_action_prior"
        if name == "demo_action_prior_no_dependency"
        else name
    )
    model = build_chunk_predictor(
        architecture,
        _model_config(config, horizon),
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
        totals = {"total": 0.0, "action": 0.0, "dependency": 0.0}
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
            positive_loss = _masked_sample_loss(
                prediction,
                values["target_actions"],
                values["target_mask"],
            )
            dependency = torch.zeros((), device=device)
            if name == "demo_action_prior":
                wrong_prediction = model(
                    values["query_context"],
                    values["wrong_context"],
                    values["wrong_actions"],
                    values["wrong_mask"],
                )
                wrong_loss = _masked_sample_loss(
                    wrong_prediction,
                    values["target_actions"],
                    values["target_mask"],
                )
                dependency = torch.relu(
                    config.dependency_margin + positive_loss - wrong_loss
                ).mean()
            action_loss = positive_loss.mean()
            total = action_loss + config.dependency_weight * dependency
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            totals["total"] += float(total.detach()) * batch_size
            totals["action"] += float(action_loss.detach()) * batch_size
            totals["dependency"] += float(dependency.detach()) * batch_size
            examples += batch_size
        scheduler.step()
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == config.epochs:
            record = {
                "model": name,
                "epoch": epoch + 1,
                "learning_rate": optimizer.param_groups[0]["lr"],
                **{key: value / examples for key, value in totals.items()},
            }
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
    return model.eval()


def _first_negative(
    pair: Mapping[str, Any],
    category: str,
    id_to_index: Mapping[str, int],
    full: torch.Tensor,
) -> int:
    for identifier in pair["hard_negatives"].get(category, []):
        if identifier in id_to_index and bool(full[id_to_index[identifier]]):
            return id_to_index[identifier]
    return -1


def _selection_indices(
    *,
    records: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    action_masks: torch.Tensor,
    retrieval_scores: torch.Tensor,
    text_mask: torch.Tensor,
    seed: int,
) -> tuple[list[int], dict[str, list[tuple[int, int]]]]:
    id_to_index = {str(row["chunk_id"]): index for index, row in enumerate(records)}
    pairs = {str(row["query_id"]): row for row in pair_rows}
    full = action_masks.all(dim=1)
    rng = random.Random(seed)
    eligible = []
    selections = {name: [] for name in SELECTION_NAMES}

    for query_index, record in enumerate(records):
        if not bool(full[query_index]):
            continue
        pair = pairs[str(record["chunk_id"])]
        positives = [
            id_to_index[identifier]
            for identifier in pair["positive_ids"]
            if identifier in id_to_index and bool(full[id_to_index[identifier]])
        ]
        if not positives:
            continue
        base_mask = _base_candidate_mask(records, query_index)
        candidate_mask = base_mask & text_mask[query_index]
        ranked = torch.argsort(
            retrieval_scores[query_index].masked_fill(
                ~candidate_mask, float("-inf")
            ),
            descending=True,
            stable=True,
        ).tolist()
        ranked = [index for index in ranked if bool(candidate_mask[index])]
        retrieved = ranked[0] if ranked else -1
        same_task = [
            index
            for index, candidate in enumerate(records)
            if bool(base_mask[index])
            and bool(full[index])
            and candidate["task"] == record["task"]
        ]
        other_task = [
            index
            for index, candidate in enumerate(records)
            if bool(full[index]) and candidate["task"] != record["task"]
        ]
        random_same_task = rng.choice(same_task) if same_task else -1
        shuffled_action = rng.choice(other_task) if other_task else -1
        wrong_phase = _first_negative(pair, "wrong_phase", id_to_index, full)
        wrong_layout = _first_negative(pair, "wrong_layout", id_to_index, full)
        wrong_task = _first_negative(
            pair,
            "geometry_collision",
            id_to_index,
            full,
        )
        controlled = (
            random_same_task,
            shuffled_action,
            wrong_phase,
            wrong_layout,
            wrong_task,
        )
        if min(controlled) < 0:
            continue

        eligible.append(query_index)
        selections["oracle"].append((positives[0], positives[0]))
        selections["retrieved"].append((retrieved, retrieved))
        selections["random_same_task"].append(
            (random_same_task, random_same_task)
        )
        selections["wrong_phase"].append((wrong_phase, wrong_phase))
        selections["wrong_layout"].append((wrong_layout, wrong_layout))
        selections["wrong_task"].append((wrong_task, wrong_task))
        selections["no_demo"].append((-1, -1))
        selections["shuffled_action"].append((retrieved, shuffled_action))
    return eligible, selections


def _condition_tensors(
    selections: Sequence[tuple[int, int]],
    embeddings: torch.Tensor,
    actions: torch.Tensor,
    action_masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    contexts = []
    demo_actions = []
    masks = []
    for context_index, action_index in selections:
        valid = context_index >= 0 and action_index >= 0
        contexts.append(
            embeddings[context_index]
            if valid
            else torch.zeros(embeddings.shape[1])
        )
        demo_actions.append(
            actions[action_index]
            if valid
            else torch.zeros_like(actions[0])
        )
        masks.append(
            action_masks[action_index]
            if valid
            else torch.zeros_like(action_masks[0])
        )
    return torch.stack(contexts), torch.stack(demo_actions), torch.stack(masks)


def _physical_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    pose_scales: torch.Tensor | Sequence[float] | None = None,
    translation_threshold_m: float = 0.05,
    rotation_threshold_rad: float = 0.25,
) -> dict[str, float]:
    if min(translation_threshold_m, rotation_threshold_rad) <= 0:
        raise ValueError("action threshold 必须为正")
    normalized_error = (prediction - target).abs().mean(dim=-1)
    prediction_physical = _denormalize_actions(prediction, pose_scales)
    target_physical = _denormalize_actions(target, pose_scales)
    translation = torch.linalg.vector_norm(
        prediction_physical[..., :3] - target_physical[..., :3], dim=-1
    )
    rotation = torch.linalg.vector_norm(
        prediction_physical[..., 3:6] - target_physical[..., 3:6], dim=-1
    )
    gripper_correct = (
        (prediction_physical[..., 6] >= 0.5)
        == (target_physical[..., 6] >= 0.5)
    )
    weights = target_mask.float()

    def masked_mean(values: torch.Tensor) -> float:
        return float((values * weights).sum() / weights.sum().clamp_min(1.0))

    step_success = (
        (translation < translation_threshold_m)
        & (rotation < rotation_threshold_rad)
        & gripper_correct
    )
    chunk_success = (step_success | ~target_mask).all(dim=1).float().mean()
    return {
        "normalized_mae": masked_mean(normalized_error),
        "translation_l2_m": masked_mean(translation),
        "rotation_l2_rad": masked_mean(rotation),
        "gripper_accuracy": masked_mean(gripper_correct.float()),
        "step_threshold_accuracy": masked_mean(step_success.float()),
        "chunk_threshold_accuracy": float(chunk_success),
    }


@torch.inference_mode()
def evaluate_predictors(
    *,
    models: Mapping[str, nn.Module],
    embeddings: torch.Tensor,
    actions: torch.Tensor,
    action_masks: torch.Tensor,
    query_indices: Sequence[int],
    selections: Mapping[str, Sequence[tuple[int, int]]],
    device: torch.device,
) -> dict[str, Any]:
    queries = embeddings[list(query_indices)]
    targets = actions[list(query_indices)]
    target_masks = action_masks[list(query_indices)]
    report: dict[str, Any] = {}
    all_models: dict[str, nn.Module | None] = {"demo_action_copy": None, **models}
    for name, model in all_models.items():
        report[name] = {}
        predictions = {}
        for condition, indices in selections.items():
            demo_context, demo_actions, demo_mask = _condition_tensors(
                indices,
                embeddings,
                actions,
                action_masks,
            )
            if name == "demo_action_copy":
                prediction = demo_actions * demo_mask.float().unsqueeze(-1)
            else:
                prediction = model(
                    queries.to(device),
                    demo_context.to(device),
                    demo_actions.to(device),
                    demo_mask.to(device),
                ).cpu()
            predictions[condition] = prediction
            report[name][condition] = {
                **_physical_metrics(prediction, targets, target_masks),
                "valid_demo_step_fraction": float(demo_mask.float().mean()),
            }
        oracle = predictions["oracle"]
        report[name]["demo_sensitivity"] = {
            "oracle_vs_wrong_phase_output_l2": float(
                torch.linalg.vector_norm(
                    oracle - predictions["wrong_phase"], dim=-1
                ).mean()
            ),
            "oracle_vs_wrong_task_output_l2": float(
                torch.linalg.vector_norm(
                    oracle - predictions["wrong_task"], dim=-1
                ).mean()
            ),
            "oracle_vs_shuffled_action_output_l2": float(
                torch.linalg.vector_norm(
                    oracle - predictions["shuffled_action"], dim=-1
                ).mean()
            ),
            "wrong_phase_minus_oracle_normalized_mae": (
                report[name]["wrong_phase"]["normalized_mae"]
                - report[name]["oracle"]["normalized_mae"]
            ),
            "no_demo_minus_oracle_normalized_mae": (
                report[name]["no_demo"]["normalized_mae"]
                - report[name]["oracle"]["normalized_mae"]
            ),
        }
    return report


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

    retriever = load_retriever(retriever_checkpoint, device)
    embedding_config = ComparisonConfig(
        batch_size=config.retriever_batch_size,
        num_workers=config.retriever_num_workers,
    )
    context_stores = {
        split: PointNetContextStore(context_root, split, cache_size=4)
        for split in ("train", "val")
    }
    action_stores = {
        split: ActionChunkStore(action_root, split, cache_size=4)
        for split in ("train", "val")
    }
    embeddings = {
        split: encode_contexts(
            retriever,
            context_stores[split],
            embedding_config,
            device,
        )
        for split in ("train", "val")
    }
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
        embeddings=embeddings["train"],
        actions=actions["train"],
        action_masks=action_masks["train"],
        records=context_stores["train"].records,
        pair_rows=pair_rows["train"],
        seed=config.seed,
    )
    log_path = temporary / "training_metrics.jsonl"
    models = {
        name: train_predictor(
            name=name,
            dataset=train_dataset,
            config=config,
            device=device,
            log_path=log_path,
        )
        for name in MODEL_NAMES
    }
    for name, model in models.items():
        payload = {
            "model": model.state_dict(),
            "model_name": name,
            "architecture": (
                "demo_action_prior"
                if name == "demo_action_prior_no_dependency"
                else name
            ),
            "dependency_loss_enabled": name == "demo_action_prior",
            "model_config": asdict(model.config),
            "train_config": asdict(config),
            "context_summary_sha256": context_hash,
            "action_summary_sha256": action_hash,
            "retriever_checkpoint_sha256": _sha256(retriever_checkpoint),
            "git_commit": _git_commit(project_root),
        }
        path = temporary / f"{name}.pt"
        torch.save(payload, path.with_suffix(".pt.tmp"))
        os.replace(path.with_suffix(".pt.tmp"), path)

    val_context_store = context_stores["val"]
    text_scores, chunk_groups, text_metadata = _load_text_scores(
        text_scores_path,
        val_context_store.records,
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
    pointnet_scores = (
        embeddings["val"] @ embeddings["val"].T + 1.0
    ) / 2.0
    normalized_text = ((text_scores + 1.0) / 2.0).clamp(0.0, 1.0)
    retrieval_scores = (
        config.text_score_weight * normalized_text
        + (1.0 - config.text_score_weight) * pointnet_scores
    )
    query_indices, selections = _selection_indices(
        records=val_context_store.records,
        pair_rows=pair_rows["val"],
        action_masks=action_masks["val"],
        retrieval_scores=retrieval_scores,
        text_mask=text_mask,
        seed=config.seed,
    )
    metrics = evaluate_predictors(
        models=models,
        embeddings=embeddings["val"],
        actions=actions["val"],
        action_masks=action_masks["val"],
        query_indices=query_indices,
        selections=selections,
        device=device,
    )
    report = {
        "protocol": {
            "split": "official val episodes; tasks overlap train",
            "target": (
                "6-step cumulative delta pose in query EEF frame "
                "+ gripper target"
            ),
            "train_demo": "future-action/effect pair-label positive",
            "retrieved_demo": "text top-10 + PointNet reranking",
            "checkpoint_selection": "fixed final epoch",
            "scope": "action-chunk mechanism pilot, not task-held-out final result",
        },
        "device": str(device),
        "cuda_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "config": asdict(config),
        "context_summary_sha256": context_hash,
        "action_summary_sha256": action_hash,
        "retriever_checkpoint_sha256": _sha256(retriever_checkpoint),
        "text_scores_sha256": _sha256(text_scores_path),
        "git_commit": _git_commit(project_root),
        "train_pairs": len(train_dataset),
        "eval_queries": len(query_indices),
        "model_parameters": {
            name: sum(parameter.numel() for parameter in model.parameters())
            for name, model in models.items()
        },
        "metrics": metrics,
    }
    temporary.joinpath("report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.joinpath("TRAINING_COMPLETE").write_text(
        "action_chunk_predictor_pilot_v1\n",
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

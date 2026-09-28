"""训练并评估 Demo-conditioned 聚合 open-loop 动作 Predictor。"""

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
    _read_jsonl,
    _sha256,
    encode_contexts,
    load_model as load_retriever,
    ComparisonConfig,
)
from dev.pointnet.dataset import PointNetContextStore
from dev.pointnet.evaluate_end_to_end_retriever import (
    _base_candidate_mask,
    _load_text_scores,
    _text_candidate_masks,
)
from dev.predictor.dataset import AggregateActionPairDataset, NEGATIVE_CATEGORIES
from dev.predictor.model import ACTION_DIM, PredictorConfig, build_predictor


ACTION_SCALES = torch.tensor(
    [0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.08], dtype=torch.float32
)
MODEL_NAMES = ("query_only", "demo_concat", "demo_action_prior")


@dataclass(frozen=True)
class TrainConfig:
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    dependency_weight: float
    dependency_margin: float
    gradient_clip_norm: float
    hidden_dim: int
    dropout: float
    residual_limit: float
    retriever_batch_size: int
    retriever_num_workers: int
    text_group_budget: int
    text_score_weight: float

    @classmethod
    def from_json(cls, path: Path) -> "TrainConfig":
        return cls(**json.loads(path.read_text()))

    def __post_init__(self) -> None:
        if min(
            self.epochs,
            self.batch_size,
            self.learning_rate,
            self.dependency_margin,
            self.gradient_clip_norm,
            self.hidden_dim,
            self.retriever_batch_size,
            self.text_group_budget,
        ) <= 0:
            raise ValueError("训练尺寸、学习率、margin 和梯度裁剪必须为正")
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


def _actions(store: PointNetContextStore) -> torch.Tensor:
    values = np.stack(
        [
            store.get(row["chunk_id"])["future_target"][:ACTION_DIM]
            for row in store.records
        ]
    )
    return torch.from_numpy(values).float() / ACTION_SCALES


def _predictor_config(config: TrainConfig) -> PredictorConfig:
    return PredictorConfig(
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        residual_limit=config.residual_limit,
    )


def _sample_loss(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    return nn.functional.smooth_l1_loss(
        prediction, target, reduction="none"
    ).mean(dim=-1)


def train_predictor(
    *,
    name: str,
    dataset: AggregateActionPairDataset,
    config: TrainConfig,
    device: torch.device,
    log_path: Path,
) -> nn.Module:
    """固定 epoch 训练，避免在 pilot val 上选择最优轮次。"""
    _seed_everything(config.seed)
    model = build_predictor(name, _predictor_config(config)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
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
        sums = {"total": 0.0, "action": 0.0, "dependency": 0.0}
        examples = 0
        for batch in loader:
            values = {
                key: value.to(device)
                for key, value in batch.items()
                if key != "query_index"
            }
            batch_size = len(values["target_action"])
            valid = torch.ones(batch_size, 1, device=device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                values["query_context"],
                values["demo_context"],
                values["demo_action"],
                valid,
            )
            positive_loss = _sample_loss(
                prediction, values["target_action"]
            )
            dependency = torch.zeros((), device=device)
            if name == "demo_action_prior":
                wrong_prediction = model(
                    values["query_context"],
                    values["wrong_context"],
                    values["wrong_action"],
                    valid,
                )
                wrong_loss = _sample_loss(
                    wrong_prediction, values["target_action"]
                )
                dependency = torch.relu(
                    config.dependency_margin + positive_loss - wrong_loss
                ).mean()
            action_loss = positive_loss.mean()
            total = action_loss + config.dependency_weight * dependency
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            sums["total"] += float(total.detach()) * batch_size
            sums["action"] += float(action_loss.detach()) * batch_size
            sums["dependency"] += float(dependency.detach()) * batch_size
            examples += batch_size
        scheduler.step()
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == config.epochs:
            record = {
                "model": name,
                "epoch": epoch + 1,
                "learning_rate": optimizer.param_groups[0]["lr"],
                **{key: value / examples for key, value in sums.items()},
            }
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
    return model.eval()


def _selection_indices(
    *,
    records: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    pointnet_scores: torch.Tensor,
    text_scores: torch.Tensor,
    text_mask: torch.Tensor,
    text_weight: float,
    seed: int,
) -> tuple[list[int], dict[str, list[tuple[int, int]]]]:
    id_to_index = {row["chunk_id"]: index for index, row in enumerate(records)}
    pairs = {row["query_id"]: row for row in pair_rows}
    eligible = []
    selections = {
        name: []
        for name in (
            "oracle",
            "retrieved",
            "random_same_task",
            "wrong_phase",
            "wrong_layout",
            "no_demo",
            "shuffled_action",
        )
    }
    normalized_text = ((text_scores + 1.0) / 2.0).clamp(0.0, 1.0)
    retrieval_scores = (
        text_weight * normalized_text + (1.0 - text_weight) * pointnet_scores
    )
    rng = random.Random(seed)
    for query_index, record in enumerate(records):
        pair = pairs[record["chunk_id"]]
        positives = [
            id_to_index[value]
            for value in pair["positive_ids"]
            if value in id_to_index
        ]
        if not positives:
            continue
        eligible.append(query_index)
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
            if bool(base_mask[index]) and candidate["task"] == record["task"]
        ]
        random_same_task = rng.choice(same_task) if same_task else -1

        def first_negative(category: str) -> int:
            for identifier in pair["hard_negatives"].get(category, []):
                if identifier in id_to_index:
                    return id_to_index[identifier]
            for fallback in NEGATIVE_CATEGORIES:
                for identifier in pair["hard_negatives"].get(fallback, []):
                    if identifier in id_to_index:
                        return id_to_index[identifier]
            return -1

        wrong_action_candidates = [
            index
            for index, candidate in enumerate(records)
            if candidate["task"] != record["task"]
        ]
        shuffled_action = rng.choice(wrong_action_candidates)
        selections["oracle"].append((positives[0], positives[0]))
        selections["retrieved"].append((retrieved, retrieved))
        selections["random_same_task"].append(
            (random_same_task, random_same_task)
        )
        wrong_phase = first_negative("wrong_phase")
        wrong_layout = first_negative("wrong_layout")
        selections["wrong_phase"].append((wrong_phase, wrong_phase))
        selections["wrong_layout"].append((wrong_layout, wrong_layout))
        selections["no_demo"].append((-1, -1))
        selections["shuffled_action"].append((retrieved, shuffled_action))
    return eligible, selections


def _condition_tensors(
    selections: Sequence[tuple[int, int]],
    embeddings: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    contexts = []
    demo_actions = []
    valid = []
    for context_index, action_index in selections:
        is_valid = context_index >= 0 and action_index >= 0
        contexts.append(
            embeddings[context_index]
            if is_valid
            else torch.zeros(embeddings.shape[1])
        )
        demo_actions.append(
            actions[action_index] if is_valid else torch.zeros(actions.shape[1])
        )
        valid.append(is_valid)
    return (
        torch.stack(contexts),
        torch.stack(demo_actions),
        torch.tensor(valid, dtype=torch.float32).reshape(-1, 1),
    )


def _physical_metrics(
    prediction: torch.Tensor, target: torch.Tensor
) -> dict[str, float]:
    prediction = prediction * ACTION_SCALES
    target = target * ACTION_SCALES
    translation = torch.linalg.vector_norm(
        prediction[:, :3] - target[:, :3], dim=-1
    )
    rotation = torch.linalg.vector_norm(
        prediction[:, 3:6] - target[:, 3:6], dim=-1
    )
    gripper = (prediction[:, 6] - target[:, 6]).abs()
    normalized_mae = ((prediction - target) / ACTION_SCALES).abs().mean(dim=-1)
    return {
        "normalized_mae": float(normalized_mae.mean()),
        "translation_l2_m": float(translation.mean()),
        "rotation_l2_rad": float(rotation.mean()),
        "gripper_abs_m": float(gripper.mean()),
        "threshold_accuracy": float(
            ((translation < 0.05) & (rotation < 0.25) & (gripper < 0.02))
            .float()
            .mean()
        ),
    }


@torch.inference_mode()
def evaluate_predictors(
    *,
    models: Mapping[str, nn.Module],
    embeddings: torch.Tensor,
    actions: torch.Tensor,
    query_indices: Sequence[int],
    selections: Mapping[str, Sequence[tuple[int, int]]],
    device: torch.device,
) -> dict[str, Any]:
    queries = embeddings[list(query_indices)]
    targets = actions[list(query_indices)]
    report: dict[str, Any] = {}
    all_models: dict[str, nn.Module | None] = {"demo_action_copy": None, **models}
    for name, model in all_models.items():
        report[name] = {}
        condition_predictions = {}
        for condition, indices in selections.items():
            demo_context, demo_action, demo_valid = _condition_tensors(
                indices, embeddings, actions
            )
            if name == "demo_action_copy":
                prediction = demo_action * demo_valid
            else:
                prediction = model(
                    queries.to(device),
                    demo_context.to(device),
                    demo_action.to(device),
                    demo_valid.to(device),
                ).cpu()
            condition_predictions[condition] = prediction
            report[name][condition] = {
                **_physical_metrics(prediction, targets),
                "valid_demo_fraction": float(demo_valid.mean()),
            }
        oracle = condition_predictions["oracle"]
        report[name]["demo_sensitivity"] = {
            "oracle_vs_wrong_phase_output_l2": float(
                torch.linalg.vector_norm(
                    oracle - condition_predictions["wrong_phase"], dim=-1
                ).mean()
            ),
            "oracle_vs_shuffled_action_output_l2": float(
                torch.linalg.vector_norm(
                    oracle - condition_predictions["shuffled_action"], dim=-1
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
    data_root: Path,
    retriever_checkpoint: Path,
    text_scores_path: Path,
    output_root: Path,
    config: TrainConfig,
    device: torch.device,
) -> None:
    if output_root.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output_root}")
    output_root.mkdir(parents=True)
    _seed_everything(config.seed)
    dataset_hash = _sha256(data_root / "summary.json")
    retriever_payload = torch.load(retriever_checkpoint, map_location="cpu")
    if retriever_payload.get("dataset_summary_sha256") != dataset_hash:
        raise ValueError("Retriever checkpoint 与数据不匹配")

    retriever = load_retriever(retriever_checkpoint, device)
    embedding_config = ComparisonConfig(
        batch_size=config.retriever_batch_size,
        num_workers=config.retriever_num_workers,
    )
    stores = {
        split: PointNetContextStore(data_root, split, cache_size=4)
        for split in ("train", "val")
    }
    embeddings = {
        split: encode_contexts(retriever, store, embedding_config, device)
        for split, store in stores.items()
    }
    actions = {split: _actions(store) for split, store in stores.items()}
    pair_rows = {
        split: _read_jsonl(data_root / f"pairs-{split}.jsonl")
        for split in ("train", "val")
    }
    train_dataset = AggregateActionPairDataset(
        embeddings=embeddings["train"],
        actions=actions["train"],
        records=stores["train"].records,
        pair_rows=pair_rows["train"],
        seed=config.seed,
    )

    log_path = output_root / "training_metrics.jsonl"
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
            "model_config": asdict(model.config),
            "train_config": asdict(config),
            "dataset_summary_sha256": dataset_hash,
            "retriever_checkpoint_sha256": _sha256(retriever_checkpoint),
            "git_commit": _git_commit(project_root),
        }
        temporary = output_root / f"{name}.pt.tmp"
        torch.save(payload, temporary)
        os.replace(temporary, output_root / f"{name}.pt")

    text_scores, chunk_groups, text_metadata = _load_text_scores(
        text_scores_path, stores["val"].records
    )
    if text_metadata.get("manifest_sha256") != _sha256(
        data_root / "manifest-val.jsonl"
    ):
        raise ValueError("文本分数与 val manifest 不匹配")
    text_mask = _text_candidate_masks(
        text_scores, chunk_groups, (config.text_group_budget,)
    )[config.text_group_budget]
    pointnet_scores = (
        embeddings["val"] @ embeddings["val"].T + 1.0
    ) / 2.0
    query_indices, selections = _selection_indices(
        records=stores["val"].records,
        pair_rows=pair_rows["val"],
        pointnet_scores=pointnet_scores,
        text_scores=text_scores,
        text_mask=text_mask,
        text_weight=config.text_score_weight,
        seed=config.seed,
    )
    metrics = evaluate_predictors(
        models=models,
        embeddings=embeddings["val"],
        actions=actions["val"],
        query_indices=query_indices,
        selections=selections,
        device=device,
    )
    metadata = {
        "protocol": {
            "target": "12-frame aggregate delta pose + gripper delta",
            "target_dim": ACTION_DIM,
            "train_demo": "pair-label positive",
            "retrieved_demo": "text top-10 groups + PointNet score",
            "checkpoint_selection": "fixed final epoch",
            "scope": "open-loop mechanism screening, not final action chunk",
        },
        "device": str(device),
        "cuda_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "config": asdict(config),
        "dataset_summary_sha256": dataset_hash,
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
    output_root.joinpath("report.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    output_root.joinpath("TRAINING_COMPLETE").write_text(
        "aggregate_open_loop_predictor_v1\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
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
        data_root=args.data_root,
        retriever_checkpoint=args.retriever_checkpoint,
        text_scores_path=args.text_scores,
        output_root=args.output_root,
        config=TrainConfig.from_json(args.config),
        device=device,
    )


if __name__ == "__main__":
    main()

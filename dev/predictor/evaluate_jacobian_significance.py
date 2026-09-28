"""用 episode-level paired bootstrap 审计 LJAT 相对 Demo copy 的增益。"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Sequence

import numpy as np
import torch

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
from dev.predictor.canonical_geometry import load_canonical_geometries
from dev.predictor.jacobian_transport_model import (
    JacobianTransportConfig,
    LocalJacobianActionTransport,
)
from dev.predictor.train_action_chunks import (
    _condition_tensors,
    _denormalize_actions,
    _load_action_tensors,
    _physical_metrics,
    _selection_indices,
)
from dev.predictor.train_jacobian_transport import TrainConfig


def _git_commit(project_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _load_transport(
    checkpoint_path: Path,
    *,
    context_hash: str,
    action_hash: str,
    device: torch.device,
) -> tuple[LocalJacobianActionTransport, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if payload.get("context_summary_sha256") != context_hash:
        raise ValueError("LJAT checkpoint 与 context 数据不匹配")
    if payload.get("action_summary_sha256") != action_hash:
        raise ValueError("LJAT checkpoint 与 action 数据不匹配")
    model = LocalJacobianActionTransport(
        JacobianTransportConfig(**payload["model_config"])
    )
    model.load_state_dict(payload["model"])
    return model.to(device).eval(), payload


def _per_query_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    pose_scales: torch.Tensor | Sequence[float] | None = None,
    translation_threshold_m: float = 0.05,
    rotation_threshold_rad: float = 0.25,
) -> dict[str, np.ndarray]:
    """返回 full-horizon query 级指标，供 episode block bootstrap 使用。"""
    if min(translation_threshold_m, rotation_threshold_rad) <= 0:
        raise ValueError("action threshold 必须为正")
    weights = mask.float()
    denominator = weights.sum(dim=1).clamp_min(1.0)
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
    step_success = (
        (translation < translation_threshold_m)
        & (rotation < rotation_threshold_rad)
        & gripper_correct
    )

    def masked_query_mean(values: torch.Tensor) -> np.ndarray:
        result = (values.float() * weights).sum(dim=1) / denominator
        return result.cpu().numpy()

    return {
        "normalized_mae": masked_query_mean(normalized_error),
        "translation_l2_m": masked_query_mean(translation),
        "rotation_l2_rad": masked_query_mean(rotation),
        "step_threshold_accuracy": masked_query_mean(step_success),
        "chunk_threshold_accuracy": (
            step_success | ~mask
        ).all(dim=1).float().cpu().numpy(),
    }


def _episode_bootstrap(
    *,
    reference: np.ndarray,
    candidate: np.ndarray,
    group_ids: Sequence[str],
    resamples: int,
    seed: int,
) -> dict[str, float | int]:
    """成组重采样 episode；delta 定义为 candidate - reference。"""
    if reference.shape != candidate.shape or len(reference) != len(group_ids):
        raise ValueError("bootstrap 输入长度不一致")
    if resamples <= 0:
        raise ValueError("resamples 必须为正")
    groups = np.asarray(group_ids)
    unique = np.unique(groups)
    members = [np.flatnonzero(groups == group) for group in unique]
    rng = np.random.default_rng(seed)
    samples = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        chosen = rng.integers(0, len(members), size=len(members))
        query_indices = np.concatenate([members[group] for group in chosen])
        samples[index] = float(
            candidate[query_indices].mean() - reference[query_indices].mean()
        )
    point = float(candidate.mean() - reference.mean())
    return {
        "candidate_minus_reference": point,
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "bootstrap_probability_delta_below_zero": float(np.mean(samples < 0.0)),
        "num_episode_groups": len(unique),
        "num_resamples": resamples,
    }


@torch.inference_mode()
def run(
    *,
    project_root: Path,
    context_root: Path,
    action_root: Path,
    retriever_checkpoint: Path,
    text_scores_path: Path,
    transport_checkpoint: Path,
    output_path: Path,
    config: TrainConfig,
    device: torch.device,
    bootstrap_resamples: int,
) -> None:
    if output_path.exists():
        raise FileExistsError(f"输出已存在，拒绝覆盖：{output_path}")
    context_hash = _sha256(context_root / "summary.json")
    action_hash = _sha256(action_root / "summary.json")
    store = PointNetContextStore(context_root, "val", cache_size=4)
    action_store = ActionChunkStore(action_root, "val", cache_size=4)
    geometries = load_canonical_geometries(store)
    actions, action_masks = _load_action_tensors(store, action_store)
    pair_rows = _read_jsonl(context_root / "pairs-val.jsonl")

    retriever = load_retriever(retriever_checkpoint, device)
    embeddings = encode_contexts(
        retriever,
        store,
        ComparisonConfig(
            batch_size=config.retriever_batch_size,
            num_workers=config.retriever_num_workers,
        ),
        device,
    )
    text_scores, chunk_groups, text_metadata = _load_text_scores(
        text_scores_path,
        store.records,
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
    pointnet_scores = (embeddings @ embeddings.T + 1.0) / 2.0
    normalized_text = ((text_scores + 1.0) / 2.0).clamp(0.0, 1.0)
    retrieval_scores = (
        config.text_score_weight * normalized_text
        + (1.0 - config.text_score_weight) * pointnet_scores
    )
    query_indices, selections = _selection_indices(
        records=store.records,
        pair_rows=pair_rows,
        action_masks=action_masks,
        retrieval_scores=retrieval_scores,
        text_mask=text_mask,
        seed=config.seed,
    )
    model, checkpoint = _load_transport(
        transport_checkpoint,
        context_hash=context_hash,
        action_hash=action_hash,
        device=device,
    )
    if checkpoint.get("train_config") != asdict(config):
        raise ValueError("评估配置与 LJAT checkpoint 的训练配置不一致")
    queries = geometries[query_indices]
    targets = actions[query_indices]
    target_masks = action_masks[query_indices]
    group_ids = [
        ":".join(
            (
                str(store.records[index]["task"]),
                str(store.records[index]["variation"]),
                str(store.records[index]["episode"]),
            )
        )
        for index in query_indices
    ]

    conditions: dict[str, Any] = {}
    for condition in ("oracle", "retrieved"):
        demo_context, demo_actions, demo_mask = _condition_tensors(
            selections[condition],
            geometries,
            actions,
            action_masks,
        )
        copy_prediction = demo_actions * demo_mask.float().unsqueeze(-1)
        transport_prediction = model(
            queries.to(device),
            demo_context.to(device),
            demo_actions.to(device),
            demo_mask.to(device),
        ).cpu()
        copy_per_query = _per_query_metrics(
            copy_prediction, targets, target_masks
        )
        transport_per_query = _per_query_metrics(
            transport_prediction, targets, target_masks
        )
        bootstrap = {}
        for offset, metric in enumerate(copy_per_query):
            bootstrap[metric] = _episode_bootstrap(
                reference=copy_per_query[metric],
                candidate=transport_per_query[metric],
                group_ids=group_ids,
                resamples=bootstrap_resamples,
                seed=config.seed + offset,
            )
        conditions[condition] = {
            "demo_action_copy": _physical_metrics(
                copy_prediction, targets, target_masks
            ),
            "local_jacobian_transport": _physical_metrics(
                transport_prediction, targets, target_masks
            ),
            "paired_episode_bootstrap": bootstrap,
        }

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "comparison": "local_jacobian_transport vs demo_action_copy",
            "delta_definition": "candidate LJAT minus reference Demo copy",
            "bootstrap_unit": "task + variation + episode",
            "checkpoint_selection": "fixed final epoch; no refit",
        },
        "device": str(device),
        "config": {
            "seed": config.seed,
            "text_group_budget": config.text_group_budget,
            "text_score_weight": config.text_score_weight,
            "bootstrap_resamples": bootstrap_resamples,
        },
        "eval_queries": len(query_indices),
        "episode_groups": len(set(group_ids)),
        "evaluator_git_commit": _git_commit(project_root),
        "transport_git_commit": checkpoint.get("git_commit", "unknown"),
        "context_summary_sha256": context_hash,
        "action_summary_sha256": action_hash,
        "retriever_checkpoint_sha256": _sha256(retriever_checkpoint),
        "text_scores_sha256": _sha256(text_scores_path),
        "transport_checkpoint_sha256": _sha256(transport_checkpoint),
        "conditions": conditions,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(f"{output_path.suffix}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--context-root", type=Path, required=True)
    parser.add_argument("--action-root", type=Path, required=True)
    parser.add_argument("--retriever-checkpoint", type=Path, required=True)
    parser.add_argument("--text-scores", type=Path, required=True)
    parser.add_argument("--transport-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
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
        output_path=arguments.output.resolve(),
        config=TrainConfig.from_json(arguments.config.resolve()),
        device=device,
        bootstrap_resamples=arguments.bootstrap_resamples,
    )


if __name__ == "__main__":
    main()

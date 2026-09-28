"""评估 top-K Demo action hypotheses 的鲁棒共识与可靠性。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

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
    _base_candidate_mask,
    _load_text_scores,
    _rank_with_episode_cap,
    _text_candidate_masks,
)
from dev.pointnet.evaluate_selective_retrieval import (
    _bootstrap_precision,
    _episode_key,
    _evaluate_threshold,
    _precision_at_coverages,
    _select_threshold,
    _split_episode_groups,
)
from dev.predictor.action_chunk_data import ActionChunkStore
from dev.predictor.canonical_geometry import load_canonical_geometries
from dev.predictor.evaluate_jacobian_significance import (
    _episode_bootstrap,
    _load_transport,
    _per_query_metrics,
)
from dev.predictor.train_action_chunks import (
    _condition_tensors,
    _load_action_tensors,
    _physical_metrics,
    _selection_indices,
)
from dev.predictor.train_jacobian_transport import TrainConfig


@dataclass(frozen=True)
class ConsensusConfig:
    """固定的多 Demo 共识与可靠性协议。"""

    k_values: tuple[int, ...] = (1, 2, 4)
    target_precision: float = 0.90
    minimum_calibration_accepts: int = 20
    bootstrap_resamples: int = 5000
    seed: int = 20260928

    def __post_init__(self) -> None:
        if not self.k_values or min(self.k_values) <= 0:
            raise ValueError("k_values 必须为非空正整数")
        if tuple(sorted(set(self.k_values))) != self.k_values:
            raise ValueError("k_values 必须严格递增且不能重复")
        if min(self.minimum_calibration_accepts, self.bootstrap_resamples) <= 0:
            raise ValueError("校准样本数和 bootstrap 次数必须为正")
        if not 0.0 < self.target_precision <= 1.0:
            raise ValueError("target_precision 必须位于 (0, 1]")


def _git_commit(project_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _rank_full_horizon_demos(
    *,
    records: Sequence[Mapping[str, Any]],
    action_masks: torch.Tensor,
    retrieval_scores: torch.Tensor,
    text_mask: torch.Tensor,
    max_k: int,
) -> tuple[list[int], torch.Tensor, torch.Tensor]:
    """为每个 full-horizon query 选择来自不同 episode 的 top-K Demo。"""
    full_horizon = action_masks.all(dim=1)
    query_indices: list[int] = []
    candidate_indices: list[list[int]] = []
    candidate_scores: list[torch.Tensor] = []
    for query_index in range(len(records)):
        if not bool(full_horizon[query_index]):
            continue
        candidate_mask = (
            _base_candidate_mask(records, query_index)
            & text_mask[query_index]
            & full_horizon
        )
        ranked = _rank_with_episode_cap(
            retrieval_scores[query_index],
            candidate_mask,
            records,
            episode_cap=1,
        )
        if len(ranked) < max_k:
            continue
        selected = ranked[:max_k]
        query_indices.append(query_index)
        candidate_indices.append(selected)
        candidate_scores.append(retrieval_scores[query_index, selected])
    if not query_indices:
        raise ValueError("没有满足 full-horizon top-K 条件的 query")
    return (
        query_indices,
        torch.tensor(candidate_indices, dtype=torch.long),
        torch.stack(candidate_scores),
    )


@torch.inference_mode()
def _predict_hypotheses(
    *,
    model: torch.nn.Module,
    geometries: torch.Tensor,
    actions: torch.Tensor,
    action_masks: torch.Tensor,
    query_indices: Sequence[int],
    candidate_indices: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """并行执行 K 次同一 LJAT；没有额外训练参数。"""
    query = geometries[list(query_indices)]
    query_count, candidate_count = candidate_indices.shape
    flat_candidates = candidate_indices.reshape(-1)
    predictions = model(
        query.repeat_interleave(candidate_count, dim=0).to(device),
        geometries[flat_candidates].to(device),
        actions[flat_candidates].to(device),
        action_masks[flat_candidates].to(device),
    ).cpu()
    return predictions.reshape(
        query_count,
        candidate_count,
        actions.shape[1],
        actions.shape[2],
    )


def _aggregate_hypotheses(
    hypotheses: torch.Tensor,
    targets: torch.Tensor,
    k_values: Sequence[int],
) -> tuple[dict[str, torch.Tensor], dict[int, torch.Tensor]]:
    """构造均值、metric medoid 和不可部署的 oracle-best 上界。"""
    predictions = {"ljat_rank1": hypotheses[:, 0]}
    disagreements: dict[int, torch.Tensor] = {}
    for k in k_values:
        selected = hypotheses[:, :k]
        flattened = selected.flatten(start_dim=2)
        distances = torch.cdist(flattened, flattened, p=1) / flattened.shape[-1]
        if k == 1:
            disagreement = torch.zeros(len(selected), dtype=selected.dtype)
        else:
            disagreement = distances.sum(dim=(1, 2)) / (k * (k - 1))
        disagreements[k] = disagreement
        if k == 1:
            continue
        medoid_indices = distances.sum(dim=2).argmin(dim=1)
        batch = torch.arange(len(selected))
        predictions[f"ljat_mean_k{k}"] = selected.mean(dim=1)
        predictions[f"ljat_medoid_k{k}"] = selected[batch, medoid_indices]
        oracle_errors = (selected - targets[:, None]).abs().mean(dim=(2, 3))
        oracle_indices = oracle_errors.argmin(dim=1)
        predictions[f"ljat_oracle_best_k{k}"] = selected[batch, oracle_indices]
    return predictions, disagreements


def _paired_bootstrap_report(
    *,
    predictions: Mapping[str, torch.Tensor],
    targets: torch.Tensor,
    target_masks: torch.Tensor,
    group_ids: Sequence[str],
    config: ConsensusConfig,
) -> dict[str, Any]:
    reference = _per_query_metrics(
        predictions["ljat_rank1"], targets, target_masks
    )
    report: dict[str, Any] = {}
    for method_offset, (name, prediction) in enumerate(predictions.items()):
        metrics = _per_query_metrics(prediction, targets, target_masks)
        if name == "ljat_rank1":
            continue
        report[name] = {
            metric: _episode_bootstrap(
                reference=reference[metric],
                candidate=metrics[metric],
                group_ids=group_ids,
                resamples=config.bootstrap_resamples,
                seed=config.seed + 10 * method_offset + metric_offset,
            )
            for metric_offset, metric in enumerate(reference)
        }
    return report


def _reliability_report(
    *,
    records: Sequence[Mapping[str, Any]],
    query_indices: Sequence[int],
    candidate_scores: torch.Tensor,
    disagreement: torch.Tensor,
    medoid_prediction: torch.Tensor,
    targets: torch.Tensor,
    target_masks: torch.Tensor,
    config: ConsensusConfig,
) -> dict[str, Any]:
    """在相同 chunk-success label 上比较 score 与动作共识置信度。"""
    success = _per_query_metrics(
        medoid_prediction, targets, target_masks
    )["chunk_threshold_accuracy"]
    confidence = {
        "retriever_top1": candidate_scores[:, 0].numpy(),
        "retriever_margin": (
            candidate_scores[:, 0] - candidate_scores[:, 1]
        ).numpy(),
        "negative_action_disagreement": -disagreement.numpy(),
    }
    calibration, test = _split_episode_groups(
        records, query_indices, config.seed
    )
    test_groups = np.asarray(
        [_episode_key(records[query_indices[index]]) for index in test]
    )
    methods: dict[str, Any] = {}
    for offset, (name, scores) in enumerate(confidence.items()):
        selected = _select_threshold(
            scores[calibration],
            success[calibration],
            target_precision=config.target_precision,
            minimum_accepts=config.minimum_calibration_accepts,
        )
        threshold = selected["threshold"]
        evaluated = _evaluate_threshold(
            scores[test], success[test], threshold
        )
        evaluated["precision_episode_bootstrap"] = _bootstrap_precision(
            scores=scores[test],
            labels=success[test],
            group_ids=test_groups,
            threshold=threshold if isinstance(threshold, float) else None,
            resamples=config.bootstrap_resamples,
            seed=config.seed + offset,
        )
        methods[name] = {
            "calibration": selected,
            "test": evaluated,
            "test_precision_at_fixed_coverages": _precision_at_coverages(
                scores[test], success[test]
            ),
        }
    return {
        "label": "K=4 medoid satisfies full 6-step action thresholds",
        "calibration_queries": len(calibration),
        "test_queries": len(test),
        "overall_chunk_success": float(success.mean()),
        "methods": methods,
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
    train_config: TrainConfig,
    consensus_config: ConsensusConfig,
    device: torch.device,
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
            batch_size=train_config.retriever_batch_size,
            num_workers=train_config.retriever_num_workers,
        ),
        device,
    )
    text_scores, chunk_groups, text_metadata = _load_text_scores(
        text_scores_path, store.records
    )
    if text_metadata.get("manifest_sha256") != _sha256(
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
    retrieval_scores = (
        train_config.text_score_weight * normalized_text
        + (1.0 - train_config.text_score_weight) * pointnet_scores
    )
    model, transport_payload = _load_transport(
        transport_checkpoint,
        context_hash=context_hash,
        action_hash=action_hash,
        device=device,
    )
    if transport_payload.get("train_config") != asdict(train_config):
        raise ValueError("评估配置与 LJAT checkpoint 的训练配置不一致")

    # 先复现旧 488-query 协议，确保 checkpoint 与数据读取没有漂移。
    legacy_queries, legacy_selections = _selection_indices(
        records=store.records,
        pair_rows=pair_rows,
        action_masks=action_masks,
        retrieval_scores=retrieval_scores,
        text_mask=text_mask,
        seed=train_config.seed,
    )
    legacy_demo_context, legacy_demo_actions, legacy_demo_mask = _condition_tensors(
        legacy_selections["retrieved"], geometries, actions, action_masks
    )
    legacy_prediction = model(
        geometries[legacy_queries].to(device),
        legacy_demo_context.to(device),
        legacy_demo_actions.to(device),
        legacy_demo_mask.to(device),
    ).cpu()
    legacy_targets = actions[legacy_queries]
    legacy_target_masks = action_masks[legacy_queries]

    query_indices, candidate_indices, candidate_scores = _rank_full_horizon_demos(
        records=store.records,
        action_masks=action_masks,
        retrieval_scores=retrieval_scores,
        text_mask=text_mask,
        max_k=max(consensus_config.k_values),
    )
    hypotheses = _predict_hypotheses(
        model=model,
        geometries=geometries,
        actions=actions,
        action_masks=action_masks,
        query_indices=query_indices,
        candidate_indices=candidate_indices,
        device=device,
    )
    targets = actions[query_indices]
    target_masks = action_masks[query_indices]
    predictions, disagreements = _aggregate_hypotheses(
        hypotheses, targets, consensus_config.k_values
    )
    predictions = {
        "demo_copy_rank1": actions[candidate_indices[:, 0]],
        **predictions,
    }
    metrics = {
        name: _physical_metrics(prediction, targets, target_masks)
        for name, prediction in predictions.items()
    }
    group_ids = [_episode_key(store.records[index]) for index in query_indices]
    bootstrap = _paired_bootstrap_report(
        predictions=predictions,
        targets=targets,
        target_masks=target_masks,
        group_ids=group_ids,
        config=consensus_config,
    )
    max_k = max(consensus_config.k_values)
    reliability = _reliability_report(
        records=store.records,
        query_indices=query_indices,
        candidate_scores=candidate_scores,
        disagreement=disagreements[max_k],
        medoid_prediction=predictions[f"ljat_medoid_k{max_k}"],
        targets=targets,
        target_masks=target_masks,
        config=consensus_config,
    )

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "retrieval": "text top-10 + PointNet; one candidate per episode",
            "candidate_requirement": "query and all candidates have full H=6 horizon",
            "aggregation": "normalized action-space mean or metric medoid",
            "checkpoint_selection": "fixed LJAT final epoch; no refit",
            "oracle_best": "analysis-only upper bound; uses target action",
        },
        "config": {
            "train": asdict(train_config),
            "consensus": asdict(consensus_config),
        },
        "device": str(device),
        "evaluator_git_commit": _git_commit(project_root),
        "transport_git_commit": transport_payload.get("git_commit", "unknown"),
        "context_summary_sha256": context_hash,
        "action_summary_sha256": action_hash,
        "retriever_checkpoint_sha256": _sha256(retriever_checkpoint),
        "text_scores_sha256": _sha256(text_scores_path),
        "transport_checkpoint_sha256": _sha256(transport_checkpoint),
        "eval_queries": len(query_indices),
        "episode_groups": len(set(group_ids)),
        "legacy_protocol_reproduction": {
            "eval_queries": len(legacy_queries),
            "demo_action_copy": _physical_metrics(
                legacy_demo_actions * legacy_demo_mask.float().unsqueeze(-1),
                legacy_targets,
                legacy_target_masks,
            ),
            "local_jacobian_transport": _physical_metrics(
                legacy_prediction, legacy_targets, legacy_target_masks
            ),
        },
        "metrics": metrics,
        "paired_episode_bootstrap_vs_ljat_rank1": bootstrap,
        "reliability": reliability,
        "disagreement_summary": {
            f"k{k}": {
                "mean": float(values.mean()),
                "p50": float(torch.quantile(values, 0.50)),
                "p95": float(torch.quantile(values, 0.95)),
            }
            for k, values in disagreements.items()
        },
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
        train_config=TrainConfig.from_json(arguments.config.resolve()),
        consensus_config=ConsensusConfig(),
        device=device,
    )


if __name__ == "__main__":
    main()

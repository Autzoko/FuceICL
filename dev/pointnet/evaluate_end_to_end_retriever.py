"""评估真实文本初筛与几何/PointNet 细排组成的端到端 Retriever。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from dev.pointnet.compare_retrievers import (
    _all_states,
    _read_jsonl,
    _sha256,
    calibrate_geometry_scales,
    encode_contexts,
    geometry_similarity,
    load_model,
    paired_bootstrap_delta,
    ComparisonConfig,
)
from dev.pointnet.dataset import PointNetContextStore


@dataclass(frozen=True)
class EndToEndConfig:
    """文本候选预算、排序和统计配置。"""

    text_group_budgets: tuple[int, ...] = (1, 3, 5, 10)
    recall_k: tuple[int, ...] = (1, 4, 10)
    text_score_weight: float = 0.15
    pointnet_fusion_weight: float = 0.50
    batch_size: int = 64
    num_workers: int = 4
    bootstrap_samples: int = 2000
    seed: int = 20260928

    def __post_init__(self) -> None:
        positive = (
            *self.text_group_budgets,
            *self.recall_k,
            self.batch_size,
            self.bootstrap_samples,
        )
        if min(positive) <= 0 or self.num_workers < 0:
            raise ValueError(
                "budgets/K/batch/bootstrap 必须为正，worker 不能为负"
            )
        for name, value in (
            ("text_score_weight", self.text_score_weight),
            ("pointnet_fusion_weight", self.pointnet_fusion_weight),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} 必须位于 [0, 1]")


def _git_commit(root: Path) -> str:
    """记录评估代码版本；失败时显式返回 unknown。"""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _load_text_scores(
    path: Path,
    records: Sequence[Mapping[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    with np.load(path) as archive:
        texts = [str(value) for value in archive["texts"].tolist()]
        group_scores = torch.from_numpy(archive["final_scores"].astype(np.float32))
        metadata = json.loads(str(archive["metadata_json"].item()))
    if group_scores.shape != (len(texts), len(texts)):
        raise ValueError("文本 group score matrix shape 错误")
    if not torch.isfinite(group_scores).all():
        raise ValueError("文本分数包含 NaN 或 Inf")
    text_to_group = {text: index for index, text in enumerate(texts)}
    if len(text_to_group) != len(texts):
        raise ValueError("文本分数产物包含重复 description")
    missing = sorted({str(row["text"]) for row in records} - set(text_to_group))
    if missing:
        raise ValueError(f"文本分数缺少 manifest descriptions：{missing[:3]}")
    chunk_groups = torch.tensor(
        [text_to_group[str(row["text"])] for row in records], dtype=torch.long
    )
    chunk_scores = group_scores[chunk_groups[:, None], chunk_groups[None, :]]
    return chunk_scores, chunk_groups, metadata


def _base_candidate_mask(
    records: Sequence[Mapping[str, Any]], query_index: int
) -> torch.Tensor:
    query = records[query_index]
    return torch.tensor(
        [
            index != query_index
            and not (
                candidate["task"] == query["task"]
                and candidate["episode"] == query["episode"]
            )
            for index, candidate in enumerate(records)
        ],
        dtype=torch.bool,
    )


def _text_candidate_masks(
    text_scores: torch.Tensor,
    chunk_groups: torch.Tensor,
    budgets: Sequence[int],
) -> dict[int, torch.Tensor]:
    group_count = int(chunk_groups.max()) + 1
    group_scores = torch.empty((group_count, group_count), dtype=torch.float32)
    for query_group in range(group_count):
        query_chunk = int(torch.nonzero(chunk_groups == query_group)[0])
        for candidate_group in range(group_count):
            candidate_chunk = int(torch.nonzero(chunk_groups == candidate_group)[0])
            group_scores[query_group, candidate_group] = text_scores[
                query_chunk, candidate_chunk
            ]
    group_ranking = torch.argsort(
        group_scores, dim=1, descending=True, stable=True
    )
    masks = {}
    for budget in budgets:
        selected = torch.zeros_like(group_scores, dtype=torch.bool)
        selected.scatter_(1, group_ranking[:, : min(budget, group_count)], True)
        masks[int(budget)] = selected[chunk_groups[:, None], chunk_groups[None, :]]
    return masks


def _rank_with_episode_cap(
    scores: torch.Tensor,
    mask: torch.Tensor,
    records: Sequence[Mapping[str, Any]],
    *,
    episode_cap: int | None,
) -> list[int]:
    ranked = torch.argsort(
        scores.masked_fill(~mask, float("-inf")), descending=True, stable=True
    ).tolist()
    ranked = [index for index in ranked if bool(mask[index])]
    if episode_cap is None:
        return ranked
    counts: dict[tuple[str, int], int] = {}
    diverse = []
    for index in ranked:
        record = records[index]
        key = (str(record["task"]), int(record["episode"]))
        if counts.get(key, 0) >= episode_cap:
            continue
        counts[key] = counts.get(key, 0) + 1
        diverse.append(index)
    return diverse


def evaluate_pipeline_scores(
    scores: torch.Tensor,
    text_mask: torch.Tensor,
    records: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    config: EndToEndConfig,
    *,
    episode_cap: int | None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """固定全局 eligible query 分母，文本漏召回会被计为失败。"""
    count = len(records)
    if scores.shape != (count, count) or text_mask.shape != (count, count):
        raise ValueError("score/mask matrix 与 manifest 不一致")
    id_to_index = {row["chunk_id"]: index for index, row in enumerate(records)}
    pairs = {row["query_id"]: row for row in pair_rows}
    hard_categories = (
        "wrong_phase",
        "wrong_gripper",
        "wrong_layout",
        "geometry_collision",
    )
    hits = {value: 0 for value in config.recall_k}
    intrusions = {
        category: {value: 0 for value in config.recall_k}
        for category in hard_categories
    }
    hard_available = {category: 0 for category in hard_categories}
    task_precision = {value: [] for value in config.recall_k}
    candidate_counts = []
    reciprocal_ranks = []
    stage1_positive_hits = 0
    stage1_task_hits = 0
    eligible = 0
    outcomes: dict[str, dict[str, Any]] = {}

    for query_index, record in enumerate(records):
        pair = pairs[record["chunk_id"]]
        positives = {
            id_to_index[value]
            for value in pair["positive_ids"]
            if value in id_to_index
        }
        base_mask = _base_candidate_mask(records, query_index)
        positives = {index for index in positives if bool(base_mask[index])}
        if not positives:
            continue
        eligible += 1
        candidate_mask = base_mask & text_mask[query_index]
        candidate_counts.append(int(candidate_mask.sum()))
        stage1_positive_hits += int(
            any(bool(candidate_mask[index]) for index in positives)
        )
        stage1_task_hits += int(
            any(
                bool(candidate_mask[index])
                and records[index]["task"] == record["task"]
                for index in range(count)
            )
        )
        ranked = _rank_with_episode_cap(
            scores[query_index],
            candidate_mask,
            records,
            episode_cap=episode_cap,
        )
        positive_ranks = [
            rank
            for rank, candidate_index in enumerate(ranked, start=1)
            if candidate_index in positives
        ]
        best_rank = min(positive_ranks) if positive_ranks else None
        reciprocal_ranks.append(0.0 if best_rank is None else 1.0 / best_rank)
        query_hits = {
            value: int(best_rank is not None and best_rank <= value)
            for value in config.recall_k
        }
        for value, hit in query_hits.items():
            hits[value] += hit
            top = ranked[:value]
            task_precision[value].append(
                sum(records[index]["task"] == record["task"] for index in top)
                / max(len(top), 1)
            )

        for category in hard_categories:
            negative_indices = {
                id_to_index[value]
                for value in pair["hard_negatives"].get(category, [])
                if value in id_to_index and bool(candidate_mask[id_to_index[value]])
            }
            if not negative_indices:
                continue
            hard_available[category] += 1
            for value in config.recall_k:
                intrusions[category][value] += int(
                    bool(negative_indices.intersection(ranked[:value]))
                )
        outcomes[str(record["chunk_id"])] = {
            "rank": best_rank,
            "hits": {str(key): value for key, value in query_hits.items()},
        }

    counts = torch.tensor(candidate_counts, dtype=torch.float32)
    return (
        {
            "eligible_queries": eligible,
            "stage1_positive_coverage": stage1_positive_hits / max(eligible, 1),
            "stage1_task_coverage": stage1_task_hits / max(eligible, 1),
            "candidate_chunks": {
                "mean": float(counts.mean()),
                "p50": float(torch.quantile(counts, 0.50)),
                "p95": float(torch.quantile(counts, 0.95)),
            },
            "recall": {
                f"recall@{value}": hits[value] / max(eligible, 1)
                for value in config.recall_k
            },
            "mrr": sum(reciprocal_ranks) / max(eligible, 1),
            "task_precision": {
                f"precision@{value}": sum(task_precision[value])
                / max(len(task_precision[value]), 1)
                for value in config.recall_k
            },
            "hard_negative_intrusion": {
                category: {
                    "eligible_queries": hard_available[category],
                    **{
                        f"intrusion@{value}": (
                            intrusions[category][value] / hard_available[category]
                            if hard_available[category]
                            else None
                        )
                        for value in config.recall_k
                    },
                }
                for category in hard_categories
            },
        },
        outcomes,
    )


def evaluate(
    *,
    data_root: Path,
    checkpoint_path: Path,
    text_scores_path: Path,
    device: torch.device,
    config: EndToEndConfig,
) -> dict[str, Any]:
    started = time.perf_counter()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    dataset_hash = _sha256(data_root / "summary.json")
    if checkpoint.get("dataset_summary_sha256") != dataset_hash:
        raise ValueError("checkpoint 与数据 summary hash 不匹配")

    train_store = PointNetContextStore(data_root, "train", cache_size=4)
    scales = calibrate_geometry_scales(
        train_store, _read_jsonl(data_root / "pairs-train.jsonl")
    )
    del train_store
    val_store = PointNetContextStore(data_root, "val", cache_size=4)
    records = val_store.records
    pair_rows = _read_jsonl(data_root / "pairs-val.jsonl")
    text_scores, chunk_groups, text_metadata = _load_text_scores(
        text_scores_path, records
    )
    if text_metadata.get("manifest_sha256") != _sha256(
        data_root / "manifest-val.jsonl"
    ):
        raise ValueError("文本分数与当前 val manifest hash 不匹配")
    text_masks = _text_candidate_masks(
        text_scores, chunk_groups, config.text_group_budgets
    )

    states = _all_states(val_store)
    geometry_scores = geometry_similarity(states, scales).cpu()
    model = load_model(checkpoint_path, device)
    embeddings = encode_contexts(
        model,
        val_store,
        ComparisonConfig(
            recall_k=config.recall_k,
            fusion_weights=(config.pointnet_fusion_weight,),
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            bootstrap_samples=config.bootstrap_samples,
            seed=config.seed,
        ),
        device,
    )
    pointnet_scores = ((embeddings @ embeddings.T) + 1.0) / 2.0
    normalized_text = ((text_scores + 1.0) / 2.0).clamp(0.0, 1.0)
    local_fusion = (
        config.pointnet_fusion_weight * pointnet_scores
        + (1.0 - config.pointnet_fusion_weight) * geometry_scores
    )
    local_weight = 1.0 - config.text_score_weight
    score_matrices = {
        "text_only": normalized_text,
        "text_geometry": (
            config.text_score_weight * normalized_text
            + local_weight * geometry_scores
        ),
        "text_pointnet": (
            config.text_score_weight * normalized_text
            + local_weight * pointnet_scores
        ),
        "text_fusion": (
            config.text_score_weight * normalized_text
            + local_weight * local_fusion
        ),
    }

    main_metrics: dict[str, Any] = {}
    main_outcomes: dict[int, dict[str, dict[str, dict[str, Any]]]] = {}
    for budget in config.text_group_budgets:
        main_metrics[str(budget)] = {}
        main_outcomes[budget] = {}
        for name, scores in score_matrices.items():
            report, outcomes = evaluate_pipeline_scores(
                scores,
                text_masks[budget],
                records,
                pair_rows,
                config,
                episode_cap=1,
            )
            main_metrics[str(budget)][name] = report
            main_outcomes[budget][name] = outcomes

    primary_budget = max(config.text_group_budgets)
    no_diversity = {}
    for name, scores in score_matrices.items():
        report, _ = evaluate_pipeline_scores(
            scores,
            text_masks[primary_budget],
            records,
            pair_rows,
            config,
            episode_cap=None,
        )
        no_diversity[name] = report

    comparisons = {}
    primary_outcomes = main_outcomes[primary_budget]
    for first, second in (
        ("text_geometry", "text_pointnet"),
        ("text_geometry", "text_fusion"),
        ("text_pointnet", "text_fusion"),
    ):
        comparisons[f"{second}_minus_{first}"] = paired_bootstrap_delta(
            primary_outcomes[first],
            primary_outcomes[second],
            recall_k=4,
            samples=config.bootstrap_samples,
            seed=config.seed,
        )

    return {
        "protocol": {
            "split": "val",
            "positive_denominator": "all queries with global pair-label positives",
            "text_unit": "unique raw instruction group",
            "stage2_text_score_weight": config.text_score_weight,
            "episode_cap": 1,
            "primary_text_group_budget": primary_budget,
        },
        "config": asdict(config),
        "dataset": {
            "schema": json.loads((data_root / "summary.json").read_text())[
                "schema_version"
            ],
            "chunks": len(records),
            "text_groups": int(chunk_groups.max()) + 1,
            "dataset_summary_sha256": dataset_hash,
        },
        "checkpoint": {
            "epoch": checkpoint["epoch"],
            "git_commit": checkpoint.get("git_commit"),
        },
        "evaluator_git_commit": _git_commit(Path.cwd()),
        "text_artifact": {
            "path": str(text_scores_path),
            "metadata": text_metadata,
        },
        "metrics_by_text_group_budget": main_metrics,
        "episode_diversity_ablation": {
            "budget": primary_budget,
            "episode_cap_1": main_metrics[str(primary_budget)],
            "no_episode_cap": no_diversity,
        },
        "paired_bootstrap_primary_recall@4": comparisons,
        "runtime_seconds": time.perf_counter() - started,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--text-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA，但当前节点没有可用 GPU")
    report = evaluate(
        data_root=args.data_root,
        checkpoint_path=args.checkpoint,
        text_scores_path=args.text_scores,
        device=device,
        config=EndToEndConfig(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

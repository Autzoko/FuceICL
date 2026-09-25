"""评估 RLBench held-out query 对 train candidate 的文本与显式几何检索。"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Iterable, Mapping, Sequence

import torch

from src.components.retriever.text_retriever import TextRetriever

from .explicit_retriever import ExplicitFeatureRetriever, ExplicitRetrieverConfig
from .types import ActionSemantics, GeometricContext, RetrieverCandidate, RetrieverQuery


@dataclass(frozen=True)
class EvaluationConfig:
    recall_k: tuple[int, ...] = (1, 4, 10)
    layout_center_threshold_m: float = 0.25
    eef_position_threshold_m: float = 0.18
    text_top_k: int = 80


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _tensor(values: Sequence[float] | Sequence[Sequence[float]]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32)


def _context(record: Mapping) -> GeometricContext:
    return GeometricContext(
        active_points=_tensor(record["active_points"]),
        active_center=_tensor(record["active_center"]),
        active_extent=_tensor(record["active_extent"]),
        eef_relative_active=_tensor(record["eef_relative_active"]),
        eef_velocity=_tensor(record["eef_velocity"]),
        gripper_width=float(record["gripper_width"]),
    )


def _candidate(record: dict) -> RetrieverCandidate:
    return RetrieverCandidate(
        candidate_id=record["chunk_id"],
        text=record["text"],
        context=_context(record),
        action=ActionSemantics(
            phase=record["phase"],
            gripper_event=record["gripper_state"],
        ),
        payload=record,
    )


def _query(record: dict) -> RetrieverQuery:
    return RetrieverQuery(
        text=record["text"],
        context=_context(record),
        action=ActionSemantics(
            phase=record["phase"],
            gripper_event=record["gripper_state"],
        ),
    )


def _layout_compatible(query: Mapping, candidate: Mapping, config: EvaluationConfig) -> bool:
    query_center = _tensor(query["active_center"])
    candidate_center = _tensor(candidate["active_center"])
    query_eef = _tensor(query["eef_relative_active"][:3])
    candidate_eef = _tensor(candidate["eef_relative_active"][:3])
    return bool(
        torch.linalg.vector_norm(query_center - candidate_center)
        <= config.layout_center_threshold_m
        and torch.linalg.vector_norm(query_eef - candidate_eef)
        <= config.eef_position_threshold_m
    )


def _category(query: Mapping, candidate: Mapping, config: EvaluationConfig) -> str:
    if query["task"] != candidate["task"]:
        return "wrong_task"
    if query["phase"] != candidate["phase"]:
        return "wrong_phase"
    if query["gripper_state"] != candidate["gripper_state"]:
        return "wrong_gripper"
    if not _layout_compatible(query, candidate, config):
        return "wrong_layout"
    return "strict_positive"


def _task_phase_positive(query: Mapping, candidate: Mapping) -> bool:
    return query["task"] == candidate["task"] and query["phase"] == candidate["phase"]


def _evaluate_rankings(
    queries: Sequence[dict],
    candidates_by_id: Mapping[str, dict],
    rankings: Mapping[str, Sequence[str]],
    config: EvaluationConfig,
) -> dict:
    recall = {
        "task_phase": {str(k): 0 for k in config.recall_k},
        "strict": {str(k): 0 for k in config.recall_k},
    }
    reciprocal_ranks = {"task_phase": [], "strict": []}
    eligible = {"task_phase": 0, "strict": 0}
    intrusions = {str(k): Counter() for k in config.recall_k}
    phase_recall = defaultdict(lambda: {str(k): [0, 0] for k in config.recall_k})

    candidate_values = list(candidates_by_id.values())
    for query in queries:
        relevant = {
            "task_phase": {
                item["chunk_id"]
                for item in candidate_values
                if _task_phase_positive(query, item)
            },
            "strict": {
                item["chunk_id"]
                for item in candidate_values
                if _category(query, item, config) == "strict_positive"
            },
        }
        ranking = list(rankings[query["chunk_id"]])
        for label, identifiers in relevant.items():
            if not identifiers:
                continue
            eligible[label] += 1
            matching_ranks = [
                index + 1
                for index, identifier in enumerate(ranking)
                if identifier in identifiers
            ]
            reciprocal_ranks[label].append(
                1.0 / matching_ranks[0] if matching_ranks else 0.0
            )
            for k in config.recall_k:
                hit = any(identifier in identifiers for identifier in ranking[:k])
                recall[label][str(k)] += int(hit)
                if label == "strict":
                    values = phase_recall[query["phase"]][str(k)]
                    values[0] += int(hit)
                    values[1] += 1

        for k in config.recall_k:
            for identifier in ranking[:k]:
                intrusions[str(k)][
                    _category(query, candidates_by_id[identifier], config)
                ] += 1

    normalized_recall = {
        label: {
            f"recall@{k}": (
                count / eligible[label] if eligible[label] else None
            )
            for k, count in values.items()
        }
        for label, values in recall.items()
    }
    mrr_at_max_k = {
        label: (
            sum(values) / len(values) if values else None
        )
        for label, values in reciprocal_ranks.items()
    }
    normalized_intrusions = {}
    for k in config.recall_k:
        total = sum(intrusions[str(k)].values())
        normalized_intrusions[f"top@{k}"] = {
            category: count / total if total else None
            for category, count in sorted(intrusions[str(k)].items())
        }
    normalized_phase = {
        phase: {
            f"recall@{k}": hits / total if total else None
            for k, (hits, total) in values.items()
        }
        for phase, values in sorted(phase_recall.items())
    }
    return {
        "eligible_queries": eligible,
        "recall": normalized_recall,
        f"mrr@{max(config.recall_k)}": mrr_at_max_k,
        "hard_negative_intrusion": normalized_intrusions,
        "strict_recall_by_phase": normalized_phase,
    }


def evaluate(
    records: Sequence[dict],
    *,
    device: str,
    config: EvaluationConfig,
) -> dict:
    candidate_records = [item for item in records if item["split"] == "train"]
    query_records = [item for item in records if item["split"] == "val"]
    if not candidate_records or not query_records:
        raise ValueError("JSONL 必须同时包含 train candidates 和 val queries")

    candidates = [_candidate(item) for item in candidate_records]
    candidates_by_id = {item["chunk_id"]: item for item in candidate_records}
    max_k = max(config.recall_k)
    text_retriever = TextRetriever.from_local_models(device=device)
    retriever = ExplicitFeatureRetriever(
        text_retriever,
        config=ExplicitRetrieverConfig(
            text_top_k=min(config.text_top_k, len(candidates)),
            default_top_k=max_k,
        ),
    )

    started = time.perf_counter()
    retriever.build_index(candidates)
    index_seconds = time.perf_counter() - started

    text_rankings = {}
    explicit_rankings = {}
    latencies = {"text": [], "explicit": []}
    for record in query_records:
        started = time.perf_counter()
        text_result = text_retriever.retrieve(record["text"], top_k=max_k)
        latencies["text"].append(time.perf_counter() - started)
        text_rankings[record["chunk_id"]] = [
            hit.candidate.candidate_id for hit in text_result.hits
        ]

        started = time.perf_counter()
        explicit_result = retriever.retrieve(_query(record), top_k=max_k)
        latencies["explicit"].append(time.perf_counter() - started)
        explicit_rankings[record["chunk_id"]] = [
            hit.candidate.candidate_id for hit in explicit_result.hits
        ]

    def latency_summary(values: Iterable[float]) -> dict:
        tensor = torch.tensor(list(values), dtype=torch.float64)
        return {
            "mean_ms": float(1000.0 * tensor.mean()),
            "p50_ms": float(1000.0 * torch.quantile(tensor, 0.50)),
            "p95_ms": float(1000.0 * torch.quantile(tensor, 0.95)),
        }

    text_metrics = _evaluate_rankings(
        query_records,
        candidates_by_id,
        text_rankings,
        config,
    )
    explicit_metrics = _evaluate_rankings(
        query_records,
        candidates_by_id,
        explicit_rankings,
        config,
    )
    per_task = {}
    for task in sorted({item["task"] for item in query_records}):
        task_queries = [item for item in query_records if item["task"] == task]
        per_task[task] = {
            "query_chunks": len(task_queries),
            "text_only": _evaluate_rankings(
                task_queries,
                candidates_by_id,
                text_rankings,
                config,
            ),
            "explicit_geometry": _evaluate_rankings(
                task_queries,
                candidates_by_id,
                explicit_rankings,
                config,
            ),
        }

    return {
        "config": asdict(config),
        "dataset": {
            "candidate_chunks": len(candidate_records),
            "query_chunks": len(query_records),
            "tasks": sorted({item["task"] for item in records}),
            "candidate_episodes": len(
                {(item["task"], item["episode"]) for item in candidate_records}
            ),
            "query_episodes": len(
                {(item["task"], item["episode"]) for item in query_records}
            ),
            "skipped_expected_chunks": {
                "train": 4 * len(
                    {(item["task"], item["episode"]) for item in candidate_records}
                ) - len(candidate_records),
                "val": 4 * len(
                    {(item["task"], item["episode"]) for item in query_records}
                ) - len(query_records),
            },
        },
        "text_only": text_metrics,
        "explicit_geometry": explicit_metrics,
        "per_task": per_task,
        "timing": {
            "index_seconds": index_seconds,
            "query": {
                name: latency_summary(values) for name, values in latencies.items()
            },
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = evaluate(load_jsonl(args.input), device=args.device, config=EvaluationConfig())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

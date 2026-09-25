"""在 DROID 多描述数据上复核冻结的文本初筛分数。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.GLiNER2_Base import (  # noqa: E402
    TEXT_SCHEMA_SHA256,
    TEXT_SCHEMA_VERSION,
    GLiNERTextParser,
)
from lib.all_MiniLM_L6_v2 import MiniLMTextEncoder  # noqa: E402


ANNOTATION_KEYS = (
    "language_instruction1",
    "language_instruction2",
    "language_instruction3",
)
RAW_TEXT_WEIGHT = 0.8
OBJECT_WEIGHT = 0.2


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotations",
        type=Path,
        default=REPO_ROOT / "data/DROID/droid_language_annotations.json",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=base_dir / "cache/droid_500_text_retriever.json",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=REPO_ROOT / "lib/GLiNER2_Base/checkpoint",
    )
    parser.add_argument(
        "--minilm-dir",
        type=Path,
        default=REPO_ROOT / "lib/all_MiniLM_L6_v2/checkpoint",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=base_dir / "results/droid_500_text_retriever.json",
    )
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--refresh-cache", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_records(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """加载缓存，或用稳定 GLiNER API 重新解析固定样本。"""
    source = json.loads(args.annotations.read_text(encoding="utf-8"))
    if args.episodes <= 0 or args.episodes > len(source):
        raise ValueError("episodes 必须在数据集 episode 数量范围内")
    episode_ids = random.Random(args.seed).sample(sorted(source), args.episodes)
    metadata = {
        "schema_version": TEXT_SCHEMA_VERSION,
        "schema_sha256": TEXT_SCHEMA_SHA256,
        "source_sha256": sha256_file(args.annotations),
        "seed": args.seed,
        "episodes": args.episodes,
        "episode_ids": episode_ids,
    }
    if args.cache.is_file() and not args.refresh_cache:
        cached = json.loads(args.cache.read_text(encoding="utf-8"))
        if cached.get("metadata") == metadata:
            print(f"Using parse cache: {args.cache}")
            return metadata, cached["records"]

    parser = GLiNERTextParser(args.model_dir)
    pending = [
        {
            "episode_id": episode_id,
            "annotation_key": key,
            "text": source[episode_id][key],
        }
        for episode_id in episode_ids
        for key in ANNOTATION_KEYS
    ]
    records: list[dict[str, Any]] = []
    for start in range(0, len(pending), args.chunk_size):
        chunk = pending[start : start + args.chunk_size]
        parsed = parser.parse_many(
            [record["text"] for record in chunk],
            batch_size=args.batch_size,
        )
        for record, structure in zip(chunk, parsed, strict=True):
            records.append(
                {
                    **record,
                    "goal_operation": structure.goal_operation,
                    "operations": [span.to_dict() for span in structure.operations],
                    "objects": [span.to_dict() for span in structure.objects],
                }
            )
        print(f"Parsed {len(records)}/{len(pending)}", flush=True)

    args.cache.parent.mkdir(parents=True, exist_ok=True)
    args.cache.write_text(
        json.dumps({"metadata": metadata, "records": records}, ensure_ascii=False),
        encoding="utf-8",
    )
    return metadata, records


def split_records(
    records: Sequence[dict[str, Any]],
    episode_ids: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], np.ndarray]:
    by_episode: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        by_episode.setdefault(record["episode_id"], {})[
            record["annotation_key"]
        ] = record

    candidates: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    truth: list[int] = []
    for episode_id in episode_ids:
        annotations = by_episode[episode_id]
        candidate_index = len(candidates)
        candidates.append(annotations["language_instruction1"])
        for key in ("language_instruction2", "language_instruction3"):
            queries.append(annotations[key])
            truth.append(candidate_index)
    return candidates, queries, np.asarray(truth, dtype=np.int64)


def encode_object_sets(
    encoder: MiniLMTextEncoder,
    records: Sequence[dict[str, Any]],
) -> list[np.ndarray]:
    object_sets = [
        tuple(dict.fromkeys(span["text"] for span in record["objects"]))
        for record in records
    ]
    vocabulary = sorted({text for values in object_sets for text in values})
    if not vocabulary:
        return [np.empty((0, 384), dtype=np.float32) for _ in records]
    vectors = encoder.encode(vocabulary).numpy()
    lookup = dict(zip(vocabulary, vectors, strict=True))
    return [
        np.stack([lookup[text] for text in values])
        if values
        else np.empty((0, 384), dtype=np.float32)
        for values in object_sets
    ]


def object_set_scores(
    queries: Sequence[np.ndarray],
    candidates: Sequence[np.ndarray],
) -> np.ndarray:
    """计算与生产实现一致的双向 mean-nearest-neighbor cosine。"""
    scores = np.zeros((len(queries), len(candidates)), dtype=np.float32)
    for query_index, query in enumerate(queries):
        if not len(query):
            continue
        for candidate_index, candidate in enumerate(candidates):
            if not len(candidate):
                continue
            pairwise = query @ candidate.T
            scores[query_index, candidate_index] = 0.5 * (
                float(pairwise.max(axis=1).mean())
                + float(pairwise.max(axis=0).mean())
            )
    return scores


def ranking_metrics(scores: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    order = np.argsort(-scores, axis=1, kind="stable")
    ranks = np.argmax(order == truth[:, None], axis=1) + 1
    return {
        "recall_at_1": float(np.mean(ranks <= 1)),
        "recall_at_5": float(np.mean(ranks <= 5)),
        "recall_at_10": float(np.mean(ranks <= 10)),
        "mrr": float(np.mean(1.0 / ranks)),
        "mean_rank": float(np.mean(ranks)),
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    metadata, records = load_records(args)
    candidates, queries, truth = split_records(records, metadata["episode_ids"])
    encoder = MiniLMTextEncoder(args.minilm_dir)

    raw_scores = (
        encoder.encode([record["text"] for record in queries]).numpy()
        @ encoder.encode([record["text"] for record in candidates]).numpy().T
    )
    object_scores = object_set_scores(
        encode_object_sets(encoder, queries),
        encode_object_sets(encoder, candidates),
    )
    frozen_scores = RAW_TEXT_WEIGHT * raw_scores + OBJECT_WEIGHT * object_scores
    report = {
        "protocol": {
            **metadata,
            "queries": len(queries),
            "relevance": "same episode; strict proxy, not human task-equivalence labels",
            "score": "0.8 * raw MiniLM + 0.2 * symmetric object-set cosine",
        },
        "extraction": {
            "operation_coverage": float(
                np.mean([bool(record["operations"]) for record in records])
            ),
            "object_coverage": float(
                np.mean([bool(record["objects"]) for record in records])
            ),
        },
        "metrics": {
            "raw_minilm": ranking_metrics(raw_scores, truth),
            "frozen_text_retriever": ranking_metrics(frozen_scores, truth),
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()

"""为 RLBench manifest 的唯一任务描述预计算 GLiNER+MiniLM 分数。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from src.components.retriever.text_retriever import TextCandidate, TextRetriever


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_manifest(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not rows:
        raise ValueError("manifest 不能为空")
    if any(not row.get("text") for row in rows):
        raise ValueError("manifest 中存在空任务描述")
    return rows


def precompute(manifest: Path, output: Path, *, device: str) -> None:
    rows = _read_manifest(manifest)
    texts = sorted({str(row["text"]) for row in rows})
    identifiers = [f"text:{index:04d}" for index in range(len(texts))]
    candidates = [
        TextCandidate(identifier, text)
        for identifier, text in zip(identifiers, texts, strict=True)
    ]

    started = time.perf_counter()
    retriever = TextRetriever.from_local_models(device=device)
    retriever.build_index(candidates)
    final = np.empty((len(texts), len(texts)), dtype=np.float32)
    raw = np.empty_like(final)
    objects = np.empty_like(final)
    identifier_to_index = {
        identifier: index for index, identifier in enumerate(identifiers)
    }
    parsed_queries = []
    for query_index, text in enumerate(texts):
        result = retriever.retrieve(text, top_k=len(texts))
        parsed_queries.append(result.query.to_dict())
        for hit in result.hits:
            candidate_index = identifier_to_index[hit.candidate.candidate_id]
            final[query_index, candidate_index] = hit.score
            raw[query_index, candidate_index] = hit.raw_text_score
            objects[query_index, candidate_index] = hit.object_score

    metadata = {
        "schema_version": "rlbench-text-scores-v1",
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "unique_texts": len(texts),
        "chunks": len(rows),
        "device": device,
        "raw_text_weight": retriever.config.raw_text_weight,
        "object_weight": retriever.config.object_weight,
        "gliner_checkpoint_sha256": _sha256(
            retriever.parser.model_dir / "model.safetensors"
        ),
        "gliner_schema_sha256": _sha256(retriever.parser.schema_path),
        "minilm_checkpoint_sha256": _sha256(
            retriever.encoder.model_dir / "model.safetensors"
        ),
        "runtime_seconds": time.perf_counter() - started,
        "parsed_queries": parsed_queries,
    }
    if not all(np.isfinite(values).all() for values in (final, raw, objects)):
        raise ValueError("文本分数包含 NaN 或 Inf")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            texts=np.asarray(texts),
            final_scores=final,
            raw_scores=raw,
            object_scores=objects,
            metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
        )
    temporary.replace(output)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    precompute(args.manifest, args.output, device=args.device)


if __name__ == "__main__":
    main()

"""审计 PointNet++ 预处理产物的 shape、有限值、标签覆盖与 shard checksum。"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_ARRAYS = {
    "active_points": (3,),
    "target_points": (3,),
    "active_center": (3,),
    "active_extent": (3,),
    "target_center": (3,),
    "target_extent": (3,),
    "eef_relative_active": (9,),
    "eef_velocity": (6,),
    "future_translation": (3,),
    "future_rotation_axis_angle": (3,),
    "effect_translation": (3,),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _distribution(values: list[float]) -> dict[str, float]:
    """返回简洁分布摘要，便于审计自动标签的置信度。"""
    if not values:
        return {}
    array = np.asarray(values, dtype=np.float64)
    quantiles = np.quantile(array, (0.0, 0.05, 0.5, 0.95, 1.0))
    return {
        "min": float(quantiles[0]),
        "p05": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p95": float(quantiles[3]),
        "max": float(quantiles[4]),
    }


def inspect_dataset(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    summary = json.loads(root.joinpath("summary.json").read_text())
    persistent_handles = set(summary.get("persistent_excluded_handles", ()))
    expected = {}
    for line in root.joinpath("SHA256SUMS").read_text().splitlines():
        digest, relative = line.split("  ", 1)
        expected[relative] = digest
    checksum_errors = [
        relative
        for relative, digest in expected.items()
        if _sha256(root / relative) != digest
    ]

    report: dict[str, Any] = {"checksum_errors": checksum_errors, "splits": {}}
    for split in ("train", "val"):
        manifest = [
            json.loads(line)
            for line in root.joinpath(f"manifest-{split}.jsonl").read_text().splitlines()
            if line
        ]
        pairs = [
            json.loads(line)
            for line in root.joinpath(f"pairs-{split}.jsonl").read_text().splitlines()
            if line
        ]
        shard_rows = Counter(row["shard"] for row in manifest)
        array_errors = []
        for relative, expected_rows in shard_rows.items():
            with np.load(root / relative) as shard:
                for name, suffix in REQUIRED_ARRAYS.items():
                    if name not in shard:
                        array_errors.append(f"{relative}:missing:{name}")
                        continue
                    values = shard[name]
                    if values.shape[0] != expected_rows:
                        array_errors.append(f"{relative}:rows:{name}:{values.shape}")
                    if name.endswith("_points"):
                        valid_shape = values.ndim == 3 and values.shape[-1:] == suffix
                    else:
                        valid_shape = values.shape[1:] == suffix
                    if not valid_shape:
                        array_errors.append(f"{relative}:shape:{name}:{values.shape}")
                    if not np.isfinite(values).all():
                        array_errors.append(f"{relative}:nonfinite:{name}")
                if shard["active_points"].ndim != 3:
                    array_errors.append(f"{relative}:active_points_ndim")
                if (shard["active_extent"] <= 0).any():
                    array_errors.append(f"{relative}:nonpositive_extent")
        chunk_ids = {row["chunk_id"] for row in manifest}
        pair_errors = []
        semantic_errors = []
        positive_counts: Counter = Counter()
        negative_coverage: Counter = Counter()
        for pair in pairs:
            query_id = pair["query_id"]
            positive_ids = set(pair["positive_ids"])
            if query_id not in chunk_ids:
                pair_errors.append(f"unknown_query:{query_id}")
            missing = positive_ids - chunk_ids
            if missing:
                pair_errors.append(f"unknown_positive:{query_id}:{sorted(missing)}")
            positive_counts[len(positive_ids)] += 1
            for category, identifiers in pair["hard_negatives"].items():
                negative_ids = set(identifiers)
                missing = negative_ids - chunk_ids
                if missing:
                    pair_errors.append(
                        f"unknown_negative:{query_id}:{category}:{sorted(missing)}"
                    )
                overlap = positive_ids & negative_ids
                if overlap:
                    pair_errors.append(
                        f"positive_negative_overlap:{query_id}:{category}:{sorted(overlap)}"
                    )
                if negative_ids:
                    negative_coverage[category] += 1
        for row in manifest:
            if row["active_handle"] in persistent_handles:
                semantic_errors.append(
                    f"persistent_active_handle:{row['chunk_id']}:{row['active_handle']}"
                )
            if row["target_valid"] and row["target_handle"] in persistent_handles:
                semantic_errors.append(
                    f"persistent_target_handle:{row['chunk_id']}:{row['target_handle']}"
                )
        episode_active_handles: dict[tuple[str, int], set[int]] = defaultdict(set)
        for row in manifest:
            if row.get("active_handle_source") == "contact_locked":
                episode_active_handles[(row["task"], row["episode"])].add(
                    row["active_handle"]
                )
        for (task, episode), handles in episode_active_handles.items():
            if len(handles) > 1:
                semantic_errors.append(
                    f"contact_lock_violation:{task}:episode{episode}:{sorted(handles)}"
                )
        report["splits"][split] = {
            "chunks": len(manifest),
            "pairs": len(pairs),
            "queries_with_positive": sum(bool(row["positive_ids"]) for row in pairs),
            "positive_count_histogram": {
                str(count): frequency
                for count, frequency in sorted(positive_counts.items())
            },
            "hard_negative_query_coverage": dict(sorted(negative_coverage.items())),
            "tasks": dict(sorted(Counter(row["task"] for row in manifest).items())),
            "phases": dict(sorted(Counter(row["phase"] for row in manifest).items())),
            "phase_sources": dict(
                sorted(Counter(row["phase_source"] for row in manifest).items())
            ),
            "target_valid": sum(bool(row["target_valid"]) for row in manifest),
            "effect_valid": sum(bool(row["effect_valid"]) for row in manifest),
            "active_confidence": _distribution(
                [float(row["active_confidence"]) for row in manifest]
            ),
            "target_confidence": _distribution(
                [
                    float(row["target_confidence"])
                    for row in manifest
                    if row["target_valid"]
                ]
            ),
            "array_errors": array_errors,
            "pair_errors": pair_errors,
            "semantic_errors": semantic_errors,
        }
    report["valid"] = not checksum_errors and all(
        not values["array_errors"]
        and not values["pair_errors"]
        and not values["semantic_errors"]
        for values in report["splits"].values()
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    report = inspect_dataset(args.root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

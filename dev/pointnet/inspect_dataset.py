"""审计 PointNet++ 预处理产物的 shape、有限值、标签覆盖与 shard checksum。"""

from __future__ import annotations

import argparse
from collections import Counter
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


def inspect_dataset(root: str | Path) -> dict[str, Any]:
    root = Path(root)
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
        report["splits"][split] = {
            "chunks": len(manifest),
            "pairs": len(pairs),
            "queries_with_positive": sum(bool(row["positive_ids"]) for row in pairs),
            "tasks": dict(sorted(Counter(row["task"] for row in manifest).items())),
            "phases": dict(sorted(Counter(row["phase"] for row in manifest).items())),
            "array_errors": array_errors,
        }
    report["valid"] = not checksum_errors and all(
        not values["array_errors"] for values in report["splits"].values()
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

"""从 ManiSkill trajectory archive 无损提取指定 episodes。"""

from __future__ import annotations

import argparse
import atexit
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

import h5py


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _episode_id(episode: dict[str, Any]) -> int:
    if "episode_id" not in episode:
        raise KeyError("episode metadata 缺少 episode_id")
    return int(episode["episode_id"])


def run(
    *,
    source_h5: Path,
    source_json: Path,
    episode_ids: list[int],
    output_root: Path,
) -> None:
    if output_root.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output_root}")
    if not episode_ids or len(set(episode_ids)) != len(episode_ids):
        raise ValueError("episode IDs 必须非空且唯一")
    metadata = json.loads(source_json.read_text(encoding="utf-8"))
    episodes = {
        _episode_id(episode): episode for episode in metadata.get("episodes", [])
    }
    missing = sorted(set(episode_ids) - episodes.keys())
    if missing:
        raise KeyError(f"metadata 缺少 episodes：{missing}")

    temporary = output_root.with_name(
        f".{output_root.name}.incomplete-{os.getpid()}"
    )
    temporary.mkdir(parents=True)

    def cleanup() -> None:
        shutil.rmtree(temporary, ignore_errors=True)

    atexit.register(cleanup)
    output_h5 = temporary / "trajectory.h5"
    output_json = temporary / "trajectory.json"
    with h5py.File(source_h5, "r") as source, h5py.File(output_h5, "w") as target:
        for episode_id in episode_ids:
            key = f"traj_{episode_id}"
            if key not in source:
                raise KeyError(f"HDF5 缺少 {key}")
            source.copy(key, target)

    subset_metadata = {
        **metadata,
        "episodes": [episodes[episode_id] for episode_id in episode_ids],
    }
    output_json.write_text(
        json.dumps(subset_metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "episode_ids": episode_ids,
        "episode_seeds": [
            int(episodes[episode_id]["reset_kwargs"]["seed"])
            for episode_id in episode_ids
        ],
        "source": {
            "h5": str(source_h5),
            "h5_sha256": _sha256(source_h5),
            "json": str(source_json),
            "json_sha256": _sha256(source_json),
        },
        "output": {
            "h5_sha256": _sha256(output_h5),
            "json_sha256": _sha256(output_json),
        },
    }
    (temporary / "subset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_root)
    atexit.unregister(cleanup)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument(
        "--episode-id",
        type=int,
        action="append",
        required=True,
        dest="episode_ids",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(
        source_h5=arguments.source_h5.resolve(),
        source_json=arguments.source_json.resolve(),
        episode_ids=arguments.episode_ids,
        output_root=arguments.output_root.resolve(),
    )

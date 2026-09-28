"""审计下载的 ManiSkill ZIP 中轨迹元数据和 HDF5 结构。"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
from typing import Any
import zipfile

import h5py
import numpy as np


DEFAULT_MEMBER_PATTERN = (
    r"/(motionplanning/trajectory|"
    r"rl/trajectory\.none\.pd_ee_delta_pose\.physx_cuda)\.(json|h5)$"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metadata_summary(payload: dict[str, Any]) -> dict[str, Any]:
    episodes = payload.get("episodes", [])
    elapsed = np.asarray(
        [episode.get("elapsed_steps", 0) for episode in episodes],
        dtype=np.int64,
    )
    successes = [bool(episode.get("success", False)) for episode in episodes]
    controllers = Counter(
        str(episode.get("control_mode", "unknown")) for episode in episodes
    )
    return {
        "env_info": payload.get("env_info", {}),
        "commit_info": payload.get("commit_info", {}),
        "num_episodes": len(episodes),
        "num_success": int(sum(successes)),
        "success_rate": float(np.mean(successes)) if successes else None,
        "elapsed_steps": {
            "min": int(elapsed.min()) if elapsed.size else None,
            "max": int(elapsed.max()) if elapsed.size else None,
            "mean": float(elapsed.mean()) if elapsed.size else None,
            "sum": int(elapsed.sum()) if elapsed.size else 0,
        },
        "control_modes": dict(sorted(controllers.items())),
    }


def _h5_group_signature(group: h5py.Group, depth: int = 0) -> dict[str, Any]:
    signature: dict[str, Any] = {}
    for key in sorted(group.keys()):
        value = group[key]
        if isinstance(value, h5py.Dataset):
            signature[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
        elif depth < 1:
            signature[key] = _h5_group_signature(value, depth + 1)
        else:
            signature[key] = {"type": "group", "num_children": len(value)}
    return signature


def _h5_summary(raw: bytes) -> dict[str, Any]:
    with h5py.File(io.BytesIO(raw), "r") as handle:
        trajectory_keys = sorted(key for key in handle if key.startswith("traj_"))
        example_keys = trajectory_keys[:2]
        return {
            "num_root_groups": len(handle),
            "num_trajectories": len(trajectory_keys),
            "example_trajectories": {
                key: _h5_group_signature(handle[key]) for key in example_keys
            },
        }


def run(archive: Path, output_path: Path, member_pattern: str) -> None:
    if output_path.exists():
        raise FileExistsError(f"输出已存在，拒绝覆盖：{output_path}")
    matcher = re.compile(member_pattern)
    selected: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive) as bundle:
        for index, member in enumerate(bundle.infolist()):
            if member.is_dir() or matcher.search(member.filename) is None:
                continue
            raw = bundle.read(member)
            record: dict[str, Any] = {
                "archive_index": index,
                "member": member.filename,
                "size": member.file_size,
                "compressed_size": member.compress_size,
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
            if member.filename.endswith(".json"):
                record["metadata"] = _metadata_summary(json.loads(raw))
            elif member.filename.endswith(".h5"):
                record["hdf5"] = _h5_summary(raw)
            selected.append(record)

    if not selected:
        raise ValueError(f"没有成员匹配正则：{member_pattern}")
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive": str(archive),
        "archive_size": archive.stat().st_size,
        "archive_sha256": _sha256(archive),
        "member_pattern": member_pattern,
        "selected_members": selected,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--member-pattern", default=DEFAULT_MEMBER_PATTERN)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    run(
        archive=arguments.archive.resolve(),
        output_path=arguments.output.resolve(),
        member_pattern=arguments.member_pattern,
    )


if __name__ == "__main__":
    main()

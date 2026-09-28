"""审计 ManiSkill canonical chunk 产物的完整性、范围与 split 隔离。"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from dev.predictor.canonical_geometry import GEOMETRY_DIM


EXPECTED_ARRAYS = {"active_points", "geometry", "actions", "target_frames"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _stats(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def run(root: Path, output_path: Path) -> None:
    if output_path.exists():
        raise FileExistsError(f"输出已存在，拒绝覆盖：{output_path}")
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config = summary["config"]
    horizon = int(config["horizon"])
    points_per_object = int(config["points_per_object"])
    action_representation = config.get(
        "action_representation", "cumulative_observation_delta"
    )
    position_limit_m = float(config.get("position_limit_m", 0.1))
    rotation_scale_rad = abs(float(config.get("rotation_scale_rad", -0.1)))

    split_reports: dict[str, Any] = {}
    split_episodes: dict[str, set[int]] = {}
    for split in ("train", "val"):
        manifest_path = root / f"manifest-{split}.jsonl"
        records = _read_jsonl(manifest_path)
        identifiers = [str(record["chunk_id"]) for record in records]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(f"{split} manifest 存在重复 chunk_id")
        split_episodes[split] = {int(record["episode"]) for record in records}
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            if record["split"] != split:
                raise ValueError(f"{split} manifest 含错误 split")
            grouped[str(record["shard"])].append(record)

        values: dict[str, list[np.ndarray]] = {
            name: [] for name in EXPECTED_ARRAYS
        }
        covered_rows = 0
        for relative, shard_records in sorted(grouped.items()):
            shard_path = root / relative
            with np.load(shard_path) as archive:
                if set(archive.files) != EXPECTED_ARRAYS:
                    raise ValueError(
                        f"{relative} arrays={archive.files}，预期={sorted(EXPECTED_ARRAYS)}"
                    )
                row_count = len(archive["geometry"])
                rows = [int(record["row"]) for record in shard_records]
                if sorted(rows) != list(range(row_count)):
                    raise ValueError(f"{relative} manifest row 不连续或未完全覆盖")
                for name in EXPECTED_ARRAYS:
                    values[name].append(np.asarray(archive[name]))
                covered_rows += row_count
        if covered_rows != len(records):
            raise ValueError(f"{split} shard rows 与 manifest 数量不一致")

        arrays = {
            name: np.concatenate(parts, axis=0)
            for name, parts in values.items()
        }
        expected_shapes = {
            "active_points": (len(records), points_per_object, 3),
            "geometry": (len(records), GEOMETRY_DIM),
            "actions": (len(records), horizon, 7),
            "target_frames": (len(records), horizon),
        }
        for name, expected in expected_shapes.items():
            if arrays[name].shape != expected:
                raise ValueError(
                    f"{split}/{name} shape={arrays[name].shape}，预期={expected}"
                )
            if not np.isfinite(arrays[name]).all():
                raise ValueError(f"{split}/{name} 含 NaN/Inf")

        frames = np.asarray([int(record["frame"]) for record in records])
        expected_targets = frames[:, None] + np.arange(1, horizon + 1)[None, :]
        if not np.array_equal(arrays["target_frames"], expected_targets):
            raise ValueError(f"{split} target_frames 与 H-step stride=1 不一致")
        gripper = arrays["actions"][..., 6]
        if action_representation == "normalized_controller_command":
            if np.max(np.abs(arrays["actions"])) > 1.0001:
                raise ValueError(f"{split} normalized controller action 超出 [-1,1]")
            rotation_command_norm = np.linalg.norm(
                arrays["actions"][..., 3:6], axis=-1
            )
            if np.max(rotation_command_norm) > 1.0001:
                raise ValueError(f"{split} normalized rotation command L2 norm 超过 1")
        elif action_representation in (
            "cumulative_observation_delta",
            "canonical_controller_command",
        ):
            if gripper.min() < 0.0 or gripper.max() > 1.0:
                raise ValueError(f"{split} action gripper 超出 [0,1]")
            if action_representation == "canonical_controller_command":
                translation_norm = np.linalg.norm(
                    arrays["actions"][..., :3], axis=-1
                )
                rotation_norm = np.linalg.norm(
                    arrays["actions"][..., 3:6], axis=-1
                )
                if translation_norm.max() > np.sqrt(3.0) * position_limit_m + 1e-5:
                    raise ValueError(f"{split} canonical translation command 超限")
                # XYZ Euler command 经 SO(3) 共轭后存为 axis-angle；小角度下允许
                # 2% 的参数化差异，但不允许出现控制尺度之外的大旋转。
                if rotation_norm.max() > 1.02 * rotation_scale_rad + 1e-5:
                    raise ValueError(f"{split} canonical rotation command 超限")
        else:
            raise ValueError(f"未知 action representation：{action_representation}")
        geometry_gripper = arrays["geometry"][:, 15]
        target_valid = arrays["geometry"][:, 16]
        if geometry_gripper.min() < 0.0 or geometry_gripper.max() > 1.0:
            raise ValueError(f"{split} geometry gripper 超出 [0,1]")
        if not np.all(target_valid == 1.0):
            raise ValueError(f"{split} target_valid 应全部为 1")

        translation = np.linalg.norm(arrays["actions"][..., :3], axis=-1)
        rotation = np.linalg.norm(arrays["actions"][..., 3:6], axis=-1)
        sampled_centers = np.linalg.norm(
            arrays["active_points"].mean(axis=1), axis=-1
        )
        action_stats = {
            (
                "translation_norm_normalized"
                if action_representation == "normalized_controller_command"
                else "translation_norm_m"
            ): _stats(translation),
            (
                "rotation_norm_normalized"
                if action_representation == "normalized_controller_command"
                else "rotation_norm_rad"
            ): _stats(rotation),
            (
                "action_gripper_normalized"
                if action_representation == "normalized_controller_command"
                else "gripper_open"
            ): {
                "min": float(gripper.min()),
                "max": float(gripper.max()),
                "mean": float(gripper.mean()),
            },
        }
        split_reports[split] = {
            "chunks": len(records),
            "episodes": len(split_episodes[split]),
            "manifest_sha256": _sha256(manifest_path),
            "shards": len(grouped),
            "active_points_dtype": str(arrays["active_points"].dtype),
            "geometry_dtype": str(arrays["geometry"].dtype),
            "actions_dtype": str(arrays["actions"].dtype),
            "sampled_point_center_norm_m": _stats(sampled_centers),
            **action_stats,
            "geometry_abs_max": float(np.abs(arrays["geometry"]).max()),
        }

    overlap = split_episodes["train"] & split_episodes["val"]
    if overlap:
        raise ValueError(f"episode 跨 train/val：{sorted(overlap)}")
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "summary_sha256": _sha256(summary_path),
        "source_h5_sha256": summary["source"]["h5_sha256"],
        "array_allowlist": sorted(EXPECTED_ARRAYS),
        "privileged_state_in_shards": False,
        "action_representation": action_representation,
        "episode_overlap": [],
        "splits": split_reports,
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
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    run(arguments.root.resolve(), arguments.output.resolve())


if __name__ == "__main__":
    main()

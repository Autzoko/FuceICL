"""从 RLBench low-dimensional trajectories 生成规范 H-step action chunks。"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping
import zipfile

import numpy as np

from dev.pointnet.dataset import PointNetContextStore
from dev.predictor.action_chunk_data import (
    ActionChunkConfig,
    build_cumulative_action_chunk,
    quaternion_xyzw_to_matrix,
)
from dev.retriever.rlbench_adapter import RLBenchArchiveAdapter


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


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


class _ShardWriter:
    """按固定尺寸原子产出压缩 action shards。"""

    def __init__(self, root: Path, split: str, shard_size: int) -> None:
        self.root = root
        self.split = split
        self.shard_size = shard_size
        self.pending: list[tuple[dict[str, Any], dict[str, np.ndarray]]] = []
        self.records: list[dict[str, Any]] = []
        self.checksums: dict[str, str] = {}
        self.index = 0

    def add(
        self,
        metadata: dict[str, Any],
        arrays: dict[str, np.ndarray],
    ) -> None:
        self.pending.append((metadata, arrays))
        if len(self.pending) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        relative = f"shards/{self.split}-{self.index:05d}.npz"
        path = self.root / relative
        names = tuple(self.pending[0][1])
        np.savez_compressed(
            path,
            **{
                name: np.stack([arrays[name] for _, arrays in self.pending])
                for name in names
            },
        )
        self.checksums[relative] = _sha256(path)
        for row, (metadata, _) in enumerate(self.pending):
            self.records.append({**metadata, "shard": relative, "row": row})
        self.pending.clear()
        self.index += 1

    def close(self) -> tuple[list[dict[str, Any]], dict[str, str]]:
        self.flush()
        return self.records, self.checksums


def _episode_actions(
    *,
    raw_root: Path,
    records: list[dict[str, Any]],
    config: ActionChunkConfig,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """每个 episode 只反序列化一次，返回动作和初始旋转。"""
    adapter = RLBenchArchiveAdapter()
    grouped: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        grouped[str(record["task"])][int(record["episode"])].append(record)

    values: dict[
        str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = {}
    for task in sorted(grouped):
        archive_path = raw_root / f"{task}.zip"
        if not archive_path.is_file():
            raise FileNotFoundError(f"缺少 RLBench task archive：{archive_path}")
        with zipfile.ZipFile(archive_path) as archive:
            member_by_episode = dict(adapter._episode_members(archive))
            for episode, episode_records in sorted(grouped[task].items()):
                try:
                    member = member_by_episode[episode]
                except KeyError as error:
                    raise KeyError(
                        f"{archive_path.name} 缺少 episode{episode}"
                    ) from error
                demo, _, _, _ = adapter._read_episode(archive, member)
                for record in episode_records:
                    frame = int(record["frame"])
                    actions, valid_mask, target_frames = (
                        build_cumulative_action_chunk(
                            demo,
                            frame,
                            horizon=config.horizon,
                            frame_stride=config.frame_stride,
                        )
                    )
                    current_pose = np.asarray(
                        demo[frame].gripper_pose, dtype=np.float64
                    )
                    values[str(record["chunk_id"])] = (
                        actions,
                        valid_mask,
                        target_frames,
                        quaternion_xyzw_to_matrix(current_pose[3:7]),
                    )
    if len(values) != len(records):
        raise RuntimeError("action chunk 数量与 parent manifest 不一致")
    return values


def _preprocess_split(
    *,
    raw_data_root: Path,
    parent_root: Path,
    output_root: Path,
    split: str,
    config: ActionChunkConfig,
) -> tuple[dict[str, Any], dict[str, str]]:
    parent_manifest = parent_root / f"manifest-{split}.jsonl"
    parent_records = _read_jsonl(parent_manifest)
    generated = _episode_actions(
        raw_root=raw_data_root / split,
        records=parent_records,
        config=config,
    )
    parent_store = PointNetContextStore(parent_root, split, cache_size=4)
    writer = _ShardWriter(output_root, split, config.shard_size)
    valid_steps = Counter()
    translation_norms = []
    rotation_norms = []
    gripper_targets = Counter()
    aggregate_translation_errors = []
    aggregate_rotation_errors = []

    for record in parent_records:
        identifier = str(record["chunk_id"])
        actions, valid_mask, target_frames, current_rotation = generated[identifier]
        valid_count = int(valid_mask.sum())
        valid_steps[valid_count] += 1
        translation_norms.extend(
            np.linalg.norm(actions[valid_mask, :3], axis=1).tolist()
        )
        rotation_norms.extend(
            np.linalg.norm(actions[valid_mask, 3:6], axis=1).tolist()
        )
        gripper_targets.update(actions[valid_mask, 6].round(3).tolist())

        # H*stride 等于 parent horizon 时，应复现既有聚合监督。
        if int(record["future_frame"]) == int(target_frames[-1]):
            parent_target = parent_store.get(identifier)["future_target"]
            aggregate_translation_errors.append(
                float(
                    np.linalg.norm(
                        current_rotation @ actions[-1, :3] - parent_target[:3]
                    )
                )
            )
            aggregate_rotation_errors.append(
                float(
                    np.linalg.norm(
                        current_rotation @ actions[-1, 3:6]
                        - parent_target[3:6]
                    )
                )
            )

        writer.add(
            {
                "chunk_id": identifier,
                "split": split,
                "task": str(record["task"]),
                "episode": int(record["episode"]),
                "frame": int(record["frame"]),
                "valid_steps": valid_count,
            },
            {
                "actions": actions,
                "valid_mask": valid_mask,
                "target_frames": target_frames,
            },
        )

    output_records, checksums = writer.close()
    manifest_path = output_root / f"manifest-{split}.jsonl"
    _write_jsonl(manifest_path, output_records)
    checksums[manifest_path.name] = _sha256(manifest_path)
    expected_ids = [str(record["chunk_id"]) for record in parent_records]
    actual_ids = [str(record["chunk_id"]) for record in output_records]
    if actual_ids != expected_ids:
        raise RuntimeError(f"{split} action manifest 未保持 parent 顺序")

    translation = np.asarray(translation_norms, dtype=np.float64)
    rotation = np.asarray(rotation_norms, dtype=np.float64)
    return (
        {
            "chunks": len(output_records),
            "parent_manifest_sha256": _sha256(parent_manifest),
            "valid_steps_histogram": {
                str(key): value for key, value in sorted(valid_steps.items())
            },
            "terminal_padded_chunks": sum(
                count for steps, count in valid_steps.items() if steps < config.horizon
            ),
            "translation_norm_m": {
                "mean": float(translation.mean()),
                "p95": float(np.quantile(translation, 0.95)),
                "max": float(translation.max()),
            },
            "rotation_norm_rad": {
                "mean": float(rotation.mean()),
                "p95": float(np.quantile(rotation, 0.95)),
                "max": float(rotation.max()),
            },
            "gripper_target_counts": {
                str(key): value for key, value in sorted(gripper_targets.items())
            },
            "parent_aggregate_consistency": {
                "checked": len(aggregate_translation_errors),
                "max_translation_error_m": max(aggregate_translation_errors),
                "max_rotation_error_rad": max(aggregate_rotation_errors),
            },
        },
        checksums,
    )


def run_preprocessing(
    *,
    raw_data_root: Path,
    parent_root: Path,
    output_root: Path,
    config: ActionChunkConfig,
) -> dict[str, Any]:
    """原子生成 train/val action chunks 及完整 provenance。"""
    if output_root.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output_root}")
    parent_summary_path = parent_root / "summary.json"
    parent_summary = json.loads(parent_summary_path.read_text(encoding="utf-8"))
    temporary = output_root.with_name(
        f".{output_root.name}.incomplete-{os.getpid()}"
    )
    temporary.mkdir(parents=True)
    temporary.joinpath("shards").mkdir()
    checksums: dict[str, str] = {}
    summary: dict[str, Any] = {
        "schema_version": config.schema_version,
        "config": asdict(config),
        "action_representation": {
            "frame": "query_eef",
            "pose": "cumulative_from_query",
            "dimensions": [
                "translation_xyz_m",
                "rotation_axis_angle_rad",
                "gripper_open_target",
            ],
            "padding": "repeat_terminal_with_false_valid_mask",
        },
        "parent": {
            "root": str(parent_root),
            "schema_version": parent_summary["schema_version"],
            "summary_sha256": _sha256(parent_summary_path),
        },
        "splits": {},
    }
    try:
        for split in ("train", "val"):
            split_summary, split_checksums = _preprocess_split(
                raw_data_root=raw_data_root,
                parent_root=parent_root,
                output_root=temporary,
                split=split,
                config=config,
            )
            summary["splits"][split] = split_summary
            checksums.update(split_checksums)
            print(json.dumps({split: split_summary}, ensure_ascii=False), flush=True)
        summary_path = temporary / "summary.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        checksums[summary_path.name] = _sha256(summary_path)
        checksum_path = temporary / "SHA256SUMS"
        checksum_path.write_text(
            "".join(
                f"{digest}  {name}\n"
                for name, digest in sorted(checksums.items())
            ),
            encoding="utf-8",
        )
        temporary.rename(output_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data-root", type=Path, required=True)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_preprocessing(
        raw_data_root=args.raw_data_root,
        parent_root=args.parent_root,
        output_root=args.output_root,
        config=ActionChunkConfig.from_json(args.config),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

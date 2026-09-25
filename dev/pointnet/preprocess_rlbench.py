"""从 RLBench ZIP 构建 Tiny PointNet++ 检索训练数据。

输入只由当前/历史可观测量组成；未来 EEF/object effect 仅写入监督标签。输出为
sharded NPZ + JSONL manifest/pair labels，适合在 JUBAIL 上流式生成而无需完整解压。
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping, Sequence
import zipfile

import numpy as np

from dev.retriever.rlbench_adapter import (
    AdapterConfig,
    RLBenchArchiveAdapter,
    quaternion_xyzw_to_rotation_6d,
)


PHASES = ("approach", "contact", "manipulate", "finish")
PHASE_TO_ID = {name: index for index, name in enumerate(PHASES)}


@dataclass(frozen=True)
class EventAnchor:
    """一个由机器人事件定位的 chunk 起点。"""

    phase: str
    frame: int
    source: str

    def __post_init__(self) -> None:
        if self.phase not in PHASE_TO_ID:
            raise ValueError(f"未知 phase：{self.phase}")
        if self.frame < 0:
            raise ValueError("frame 不能为负数")


@dataclass(frozen=True)
class PointNetPreprocessConfig:
    """可复现的 RLBench pilot 预处理配置。"""

    schema_version: str
    seed: int
    cameras: tuple[str, ...]
    point_count: int
    future_horizon_frames: int
    min_segment_pixels: int
    max_segment_extent_m: float
    robot_handle_radius_m: float
    robot_handle_spread_m: float
    robot_calibration_frames: int
    shared_handle_task_fraction: float
    shared_handle_episodes: int
    min_active_confidence: float
    min_target_confidence: float
    shard_size: int
    episodes_per_task: dict[str, int]
    max_chunks: dict[str, int]
    tasks: tuple[str, ...] = ()

    @classmethod
    def from_json(cls, path: str | Path) -> "PointNetPreprocessConfig":
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        values["cameras"] = tuple(values["cameras"])
        values["tasks"] = tuple(values.get("tasks", ()))
        return cls(**values)

    def __post_init__(self) -> None:
        if not self.schema_version.strip():
            raise ValueError("schema_version 不能为空")
        integer_values = (
            self.point_count,
            self.future_horizon_frames,
            self.min_segment_pixels,
            self.shard_size,
            self.robot_calibration_frames,
            self.shared_handle_episodes,
        )
        if min(integer_values) <= 0:
            raise ValueError("点数、horizon、像素阈值和 shard_size 必须为正")
        confidences = (self.min_active_confidence, self.min_target_confidence)
        if any(not 0.0 <= value <= 1.0 for value in confidences):
            raise ValueError("confidence 阈值必须位于 [0, 1]")
        if not self.cameras:
            raise ValueError("至少需要一个相机")
        if self.robot_handle_radius_m <= 0 or self.robot_handle_spread_m <= 0:
            raise ValueError("robot handle 半径和轨迹离散阈值必须为正")
        if not 0.0 < self.shared_handle_task_fraction <= 1.0:
            raise ValueError("shared_handle_task_fraction 必须位于 (0, 1]")
        for mapping in (self.episodes_per_task, self.max_chunks):
            if not {"train", "val"}.issubset(mapping):
                raise ValueError("episodes_per_task/max_chunks 必须包含 train 和 val")
            if min(mapping.values()) <= 0:
                raise ValueError("split 限额必须为正")


@dataclass
class _Chunk:
    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]


def detect_event_anchors(gripper_open: Sequence[float]) -> tuple[EventAnchor, ...]:
    """优先用夹爪事件定位 phase，无事件时使用明确标记的时间回退。"""
    values = np.asarray(gripper_open, dtype=np.float32)
    if values.ndim != 1 or not len(values):
        raise ValueError("gripper_open 必须是一维非空序列")
    last = len(values) - 1
    closed = values < 0.5
    close_events = np.flatnonzero((~closed[:-1]) & closed[1:]) + 1
    open_events = np.flatnonzero(closed[:-1] & (~closed[1:])) + 1

    if len(close_events):
        contact = int(close_events[0])
        later_open = open_events[open_events > contact]
        release = int(later_open[0]) if len(later_open) else last
        lead = max(3, int(round(0.08 * max(1, last))))
        tail = max(2, int(round(0.04 * max(1, last))))
        anchors = (
            EventAnchor("approach", max(0, contact - lead), "gripper_event"),
            EventAnchor("contact", contact, "gripper_event"),
            EventAnchor(
                "manipulate",
                min(last, contact + max(1, (release - contact) // 2)),
                "gripper_event",
            ),
            EventAnchor("finish", min(last, release + tail), "gripper_event"),
        )
    else:
        anchors = tuple(
            EventAnchor(phase, int(round(last * fraction)), "time_fallback")
            for phase, fraction in zip(PHASES, (0.25, 0.50, 0.75, 0.90), strict=True)
        )

    # 极短 episode 可能得到重复 frame；保留更靠后的 phase 并维持时间顺序。
    by_frame = {anchor.frame: anchor for anchor in anchors}
    return tuple(by_frame[index] for index in sorted(by_frame))


def _quaternion_conjugate(quaternion: np.ndarray) -> np.ndarray:
    result = quaternion.copy()
    result[:3] *= -1.0
    return result


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return np.array(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ],
        dtype=np.float64,
    )


def _relative_axis_angle(current_xyzw: np.ndarray, future_xyzw: np.ndarray) -> np.ndarray:
    current = current_xyzw / max(np.linalg.norm(current_xyzw), 1e-12)
    future = future_xyzw / max(np.linalg.norm(future_xyzw), 1e-12)
    relative = _quaternion_multiply(future, _quaternion_conjugate(current))
    if relative[3] < 0:
        relative *= -1.0
    vector_norm = np.linalg.norm(relative[:3])
    if vector_norm <= 1e-8:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * math.atan2(vector_norm, np.clip(relative[3], -1.0, 1.0))
    return (relative[:3] / vector_norm * angle).astype(np.float32)


def _cosine_direction(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-8:
        return 0.5
    return float(np.clip(0.5 * (np.dot(first, second) / denominator + 1.0), 0.0, 1.0))


class RLBenchPointNetPreprocessor:
    """生成 PointNet++ 输入、future-effect 标签和可审计 metadata。"""

    def __init__(self, config: PointNetPreprocessConfig) -> None:
        self.config = config
        self.adapter = RLBenchArchiveAdapter(
            AdapterConfig(
                cameras=config.cameras,
                point_count=config.point_count,
                min_segment_pixels=config.min_segment_pixels,
                max_segment_extent_m=config.max_segment_extent_m,
                robot_handle_radius_m=config.robot_handle_radius_m,
            )
        )

    @staticmethod
    def _rotation_matrix(quaternion_xyzw: Sequence[float]) -> np.ndarray:
        """将 xyzw 四元数转为旋转矩阵。"""
        x, y, z, w = np.asarray(quaternion_xyzw, dtype=np.float64)
        norm = np.linalg.norm((x, y, z, w))
        if norm <= 1e-12:
            raise ValueError("四元数退化")
        x, y, z, w = np.asarray((x, y, z, w)) / norm
        return np.asarray(
            [
                [
                    1 - 2 * (y * y + z * z),
                    2 * (x * y - z * w),
                    2 * (x * z + y * w),
                ],
                [
                    2 * (x * y + z * w),
                    1 - 2 * (x * x + z * z),
                    2 * (y * z - x * w),
                ],
                [
                    2 * (x * z - y * w),
                    2 * (y * z + x * w),
                    1 - 2 * (x * x + y * y),
                ],
            ],
            dtype=np.float64,
        )

    def _robot_handles(
        self,
        demo: Any,
        calibration_frames: Sequence[int],
        frame_segments: Mapping[int, Mapping[int, Any]],
    ) -> set[int]:
        """用接触前多帧轨迹识别跟随 EEF 刚性运动的机器人 segment。

        位置先转到 EEF 局部系；机器人 link 的局部位置在多帧间稳定，
        而作业物体在接触前不会刚性跟随 EEF。不依赖任务或 handle ID 硬编码。
        """
        local_tracks: dict[int, list[np.ndarray]] = defaultdict(list)
        for frame in calibration_frames:
            pose = np.asarray(demo[frame].gripper_pose, dtype=np.float64)
            rotation = self._rotation_matrix(pose[3:7])
            for handle, segment in frame_segments[frame].items():
                local = rotation.T @ (segment.center - pose[:3])
                local_tracks[handle].append(local)

        minimum_observations = max(2, math.ceil(0.5 * len(calibration_frames)))
        robot_handles = set()
        for handle, track in local_tracks.items():
            if len(track) < minimum_observations:
                continue
            values = np.stack(track)
            median = np.median(values, axis=0)
            distance = float(np.linalg.norm(median))
            spread = float(
                np.quantile(np.linalg.norm(values - median, axis=1), 0.90)
            )
            if (
                distance <= self.config.robot_handle_radius_m
                and spread <= self.config.robot_handle_spread_m
            ):
                robot_handles.add(handle)
        return robot_handles

    def _calibration_frames(
        self,
        anchors: Sequence[EventAnchor],
    ) -> list[int]:
        contact_frame = next(
            (anchor.frame for anchor in anchors if anchor.phase == "contact"),
            anchors[0].frame,
        )
        count = min(self.config.robot_calibration_frames, contact_frame + 1)
        return sorted(
            set(
                np.linspace(
                    0,
                    contact_frame,
                    count,
                    dtype=np.int64,
                ).tolist()
            )
        )

    def discover_persistent_handles(
        self,
        data_root: Path,
        *,
        split: str = "train",
    ) -> tuple[set[int], dict[int, int]]:
        """从多任务 mask 中自动发现 Panda/公共场景 handle。

        RLBench 先加载机器人再加载 task scene，因此机器人和公共场景
        handle 会跨大多数任务稳定出现。阈值由任务占比定义，不写死 ID。
        """
        tasks = _available_tasks(data_root, split, ())
        task_counts: Counter = Counter()
        for task in tasks:
            with zipfile.ZipFile(data_root / split / f"{task}.zip") as archive:
                members = self.adapter._episode_members(archive)
                if not members:
                    continue
                visible_handles = set()
                for _, member in members[: self.config.shared_handle_episodes]:
                    demo, _, _, prefix = self.adapter._read_episode(archive, member)
                    anchors = detect_event_anchors(
                        [float(observation.gripper_open) for observation in demo]
                    )
                    for frame in self._calibration_frames(anchors):
                        visible_handles.update(
                            self.adapter._frame_segments(
                                archive,
                                prefix,
                                demo[frame],
                                frame,
                            )
                        )
                task_counts.update(visible_handles)
        minimum_tasks = math.ceil(
            self.config.shared_handle_task_fraction * len(tasks)
        )
        persistent = {
            handle
            for handle, task_count in task_counts.items()
            if task_count >= minimum_tasks
        }
        return persistent, dict(task_counts)

    def _active_candidate(
        self,
        current_segments: Mapping[int, Any],
        future_segments: Mapping[int, Any],
        robot_handles: set[int],
        current_eef: np.ndarray,
        future_eef: np.ndarray,
        phase: str,
        preferred_handle: int | None = None,
    ) -> tuple[int, float] | None:
        scored = []
        eef_motion = future_eef - current_eef
        for handle, segment in current_segments.items():
            if handle in robot_handles:
                continue
            if preferred_handle is not None and handle != preferred_handle:
                continue
            distance = float(np.linalg.norm(segment.center - current_eef))
            distance_score = math.exp(-distance / 0.18)
            future = future_segments.get(handle)
            if future is None:
                motion_score = 0.0
                coupling_score = 0.5
            else:
                object_motion = future.center - segment.center
                motion_score = min(1.0, float(np.linalg.norm(object_motion)) / 0.05)
                coupling_score = _cosine_direction(object_motion, eef_motion)
            if phase in {"approach", "contact"}:
                confidence = 0.75 * distance_score + 0.15 * motion_score + 0.10 * coupling_score
            else:
                confidence = 0.50 * distance_score + 0.25 * motion_score + 0.25 * coupling_score
            scored.append((confidence, -distance, handle))
        if not scored:
            return None
        confidence, _, handle = max(scored)
        return handle, float(confidence)

    def _target_candidate(
        self,
        active_handle: int,
        current_segments: Mapping[int, Any],
        future_segments: Mapping[int, Any],
        robot_handles: set[int],
    ) -> tuple[int, float] | None:
        active_future = future_segments.get(active_handle)
        active_current = current_segments.get(active_handle)
        if active_future is None or active_current is None:
            return None
        active_motion = float(np.linalg.norm(active_future.center - active_current.center))
        if active_motion < 0.015:
            return None

        scored = []
        for handle, segment in current_segments.items():
            if handle == active_handle or handle in robot_handles:
                continue
            future = future_segments.get(handle)
            if future is None:
                continue
            target_motion = float(np.linalg.norm(future.center - segment.center))
            final_distance = float(np.linalg.norm(active_future.center - future.center))
            if target_motion > 0.02 or not 0.025 <= final_distance <= 0.30:
                continue
            stability = math.exp(-target_motion / 0.01)
            proximity = math.exp(-final_distance / 0.15)
            confidence = 0.55 * stability + 0.45 * proximity
            scored.append((confidence, -final_distance, handle))
        if not scored:
            return None
        confidence, _, handle = max(scored)
        if confidence < self.config.min_target_confidence:
            return None
        return handle, float(confidence)

    def _make_chunk(
        self,
        *,
        split: str,
        task: str,
        episode: int,
        demo: Any,
        text: str,
        variation: int,
        anchor: EventAnchor,
        robot_handles: set[int],
        persistent_handles: set[int],
        frame_segments: Mapping[int, Mapping[int, Any]],
        episode_active_handle: int | None,
    ) -> _Chunk | None:
        frame = anchor.frame
        future_frame = min(len(demo) - 1, frame + self.config.future_horizon_frames)
        current_segments = frame_segments[frame]
        future_segments = frame_segments[future_frame]
        current_pose = np.asarray(demo[frame].gripper_pose, dtype=np.float64)
        future_pose = np.asarray(demo[future_frame].gripper_pose, dtype=np.float64)
        selected = self._active_candidate(
            current_segments,
            future_segments,
            robot_handles,
            current_pose[:3],
            future_pose[:3],
            anchor.phase,
            episode_active_handle,
        )
        if selected is None:
            return None
        active_handle, active_confidence = selected
        if active_confidence < self.config.min_active_confidence:
            return None

        active = current_segments[active_handle]
        active_points = self.adapter._sample_points(
            active.points,
            self.config.point_count,
        ) - active.center
        target_selection = self._target_candidate(
            active_handle,
            current_segments,
            future_segments,
            robot_handles,
        )
        target_valid = target_selection is not None
        if target_selection is None:
            target_handle = -1
            target_confidence = 0.0
            target_points = np.zeros_like(active_points)
            target_center = np.zeros(3, dtype=np.float32)
            target_extent = np.zeros(3, dtype=np.float32)
        else:
            target_handle, target_confidence = target_selection
            target = current_segments[target_handle]
            target_points = self.adapter._sample_points(
                target.points,
                self.config.point_count,
            ) - target.center
            target_center = target.center.astype(np.float32)
            target_extent = target.extent.astype(np.float32)

        gripper_joints = np.asarray(demo[frame].gripper_joint_positions, dtype=np.float32)
        future_gripper_joints = np.asarray(
            demo[future_frame].gripper_joint_positions,
            dtype=np.float32,
        )
        gripper_width = float(np.abs(gripper_joints).sum())
        future_gripper_width = float(np.abs(future_gripper_joints).sum())
        rotation_6d = quaternion_xyzw_to_rotation_6d(current_pose[3:7])
        eef_relative = np.concatenate((current_pose[:3] - active.center, rotation_6d))

        active_future = future_segments.get(active_handle)
        effect_valid = active_future is not None
        effect_translation = (
            active_future.center - active.center
            if effect_valid
            else np.zeros(3, dtype=np.float64)
        )
        future_translation = future_pose[:3] - current_pose[:3]
        future_rotation = _relative_axis_angle(current_pose[3:7], future_pose[3:7])
        chunk_id = f"{split}:{task}:episode{episode}:frame{frame}"
        metadata = {
            "chunk_id": chunk_id,
            "split": split,
            "task": task,
            "episode": episode,
            "variation": variation,
            "frame": frame,
            "future_frame": future_frame,
            "phase": anchor.phase,
            "phase_source": anchor.source,
            "text": text,
            "active_handle": int(active_handle),
            "active_handle_source": (
                "contact_locked"
                if episode_active_handle is not None
                else "per_frame"
            ),
            "target_handle": int(target_handle),
            "active_confidence": active_confidence,
            "target_confidence": target_confidence,
            "target_valid": target_valid,
            "effect_valid": effect_valid,
            "robot_handle_count": len(robot_handles),
            "persistent_handle_count": len(persistent_handles),
            "robot_filter_source": "cross_task_persistent+precontact_eef_rigid",
            "gripper_state": "open" if float(demo[frame].gripper_open) >= 0.5 else "closed",
            "active_center": active.center.astype(float).tolist(),
            "active_extent": active.extent.astype(float).tolist(),
            "eef_relative_position": eef_relative[:3].astype(float).tolist(),
            "future_translation": future_translation.astype(float).tolist(),
            "future_rotation_axis_angle": future_rotation.astype(float).tolist(),
            "effect_translation": effect_translation.astype(float).tolist(),
        }
        arrays = {
            "active_points": active_points.astype(np.float32),
            "target_points": target_points.astype(np.float32),
            "active_center": active.center.astype(np.float32),
            "active_extent": active.extent.astype(np.float32),
            "target_center": target_center,
            "target_extent": target_extent,
            "eef_relative_active": eef_relative.astype(np.float32),
            "eef_velocity": np.asarray(self.adapter._velocity(demo, frame), dtype=np.float32),
            "gripper_width": np.asarray(gripper_width, dtype=np.float32),
            "target_valid": np.asarray(target_valid, dtype=np.bool_),
            "phase_id": np.asarray(PHASE_TO_ID[anchor.phase], dtype=np.int8),
            "active_confidence": np.asarray(active_confidence, dtype=np.float32),
            "future_translation": future_translation.astype(np.float32),
            "future_rotation_axis_angle": future_rotation.astype(np.float32),
            "future_gripper_delta": np.asarray(
                future_gripper_width - gripper_width,
                dtype=np.float32,
            ),
            "effect_translation": effect_translation.astype(np.float32),
            "effect_valid": np.asarray(effect_valid, dtype=np.bool_),
        }
        return _Chunk(metadata=metadata, arrays=arrays)

    def extract_episode(
        self,
        archive: zipfile.ZipFile,
        *,
        split: str,
        task: str,
        episode: int,
        low_dim_member: str,
        persistent_handles: set[int],
    ) -> list[_Chunk]:
        demo, descriptions, variation, prefix = self.adapter._read_episode(
            archive,
            low_dim_member,
        )
        anchors = detect_event_anchors(
            [float(observation.gripper_open) for observation in demo]
        )
        calibration_frames = self._calibration_frames(anchors)
        required_frames = set(calibration_frames)
        for anchor in anchors:
            required_frames.add(anchor.frame)
            required_frames.add(
                min(len(demo) - 1, anchor.frame + self.config.future_horizon_frames)
            )
        frame_segments = {
            frame: self.adapter._frame_segments(
                archive,
                prefix,
                demo[frame],
                frame,
            )
            for frame in sorted(required_frames)
        }
        robot_handles = persistent_handles | self._robot_handles(
            demo, calibration_frames, frame_segments
        )
        episode_active_handle = None
        contact_anchor = next(
            (
                anchor
                for anchor in anchors
                if anchor.phase == "contact" and anchor.source == "gripper_event"
            ),
            None,
        )
        if contact_anchor is not None:
            frame = contact_anchor.frame
            future_frame = min(
                len(demo) - 1,
                frame + self.config.future_horizon_frames,
            )
            current_pose = np.asarray(demo[frame].gripper_pose, dtype=np.float64)
            future_pose = np.asarray(
                demo[future_frame].gripper_pose,
                dtype=np.float64,
            )
            selected = self._active_candidate(
                frame_segments[frame],
                frame_segments[future_frame],
                robot_handles,
                current_pose[:3],
                future_pose[:3],
                contact_anchor.phase,
            )
            if selected is not None:
                episode_active_handle = selected[0]
        chunks = []
        for anchor in anchors:
            chunk = self._make_chunk(
                split=split,
                task=task,
                episode=episode,
                demo=demo,
                text=descriptions[0],
                variation=variation,
                anchor=anchor,
                robot_handles=robot_handles,
                persistent_handles=persistent_handles,
                frame_segments=frame_segments,
                episode_active_handle=episode_active_handle,
            )
            if chunk is not None:
                chunks.append(chunk)
        return chunks


class _ShardWriter:
    def __init__(self, root: Path, split: str, shard_size: int) -> None:
        self.root = root
        self.split = split
        self.shard_size = shard_size
        self.pending: list[_Chunk] = []
        self.metadata: list[dict[str, Any]] = []
        self.shard_index = 0
        self.checksums: dict[str, str] = {}

    def add(self, chunk: _Chunk) -> None:
        self.pending.append(chunk)
        if len(self.pending) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        name = f"{self.split}-{self.shard_index:05d}.npz"
        path = self.root / "shards" / name
        arrays = {
            key: np.stack([chunk.arrays[key] for chunk in self.pending])
            for key in self.pending[0].arrays
        }
        np.savez(path, **arrays)
        self.checksums[f"shards/{name}"] = _sha256(path)
        for row, chunk in enumerate(self.pending):
            self.metadata.append(
                {
                    **chunk.metadata,
                    "shard": f"shards/{name}",
                    "row": row,
                }
            )
        self.pending.clear()
        self.shard_index += 1

    def close(self) -> list[dict[str, Any]]:
        self.flush()
        return self.metadata


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _vector_distance(first: Sequence[float], second: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(first) - np.asarray(second)))


def build_pair_labels(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """用跨 episode future action/effect 构造 positives 与互斥 hard negatives。"""
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_task[str(record["task"])].append(record)

    labels = []
    for query in records:
        same_task = [
            item
            for item in by_task[str(query["task"])]
            if item["episode"] != query["episode"]
        ]

        def action_distance(candidate: Mapping[str, Any]) -> float:
            translation = _vector_distance(
                query["future_translation"],
                candidate["future_translation"],
            )
            rotation = _vector_distance(
                query["future_rotation_axis_angle"],
                candidate["future_rotation_axis_angle"],
            )
            effect = 0.0
            if query["effect_valid"] and candidate["effect_valid"]:
                effect = _vector_distance(
                    query["effect_translation"],
                    candidate["effect_translation"],
                )
            return translation + 0.15 * rotation + effect

        def layout_compatible(candidate: Mapping[str, Any]) -> bool:
            return (
                _vector_distance(query["active_center"], candidate["active_center"])
                <= 0.25
                and _vector_distance(
                    query["eef_relative_position"],
                    candidate["eef_relative_position"],
                )
                <= 0.18
            )

        positives = [
            item
            for item in same_task
            if item["phase"] == query["phase"]
            and item["gripper_state"] == query["gripper_state"]
            and action_distance(item) <= 0.20
            and layout_compatible(item)
        ]
        positives.sort(key=action_distance)

        wrong_phase = [item for item in same_task if item["phase"] != query["phase"]]
        wrong_phase.sort(key=action_distance)
        wrong_gripper = [
            item
            for item in same_task
            if item["phase"] == query["phase"]
            and item["gripper_state"] != query["gripper_state"]
        ]
        wrong_gripper.sort(key=action_distance)
        wrong_layout = [
            item
            for item in same_task
            if item["phase"] == query["phase"]
            and item["gripper_state"] == query["gripper_state"]
            and not layout_compatible(item)
        ]
        wrong_layout.sort(key=action_distance)

        geometry_collisions = [
            item
            for item in records
            if item["task"] != query["task"] and item["phase"] == query["phase"]
        ]
        geometry_collisions.sort(
            key=lambda item: _vector_distance(
                query["active_extent"],
                item["active_extent"],
            )
        )
        labels.append(
            {
                "query_id": query["chunk_id"],
                "positive_ids": [item["chunk_id"] for item in positives[:4]],
                "hard_negatives": {
                    "wrong_phase": [item["chunk_id"] for item in wrong_phase[:8]],
                    "wrong_gripper": [item["chunk_id"] for item in wrong_gripper[:8]],
                    "wrong_layout": [item["chunk_id"] for item in wrong_layout[:8]],
                    "geometry_collision": [
                        item["chunk_id"] for item in geometry_collisions[:8]
                    ],
                },
            }
        )
    return labels


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _available_tasks(root: Path, split: str, selected: Sequence[str]) -> list[str]:
    tasks = sorted(path.stem for path in (root / split).glob("*.zip"))
    if selected:
        missing = sorted(set(selected) - set(tasks))
        if missing:
            raise FileNotFoundError(f"{split} 缺少 task archives：{missing}")
        return list(selected)
    if not tasks:
        raise FileNotFoundError(f"{root / split} 没有 task ZIP")
    return tasks


def _episode_schedule(
    archives: Mapping[str, zipfile.ZipFile],
    adapter: RLBenchArchiveAdapter,
    episode_limit: int,
) -> Iterable[tuple[str, int, str]]:
    members = {
        task: adapter._episode_members(archive)[:episode_limit]
        for task, archive in archives.items()
    }
    maximum = max(len(values) for values in members.values())
    for episode_offset in range(maximum):
        for task in sorted(members):
            if episode_offset < len(members[task]):
                episode, member = members[task][episode_offset]
                yield task, episode, member


def preprocess_split(
    preprocessor: RLBenchPointNetPreprocessor,
    root: Path,
    output: Path,
    split: str,
    persistent_handles: set[int],
) -> tuple[list[dict[str, Any]], dict[str, str], Counter]:
    config = preprocessor.config
    tasks = _available_tasks(root, split, config.tasks)
    archives = {
        task: zipfile.ZipFile(root / split / f"{task}.zip") for task in tasks
    }
    writer = _ShardWriter(output, split, config.shard_size)
    stats: Counter = Counter()
    try:
        schedule = _episode_schedule(
            archives,
            preprocessor.adapter,
            config.episodes_per_task[split],
        )
        for task, episode, member in schedule:
            if len(writer.metadata) + len(writer.pending) >= config.max_chunks[split]:
                break
            stats["episodes_seen"] += 1
            chunks = preprocessor.extract_episode(
                archives[task],
                split=split,
                task=task,
                episode=episode,
                low_dim_member=member,
                persistent_handles=persistent_handles,
            )
            stats["chunks_accepted"] += len(chunks)
            stats["chunks_rejected"] += len(PHASES) - len(chunks)
            for chunk in chunks:
                if len(writer.metadata) + len(writer.pending) >= config.max_chunks[split]:
                    break
                writer.add(chunk)
    finally:
        for archive in archives.values():
            archive.close()
    metadata = writer.close()
    return metadata, writer.checksums, stats


def run_preprocessing(
    *,
    data_root: Path,
    output_root: Path,
    config: PointNetPreprocessConfig,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output_root}")
    temporary = output_root.with_name(f".{output_root.name}.incomplete-{os.getpid()}")
    temporary.mkdir(parents=True)
    temporary.joinpath("shards").mkdir()
    preprocessor = RLBenchPointNetPreprocessor(config)
    all_checksums: dict[str, str] = {}
    summary: dict[str, Any] = {
        "schema_version": config.schema_version,
        "config": asdict(config),
        "splits": {},
    }
    try:
        persistent_handles, handle_task_counts = (
            preprocessor.discover_persistent_handles(data_root)
        )
        summary["persistent_excluded_handles"] = sorted(persistent_handles)
        summary["persistent_handle_task_counts"] = {
            str(handle): handle_task_counts[handle]
            for handle in sorted(persistent_handles)
        }
        for split in ("train", "val"):
            metadata, checksums, stats = preprocess_split(
                preprocessor,
                data_root,
                temporary,
                split,
                persistent_handles,
            )
            manifest_path = temporary / f"manifest-{split}.jsonl"
            pairs_path = temporary / f"pairs-{split}.jsonl"
            _write_jsonl(manifest_path, metadata)
            pairs = build_pair_labels(metadata)
            _write_jsonl(pairs_path, pairs)
            all_checksums.update(checksums)
            all_checksums[manifest_path.name] = _sha256(manifest_path)
            all_checksums[pairs_path.name] = _sha256(pairs_path)
            summary["splits"][split] = {
                "chunks": len(metadata),
                "queries_with_positive": sum(bool(row["positive_ids"]) for row in pairs),
                "tasks": dict(sorted(Counter(row["task"] for row in metadata).items())),
                "phases": dict(sorted(Counter(row["phase"] for row in metadata).items())),
                "phase_sources": dict(
                    sorted(Counter(row["phase_source"] for row in metadata).items())
                ),
                "target_valid": sum(bool(row["target_valid"]) for row in metadata),
                "effect_valid": sum(bool(row["effect_valid"]) for row in metadata),
                "stats": dict(stats),
            }
            print(
                f"{split}: {len(metadata)} chunks, "
                f"{summary['splits'][split]['queries_with_positive']} queries with positives",
                flush=True,
            )
        summary_path = temporary / "summary.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        all_checksums[summary_path.name] = _sha256(summary_path)
        checksum_path = temporary / "SHA256SUMS"
        checksum_path.write_text(
            "".join(f"{digest}  {name}\n" for name, digest in sorted(all_checksums.items())),
            encoding="utf-8",
        )
        temporary.rename(output_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = PointNetPreprocessConfig.from_json(args.config)
    summary = run_preprocessing(
        data_root=args.data_root,
        output_root=args.output_root,
        config=config,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

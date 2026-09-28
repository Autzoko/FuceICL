"""RLBench action chunk 的规范表示与轻量数据读取接口。"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ACTION_DIM = 7


@dataclass(frozen=True)
class ActionChunkConfig:
    """action chunk 预处理的稳定配置。"""

    schema_version: str
    horizon: int
    frame_stride: int
    shard_size: int

    @classmethod
    def from_json(cls, path: str | Path) -> "ActionChunkConfig":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def __post_init__(self) -> None:
        if not self.schema_version.strip():
            raise ValueError("schema_version 不能为空")
        if min(self.horizon, self.frame_stride, self.shard_size) <= 0:
            raise ValueError("horizon、frame_stride 和 shard_size 必须为正")


def quaternion_xyzw_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    """将 xyzw 四元数转换为 ``float64[3, 3]`` 旋转矩阵。"""
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm((x, y, z, w)))
    if norm <= 1e-12:
        raise ValueError("四元数退化")
    x, y, z, w = np.asarray((x, y, z, w), dtype=np.float64) / norm
    return np.asarray(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def matrix_to_axis_angle(rotation: np.ndarray) -> np.ndarray:
    """稳定地将旋转矩阵转换为 axis-angle 向量。"""
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"rotation shape 错误：{matrix.shape}")
    cosine = float(np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle <= 1e-8:
        return np.zeros(3, dtype=np.float32)
    if np.pi - angle <= 1e-5:
        # pi 附近用 R + I 的主列恢复旋转轴，
        # 避免 sin(angle) 数值退化。
        symmetric = matrix + np.eye(3, dtype=np.float64)
        axis = symmetric[:, int(np.argmax(np.linalg.norm(symmetric, axis=0)))]
        norm = float(np.linalg.norm(axis))
        if norm <= 1e-8:
            raise ValueError("无法恢复 pi 旋转的轴")
        axis = axis / norm
    else:
        axis = np.asarray(
            [
                matrix[2, 1] - matrix[1, 2],
                matrix[0, 2] - matrix[2, 0],
                matrix[1, 0] - matrix[0, 1],
            ],
            dtype=np.float64,
        ) / (2.0 * np.sin(angle))
    return (axis * angle).astype(np.float32)


def build_cumulative_action_chunk(
    demo: Any,
    frame: int,
    *,
    horizon: int,
    frame_stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构造查询时刻 EEF 坐标系中的累计 SE(3) action chunk。

    每个 token 为 ``[local_translation(3), local_axis_angle(3), gripper_open]``。
    位姿都相对同一个查询时刻，而不是逐步积分，因而不会累积表示误差。
    越过 episode 末尾的 token 用最后一帧填充并由 ``valid_mask`` 排除。
    """
    if not 0 <= frame < len(demo):
        raise IndexError(f"frame={frame} 超出 episode 长度 {len(demo)}")
    if min(horizon, frame_stride) <= 0:
        raise ValueError("horizon 与 frame_stride 必须为正")

    current_pose = np.asarray(demo[frame].gripper_pose, dtype=np.float64)
    if current_pose.shape != (7,):
        raise ValueError(f"gripper_pose shape 错误：{current_pose.shape}")
    current_rotation = quaternion_xyzw_to_matrix(current_pose[3:7])
    actions = np.zeros((horizon, ACTION_DIM), dtype=np.float32)
    valid_mask = np.zeros(horizon, dtype=np.bool_)
    target_frames = np.zeros(horizon, dtype=np.int32)
    last_frame = len(demo) - 1

    for step in range(horizon):
        requested_frame = frame + (step + 1) * frame_stride
        target_frame = min(last_frame, requested_frame)
        target_pose = np.asarray(demo[target_frame].gripper_pose, dtype=np.float64)
        target_rotation = quaternion_xyzw_to_matrix(target_pose[3:7])
        local_translation = current_rotation.T @ (
            target_pose[:3] - current_pose[:3]
        )
        local_rotation = current_rotation.T @ target_rotation
        actions[step, :3] = local_translation.astype(np.float32)
        actions[step, 3:6] = matrix_to_axis_angle(local_rotation)
        actions[step, 6] = float(np.clip(demo[target_frame].gripper_open, 0.0, 1.0))
        valid_mask[step] = requested_frame <= last_frame
        target_frames[step] = target_frame

    if not np.isfinite(actions).all():
        raise ValueError("action chunk 包含 NaN 或 Inf")
    return actions, valid_mask, target_frames


class _ShardCache:
    """ActionChunkStore 使用的进程内 shard LRU。"""

    def __init__(self, root: Path, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("cache capacity 必须为正")
        self.root = root
        self.capacity = capacity
        self.values: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    def get(self, relative: str) -> Mapping[str, np.ndarray]:
        cached = self.values.pop(relative, None)
        if cached is None:
            with np.load(self.root / relative) as archive:
                cached = {name: archive[name] for name in archive.files}
            if len(self.values) >= self.capacity:
                self.values.popitem(last=False)
        self.values[relative] = cached
        return cached


class ActionChunkStore:
    """按 parent chunk ID 读取规范 action chunk。"""

    def __init__(self, root: str | Path, split: str, cache_size: int = 4) -> None:
        self.root = Path(root)
        self.split = split
        manifest = self.root / f"manifest-{split}.jsonl"
        self.records = [
            json.loads(line)
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line
        ]
        self.records_by_id = {
            str(record["chunk_id"]): record for record in self.records
        }
        if len(self.records_by_id) != len(self.records):
            raise ValueError(f"{split} action manifest 存在重复 chunk_id")
        self.cache = _ShardCache(self.root, cache_size)

    def get(self, identifier: str) -> dict[str, np.ndarray]:
        record = self.records_by_id[identifier]
        arrays = self.cache.get(str(record["shard"]))
        row = int(record["row"])
        return {
            "actions": arrays["actions"][row].astype(np.float32),
            "valid_mask": arrays["valid_mask"][row].astype(np.bool_),
            "target_frames": arrays["target_frames"][row].astype(np.int32),
        }

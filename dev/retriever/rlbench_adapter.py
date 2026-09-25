"""从 RLBench 原始 ZIP 生成统一的检索 chunk 记录。

该 adapter 只使用公开观测：任务文本、RGB 编码的 GT mask、depth、相机标定、
末端位姿和夹爪状态。``task_low_dim_state`` 不进入检索 key，避免任务对象数量和
排列方式泄漏任务身份。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import io
import json
from pathlib import Path
import pickle
import re
import struct
import subprocess
from typing import Any, Iterable, Optional, Sequence, Union
import zipfile

import numpy as np


DEPTH_SCALE = float(2**24 - 1)
_EPISODE_PATTERN = re.compile(r"/episodes/episode(\d+)/low_dim_obs\.pkl$")


class _Demo:
    """用于读取未安装 RLBench 时的可信官方 pickle。"""

    def __len__(self) -> int:
        return len(self._observations)

    def __getitem__(self, index: int) -> Any:
        return self._observations[index]


class _Observation:
    pass


class _RLBenchUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if (module, name) == ("rlbench.demo", "Demo"):
            return _Demo
        if (module, name) == ("rlbench.backend.observation", "Observation"):
            return _Observation
        return super().find_class(module, name)


def _loads_rlbench_pickle(data: bytes) -> Any:
    # 数据必须来自可信的 RLBench 官方归档；pickle 不适用于不可信输入。
    return _RLBenchUnpickler(io.BytesIO(data)).load()


def _png_size(data: bytes) -> tuple[int, int]:
    if data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        raise ValueError("输入不是合法 PNG")
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def decode_png_rgb(data: bytes) -> np.ndarray:
    """将 PNG 解码为 ``uint8[H,W,3]``，无 Pillow 时回退到 ImageMagick。"""
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    except ImportError:
        width, height = _png_size(data)
        process = subprocess.run(
            ["convert", "png:-", "rgb:-"],
            input=data,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        values = np.frombuffer(process.stdout, dtype=np.uint8)
        expected = width * height * 3
        if values.size != expected:
            raise RuntimeError(
                f"ImageMagick 解码长度错误：expected={expected}, got={values.size}"
            )
        return values.reshape(height, width, 3)


def decode_depth(depth_rgb: np.ndarray, near: float, far: float) -> np.ndarray:
    """按 RLBench 官方规则将 24-bit depth PNG 转为米。"""
    encoded = np.sum(
        depth_rgb.astype(np.float64) * np.array([65536.0, 256.0, 1.0]),
        axis=2,
    )
    normalized = encoded / DEPTH_SCALE
    return near + normalized * (far - near)


def decode_mask(mask_rgb: np.ndarray) -> np.ndarray:
    """按 RLBench 官方规则将 RGB mask 转为 CoppeliaSim object handle。"""
    values = mask_rgb.astype(np.int64)
    return values[..., 0] + 256 * values[..., 1] + 65536 * values[..., 2]


def pointcloud_from_depth(
    depth_m: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    """复现 PyRep 的 depth-to-world 反投影。"""
    height, width = depth_m.shape
    pixel_x, pixel_y = np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(height, dtype=np.float64),
    )
    pixels = np.stack((pixel_x, pixel_y, np.ones_like(pixel_x)), axis=-1)
    pixels = pixels * depth_m[..., None]

    rotation = extrinsics[:3, :3]
    camera_center = extrinsics[:3, 3:4]
    rotation_inverse = rotation.T
    world_to_camera = np.concatenate(
        (rotation_inverse, -rotation_inverse @ camera_center),
        axis=1,
    )
    projection = intrinsics @ world_to_camera
    projection_homogeneous = np.concatenate(
        (projection, np.array([[0.0, 0.0, 0.0, 1.0]])),
        axis=0,
    )
    inverse = np.linalg.inv(projection_homogeneous)[:3]
    homogeneous_pixels = np.concatenate(
        (pixels, np.ones((height, width, 1), dtype=np.float64)),
        axis=-1,
    )
    return homogeneous_pixels @ inverse.T


def quaternion_xyzw_to_rotation_6d(quaternion: Sequence[float]) -> list[float]:
    """将 RLBench 的 xyzw 四元数转为旋转矩阵前两列。"""
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm([x, y, z, w])
    if norm <= 1e-12:
        raise ValueError("四元数退化")
    x, y, z, w = np.array([x, y, z, w]) / norm
    matrix = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return np.concatenate((matrix[:, 0], matrix[:, 1])).astype(float).tolist()


@dataclass(frozen=True)
class RLBenchChunkRecord:
    """可跨机器传输的 RLBench query/candidate key。"""

    chunk_id: str
    split: str
    task: str
    episode: int
    variation: int
    frame: int
    phase: str
    text: str
    active_handle: int
    active_points: list[list[float]]
    active_center: list[float]
    active_extent: list[float]
    eef_relative_active: list[float]
    eef_velocity: list[float]
    gripper_width: float
    gripper_state: str


@dataclass(frozen=True)
class AdapterConfig:
    cameras: tuple[str, ...] = ("front", "overhead")
    phase_fractions: tuple[float, ...] = (0.25, 0.50, 0.75, 0.90)
    phase_names: tuple[str, ...] = ("approach", "contact", "manipulate", "finish")
    point_count: int = 128
    min_segment_pixels: int = 24
    max_segment_extent_m: float = 0.50
    robot_handle_radius_m: float = 0.22

    def __post_init__(self) -> None:
        if len(self.phase_fractions) != len(self.phase_names):
            raise ValueError("phase_fractions 与 phase_names 长度必须一致")
        if not self.phase_fractions or any(
            not 0.0 <= value <= 1.0 for value in self.phase_fractions
        ):
            raise ValueError("phase fraction 必须位于 [0, 1]")
        if self.point_count <= 0 or self.min_segment_pixels <= 0:
            raise ValueError("点数与最小像素数必须为正")


@dataclass
class _Segment:
    points: np.ndarray
    center: np.ndarray
    extent: np.ndarray


class RLBenchArchiveAdapter:
    """直接从 task ZIP 读取 episode，不需要完整解压数据集。"""

    def __init__(self, config: Optional[AdapterConfig] = None) -> None:
        self.config = config or AdapterConfig()

    @staticmethod
    def _episode_members(archive: zipfile.ZipFile) -> list[tuple[int, str]]:
        members = []
        for name in archive.namelist():
            match = _EPISODE_PATTERN.search(name)
            if match:
                members.append((int(match.group(1)), name))
        return sorted(members)

    @staticmethod
    def _read_episode(
        archive: zipfile.ZipFile,
        low_dim_member: str,
    ) -> tuple[Any, list[str], int, str]:
        demo = _loads_rlbench_pickle(archive.read(low_dim_member))
        prefix = low_dim_member.rsplit("/", 1)[0]
        descriptions = _loads_rlbench_pickle(
            archive.read(f"{prefix}/variation_descriptions.pkl")
        )
        variation = int(
            _loads_rlbench_pickle(archive.read(f"{prefix}/variation_number.pkl"))
        )
        return demo, list(descriptions), variation, prefix

    def _frame_segments(
        self,
        archive: zipfile.ZipFile,
        prefix: str,
        observation: Any,
        frame: int,
    ) -> dict[int, _Segment]:
        grouped: dict[int, list[np.ndarray]] = {}
        for camera in self.config.cameras:
            depth_rgb = decode_png_rgb(
                archive.read(f"{prefix}/{camera}_depth/{frame}.png")
            )
            mask_rgb = decode_png_rgb(
                archive.read(f"{prefix}/{camera}_mask/{frame}.png")
            )
            near = float(observation.misc[f"{camera}_camera_near"])
            far = float(observation.misc[f"{camera}_camera_far"])
            depth = decode_depth(depth_rgb, near, far)
            handles = decode_mask(mask_rgb)
            points = pointcloud_from_depth(
                depth,
                np.asarray(observation.misc[f"{camera}_camera_extrinsics"]),
                np.asarray(observation.misc[f"{camera}_camera_intrinsics"]),
            )
            unique, counts = np.unique(handles, return_counts=True)
            for handle, count in zip(unique.tolist(), counts.tolist()):
                if handle == 0 or count < self.config.min_segment_pixels:
                    continue
                selected = points[handles == handle]
                selected = selected[np.isfinite(selected).all(axis=1)]
                if len(selected):
                    grouped.setdefault(int(handle), []).append(selected)

        segments: dict[int, _Segment] = {}
        for handle, clouds in grouped.items():
            points = np.concatenate(clouds, axis=0)
            lower, upper = np.quantile(points, (0.05, 0.95), axis=0)
            extent = np.maximum(upper - lower, 1e-4)
            center = np.median(points, axis=0)
            if not (-1.2 <= center[0] <= 1.2 and -1.2 <= center[1] <= 1.2):
                continue
            if not (0.55 <= center[2] <= 1.8):
                continue
            if float(np.max(extent)) > self.config.max_segment_extent_m:
                continue
            segments[handle] = _Segment(points=points, center=center, extent=extent)
        return segments

    @staticmethod
    def _sample_points(points: np.ndarray, count: int) -> np.ndarray:
        if len(points) >= count:
            indices = np.linspace(0, len(points) - 1, count, dtype=np.int64)
            return points[indices]
        repeats = int(np.ceil(count / len(points)))
        return np.tile(points, (repeats, 1))[:count]

    def _select_active_handle(
        self,
        segments: dict[int, _Segment],
        eef_position: np.ndarray,
        robot_handles: set[int],
    ) -> Optional[int]:
        candidates = [
            (float(np.linalg.norm(segment.center - eef_position)), handle)
            for handle, segment in segments.items()
            if handle not in robot_handles
        ]
        return min(candidates)[1] if candidates else None

    @staticmethod
    def _velocity(demo: Any, frame: int) -> list[float]:
        previous = max(0, frame - 3)
        delta_steps = max(1, frame - previous)
        current_pose = np.asarray(demo[frame].gripper_pose, dtype=np.float64)
        previous_pose = np.asarray(demo[previous].gripper_pose, dtype=np.float64)
        # 这里只比较局部状态连续性，不假设不同数据集具有完全一致的控制频率。
        linear = (current_pose[:3] - previous_pose[:3]) / delta_steps
        angular_proxy = (current_pose[3:6] - previous_pose[3:6]) / delta_steps
        return np.concatenate((linear, angular_proxy)).astype(float).tolist()

    def _episode_records(
        self,
        archive: zipfile.ZipFile,
        task: str,
        split: str,
        episode: int,
        low_dim_member: str,
    ) -> list[RLBenchChunkRecord]:
        demo, descriptions, variation, prefix = self._read_episode(
            archive,
            low_dim_member,
        )
        text = descriptions[0]
        frames = [
            min(len(demo) - 1, int(round((len(demo) - 1) * fraction)))
            for fraction in self.config.phase_fractions
        ]
        required_frames = sorted(set([0] + frames))
        frame_segments = {
            frame: self._frame_segments(
                archive,
                prefix,
                demo[frame],
                frame,
            )
            for frame in required_frames
        }

        initial_eef = np.asarray(demo[0].gripper_pose[:3], dtype=np.float64)
        robot_handles = {
            handle
            for handle, segment in frame_segments[0].items()
            if np.linalg.norm(segment.center - initial_eef)
            <= self.config.robot_handle_radius_m
        }

        records = []
        for phase, frame in zip(self.config.phase_names, frames):
            observation = demo[frame]
            eef_pose = np.asarray(observation.gripper_pose, dtype=np.float64)
            handle = self._select_active_handle(
                frame_segments[frame],
                eef_pose[:3],
                robot_handles,
            )
            if handle is None:
                continue
            segment = frame_segments[frame][handle]
            sampled = self._sample_points(segment.points, self.config.point_count)
            centered = sampled - segment.center
            rotation_6d = quaternion_xyzw_to_rotation_6d(eef_pose[3:7])
            relative = np.concatenate((eef_pose[:3] - segment.center, rotation_6d))
            gripper_joints = np.asarray(
                observation.gripper_joint_positions,
                dtype=np.float64,
            )
            gripper_width = float(np.sum(np.abs(gripper_joints)))
            gripper_state = "open" if float(observation.gripper_open) >= 0.5 else "closed"
            records.append(
                RLBenchChunkRecord(
                    chunk_id=f"{split}:{task}:episode{episode}:frame{frame}",
                    split=split,
                    task=task,
                    episode=episode,
                    variation=variation,
                    frame=frame,
                    phase=phase,
                    text=text,
                    active_handle=handle,
                    active_points=centered.astype(float).tolist(),
                    active_center=segment.center.astype(float).tolist(),
                    active_extent=segment.extent.astype(float).tolist(),
                    eef_relative_active=relative.astype(float).tolist(),
                    eef_velocity=self._velocity(demo, frame),
                    gripper_width=gripper_width,
                    gripper_state=gripper_state,
                )
            )
        return records

    def extract_archive(
        self,
        archive_path: Union[str, Path],
        *,
        split: str,
        episode_limit: int,
    ) -> list[RLBenchChunkRecord]:
        """抽取单个 task archive 的前 N 个 episode。"""
        path = Path(archive_path)
        task = path.stem
        records: list[RLBenchChunkRecord] = []
        with zipfile.ZipFile(path) as archive:
            members = self._episode_members(archive)[:episode_limit]
            for episode, member in members:
                records.extend(
                    self._episode_records(
                        archive,
                        task,
                        split,
                        episode,
                        member,
                    )
                )
        return records


def write_jsonl(
    records: Iterable[RLBenchChunkRecord],
    path: Union[str, Path],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--candidate-episodes", type=int, default=8)
    parser.add_argument("--query-episodes", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    adapter = RLBenchArchiveAdapter()
    records = []
    for split, limit in (
        ("train", args.candidate_episodes),
        ("val", args.query_episodes),
    ):
        for task in args.tasks:
            archive = args.root / split / f"{task}.zip"
            print(f"extracting {split}/{task}: {archive}", flush=True)
            records.extend(
                adapter.extract_archive(archive, split=split, episode_limit=limit)
            )
    write_jsonl(records, args.output)
    print(f"wrote {len(records)} chunks to {args.output}", flush=True)


if __name__ == "__main__":
    main()

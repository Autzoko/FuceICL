"""把 ManiSkill pointcloud replay 转为感知式 canonical geometry/action chunks。"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping

import h5py
import numpy as np

from dev.predictor.action_chunk_data import matrix_to_axis_angle
from dev.predictor.canonical_geometry import FEATURE_NAMES, GEOMETRY_DIM


@dataclass(frozen=True)
class ManiSkillChunkConfig:
    """单活动物体 ManiSkill 任务的固定预处理协议。"""

    schema_version: str
    seed: int
    horizon: int
    frame_stride: int
    points_per_object: int
    label_probe_frames: int
    minimum_label_points: int
    maximum_centroid_error_m: float
    validation_episodes: int
    shard_size: int
    action_representation: str = "cumulative_observation_delta"
    position_limit_m: float = 0.1
    rotation_scale_rad: float = -0.1
    task_id: str = "PickCube-v1"
    active_actor_name: str = "cube"
    target_position_source: str = "observation_goal_pos"
    target_actor_name: str | None = None
    store_geometry_sequence: bool = False

    @classmethod
    def from_json(cls, path: Path) -> "ManiSkillChunkConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def __post_init__(self) -> None:
        if not self.schema_version.strip():
            raise ValueError("schema_version 不能为空")
        positive = (
            self.horizon,
            self.frame_stride,
            self.points_per_object,
            self.label_probe_frames,
            self.minimum_label_points,
            self.maximum_centroid_error_m,
            self.validation_episodes,
            self.shard_size,
        )
        if min(positive) <= 0:
            raise ValueError("chunk、点数、阈值与 episode 数必须为正")
        if self.action_representation not in (
            "cumulative_observation_delta",
            "canonical_controller_command",
        ):
            raise ValueError("未知 action representation")
        if (
            self.action_representation == "canonical_controller_command"
            and self.frame_stride != 1
        ):
            raise ValueError("controller command chunks 当前只支持 frame_stride=1")
        if self.position_limit_m <= 0 or self.rotation_scale_rad == 0:
            raise ValueError("controller position/rotation scale 非法")
        if not self.task_id.strip() or not self.active_actor_name.strip():
            raise ValueError("task_id 与 active_actor_name 不能为空")
        if self.target_position_source not in (
            "observation_goal_pos",
            "segmented_actor_centroid",
        ):
            raise ValueError("未知 target position source")
        if (
            self.target_position_source == "segmented_actor_centroid"
            and not self.target_actor_name
        ):
            raise ValueError("segmented actor target 必须提供 target_actor_name")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """将 ManiSkill/SAPIEN ``wxyz`` quaternion 转为旋转矩阵。"""
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm((w, x, y, z)))
    if norm <= 1e-12:
        raise ValueError("TCP quaternion 退化")
    w, x, y, z = np.asarray((w, x, y, z), dtype=np.float64) / norm
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


def _shape_sigmas(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    denominator = max(len(points) - 1, 1)
    covariance = centered.T @ centered / denominator
    eigenvalues = np.linalg.eigvalsh(covariance).clip(min=0.0)
    return np.sqrt(eigenvalues)[::-1].astype(np.float32)


def _gripper_open(qpos: np.ndarray) -> float:
    """Panda 两个 4 cm finger joints 映射为 [0,1] opening。"""
    if qpos.shape[-1] < 2:
        raise ValueError("Panda qpos 缺少 finger joints")
    width = float(qpos[-2] + qpos[-1])
    return float(np.clip(width / 0.08, 0.0, 1.0))


def _label_centroids(
    xyzw: np.ndarray,
    segmentation: np.ndarray,
    minimum_points: int,
) -> dict[int, tuple[np.ndarray, int]]:
    valid = np.abs(xyzw[:, 3]) > 0.5
    labels = segmentation.reshape(-1)
    result: dict[int, tuple[np.ndarray, int]] = {}
    for label in np.unique(labels[valid]):
        mask = valid & (labels == label)
        count = int(mask.sum())
        if count < minimum_points:
            continue
        points = np.asarray(xyzw[mask, :3], dtype=np.float64)
        result[int(label)] = (points.mean(axis=0), count)
    return result


def _infer_actor_label(
    *,
    xyzw: h5py.Dataset,
    segmentation: h5py.Dataset,
    actor_positions: np.ndarray,
    config: ManiSkillChunkConfig,
    role: str,
) -> tuple[int, dict[str, Any]]:
    """用少量离线 actor supervision 自动匹配观测中的 segmentation label。"""
    votes: Counter[int] = Counter()
    distances: dict[int, list[float]] = {}
    probe_count = min(config.label_probe_frames, len(actor_positions))
    for frame in range(probe_count):
        candidates = _label_centroids(
            np.asarray(xyzw[frame]),
            np.asarray(segmentation[frame]),
            config.minimum_label_points,
        )
        if not candidates:
            raise ValueError(f"frame {frame} 没有满足点数门槛的 segmentation label")
        label, (centroid, _) = min(
            candidates.items(),
            key=lambda item: float(
                np.linalg.norm(item[1][0] - actor_positions[frame])
            ),
        )
        distance = float(np.linalg.norm(centroid - actor_positions[frame]))
        votes[label] += 1
        distances.setdefault(label, []).append(distance)
    label = min(
        votes,
        key=lambda value: (-votes[value], float(np.mean(distances[value])), value),
    )
    probe_errors = distances[label]
    if max(probe_errors) > config.maximum_centroid_error_m:
        raise ValueError(
            f"{role} label={label} probe centroid error="
            f"{max(probe_errors):.4f} m 超限"
        )
    return label, {
        "votes": {str(key): value for key, value in sorted(votes.items())},
        "selected_probe_error_max_m": max(probe_errors),
        "selected_probe_error_mean_m": float(np.mean(probe_errors)),
    }


def _segmented_points(
    xyzw: np.ndarray,
    segmentation: np.ndarray,
    label: int,
) -> np.ndarray:
    valid = np.abs(xyzw[:, 3]) > 0.5
    mask = valid & (segmentation.reshape(-1) == label)
    return np.asarray(xyzw[mask, :3], dtype=np.float64)


def _sample_centered_points(
    points: np.ndarray,
    center: np.ndarray,
    *,
    count: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(points), size=count, replace=len(points) < count)
    return (points[indices] - center).astype(np.float32)


def _canonical_geometry(
    *,
    active_points: np.ndarray,
    active_center: np.ndarray,
    goal_position: np.ndarray,
    tcp_poses: np.ndarray,
    qpos: np.ndarray,
    frame: int,
) -> np.ndarray:
    pose = np.asarray(tcp_poses[frame], dtype=np.float64)
    rotation = _quaternion_wxyz_to_matrix(pose[3:7])
    active_in_eef = rotation.T @ (active_center - pose[:3])
    target_from_active = rotation.T @ (goal_position - active_center)
    world_velocity = (
        np.zeros(3, dtype=np.float64)
        if frame == 0
        else pose[:3] - np.asarray(tcp_poses[frame - 1, :3], dtype=np.float64)
    )
    local_velocity = rotation.T @ world_velocity
    geometry = np.concatenate(
        (
            active_in_eef,
            target_from_active,
            _shape_sigmas(active_points),
            np.zeros(3, dtype=np.float32),
            local_velocity,
            np.asarray([_gripper_open(qpos[frame]), 1.0]),
        )
    ).astype(np.float32)
    if geometry.shape != (GEOMETRY_DIM,) or not np.isfinite(geometry).all():
        raise ValueError(f"canonical geometry shape/value 错误：{geometry.shape}")
    return geometry


def _observed_frame_geometry(
    *,
    xyzw: np.ndarray,
    segmentation: np.ndarray,
    active_label: int,
    active_position: np.ndarray,
    target_label: int | None,
    target_position: np.ndarray,
    tcp_poses: np.ndarray,
    qpos: np.ndarray,
    frame: int,
    config: ManiSkillChunkConfig,
) -> tuple[np.ndarray, np.ndarray, float, int | None, float | None] | None:
    """从单帧观测构造 geometry；仿真位姿仅用于离线分割质心审计。"""
    active_points = _segmented_points(xyzw, segmentation, active_label)
    if len(active_points) < config.minimum_label_points:
        return None
    active_center = active_points.mean(axis=0)
    active_error = float(np.linalg.norm(active_center - active_position))
    if active_error > config.maximum_centroid_error_m:
        raise ValueError(
            f"frame={frame} active centroid error={active_error:.4f} m 超限"
        )

    target_point_count: int | None = None
    target_error: float | None = None
    goal_position = target_position
    if target_label is not None:
        target_points = _segmented_points(xyzw, segmentation, target_label)
        target_point_count = len(target_points)
        if target_point_count < config.minimum_label_points:
            return None
        goal_position = target_points.mean(axis=0)
        target_error = float(np.linalg.norm(goal_position - target_position))
        if target_error > config.maximum_centroid_error_m:
            raise ValueError(
                f"frame={frame} target centroid error={target_error:.4f} m 超限"
            )
    geometry = _canonical_geometry(
        active_points=active_points,
        active_center=active_center,
        goal_position=goal_position,
        tcp_poses=tcp_poses,
        qpos=qpos,
        frame=frame,
    )
    return (
        active_points,
        geometry,
        active_error,
        target_point_count,
        target_error,
    )


def _canonical_action_chunk(
    *,
    tcp_poses: np.ndarray,
    qpos: np.ndarray,
    frame: int,
    config: ManiSkillChunkConfig,
) -> tuple[np.ndarray, np.ndarray]:
    query_pose = np.asarray(tcp_poses[frame], dtype=np.float64)
    query_rotation = _quaternion_wxyz_to_matrix(query_pose[3:7])
    actions = np.empty((config.horizon, 7), dtype=np.float32)
    target_frames = np.empty(config.horizon, dtype=np.int32)
    for step in range(config.horizon):
        target_frame = frame + (step + 1) * config.frame_stride
        target_pose = np.asarray(tcp_poses[target_frame], dtype=np.float64)
        target_rotation = _quaternion_wxyz_to_matrix(target_pose[3:7])
        actions[step, :3] = (
            query_rotation.T @ (target_pose[:3] - query_pose[:3])
        ).astype(np.float32)
        actions[step, 3:6] = matrix_to_axis_angle(
            query_rotation.T @ target_rotation
        )
        actions[step, 6] = _gripper_open(qpos[target_frame])
        target_frames[step] = target_frame
    if not np.isfinite(actions).all():
        raise ValueError("canonical action chunk 含 NaN/Inf")
    return actions, target_frames


def _canonical_controller_action_chunk(
    *,
    controller_actions: np.ndarray,
    tcp_poses: np.ndarray,
    frame: int,
    config: ManiSkillChunkConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """把实际 controller commands 转为各执行时刻 EEF 中的物理命令。"""
    # 延迟导入避免 action bridge 的 observation helper 形成模块初始化环。
    from dev.simulator.maniskill_action_bridge import (
        canonical_to_controller,
        controller_to_canonical_unclipped,
    )

    stop = frame + config.horizon
    normalized = np.asarray(controller_actions[frame:stop], dtype=np.float32)
    if normalized.shape != (config.horizon, 7):
        raise ValueError(f"controller action chunk shape 错误：{normalized.shape}")
    if not np.isfinite(normalized).all() or np.max(np.abs(normalized)) > 1.0001:
        raise ValueError("normalized controller action 含非法值")
    if np.max(np.linalg.norm(normalized[:, 3:6], axis=1)) > 1.0001:
        raise ValueError("normalized rotation command L2 norm 超过 1")
    actions = np.stack(
        [
            controller_to_canonical_unclipped(
                normalized[index],
                tcp_poses[frame + index],
                position_limit_m=config.position_limit_m,
                rotation_scale_rad=config.rotation_scale_rad,
            )
            for index in range(config.horizon)
        ]
    ).astype(np.float32)
    reconstructed = np.stack(
        [
            canonical_to_controller(
                actions[index],
                tcp_poses[frame + index],
                position_limit_m=config.position_limit_m,
                rotation_scale_rad=config.rotation_scale_rad,
            ).value
            for index in range(config.horizon)
        ]
    )
    maximum_error = float(np.max(np.abs(reconstructed - normalized)))
    if maximum_error > 1e-5:
        raise ValueError(
            "controller/canonical command round-trip error="
            f"{maximum_error:.3e}"
        )
    target_frames = np.arange(frame + 1, stop + 1, dtype=np.int32)
    return actions, target_frames


class _ShardWriter:
    def __init__(self, root: Path, split: str, shard_size: int) -> None:
        self.root = root
        self.split = split
        self.shard_size = shard_size
        self.pending: list[tuple[dict[str, Any], dict[str, np.ndarray]]] = []
        self.records: list[dict[str, Any]] = []
        self.checksums: dict[str, str] = {}
        self.index = 0

    def add(self, metadata: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
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


def _episode_split(keys: list[str], validation_episodes: int, seed: int) -> dict[str, str]:
    if validation_episodes >= len(keys):
        raise ValueError("validation episodes 必须少于总 episodes")
    ranked = sorted(
        keys,
        key=lambda key: hashlib.sha256(f"{seed}:{key}".encode()).hexdigest(),
    )
    validation = set(ranked[:validation_episodes])
    return {key: "val" if key in validation else "train" for key in keys}


def run(
    *,
    h5_path: Path,
    metadata_path: Path,
    output_root: Path,
    config: ManiSkillChunkConfig,
) -> None:
    if output_root.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output_root}")
    temporary = output_root.with_name(f".{output_root.name}.incomplete-{os.getpid()}")
    temporary.mkdir(parents=True)
    temporary.joinpath("shards").mkdir()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source_episodes = {
        int(record["episode_id"]): record
        for record in metadata.get("episodes", [])
    }
    if len(source_episodes) != len(metadata.get("episodes", [])):
        raise ValueError("source metadata episode_id 重复")
    source_task = str(metadata.get("env_info", {}).get("env_id", ""))
    if source_task and source_task != config.task_id:
        raise ValueError(
            f"source env_id={source_task} 与 config task_id={config.task_id} 不一致"
        )
    summary: dict[str, Any] = {
        "schema_version": config.schema_version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "feature_names": FEATURE_NAMES,
        "source": {
            "h5": str(h5_path),
            "h5_sha256": _sha256(h5_path),
            "metadata": str(metadata_path),
            "metadata_sha256": _sha256(metadata_path),
            "env_info": metadata.get("env_info", {}),
        },
        "representations": {
            "tcp_quaternion": "wxyz (ManiSkill/SAPIEN raw pose)",
            "geometry": "17D query-EEF canonical geometry",
            "action": (
                "physical per-step EEF command inverted from executed "
                "pd_ee_delta_pose actions"
                if config.action_representation
                == "canonical_controller_command"
                else "cumulative query-EEF [translation, axis-angle, gripper_open]"
            ),
            "pointcloud": "active-object points centered by observed centroid",
            "geometry_sequence": (
                "Demo-only H+1 observed canonical geometries"
                if config.store_geometry_sequence
                else "not stored"
            ),
        },
        "privileged_supervision": {
            "active": (
                f"{config.active_actor_name} actor position is used only to identify "
                "a segmentation label; stored geometry uses the observed point centroid"
            ),
            "target": (
                f"{config.target_actor_name} actor position is used only to identify "
                "a segmentation label; stored target position uses the observed point "
                "centroid"
                if config.target_position_source == "segmented_actor_centroid"
                else "stored target position comes from obs/extra/goal_pos"
            ),
        },
        "splits": {},
    }
    try:
        with h5py.File(h5_path, "r") as handle:
            keys = sorted(
                (key for key in handle if key.startswith("traj_")),
                key=lambda value: int(value.split("_")[1]),
            )
            if len(keys) != len(metadata.get("episodes", [])):
                raise ValueError("HDF5 trajectory 数与 metadata episodes 不一致")
            split_by_episode = _episode_split(
                keys, config.validation_episodes, config.seed
            )
            writers = {
                split: _ShardWriter(temporary, split, config.shard_size)
                for split in ("train", "val")
            }
            diagnostics = {split: [] for split in writers}
            for key in keys:
                trajectory = handle[key]
                episode = int(key.split("_")[1])
                if episode not in source_episodes:
                    raise ValueError(f"{key} 在 source metadata 中缺失")
                source_episode = source_episodes[episode]
                source_seed_value = source_episode.get("episode_seed")
                if source_seed_value is None:
                    source_seed_value = source_episode["reset_kwargs"]["seed"]
                source_episode_seed = int(source_seed_value)
                xyzw = trajectory["obs/pointcloud/xyzw"]
                segmentation = trajectory["obs/pointcloud/segmentation"]
                tcp_poses = np.asarray(trajectory["obs/extra/tcp_pose"])
                qpos = np.asarray(trajectory["obs/agent/qpos"])
                controller_actions = np.asarray(trajectory["actions"])
                actor_path = f"env_states/actors/{config.active_actor_name}"
                if actor_path not in trajectory:
                    raise KeyError(f"{key} 缺少 active actor path：{actor_path}")
                active_positions = np.asarray(
                    trajectory[actor_path][:, :3]
                )
                if len(controller_actions) != len(tcp_poses) - 1:
                    raise ValueError(f"{key} controller action/observation 长度不一致")
                active_label, active_label_diagnostic = _infer_actor_label(
                    xyzw=xyzw,
                    segmentation=segmentation,
                    actor_positions=active_positions,
                    config=config,
                    role="active",
                )
                target_label: int | None = None
                target_label_diagnostic: dict[str, Any] | None = None
                target_positions: np.ndarray | None = None
                if config.target_position_source == "observation_goal_pos":
                    goal_path = "obs/extra/goal_pos"
                    if goal_path not in trajectory:
                        raise KeyError(f"{key} 缺少 target observation：{goal_path}")
                    target_positions = np.asarray(trajectory[goal_path])
                else:
                    target_actor_path = (
                        f"env_states/actors/{config.target_actor_name}"
                    )
                    if target_actor_path not in trajectory:
                        raise KeyError(
                            f"{key} 缺少 target actor path：{target_actor_path}"
                        )
                    target_positions = np.asarray(
                        trajectory[target_actor_path][:, :3]
                    )
                    target_label, target_label_diagnostic = _infer_actor_label(
                        xyzw=xyzw,
                        segmentation=segmentation,
                        actor_positions=target_positions,
                        config=config,
                        role="target",
                    )
                if len(active_positions) != len(tcp_poses) or len(target_positions) != len(
                    tcp_poses
                ):
                    raise ValueError(f"{key} actor/target 与 observation 长度不一致")
                split = split_by_episode[key]
                frame_errors = []
                frame_point_counts = []
                target_frame_errors = []
                target_frame_point_counts = []
                skipped_low_visibility = []
                max_query_frame = len(tcp_poses) - 1 - (
                    config.horizon * config.frame_stride
                )
                for frame in range(max_query_frame + 1):
                    observed = _observed_frame_geometry(
                        xyzw=np.asarray(xyzw[frame]),
                        segmentation=np.asarray(segmentation[frame]),
                        active_label=active_label,
                        active_position=active_positions[frame],
                        target_label=target_label,
                        target_position=target_positions[frame],
                        tcp_poses=tcp_poses,
                        qpos=qpos,
                        frame=frame,
                        config=config,
                    )
                    if observed is None:
                        skipped_low_visibility.append(
                            {"frame": frame, "role": "query_or_target"}
                        )
                        continue
                    (
                        points,
                        geometry,
                        centroid_error,
                        target_point_count,
                        target_centroid_error,
                    ) = observed
                    center = points.mean(axis=0)
                    identifier = (
                        f"maniskill:{config.task_id}:{key}:f{frame:03d}"
                    )
                    seed = int.from_bytes(
                        hashlib.sha256(identifier.encode()).digest()[:8], "little"
                    )
                    sampled = _sample_centered_points(
                        points,
                        center,
                        count=config.points_per_object,
                        seed=seed,
                    )
                    if (
                        config.action_representation
                        == "canonical_controller_command"
                    ):
                        actions, target_frames = (
                            _canonical_controller_action_chunk(
                                controller_actions=controller_actions,
                                tcp_poses=tcp_poses,
                                frame=frame,
                                config=config,
                            )
                        )
                    else:
                        actions, target_frames = _canonical_action_chunk(
                            tcp_poses=tcp_poses,
                            qpos=qpos,
                            frame=frame,
                            config=config,
                        )

                    geometry_sequence: np.ndarray | None = None
                    if config.store_geometry_sequence:
                        sequence = [geometry]
                        sequence_visible = True
                        for sequence_frame in target_frames.tolist():
                            future = _observed_frame_geometry(
                                xyzw=np.asarray(xyzw[sequence_frame]),
                                segmentation=np.asarray(
                                    segmentation[sequence_frame]
                                ),
                                active_label=active_label,
                                active_position=active_positions[sequence_frame],
                                target_label=target_label,
                                target_position=target_positions[sequence_frame],
                                tcp_poses=tcp_poses,
                                qpos=qpos,
                                frame=sequence_frame,
                                config=config,
                            )
                            if future is None:
                                skipped_low_visibility.append(
                                    {
                                        "frame": frame,
                                        "role": "geometry_sequence",
                                        "missing_frame": sequence_frame,
                                    }
                                )
                                sequence_visible = False
                                break
                            sequence.append(future[1])
                        if not sequence_visible:
                            continue
                        geometry_sequence = np.stack(sequence).astype(np.float32)

                    arrays = {
                        "active_points": sampled,
                        "geometry": geometry,
                        "actions": actions,
                        "target_frames": target_frames,
                    }
                    if geometry_sequence is not None:
                        arrays["geometry_sequence"] = geometry_sequence
                    writers[split].add(
                        {
                            "chunk_id": identifier,
                            "split": split,
                            "task": config.task_id,
                            "episode": episode,
                            "source_episode_seed": source_episode_seed,
                            "frame": frame,
                            "active_label": active_label,
                            "active_point_count": len(points),
                            "centroid_error_m": centroid_error,
                            "target_label": target_label,
                            "target_point_count": target_point_count,
                            "target_centroid_error_m": target_centroid_error,
                        },
                        arrays,
                    )
                    frame_errors.append(centroid_error)
                    frame_point_counts.append(len(points))
                    if target_point_count is not None:
                        target_frame_point_counts.append(target_point_count)
                    if target_centroid_error is not None:
                        target_frame_errors.append(target_centroid_error)
                if not frame_errors:
                    raise ValueError(f"{key} 没有满足可见点门槛的 query frame")
                diagnostics[split].append(
                    {
                        "trajectory": key,
                        "source_episode_seed": source_episode_seed,
                        "active_label": active_label,
                        "target_label": target_label,
                        "chunks": len(frame_errors),
                        "skipped_low_visibility": skipped_low_visibility,
                        "point_count_min": min(frame_point_counts),
                        "point_count_max": max(frame_point_counts),
                        "centroid_error_max_m": max(frame_errors),
                        "active_label_inference": active_label_diagnostic,
                        "target_label_inference": target_label_diagnostic,
                        "target_point_count_min": (
                            min(target_frame_point_counts)
                            if target_frame_point_counts
                            else None
                        ),
                        "target_point_count_max": (
                            max(target_frame_point_counts)
                            if target_frame_point_counts
                            else None
                        ),
                        "target_centroid_error_max_m": (
                            max(target_frame_errors) if target_frame_errors else None
                        ),
                    }
                )

        checksums: dict[str, str] = {}
        for split, writer in writers.items():
            records, shard_checksums = writer.close()
            manifest = temporary / f"manifest-{split}.jsonl"
            _write_jsonl(manifest, records)
            checksums.update(shard_checksums)
            checksums[manifest.name] = _sha256(manifest)
            errors = np.asarray(
                [record["centroid_error_m"] for record in records],
                dtype=np.float64,
            )
            target_errors = np.asarray(
                [
                    record["target_centroid_error_m"]
                    for record in records
                    if record["target_centroid_error_m"] is not None
                ],
                dtype=np.float64,
            )
            target_point_counts = np.asarray(
                [
                    record["target_point_count"]
                    for record in records
                    if record["target_point_count"] is not None
                ],
                dtype=np.int64,
            )
            summary["splits"][split] = {
                "episodes": len(diagnostics[split]),
                "chunks": len(records),
                "skipped_low_visibility": sum(
                    len(record["skipped_low_visibility"])
                    for record in diagnostics[split]
                ),
                "centroid_error_m": {
                    "mean": float(errors.mean()),
                    "p95": float(np.quantile(errors, 0.95)),
                    "max": float(errors.max()),
                },
                "target_observation": (
                    {
                        "point_count_min": int(target_point_counts.min()),
                        "point_count_max": int(target_point_counts.max()),
                        "centroid_error_m": {
                            "mean": float(target_errors.mean()),
                            "p95": float(np.quantile(target_errors, 0.95)),
                            "max": float(target_errors.max()),
                        },
                    }
                    if target_errors.size
                    else {"source": "obs/extra/goal_pos"}
                ),
                "trajectories": diagnostics[split],
            }
        summary_path = temporary / "summary.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        checksums[summary_path.name] = _sha256(summary_path)
        temporary.joinpath("SHA256SUMS").write_text(
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
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    run(
        h5_path=arguments.h5.resolve(),
        metadata_path=arguments.metadata.resolve(),
        output_root=arguments.output_root.resolve(),
        config=ManiSkillChunkConfig.from_json(arguments.config.resolve()),
    )


if __name__ == "__main__":
    main()

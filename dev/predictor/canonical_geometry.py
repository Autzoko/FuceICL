"""从 Retriever context 构造 EEF-canonical、低维且可解释的几何状态。"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import torch

from dev.pointnet.dataset import PointNetContextStore, STATE_DIM


FEATURE_NAMES = (
    "active_in_eef_x",
    "active_in_eef_y",
    "active_in_eef_z",
    "target_from_active_eef_x",
    "target_from_active_eef_y",
    "target_from_active_eef_z",
    "active_shape_sigma_1",
    "active_shape_sigma_2",
    "active_shape_sigma_3",
    "target_shape_sigma_1",
    "target_shape_sigma_2",
    "target_shape_sigma_3",
    "eef_local_velocity_x",
    "eef_local_velocity_y",
    "eef_local_velocity_z",
    "gripper_width",
    "target_valid",
)
GEOMETRY_DIM = len(FEATURE_NAMES)


def _rotation_from_6d(rotation_6d: torch.Tensor) -> torch.Tensor:
    """用 Gram-Schmidt 将矩阵前两列恢复成正交旋转矩阵。"""
    first = torch.nn.functional.normalize(rotation_6d[..., :3], dim=-1)
    second_raw = rotation_6d[..., 3:6]
    second = second_raw - (first * second_raw).sum(dim=-1, keepdim=True) * first
    second = torch.nn.functional.normalize(second, dim=-1)
    third = torch.linalg.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


def _shape_sigmas(points: torch.Tensor) -> torch.Tensor:
    """返回点云协方差主轴标准差，数值对整体旋转不变。"""
    centered = points - points.mean(dim=1, keepdim=True)
    denominator = max(points.shape[1] - 1, 1)
    covariance = centered.transpose(1, 2) @ centered / denominator
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    return torch.sqrt(eigenvalues).flip(dims=(-1,))


def canonical_geometry_from_tensors(
    *,
    active_points: torch.Tensor,
    target_points: torch.Tensor,
    state: torch.Tensor,
    target_valid: torch.Tensor,
) -> torch.Tensor:
    """将一个 batch 的 context 转为 17D canonical geometry。"""
    if active_points.ndim != 3 or active_points.shape[-1] != 3:
        raise ValueError("active_points shape 必须为 [B, N, 3]")
    if target_points.shape != active_points.shape:
        raise ValueError("target_points 与 active_points shape 必须一致")
    if state.ndim != 2 or state.shape[1] != STATE_DIM:
        raise ValueError(f"state shape 必须为 [B, {STATE_DIM}]")
    if len(state) != len(active_points):
        raise ValueError("point cloud 与 state batch 数量不一致")
    valid = target_valid.bool().reshape(-1, 1)
    rotation = _rotation_from_6d(state[:, 15:21])

    def to_eef(vector: torch.Tensor) -> torch.Tensor:
        return torch.bmm(rotation.transpose(1, 2), vector.unsqueeze(-1)).squeeze(-1)

    active_in_eef = to_eef(-state[:, 12:15])
    target_from_active = to_eef(state[:, 6:9]) * valid
    active_shape = _shape_sigmas(active_points)
    target_shape = _shape_sigmas(target_points) * valid
    local_velocity = to_eef(state[:, 21:24])
    geometry = torch.cat(
        (
            active_in_eef,
            target_from_active,
            active_shape,
            target_shape,
            local_velocity,
            state[:, 27:28],
            valid.float(),
        ),
        dim=-1,
    ).float()
    if geometry.shape[1] != GEOMETRY_DIM:
        raise RuntimeError(f"canonical geometry shape 错误：{geometry.shape}")
    if not torch.isfinite(geometry).all():
        raise ValueError("canonical geometry 包含 NaN 或 Inf")
    return geometry


def load_canonical_geometries(
    store: PointNetContextStore,
    *,
    batch_size: int = 256,
) -> torch.Tensor:
    """按 manifest 顺序从 context store 批量构造 canonical geometry。"""
    if batch_size <= 0:
        raise ValueError("batch_size 必须为正")
    outputs = []
    for start in range(0, len(store.records), batch_size):
        records = store.records[start : start + batch_size]
        values = [store.get(str(record["chunk_id"])) for record in records]
        outputs.append(
            canonical_geometry_from_tensors(
                active_points=torch.from_numpy(
                    np.stack([value["active_points"] for value in values])
                ),
                target_points=torch.from_numpy(
                    np.stack([value["target_points"] for value in values])
                ),
                state=torch.from_numpy(
                    np.stack([value["state"] for value in values])
                ),
                target_valid=torch.from_numpy(
                    np.stack([value["target_valid"] for value in values])
                ),
            )
        )
    return torch.cat(outputs, dim=0)


def geometry_statistics(geometry: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """计算训练集标准化统计，并保护近常量维度。"""
    if geometry.ndim != 2 or geometry.shape[1] != GEOMETRY_DIM:
        raise ValueError(f"geometry shape 必须为 [B, {GEOMETRY_DIM}]")
    mean = geometry.mean(dim=0)
    std = geometry.std(dim=0, unbiased=False).clamp_min(1e-4)
    return mean, std

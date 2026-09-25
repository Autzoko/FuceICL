"""纯 PyTorch Tiny PointNet++，不依赖自定义 CUDA 算子。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as functional


def _normalization_groups(channels: int) -> int:
    """选择能够整除通道数的较小 GroupNorm 分组数。"""
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


def _index_points(points: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """按 batch 索引点或特征，支持 ``[B, ...]`` 形状的 indices。"""
    batch_indices = torch.arange(points.shape[0], device=points.device)
    view_shape = (points.shape[0],) + (1,) * (indices.ndim - 1)
    batch_indices = batch_indices.view(view_shape).expand_as(indices)
    return points[batch_indices, indices]


def _farthest_point_sample(
    xyz: torch.Tensor,
    valid_mask: torch.Tensor,
    sample_count: int,
) -> torch.Tensor:
    """Mask-aware FPS；有效点不足时允许重复采样。"""
    batch_size, point_count, _ = xyz.shape
    centroids = torch.zeros(
        (batch_size, sample_count),
        dtype=torch.long,
        device=xyz.device,
    )
    distances = torch.full(
        (batch_size, point_count),
        float("inf"),
        dtype=xyz.dtype,
        device=xyz.device,
    )
    weights = valid_mask.to(dtype=xyz.dtype).unsqueeze(-1)
    cloud_center = (xyz * weights).sum(dim=1) / weights.sum(dim=1)
    distance_to_center = torch.sum((xyz - cloud_center[:, None, :]) ** 2, dim=-1)
    distance_to_center = distance_to_center.masked_fill(~valid_mask, -1.0)
    # 从离质心最远点开始，避免 FPS 由输入点的排列顺序决定。
    farthest = distance_to_center.argmax(dim=1)
    batch_indices = torch.arange(batch_size, device=xyz.device)

    for index in range(sample_count):
        centroids[:, index] = farthest
        centroid = xyz[batch_indices, farthest].unsqueeze(1)
        squared_distance = torch.sum((xyz - centroid) ** 2, dim=-1)
        distances = torch.minimum(distances, squared_distance)
        distances = distances.masked_fill(~valid_mask, -1.0)
        farthest = distances.argmax(dim=1)
    return centroids


def _knn_indices(
    xyz: torch.Tensor,
    centers: torch.Tensor,
    valid_mask: torch.Tensor,
    neighbor_count: int,
) -> torch.Tensor:
    """查找每个中心的有效近邻；有效点较少时复制最近点。"""
    point_count = xyz.shape[1]
    selected_count = min(neighbor_count, point_count)
    distances = torch.cdist(centers, xyz)
    distances = distances.masked_fill(~valid_mask[:, None, :], float("inf"))
    selected_distances, indices = distances.topk(
        selected_count,
        dim=-1,
        largest=False,
        sorted=False,
    )
    fallback = distances.argmin(dim=-1, keepdim=True).expand_as(indices)
    indices = torch.where(selected_distances.isfinite(), indices, fallback)
    if selected_count < neighbor_count:
        padding = indices[..., :1].expand(
            *indices.shape[:-1],
            neighbor_count - selected_count,
        )
        indices = torch.cat((indices, padding), dim=-1)
    return indices


class _SetAbstraction(nn.Module):
    """PointNet++ 的 FPS + kNN grouping + local PointNet 模块。"""

    def __init__(
        self,
        *,
        sample_count: int,
        neighbor_count: int,
        input_channels: int,
        mlp_channels: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.sample_count = sample_count
        self.neighbor_count = neighbor_count

        layers: list[nn.Module] = []
        channels = input_channels + 3
        for output_channels in mlp_channels:
            layers.extend(
                (
                    nn.Conv2d(channels, output_channels, kernel_size=1, bias=False),
                    nn.GroupNorm(
                        _normalization_groups(output_channels),
                        output_channels,
                    ),
                    nn.SiLU(),
                )
            )
            channels = output_channels
        self.local_mlp = nn.Sequential(*layers)

    def forward(
        self,
        xyz: torch.Tensor,
        features: torch.Tensor | None,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample_indices = _farthest_point_sample(
            xyz,
            valid_mask,
            self.sample_count,
        )
        centers = _index_points(xyz, sample_indices)
        neighbor_indices = _knn_indices(
            xyz,
            centers,
            valid_mask,
            self.neighbor_count,
        )
        grouped_xyz = _index_points(xyz, neighbor_indices) - centers.unsqueeze(2)
        grouped = grouped_xyz
        if features is not None:
            grouped_features = _index_points(features, neighbor_indices)
            grouped = torch.cat((grouped_xyz, grouped_features), dim=-1)

        encoded = self.local_mlp(grouped.permute(0, 3, 1, 2))
        pooled = encoded.max(dim=-1).values.transpose(1, 2)
        center_mask = torch.ones(
            centers.shape[:2],
            dtype=torch.bool,
            device=centers.device,
        )
        return centers, pooled, center_mask


@dataclass(frozen=True)
class TinyPointNetPPConfig:
    """Tiny PointNet++ 结构配置。"""

    output_dim: int = 128
    sa1_points: int = 64
    sa1_neighbors: int = 24
    sa2_points: int = 16
    sa2_neighbors: int = 16

    def __post_init__(self) -> None:
        values = (
            self.output_dim,
            self.sa1_points,
            self.sa1_neighbors,
            self.sa2_points,
            self.sa2_neighbors,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in values
        ):
            raise TypeError("PointNet++ 配置必须是整数")
        if min(values) <= 0:
            raise ValueError("PointNet++ 配置必须大于 0")


class TinyPointNetPPEncoder(nn.Module):
    """将 partial point cloud 编码为 L2-normalized embedding。

    输入点云 shape 为 ``[B, N, 3]``，单位和中心化方式由调用方统一。
    可选 ``valid_mask`` shape 为 ``[B, N]``，用于忽略 padding 或无效深度点。
    """

    def __init__(self, config: TinyPointNetPPConfig | None = None) -> None:
        super().__init__()
        self.config = config or TinyPointNetPPConfig()
        self.sa1 = _SetAbstraction(
            sample_count=self.config.sa1_points,
            neighbor_count=self.config.sa1_neighbors,
            input_channels=0,
            mlp_channels=(32, 32, 64),
        )
        self.sa2 = _SetAbstraction(
            sample_count=self.config.sa2_points,
            neighbor_count=self.config.sa2_neighbors,
            input_channels=64,
            mlp_channels=(64, 96, 128),
        )
        self.projection = nn.Sequential(
            nn.Linear(128, 128, bias=False),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, self.config.output_dim),
        )

    @staticmethod
    def _validate_inputs(
        points: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if not isinstance(points, torch.Tensor):
            raise TypeError("points 必须是 torch.Tensor")
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError("points shape 必须为 [B, N, 3]")
        if points.shape[0] == 0 or points.shape[1] == 0:
            raise ValueError("points 不能为空")
        if not points.is_floating_point():
            raise TypeError("points 必须是浮点 Tensor")
        if not torch.isfinite(points).all():
            raise ValueError("points 包含 NaN 或 Inf")

        if valid_mask is None:
            mask = torch.ones(
                points.shape[:2],
                dtype=torch.bool,
                device=points.device,
            )
        else:
            if valid_mask.shape != points.shape[:2]:
                raise ValueError("valid_mask shape 必须为 [B, N]")
            mask = valid_mask.to(device=points.device, dtype=torch.bool)
        if not mask.any(dim=1).all():
            raise ValueError("每个点云至少需要一个有效点")
        return mask

    def forward(
        self,
        points: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mask = self._validate_inputs(points, valid_mask)
        xyz1, features1, mask1 = self.sa1(points, None, mask)
        _, features2, _ = self.sa2(xyz1, features1, mask1)
        global_features = features2.max(dim=1).values
        embeddings = self.projection(global_features)
        return functional.normalize(embeddings, p=2, dim=-1)

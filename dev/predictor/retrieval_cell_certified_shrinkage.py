"""按 Retriever cell 约束 Demo residual 的轻量支持域证书。"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RetrievalCellCertificate:
    """固定校准 bank 上的 retrieval-cell Lipschitz 下界。"""

    support_geometry: torch.Tensor
    support_benefit: torch.Tensor
    support_cell: torch.Tensor
    geometry_mean: torch.Tensor
    geometry_std: torch.Tensor
    lipschitz_constant: float

    def __post_init__(self) -> None:
        geometry = self.support_geometry
        if geometry.ndim != 2 or len(geometry) == 0:
            raise ValueError("support_geometry 必须是非空二维张量")
        if self.support_benefit.shape != (len(geometry),):
            raise ValueError("support_benefit shape 错误")
        if self.support_cell.shape != (len(geometry),):
            raise ValueError("support_cell shape 错误")
        if self.geometry_mean.shape != (geometry.shape[1],):
            raise ValueError("geometry_mean shape 错误")
        if self.geometry_std.shape != self.geometry_mean.shape:
            raise ValueError("geometry_std shape 错误")
        if not bool(torch.isfinite(geometry).all()):
            raise ValueError("support_geometry 包含非有限值")
        if not bool(torch.isfinite(self.support_benefit).all()):
            raise ValueError("support_benefit 包含非有限值")
        if not bool(
            ((self.support_benefit >= 0.0) & (self.support_benefit <= 1.0)).all()
        ):
            raise ValueError("support_benefit 必须位于 [0,1]")
        if not bool((self.geometry_std > 0.0).all()):
            raise ValueError("geometry_std 必须为正")
        if not torch.isfinite(torch.tensor(self.lipschitz_constant)):
            raise ValueError("lipschitz_constant 必须有限")
        if self.lipschitz_constant < 0.0:
            raise ValueError("lipschitz_constant 不能为负")

    @property
    def geometry_dim(self) -> int:
        return int(self.support_geometry.shape[1])

    def lower_bound(
        self,
        query_geometry: torch.Tensor,
        query_cell: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回收益下界、同 cell 最近距离与是否具有校准支持。"""
        if query_geometry.ndim != 2:
            raise ValueError("query_geometry 必须是二维张量")
        if query_geometry.shape[1] != self.geometry_dim:
            raise ValueError("query geometry 维度与证书不一致")
        if query_cell.shape != (len(query_geometry),):
            raise ValueError("query_cell shape 错误")
        device = query_geometry.device
        dtype = query_geometry.dtype
        support = self.support_geometry.to(device=device, dtype=dtype)
        mean = self.geometry_mean.to(device=device, dtype=dtype)
        std = self.geometry_std.to(device=device, dtype=dtype)
        benefit = self.support_benefit.to(device=device, dtype=dtype)
        support_cell = self.support_cell.to(device=device)
        standardized_query = (query_geometry - mean) / std
        standardized_support = (support - mean) / std
        distances = torch.cdist(
            standardized_query,
            standardized_support,
        ) / self.geometry_dim**0.5
        same_cell = query_cell.to(device=device)[:, None] == support_cell[None]
        has_support = same_cell.any(dim=1)
        infinity = torch.full_like(distances, torch.inf)
        nearest_distance = torch.where(same_cell, distances, infinity).min(
            dim=1
        ).values
        nearest_distance = torch.where(
            has_support,
            nearest_distance,
            torch.full_like(nearest_distance, torch.inf),
        )
        candidates = benefit[None] - self.lipschitz_constant * distances
        candidates = candidates.masked_fill(~same_cell, -torch.inf)
        lower = candidates.max(dim=1).values
        lower = torch.where(has_support, lower.clamp(0.0, 1.0), 0.0)
        return lower, nearest_distance, has_support


def empirical_cell_lipschitz_constant(
    geometry: torch.Tensor,
    benefit: torch.Tensor,
    cell: torch.Tensor,
    *,
    geometry_mean: torch.Tensor,
    geometry_std: torch.Tensor,
    minimum_distance: float = 1e-6,
) -> tuple[float, dict[str, int | float]]:
    """计算同 retrieval cell 校准点对的最大经验斜率。"""
    if geometry.ndim != 2 or len(geometry) == 0:
        raise ValueError("geometry 必须是非空二维张量")
    if benefit.shape != (len(geometry),) or cell.shape != (len(geometry),):
        raise ValueError("benefit/cell shape 错误")
    if geometry_mean.shape != (geometry.shape[1],):
        raise ValueError("geometry_mean shape 错误")
    if geometry_std.shape != geometry_mean.shape:
        raise ValueError("geometry_std shape 错误")
    if minimum_distance <= 0.0:
        raise ValueError("minimum_distance 必须为正")
    standardized = (geometry - geometry_mean) / geometry_std
    distances = torch.pdist(standardized) / geometry.shape[1] ** 0.5
    rows, columns = torch.triu_indices(
        len(geometry),
        len(geometry),
        offset=1,
        device=geometry.device,
    )
    same_cell = cell[rows] == cell[columns]
    valid = same_cell & (distances >= minimum_distance)
    if not bool(valid.any()):
        raise ValueError("没有可用于估计 Lipschitz 常数的同 cell 点对")
    slopes = (benefit[rows] - benefit[columns]).abs() / distances.clamp_min(
        minimum_distance
    )
    selected = slopes[valid]
    return float(selected.max()), {
        "total_pairs": int(len(distances)),
        "same_cell_pairs": int(same_cell.sum()),
        "valid_pairs": int(valid.sum()),
        "cells_with_support": int(torch.unique(cell).numel()),
        "maximum_slope": float(selected.max()),
        "median_slope": float(selected.median()),
    }


def support_certified_gate(
    base_gate: torch.Tensor,
    benefit_lower_bound: torch.Tensor,
) -> torch.Tensor:
    """将 learned gate 投影到不劣于 Demo copy 的认证区间。"""
    if base_gate.shape != benefit_lower_bound.shape:
        raise ValueError("base_gate 与 benefit_lower_bound shape 不一致")
    if not bool(((base_gate >= 0.0) & (base_gate <= 1.0)).all()):
        raise ValueError("base_gate 必须位于 [0,1]")
    if not bool(
        ((benefit_lower_bound >= 0.0) & (benefit_lower_bound <= 1.0)).all()
    ):
        raise ValueError("benefit_lower_bound 必须位于 [0,1]")
    return torch.minimum(base_gate, 2.0 * benefit_lower_bound).clamp(0.0, 1.0)


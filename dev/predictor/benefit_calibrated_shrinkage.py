"""冻结 Demo residual 的理论可解释收益校准收缩。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from dev.predictor.query_aligned_transport import (
    QueryAlignedLowRankDemoTransport,
)


GATE_FEATURE_DIM = 40


@dataclass(frozen=True)
class BenefitGateConfig:
    """小型 chunk-level shrinkage gate 配置。"""

    input_dim: int = GATE_FEATURE_DIM
    hidden_dim: int = 32

    def __post_init__(self) -> None:
        if min(self.input_dim, self.hidden_dim) <= 0:
            raise ValueError("gate 尺寸必须为正")
        if self.input_dim != GATE_FEATURE_DIM:
            raise ValueError(f"gate input_dim 必须为 {GATE_FEATURE_DIM}")


class BenefitCalibratedGate(nn.Module):
    """从 inference-only diagnostics 预测 `[0,1]` residual gate。"""

    def __init__(
        self,
        config: BenefitGateConfig | None = None,
        *,
        feature_mean: torch.Tensor | None = None,
        feature_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.config = config or BenefitGateConfig()
        mean = (
            torch.zeros(self.config.input_dim)
            if feature_mean is None
            else feature_mean.float()
        )
        std = (
            torch.ones(self.config.input_dim)
            if feature_std is None
            else feature_std.float()
        )
        if mean.shape != (self.config.input_dim,) or std.shape != mean.shape:
            raise ValueError("feature mean/std shape 错误")
        self.register_buffer("feature_mean", mean)
        self.register_buffer("feature_std", std.clamp_min(1e-5))
        self.network = nn.Sequential(
            nn.LayerNorm(self.config.input_dim),
            nn.Linear(self.config.input_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = (features - self.feature_mean) / self.feature_std
        return torch.sigmoid(self.network(normalized)).squeeze(-1)


def _attention_statistics(
    attention: torch.Tensor,
    demo_mask: torch.Tensor,
) -> torch.Tensor:
    valid = demo_mask.bool()
    key_mask = valid[:, None, None, :]
    probabilities = attention.masked_fill(~key_mask, 0.0).clamp_min(0.0)
    probabilities = probabilities / probabilities.sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(1e-8)
    entropy = -(
        probabilities * probabilities.clamp_min(1e-8).log()
    ).sum(dim=-1)
    valid_count = valid.sum(dim=1).clamp_min(2).float().log()
    entropy = entropy / valid_count[:, None, None]
    peak = probabilities.max(dim=-1).values
    return torch.stack(
        (
            entropy.mean(dim=(1, 2)),
            entropy.std(dim=(1, 2), unbiased=False),
            peak.mean(dim=(1, 2)),
            peak.std(dim=(1, 2), unbiased=False),
        ),
        dim=-1,
    )


@torch.inference_mode()
def frozen_transport_features(
    transport: QueryAlignedLowRankDemoTransport,
    query_geometry: torch.Tensor,
    demo_geometry: torch.Tensor,
    demo_actions: torch.Tensor,
    demo_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """单次冻结 transport 前向，返回预测与 40D gate 特征。"""
    prediction, delta, attention = transport.forward_with_diagnostics(
        query_geometry,
        demo_geometry,
        demo_actions,
        demo_mask,
    )
    residual_norm = torch.linalg.vector_norm(
        prediction[..., :3] - demo_actions[..., :3],
        dim=-1,
    )
    valid = demo_mask.float()
    denominator = valid.sum(dim=1).clamp_min(1.0)
    residual_mean = (residual_norm * valid).sum(dim=1) / denominator
    residual_max = residual_norm.masked_fill(
        ~demo_mask.bool(),
        0.0,
    ).max(dim=1).values
    residual_statistics = torch.stack((residual_mean, residual_max), dim=-1)
    features = torch.cat(
        (
            delta,
            delta.abs(),
            residual_statistics,
            _attention_statistics(attention, demo_mask),
        ),
        dim=-1,
    )
    if features.shape[-1] != GATE_FEATURE_DIM:
        raise RuntimeError("gate feature dimension 不符合冻结协议")
    return prediction, features


def optimal_shrinkage_target(
    copy: torch.Tensor,
    transported: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 translation squared loss 上的解析 `g*` 与 residual energy。"""
    valid = mask.float().unsqueeze(-1)
    residual = (transported[..., :3] - copy[..., :3]) * valid
    target_offset = (target[..., :3] - copy[..., :3]) * valid
    numerator = (residual * target_offset).sum(dim=(1, 2))
    energy = residual.square().sum(dim=(1, 2))
    gate = (numerator / energy.clamp_min(1e-8)).clamp(0.0, 1.0)
    gate = torch.where(energy > 1e-8, gate, torch.zeros_like(gate))
    return gate, energy


def constant_optimal_shrinkage(
    copy: torch.Tensor,
    transported: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """返回 calibration set 上的单一最小二乘收缩系数。"""
    valid = mask.float().unsqueeze(-1)
    residual = (transported[..., :3] - copy[..., :3]) * valid
    target_offset = (target[..., :3] - copy[..., :3]) * valid
    numerator = (residual * target_offset).sum()
    energy = residual.square().sum()
    return (numerator / energy.clamp_min(1e-8)).clamp(0.0, 1.0)


def apply_shrinkage(
    copy: torch.Tensor,
    transported: torch.Tensor,
    gate: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """沿冻结 residual 做凸组合，并保持无 Demo 时严格零输出。"""
    if gate.ndim != 1 or len(gate) != len(copy):
        raise ValueError("gate 必须为 batch 长度的一维张量")
    if bool(((gate < 0.0) | (gate > 1.0)).any()):
        raise ValueError("gate 必须位于 [0,1]")
    prediction = copy + gate[:, None, None] * (transported - copy)
    return prediction * mask.float().unsqueeze(-1)


def shrinkage_excess_identity(
    residual_energy: torch.Tensor,
    predicted_gate: torch.Tensor,
    optimal_gate: torch.Tensor,
) -> torch.Tensor:
    """内点情形的理论超额 squared loss 项，供诊断使用。"""
    return residual_energy * (predicted_gate - optimal_gate).square()

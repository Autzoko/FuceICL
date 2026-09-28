"""Permutation-equivariant 的轻量 Demo action-utility selector。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class UtilitySelectorConfig:
    """Selector 容量配置；候选数不影响参数量。"""

    input_dim: int
    hidden_dim: int = 64

    def __post_init__(self) -> None:
        if min(self.input_dim, self.hidden_dim) <= 0:
            raise ValueError("input_dim 和 hidden_dim 必须为正")


class DemoUtilitySelector(nn.Module):
    """用共享编码器和 set mean 为每个 Demo 候选预测 utility score。"""

    def __init__(
        self,
        config: UtilitySelectorConfig,
        *,
        feature_mean: torch.Tensor | None = None,
        feature_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        mean = (
            torch.zeros(config.input_dim)
            if feature_mean is None
            else feature_mean.float()
        )
        std = (
            torch.ones(config.input_dim)
            if feature_std is None
            else feature_std.float()
        )
        if mean.shape != (config.input_dim,) or std.shape != mean.shape:
            raise ValueError("feature mean/std shape 错误")
        self.register_buffer("feature_mean", mean)
        self.register_buffer("feature_std", std.clamp_min(1e-5))
        self.candidate_encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
        )
        self.score_head = nn.Sequential(
            nn.Linear(2 * config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != self.config.input_dim:
            raise ValueError("features 必须是 [batch, candidates, input_dim]")
        normalized = (features - self.feature_mean) / self.feature_std
        candidate = self.candidate_encoder(normalized)
        set_context = candidate.mean(dim=1, keepdim=True).expand_as(candidate)
        return self.score_head(
            torch.cat((candidate, set_context), dim=-1)
        ).squeeze(-1)

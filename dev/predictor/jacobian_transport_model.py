"""基于 canonical geometry 差分的一阶 Demo action transport。"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from dev.predictor.canonical_geometry import GEOMETRY_DIM


ACTION_DIM = 7


@dataclass(frozen=True)
class JacobianTransportConfig:
    """局部 Jacobian action transport 的轻量配置。"""

    horizon: int = 6
    geometry_dim: int = GEOMETRY_DIM
    action_dim: int = ACTION_DIM
    hidden_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    feedforward_dim: int = 256
    dropout: float = 0.1
    residual_limit: float = 2.0

    def __post_init__(self) -> None:
        positive = (
            self.horizon,
            self.geometry_dim,
            self.action_dim,
            self.hidden_dim,
            self.num_layers,
            self.num_heads,
            self.feedforward_dim,
        )
        if min(positive) <= 0:
            raise ValueError("模型尺寸必须为正")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim 必须能被 num_heads 整除")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout 必须位于 [0, 1)")
        if self.residual_limit <= 0:
            raise ValueError("residual_limit 必须为正")


class LocalJacobianActionTransport(nn.Module):
    """用学习的一阶 Jacobian 将 Demo actions 搬运到 query geometry。

    ``query_geometry == demo_geometry`` 时残差严格为零；
    query 只能通过几何差分进入 Jacobian 乘法，不能独立生成动作。
    没有有效 Demo 时输出也严格为零。
    """

    def __init__(
        self,
        config: JacobianTransportConfig | None = None,
        *,
        geometry_mean: torch.Tensor | None = None,
        geometry_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.config = config or JacobianTransportConfig()
        mean = (
            torch.zeros(self.config.geometry_dim)
            if geometry_mean is None
            else geometry_mean.float()
        )
        std = (
            torch.ones(self.config.geometry_dim)
            if geometry_std is None
            else geometry_std.float()
        )
        if mean.shape != (self.config.geometry_dim,) or std.shape != mean.shape:
            raise ValueError("geometry mean/std shape 错误")
        self.register_buffer("geometry_mean", mean)
        self.register_buffer("geometry_std", std.clamp_min(1e-4))
        self.action_projection = nn.Linear(
            self.config.action_dim,
            self.config.hidden_dim,
        )
        self.demo_geometry_projection = nn.Linear(
            self.config.geometry_dim,
            self.config.hidden_dim,
        )
        self.positions = nn.Parameter(
            torch.empty(self.config.horizon, self.config.hidden_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.config.hidden_dim,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.feedforward_dim,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.demo_encoder = nn.TransformerEncoder(
            layer,
            num_layers=self.config.num_layers,
            norm=nn.LayerNorm(self.config.hidden_dim),
            enable_nested_tensor=False,
        )
        self.jacobian_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(
                self.config.hidden_dim,
                self.config.action_dim * self.config.geometry_dim,
            ),
        )
        nn.init.normal_(self.positions, std=0.02)

    def forward(
        self,
        query_geometry: torch.Tensor,
        demo_geometry: torch.Tensor,
        demo_actions: torch.Tensor,
        demo_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid = demo_mask.bool()
        mask = valid.float().unsqueeze(-1)
        global_valid = valid.any(dim=1, keepdim=True).float()
        query = (query_geometry - self.geometry_mean) / self.geometry_std
        demo = (demo_geometry - self.geometry_mean) / self.geometry_std
        demo = demo * global_valid
        delta = (query - demo) * global_valid
        tokens = (
            self.action_projection(demo_actions * mask)
            + self.demo_geometry_projection(demo)[:, None, :]
            + self.positions[None, :, :]
        ) * mask
        safe_valid = valid.clone()
        no_demo = ~safe_valid.any(dim=1)
        safe_valid[no_demo, 0] = True
        memory = self.demo_encoder(tokens, src_key_padding_mask=~safe_valid)
        jacobian = self.jacobian_head(memory).reshape(
            -1,
            self.config.horizon,
            self.config.action_dim,
            self.config.geometry_dim,
        )
        linear_residual = torch.einsum("btag,bg->bta", jacobian, delta)
        residual = self.config.residual_limit * torch.tanh(
            linear_residual / math.sqrt(self.config.geometry_dim)
        )
        return (demo_actions + residual) * mask

"""Query-aligned、严格 Demo-residual 的低秩 action transport。"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from dev.predictor.canonical_geometry import GEOMETRY_DIM


ACTION_DIM = 7


@dataclass(frozen=True)
class QueryAlignedTransportConfig:
    """轻量 query-to-Demo cross-attention transport 配置。"""

    horizon: int = 6
    geometry_dim: int = GEOMETRY_DIM
    action_dim: int = ACTION_DIM
    transported_action_dim: int = 3
    rank: int = 4
    hidden_dim: int = 96
    num_layers: int = 2
    num_heads: int = 4
    feedforward_dim: int = 192
    dropout: float = 0.1
    residual_limit: float = 2.0

    def __post_init__(self) -> None:
        positive = (
            self.horizon,
            self.geometry_dim,
            self.action_dim,
            self.transported_action_dim,
            self.rank,
            self.hidden_dim,
            self.num_layers,
            self.num_heads,
            self.feedforward_dim,
            self.residual_limit,
        )
        if min(positive) <= 0:
            raise ValueError("模型尺寸必须为正")
        if self.transported_action_dim > self.action_dim:
            raise ValueError("transported action dim 不能超过 action dim")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim 必须能被 num_heads 整除")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout 必须位于 [0,1)")


class QueryAlignedLowRankDemoTransport(nn.Module):
    """由 query offset 对 Demo tokens 做 cross-attention 后生成低秩算子。

    Query 没有独立 action head，只能通过 geometry delta 选择 Demo memory，
    并进入低秩乘法。相同 geometry 严格退化为 Demo copy；
    无 Demo 时严格输出零。
    """

    def __init__(
        self,
        config: QueryAlignedTransportConfig | None = None,
        *,
        geometry_mean: torch.Tensor | None = None,
        geometry_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.config = config or QueryAlignedTransportConfig()
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
        self.delta_projection = nn.Linear(
            self.config.geometry_dim,
            self.config.hidden_dim,
        )
        self.memory_positions = nn.Parameter(
            torch.empty(self.config.horizon, self.config.hidden_dim)
        )
        self.query_positions = nn.Parameter(
            torch.empty(self.config.horizon, self.config.hidden_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.hidden_dim,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.feedforward_dim,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.demo_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.config.num_layers,
            norm=nn.LayerNorm(self.config.hidden_dim),
            enable_nested_tensor=False,
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.config.hidden_dim,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(self.config.hidden_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.feedforward_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.feedforward_dim, self.config.hidden_dim),
        )
        self.output_norm = nn.LayerNorm(self.config.hidden_dim)
        self.left_head = nn.Linear(
            self.config.hidden_dim,
            self.config.transported_action_dim * self.config.rank,
        )
        self.right_head = nn.Linear(
            self.config.hidden_dim,
            self.config.rank * self.config.geometry_dim,
        )
        nn.init.normal_(self.memory_positions, std=0.02)
        nn.init.normal_(self.query_positions, std=0.02)

    def transport_factors(
        self,
        query_geometry: torch.Tensor,
        demo_geometry: torch.Tensor,
        demo_actions: torch.Tensor,
        demo_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回低秩算子、normalized delta 与 attention weights。"""
        valid = demo_mask.bool()
        mask = valid.float().unsqueeze(-1)
        global_valid = valid.any(dim=1, keepdim=True).float()
        query = (query_geometry - self.geometry_mean) / self.geometry_std
        demo = (demo_geometry - self.geometry_mean) / self.geometry_std
        demo = demo * global_valid
        delta = (query - demo) * global_valid
        memory_tokens = (
            self.action_projection(demo_actions * mask)
            + self.demo_geometry_projection(demo)[:, None]
            + self.memory_positions[None]
        ) * mask
        safe_valid = valid.clone()
        no_demo = ~safe_valid.any(dim=1)
        safe_valid[no_demo, 0] = True
        memory = self.demo_encoder(
            memory_tokens,
            src_key_padding_mask=~safe_valid,
        )
        query_tokens = (
            self.delta_projection(delta)[:, None]
            + self.query_positions[None]
        )
        attended, attention = self.cross_attention(
            query_tokens,
            memory,
            memory,
            key_padding_mask=~safe_valid,
            need_weights=True,
            average_attn_weights=False,
        )
        aligned = self.attention_norm(query_tokens + attended)
        aligned = self.output_norm(aligned + self.feedforward(aligned))
        aligned = aligned * global_valid[:, None]
        left = self.left_head(aligned).reshape(
            -1,
            self.config.horizon,
            self.config.transported_action_dim,
            self.config.rank,
        )
        right = self.right_head(aligned).reshape(
            -1,
            self.config.horizon,
            self.config.rank,
            self.config.geometry_dim,
        )
        return left, right, delta, attention

    def forward(
        self,
        query_geometry: torch.Tensor,
        demo_geometry: torch.Tensor,
        demo_actions: torch.Tensor,
        demo_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid = demo_mask.bool()
        mask = valid.float().unsqueeze(-1)
        left, right, delta, _ = self.transport_factors(
            query_geometry,
            demo_geometry,
            demo_actions,
            valid,
        )
        latent = torch.einsum("btrg,bg->btr", right, delta)
        continuous = torch.einsum("btar,btr->bta", left, latent)
        continuous = self.config.residual_limit * torch.tanh(
            continuous / math.sqrt(self.config.geometry_dim * self.config.rank)
        )
        residual = torch.zeros_like(demo_actions)
        residual[..., : self.config.transported_action_dim] = continuous
        return (demo_actions + residual) * mask

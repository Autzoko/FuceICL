"""Demo-conditioned 低秩局部 action transport。"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from dev.predictor.canonical_geometry import GEOMETRY_DIM


ACTION_DIM = 7


@dataclass(frozen=True)
class LowRankTransportConfig:
    """低秩搬运算子的轻量固定配置。"""

    horizon: int = 6
    geometry_dim: int = GEOMETRY_DIM
    action_dim: int = ACTION_DIM
    transported_action_dim: int = 6
    rank: int = 4
    hidden_dim: int = 96
    num_layers: int = 2
    num_heads: int = 4
    feedforward_dim: int = 192
    dropout: float = 0.1
    residual_limit: float = 2.0
    geometry_mask: tuple[float, ...] | None = None
    transition_context: bool = False

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
            raise ValueError("transported_action_dim 不能超过 action_dim")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim 必须能被 num_heads 整除")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout 必须位于 [0, 1)")
        if self.geometry_mask is not None:
            if len(self.geometry_mask) != self.geometry_dim:
                raise ValueError("geometry_mask 长度必须等于 geometry_dim")
            if any(value not in (0.0, 1.0) for value in self.geometry_mask):
                raise ValueError("geometry_mask 只允许 0/1")


class LowRankDemoActionTransport(nn.Module):
    """由 Demo 生成低秩 Jacobian，并搬运其 action chunk。

    Query 只能通过 canonical geometry difference 进入低秩乘法；因此不存在
    query-only action shortcut。相同几何严格返回 Demo action，无有效 Demo 时严格
    返回零。默认只搬运连续 6D EEF command，gripper 保留 Demo 动作先验。
    """

    def __init__(
        self,
        config: LowRankTransportConfig | None = None,
        *,
        geometry_mean: torch.Tensor | None = None,
        geometry_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.config = config or LowRankTransportConfig()
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
        feature_mask = (
            torch.ones(self.config.geometry_dim)
            if self.config.geometry_mask is None
            else torch.tensor(self.config.geometry_mask, dtype=torch.float32)
        )
        self.register_buffer(
            "geometry_feature_mask",
            feature_mask,
            persistent=False,
        )

        self.action_projection = nn.Linear(
            self.config.action_dim,
            self.config.hidden_dim,
        )
        self.geometry_projection = nn.Linear(
            self.config.geometry_dim,
            self.config.hidden_dim,
        )
        self.transition_projection = (
            nn.Linear(self.config.geometry_dim, self.config.hidden_dim)
            if self.config.transition_context
            else None
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
        self.left_head = nn.Linear(
            self.config.hidden_dim,
            self.config.transported_action_dim * self.config.rank,
        )
        self.right_head = nn.Linear(
            self.config.hidden_dim,
            self.config.rank * self.config.geometry_dim,
        )
        nn.init.normal_(self.positions, std=0.02)

    def _encode_factors(
        self,
        demo_geometry: torch.Tensor,
        demo_actions: torch.Tensor,
        valid: torch.Tensor,
        demo_geometry_sequence: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = valid.float().unsqueeze(-1)
        global_valid = valid.any(dim=1, keepdim=True).float()
        normalized_demo = (
            (demo_geometry - self.geometry_mean)
            / self.geometry_std
            * self.geometry_feature_mask
            * global_valid
        )
        if self.config.transition_context:
            expected = (
                len(demo_actions),
                self.config.horizon + 1,
                self.config.geometry_dim,
            )
            if demo_geometry_sequence is None or demo_geometry_sequence.shape != expected:
                raise ValueError(
                    "transition context 要求 demo_geometry_sequence shape="
                    f"{expected}"
                )
            sequence = (
                (demo_geometry_sequence - self.geometry_mean)
                / self.geometry_std
                * self.geometry_feature_mask
                * global_valid[:, None, :]
            )
            transitions = (
                (demo_geometry_sequence[:, 1:] - demo_geometry_sequence[:, :-1])
                / self.geometry_std
                * self.geometry_feature_mask
                * global_valid[:, None, :]
            )
            geometry_tokens = self.geometry_projection(sequence[:, :-1])
            if self.transition_projection is None:
                raise RuntimeError("transition projection 未初始化")
            geometry_tokens = geometry_tokens + self.transition_projection(transitions)
        else:
            geometry_tokens = self.geometry_projection(normalized_demo)[:, None, :]
        tokens = (
            self.action_projection(demo_actions * mask)
            + geometry_tokens
            + self.positions[None, :, :]
        ) * mask
        safe_valid = valid.clone()
        no_demo = ~safe_valid.any(dim=1)
        safe_valid[no_demo, 0] = True
        memory = self.demo_encoder(tokens, src_key_padding_mask=~safe_valid)
        left = self.left_head(memory).reshape(
            -1,
            self.config.horizon,
            self.config.transported_action_dim,
            self.config.rank,
        )
        right = self.right_head(memory).reshape(
            -1,
            self.config.horizon,
            self.config.rank,
            self.config.geometry_dim,
        )
        return left * mask.unsqueeze(-1), right * mask.unsqueeze(-1)

    def transport_factors(
        self,
        demo_geometry: torch.Tensor,
        demo_actions: torch.Tensor,
        demo_mask: torch.Tensor,
        demo_geometry_sequence: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回每个 action token 的低秩算子因子，供正则和诊断使用。"""
        return self._encode_factors(
            demo_geometry,
            demo_actions,
            demo_mask.bool(),
            demo_geometry_sequence,
        )

    def operator_frobenius_upper_bound(
        self,
        demo_geometry: torch.Tensor,
        demo_actions: torch.Tensor,
        demo_mask: torch.Tensor,
        demo_geometry_sequence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """返回 ``||U||_F ||V||_F``，作为每个 token 的稳定性上界。"""
        left, right = self.transport_factors(
            demo_geometry,
            demo_actions,
            demo_mask,
            demo_geometry_sequence,
        )
        return torch.linalg.matrix_norm(left) * torch.linalg.matrix_norm(right)

    def forward(
        self,
        query_geometry: torch.Tensor,
        demo_geometry: torch.Tensor,
        demo_actions: torch.Tensor,
        demo_mask: torch.Tensor,
        demo_geometry_sequence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        valid = demo_mask.bool()
        mask = valid.float().unsqueeze(-1)
        global_valid = valid.any(dim=1, keepdim=True).float()
        delta = (
            (query_geometry - demo_geometry)
            / self.geometry_std
            * self.geometry_feature_mask
            * global_valid
        )
        left, right = self._encode_factors(
            demo_geometry,
            demo_actions,
            valid,
            demo_geometry_sequence,
        )
        latent = torch.einsum("btrg,bg->btr", right, delta)
        continuous = torch.einsum("btar,btr->bta", left, latent)
        continuous = self.config.residual_limit * torch.tanh(
            continuous / math.sqrt(self.config.geometry_dim * self.config.rank)
        )
        residual = torch.zeros_like(demo_actions)
        residual[..., : self.config.transported_action_dim] = continuous
        return (demo_actions + residual) * mask

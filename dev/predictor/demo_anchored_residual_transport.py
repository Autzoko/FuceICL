"""以 Demo action 为硬锚点的低秩局部 residual transport。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class DemoAnchoredTransportConfig:
    """轻量 transport 的固定容量与结构约束。"""

    state_dim: int = 21
    action_horizon: int = 6
    rank: int = 4
    maximum_translation_residual_m: float = 0.001

    def __post_init__(self) -> None:
        positive = (
            self.state_dim,
            self.action_horizon,
            self.rank,
            self.maximum_translation_residual_m,
        )
        if min(positive) <= 0:
            raise ValueError("Demo anchored transport 配置必须为正")


class DemoAnchoredResidualTransport(nn.Module):
    """对 Demo translation 施加低秩、零偏置、有界残差。"""

    def __init__(
        self,
        config: DemoAnchoredTransportConfig | None = None,
        *,
        state_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.config = config or DemoAnchoredTransportConfig()
        standard_deviation = (
            torch.ones(self.config.state_dim)
            if state_std is None
            else state_std.float()
        )
        if standard_deviation.shape != (self.config.state_dim,):
            raise ValueError("state_std shape 错误")
        if not bool(torch.isfinite(standard_deviation).all()):
            raise ValueError("state_std 含 NaN/Inf")
        self.register_buffer("state_std", standard_deviation.clamp_min(1e-5))
        self.reduce = nn.Linear(self.config.state_dim, self.config.rank, bias=False)
        self.expand = nn.Linear(
            self.config.rank,
            self.config.action_horizon * 3,
            bias=False,
        )

    def forward(
        self,
        query_state: torch.Tensor,
        demo_state: torch.Tensor,
        demo_action: torch.Tensor,
        demo_mask: torch.Tensor,
    ) -> torch.Tensor:
        """预测 `[B,H,7]` action；无 Demo 时严格为零。"""
        if query_state.ndim != 2 or query_state.shape != demo_state.shape:
            raise ValueError("query/demo state shape 错误")
        batch = query_state.shape[0]
        if query_state.shape[1] != self.config.state_dim:
            raise ValueError("state feature dimension 错误")
        expected_action = (batch, self.config.action_horizon, 7)
        if demo_action.shape != expected_action or demo_mask.shape != (batch,):
            raise ValueError("demo action/mask shape 错误")
        if not all(
            bool(torch.isfinite(value).all())
            for value in (query_state, demo_state, demo_action, demo_mask)
        ):
            raise ValueError("transport 输入含 NaN/Inf")
        if bool(((demo_mask < 0.0) | (demo_mask > 1.0)).any()):
            raise ValueError("demo_mask 必须在 [0,1]")

        delta = (query_state - demo_state) / self.state_std
        residual = self.expand(self.reduce(delta)).view(
            batch,
            self.config.action_horizon,
            3,
        )
        residual = self.config.maximum_translation_residual_m * torch.tanh(
            residual
        )
        translation = demo_action[..., :3] + residual
        output = torch.cat((translation, demo_action[..., 3:]), dim=-1)
        return output * demo_mask[:, None, None]

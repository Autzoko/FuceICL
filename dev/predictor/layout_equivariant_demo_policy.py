"""以 layout-transported Demo action 为硬先验的轻量 residual policy。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class LayoutEquivariantDemoPolicyConfig:
    """Demo-anchored predictor 的容量与物理边界。"""

    state_dim: int = 21
    action_horizon: int = 6
    hidden_dim: int = 64
    maximum_translation_residual_m: float = 0.002
    maximum_rotation_residual_rad: float = 0.01
    translation_action_scale_m: float = 0.01
    rotation_action_scale_rad: float = 0.1

    def __post_init__(self) -> None:
        positive = (
            self.state_dim,
            self.action_horizon,
            self.hidden_dim,
            self.maximum_translation_residual_m,
            self.maximum_rotation_residual_rad,
            self.translation_action_scale_m,
            self.rotation_action_scale_rad,
        )
        if min(positive) <= 0:
            raise ValueError("policy 配置必须为正")


class LayoutEquivariantDemoPolicy(nn.Module):
    """预测有界 residual；operation 与主动作方向由 Demo anchor 提供。"""

    def __init__(
        self,
        config: LayoutEquivariantDemoPolicyConfig | None = None,
        *,
        state_mean: torch.Tensor | None = None,
        state_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.config = config or LayoutEquivariantDemoPolicyConfig()
        mean = (
            torch.zeros(self.config.state_dim)
            if state_mean is None
            else state_mean.float()
        )
        standard_deviation = (
            torch.ones(self.config.state_dim)
            if state_std is None
            else state_std.float()
        )
        expected = (self.config.state_dim,)
        if mean.shape != expected or standard_deviation.shape != expected:
            raise ValueError("state normalizer shape 错误")
        if not bool(torch.isfinite(mean).all()) or not bool(
            torch.isfinite(standard_deviation).all()
        ):
            raise ValueError("state normalizer 含 NaN/Inf")
        self.register_buffer("state_mean", mean)
        self.register_buffer("state_std", standard_deviation.clamp_min(1e-5))

        action_features = self.config.action_horizon * 7
        input_dim = 2 * self.config.state_dim + action_features
        output_dim = self.config.action_horizon * 6
        self.residual = nn.Sequential(
            nn.Linear(input_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, output_dim),
        )

    def _action_features(self, action: torch.Tensor) -> torch.Tensor:
        scale = torch.ones_like(action)
        scale[..., :3] = self.config.translation_action_scale_m
        scale[..., 3:6] = self.config.rotation_action_scale_rad
        return action / scale

    def forward(
        self,
        query_state: torch.Tensor,
        demo_state: torch.Tensor,
        transported_demo_action: torch.Tensor,
        demo_mask: torch.Tensor,
    ) -> torch.Tensor:
        """返回 `[B,H,7]` action chunk；无 Demo 时严格为零。"""
        if query_state.ndim != 2 or query_state.shape != demo_state.shape:
            raise ValueError("query/demo state shape 错误")
        batch = query_state.shape[0]
        if query_state.shape[1] != self.config.state_dim:
            raise ValueError("state dimension 错误")
        expected_action = (batch, self.config.action_horizon, 7)
        if transported_demo_action.shape != expected_action:
            raise ValueError("transported Demo action shape 错误")
        if demo_mask.shape != (batch,):
            raise ValueError("demo_mask shape 错误")
        values = (
            query_state,
            demo_state,
            transported_demo_action,
            demo_mask,
        )
        if not all(bool(torch.isfinite(value).all()) for value in values):
            raise ValueError("policy 输入含 NaN/Inf")
        if bool(((demo_mask < 0.0) | (demo_mask > 1.0)).any()):
            raise ValueError("demo_mask 必须在 [0,1]")

        query_normalized = (query_state - self.state_mean) / self.state_std
        demo_normalized = (demo_state - self.state_mean) / self.state_std
        delta = query_normalized - demo_normalized
        features = torch.cat(
            (
                query_normalized,
                delta,
                self._action_features(transported_demo_action).flatten(1),
            ),
            dim=1,
        )
        residual = self.residual(features).view(
            batch, self.config.action_horizon, 6
        )
        translation = self.config.maximum_translation_residual_m * torch.tanh(
            residual[..., :3]
        )
        rotation = self.config.maximum_rotation_residual_rad * torch.tanh(
            residual[..., 3:6]
        )
        # 同一状态/Demo 时 gate=0，确保 identity anchor 精确成立。
        identity_gate = torch.clamp(
            torch.linalg.vector_norm(delta, dim=1)
            / self.config.state_dim**0.5,
            max=1.0,
        )
        bounded = torch.cat((translation, rotation), dim=-1)
        corrected = torch.cat(
            (
                transported_demo_action[..., :6]
                + identity_gate[:, None, None] * bounded,
                transported_demo_action[..., 6:],
            ),
            dim=-1,
        )
        return corrected * demo_mask[:, None, None]

"""仅在 query--Demo belief edge 上学习修正的轻量 action-prior Predictor。"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass(frozen=True)
class DemoRelativePolicyConfig:
    """Demo-relative Predictor 的容量与物理边界。"""

    state_dim: int = 26
    action_horizon: int = 6
    hidden_dim: int = 64
    maximum_translation_residual_m: float = 0.002
    maximum_rotation_residual_rad: float = 0.01
    translation_action_scale_m: float = 0.01
    rotation_action_scale_rad: float = 0.1

    def __post_init__(self) -> None:
        dimensions = (self.state_dim, self.action_horizon, self.hidden_dim)
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in dimensions
        ) or min(dimensions) <= 0:
            raise ValueError("policy 维度必须是正整数")
        scales = (
            self.maximum_translation_residual_m,
            self.maximum_rotation_residual_rad,
            self.translation_action_scale_m,
            self.rotation_action_scale_rad,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
            for value in scales
        ):
            raise ValueError("policy 物理尺度必须是有限正数")


def _floating_tensor(value: torch.Tensor, *, name: str) -> None:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} 必须是浮点 Tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} 含 NaN/Inf")


class DemoRelativePolicy(nn.Module):
    """用 belief difference 修正 raw Demo action，不含 absolute query head。"""

    def __init__(
        self,
        config: DemoRelativePolicyConfig | None = None,
        *,
        state_mean: torch.Tensor | None = None,
        state_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.config = config or DemoRelativePolicyConfig()
        mean = (
            torch.zeros(self.config.state_dim)
            if state_mean is None
            else state_mean.detach().float()
        )
        standard_deviation = (
            torch.ones(self.config.state_dim)
            if state_std is None
            else state_std.detach().float()
        )
        expected = (self.config.state_dim,)
        if mean.shape != expected or standard_deviation.shape != expected:
            raise ValueError("state normalizer shape 错误")
        if not bool(torch.isfinite(mean).all()) or not bool(
            torch.isfinite(standard_deviation).all()
        ):
            raise ValueError("state normalizer 含 NaN/Inf")
        if bool((standard_deviation <= 0).any()):
            raise ValueError("state_std 必须严格为正")
        self.register_buffer("state_mean", mean)
        self.register_buffer("state_std", standard_deviation.clamp_min(1e-5))

        action_features = self.config.action_horizon * 7
        input_dim = self.config.state_dim + action_features
        output_dim = self.config.action_horizon * 6
        self.residual = nn.Sequential(
            nn.Linear(input_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, output_dim),
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _action_features(self, action: torch.Tensor) -> torch.Tensor:
        scale = torch.ones_like(action)
        scale[..., :3] = self.config.translation_action_scale_m
        scale[..., 3:6] = self.config.rotation_action_scale_rad
        return action / scale

    def forward(
        self,
        query_state: torch.Tensor,
        demo_state: torch.Tensor,
        raw_demo_action: torch.Tensor,
        demo_mask: torch.Tensor,
    ) -> torch.Tensor:
        """返回 `[B,H,7]` action；query 仅以与 Demo 的差值进入网络。"""
        tensors = {
            "query_state": query_state,
            "demo_state": demo_state,
            "raw_demo_action": raw_demo_action,
            "demo_mask": demo_mask,
        }
        for name, value in tensors.items():
            _floating_tensor(value, name=name)
            if value.device != self.state_mean.device:
                raise ValueError(f"{name} device 与模型不一致")
            if value.dtype != self.state_mean.dtype:
                raise TypeError(f"{name} dtype 与模型不一致")
        if query_state.ndim != 2 or query_state.shape != demo_state.shape:
            raise ValueError("query/demo state shape 错误")
        batch = query_state.shape[0]
        if query_state.shape[1] != self.config.state_dim:
            raise ValueError("state dimension 错误")
        expected_action = (batch, self.config.action_horizon, 7)
        if raw_demo_action.shape != expected_action:
            raise ValueError("raw Demo action shape 错误")
        if demo_mask.shape != (batch,):
            raise ValueError("demo_mask shape 错误")
        if bool(((demo_mask < 0.0) | (demo_mask > 1.0)).any()):
            raise ValueError("demo_mask 必须位于 [0,1]")

        # mean 在差分中抵消；buffer 仍保留 checkpoint normalizer 契约。
        delta = (query_state - demo_state) / self.state_std
        features = torch.cat(
            (delta, self._action_features(raw_demo_action).flatten(1)),
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
        identity_gate = torch.clamp(
            torch.linalg.vector_norm(delta, dim=1)
            / self.config.state_dim**0.5,
            max=1.0,
        )
        bounded = torch.cat((translation, rotation), dim=-1)
        corrected = torch.cat(
            (
                raw_demo_action[..., :6]
                + identity_gate[:, None, None] * bounded,
                raw_demo_action[..., 6:],
            ),
            dim=-1,
        )
        return corrected * demo_mask[:, None, None]

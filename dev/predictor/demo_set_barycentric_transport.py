"""受 Retriever prior 约束的 Demo-set 逐步凸组合模型。"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass(frozen=True)
class DemoSetBarycentricConfig:
    """DSBT 容量、输入形状与结构消融配置。"""

    geometry_dim: int = 17
    action_horizon: int = 6
    token_dim: int = 48
    scorer_hidden_dim: int = 32
    maximum_odds_distortion: float = 4.0
    use_set_context: bool = True
    per_step_weights: bool = True

    def __post_init__(self) -> None:
        positive = (
            self.geometry_dim,
            self.action_horizon,
            self.token_dim,
            self.scorer_hidden_dim,
        )
        if min(positive) <= 0:
            raise ValueError("DSBT shape 与 hidden dimensions 必须为正")
        if self.maximum_odds_distortion <= 1.0:
            raise ValueError("maximum_odds_distortion 必须大于 1")


class DemoSetBarycentricTransport(nn.Module):
    """联合编码 Demo set，并生成 prior-anchored 逐步 mixture weights。"""

    def __init__(
        self,
        config: DemoSetBarycentricConfig | None = None,
        *,
        geometry_delta_mean: torch.Tensor | None = None,
        geometry_delta_std: torch.Tensor | None = None,
        action_delta_mean: torch.Tensor | None = None,
        action_delta_std: torch.Tensor | None = None,
        distance_mean: torch.Tensor | None = None,
        distance_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.config = config or DemoSetBarycentricConfig()
        geometry_shape = (self.config.geometry_dim,)
        action_shape = (self.config.action_horizon, 3)
        self.register_buffer(
            "geometry_delta_mean",
            self._normalizer(geometry_delta_mean, geometry_shape, 0.0),
        )
        self.register_buffer(
            "geometry_delta_std",
            self._normalizer(geometry_delta_std, geometry_shape, 1.0).clamp_min(
                1e-5
            ),
        )
        self.register_buffer(
            "action_delta_mean",
            self._normalizer(action_delta_mean, action_shape, 0.0),
        )
        self.register_buffer(
            "action_delta_std",
            self._normalizer(action_delta_std, action_shape, 1.0).clamp_min(
                1e-5
            ),
        )
        self.register_buffer(
            "distance_mean",
            self._normalizer(distance_mean, (), 0.0),
        )
        self.register_buffer(
            "distance_std",
            self._normalizer(distance_std, (), 1.0).clamp_min(1e-5),
        )
        token_input_dim = (
            self.config.geometry_dim + self.config.action_horizon * 3 + 1
        )
        self.token_encoder = nn.Sequential(
            nn.LayerNorm(token_input_dim),
            nn.Linear(token_input_dim, self.config.token_dim),
            nn.GELU(),
            nn.Linear(self.config.token_dim, self.config.token_dim),
        )
        scorer_input_dim = self.config.token_dim + 4
        if self.config.use_set_context:
            scorer_input_dim += self.config.token_dim
        self.scorer = nn.Sequential(
            nn.LayerNorm(scorer_input_dim),
            nn.Linear(scorer_input_dim, self.config.scorer_hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.scorer_hidden_dim, 1),
        )

    @staticmethod
    def _normalizer(
        value: torch.Tensor | None,
        shape: tuple[int, ...],
        default: float,
    ) -> torch.Tensor:
        output = torch.full(shape, default) if value is None else value.float()
        if output.shape != shape or not bool(torch.isfinite(output).all()):
            raise ValueError(f"DSBT normalizer shape 必须为 {shape} 且 finite")
        return output

    @property
    def residual_logit_bound(self) -> float:
        """单候选相对 prior logit 的对称界。"""
        return 0.5 * math.log(self.config.maximum_odds_distortion)

    def forward(
        self,
        query_geometry: torch.Tensor,
        demo_geometry: torch.Tensor,
        hypotheses: torch.Tensor,
        distances: torch.Tensor,
        prior: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        """返回 `[B,K,H]` weights；空 Demo set 严格返回零。"""
        if query_geometry.ndim != 2 or query_geometry.shape[1] != (
            self.config.geometry_dim
        ):
            raise ValueError("query_geometry shape 错误")
        batch, candidates = demo_geometry.shape[:2]
        expected_demo = (batch, candidates, self.config.geometry_dim)
        expected_action = (batch, candidates, self.config.action_horizon, 7)
        if demo_geometry.shape != expected_demo:
            raise ValueError("demo_geometry shape 错误")
        if hypotheses.shape != expected_action:
            raise ValueError("hypotheses shape 错误")
        if any(
            value.shape != (batch, candidates)
            for value in (distances, prior, candidate_mask)
        ):
            raise ValueError("DSBT distance/prior/mask shape 错误")
        if not all(
            bool(torch.isfinite(value).all())
            for value in (query_geometry, demo_geometry, hypotheses, distances, prior)
        ):
            raise ValueError("DSBT 输入包含非有限值")
        if bool((prior < 0.0).any()):
            raise ValueError("Retriever prior 必须非负")
        valid = candidate_mask.bool()
        has_candidate = valid.any(dim=1)
        if bool((has_candidate & ~valid[:, 0]).any()):
            raise ValueError("存在候选时 rank-1 candidate 必须有效")
        masked_prior = prior * valid.float()
        normalized_prior = masked_prior / masked_prior.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1e-12)
        normalized_prior = normalized_prior * has_candidate[:, None].float()
        if bool((valid & (normalized_prior <= 0.0)).any()):
            raise ValueError("有效候选的 Retriever prior 必须为正")

        translation = hypotheses[..., :3]
        prior_translation = (
            translation * normalized_prior[:, :, None, None]
        ).sum(dim=1)
        geometry_delta = query_geometry[:, None] - demo_geometry
        action_delta = translation - prior_translation[:, None]
        normalized_geometry = (
            geometry_delta - self.geometry_delta_mean
        ) / self.geometry_delta_std
        normalized_action = (
            action_delta - self.action_delta_mean
        ) / self.action_delta_std
        normalized_distance = (
            distances - self.distance_mean
        ) / self.distance_std
        token_input = torch.cat(
            (
                normalized_geometry,
                normalized_action.flatten(2),
                normalized_distance[..., None],
            ),
            dim=-1,
        )
        tokens = self.token_encoder(token_input)
        pooled = (tokens * normalized_prior[..., None]).sum(dim=1)
        pooled = pooled[:, None, None].expand(
            -1,
            candidates,
            self.config.action_horizon,
            -1,
        )
        token_steps = tokens[:, :, None].expand(
            -1,
            -1,
            self.config.action_horizon,
            -1,
        )
        step_action = normalized_action
        step_coordinate = torch.linspace(
            -1.0,
            1.0,
            self.config.action_horizon,
            device=hypotheses.device,
            dtype=hypotheses.dtype,
        )[None, None, :, None].expand(batch, candidates, -1, -1)
        scorer_parts = [token_steps]
        if self.config.use_set_context:
            scorer_parts.append(pooled)
        scorer_parts.extend((step_action, step_coordinate))
        scorer_input = torch.cat(scorer_parts, dim=-1)
        residual = self.scorer(scorer_input).squeeze(-1)
        if not self.config.per_step_weights:
            residual = residual.mean(dim=2, keepdim=True).expand_as(residual)
        residual = self.residual_logit_bound * torch.tanh(residual)
        logits = normalized_prior.clamp_min(1e-12).log()[..., None] + residual
        logits = logits.masked_fill(~valid[..., None], -torch.inf)
        safe_logits = torch.where(
            has_candidate[:, None, None],
            logits,
            torch.zeros_like(logits),
        )
        weights = torch.softmax(safe_logits, dim=1) * valid[..., None].float()
        return weights * has_candidate[:, None, None].float()


def mix_step_barycentric(
    hypotheses: torch.Tensor,
    weights: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    """逐 action-step 凸混合 translation，并复制 rank-1 离散先验。"""
    if hypotheses.ndim != 4 or hypotheses.shape[-1] != 7:
        raise ValueError("hypotheses 必须为 [B,K,H,7]")
    if weights.shape != hypotheses.shape[:3]:
        raise ValueError("weights 必须为 [B,K,H]")
    if candidate_mask.shape != hypotheses.shape[:2]:
        raise ValueError("candidate_mask 必须为 [B,K]")
    if bool(((weights < 0.0) | (~torch.isfinite(weights))).any()):
        raise ValueError("DSBT weights 必须 finite 且非负")
    valid = candidate_mask.bool()
    has_candidate = valid.any(dim=1)
    if bool((has_candidate & ~valid[:, 0]).any()):
        raise ValueError("存在候选时 rank-1 candidate 必须有效")
    normalized = weights * valid[..., None].float()
    normalized = normalized / normalized.sum(dim=1, keepdim=True).clamp_min(
        1e-12
    )
    translation = (hypotheses[..., :3] * normalized[..., None]).sum(dim=1)
    discrete = hypotheses[:, 0, :, 3:]
    output = torch.cat((translation, discrete), dim=-1)
    return output * has_candidate[:, None, None].float()

"""受 Retriever prior 约束的轻量多 Demo action mixture。"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


CANDIDATE_FEATURE_DIM = 42


@dataclass(frozen=True)
class RetrieverAnchoredMixtureConfig:
    """RADM scorer 的冻结容量和 odds trust region。"""

    input_dim: int = CANDIDATE_FEATURE_DIM
    hidden_dim: int = 32
    maximum_odds_distortion: float = 4.0

    def __post_init__(self) -> None:
        if min(self.input_dim, self.hidden_dim) <= 0:
            raise ValueError("RADM input/hidden dimension 必须为正")
        if self.input_dim != CANDIDATE_FEATURE_DIM:
            raise ValueError(f"RADM input_dim 必须为 {CANDIDATE_FEATURE_DIM}")
        if self.maximum_odds_distortion <= 1.0:
            raise ValueError("maximum_odds_distortion 必须大于 1")


class RetrieverAnchoredDemoMixture(nn.Module):
    """共享候选 scorer，并限制 posterior 相对 Retriever prior 的偏移。"""

    def __init__(
        self,
        config: RetrieverAnchoredMixtureConfig | None = None,
        *,
        feature_mean: torch.Tensor | None = None,
        feature_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.config = config or RetrieverAnchoredMixtureConfig()
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
            raise ValueError("RADM feature mean/std shape 错误")
        self.register_buffer("feature_mean", mean)
        self.register_buffer("feature_std", std.clamp_min(1e-5))
        self.scorer = nn.Sequential(
            nn.LayerNorm(self.config.input_dim),
            nn.Linear(self.config.input_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, 1),
        )

    @property
    def residual_logit_bound(self) -> float:
        """单候选相对 prior logit 的对称界。"""
        return 0.5 * math.log(self.config.maximum_odds_distortion)

    def forward(
        self,
        features: torch.Tensor,
        prior: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        """返回 `[B,K]` posterior；无候选行严格返回零。"""
        if features.ndim != 3 or features.shape[-1] != self.config.input_dim:
            raise ValueError("RADM features 必须为 [B,K,42]")
        if prior.shape != features.shape[:2] or candidate_mask.shape != prior.shape:
            raise ValueError("RADM prior/mask shape 与候选不一致")
        valid = candidate_mask.bool()
        if bool((prior < 0.0).any()) or not bool(torch.isfinite(prior).all()):
            raise ValueError("Retriever prior 必须 finite 且非负")
        normalized = (features - self.feature_mean) / self.feature_std
        residual_logits = self.residual_logit_bound * torch.tanh(
            self.scorer(normalized).squeeze(-1)
        )
        logits = prior.clamp_min(1e-12).log() + residual_logits
        logits = logits.masked_fill(~valid, -torch.inf)
        has_candidate = valid.any(dim=1)
        safe_logits = torch.where(
            has_candidate[:, None],
            logits,
            torch.zeros_like(logits),
        )
        posterior = torch.softmax(safe_logits, dim=1) * valid.float()
        posterior = posterior * has_candidate[:, None].float()
        return posterior


def retrieval_prior(
    distances: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    """用固定单位温度把 standardized geometry distance 转为 prior。"""
    if distances.ndim != 2 or candidate_mask.shape != distances.shape:
        raise ValueError("distance/mask 必须为相同的 [B,K]")
    if not bool(torch.isfinite(distances).all()):
        raise ValueError("retrieval distance 含非有限值")
    valid = candidate_mask.bool()
    has_candidate = valid.any(dim=1)
    logits = (-distances).masked_fill(~valid, -torch.inf)
    safe_logits = torch.where(
        has_candidate[:, None],
        logits,
        torch.zeros_like(logits),
    )
    prior = torch.softmax(safe_logits, dim=1) * valid.float()
    return prior * has_candidate[:, None].float()


def mix_demo_hypotheses(
    hypotheses: torch.Tensor,
    weights: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    """凸混合 translation，并严格复制 rank-1 rotation/gripper 先验。"""
    if hypotheses.ndim != 4 or hypotheses.shape[-1] != 7:
        raise ValueError("hypotheses 必须为 [B,K,H,7]")
    if weights.shape != hypotheses.shape[:2] or candidate_mask.shape != weights.shape:
        raise ValueError("mixture weights/mask shape 与 hypotheses 不一致")
    valid = candidate_mask.bool()
    if bool(((weights < 0.0) | (~torch.isfinite(weights))).any()):
        raise ValueError("mixture weights 必须 finite 且非负")
    has_candidate = valid.any(dim=1)
    if bool((has_candidate & ~valid[:, 0]).any()):
        raise ValueError("存在候选时 rank-1 candidate 必须有效")
    normalized = weights * valid.float()
    normalized = normalized / normalized.sum(dim=1, keepdim=True).clamp_min(1e-12)
    translation = (
        hypotheses[..., :3] * normalized[:, :, None, None]
    ).sum(dim=1)
    discrete_prior = hypotheses[:, 0, :, 3:]
    output = torch.cat((translation, discrete_prior), dim=-1)
    return output * has_candidate[:, None, None].float()

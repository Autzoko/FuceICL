"""从 retrieved Demo transitions 解析辨识局部 translation policy。"""

from __future__ import annotations

import math

import torch


POSITION_FEATURE_DIM = 6
TRANSLATION_ACTION_DIM = 3


def _transition_mask(
    actions: torch.Tensor,
    transition_mask: torch.Tensor | None,
) -> torch.Tensor:
    expected = actions.shape[:2]
    if transition_mask is None:
        return torch.ones(expected, dtype=torch.bool, device=actions.device)
    if transition_mask.shape != expected:
        raise ValueError(
            f"transition mask 应为 {expected}，实际为 {transition_mask.shape}"
        )
    return transition_mask.to(device=actions.device, dtype=torch.bool)


def fit_local_translation_operator(
    demo_geometry_sequence: torch.Tensor,
    demo_actions: torch.Tensor,
    *,
    position_scale_m: float,
    ridge_lambda: float,
    transition_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """逐 Demo 拟合 masked 6D geometry 到 3D translation ridge 算子。"""
    if position_scale_m <= 0 or ridge_lambda <= 0:
        raise ValueError("position scale 与 ridge lambda 必须为正")
    horizon = demo_actions.shape[1]
    expected = (len(demo_actions), horizon + 1)
    if demo_geometry_sequence.shape[:2] != expected:
        raise ValueError(
            "Demo geometry sequence 与 action horizon 不匹配："
            f"expected={expected}, actual={demo_geometry_sequence.shape[:2]}"
        )
    mask = _transition_mask(demo_actions, transition_mask)
    features = (
        demo_geometry_sequence[:, :horizon, :POSITION_FEATURE_DIM]
        / position_scale_m
    )
    translations = demo_actions[..., :TRANSLATION_ACTION_DIM]
    weights = mask.to(features.dtype)[..., None]
    counts = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    feature_mean = (features * weights).sum(dim=1, keepdim=True) / counts
    action_mean = (translations * weights).sum(dim=1, keepdim=True) / counts
    centered_features = (features - feature_mean) * weights
    centered_actions = (translations - action_mean) * weights
    transpose = centered_features.transpose(1, 2)
    identity = torch.eye(
        POSITION_FEATURE_DIM,
        dtype=features.dtype,
        device=features.device,
    )[None]
    regularized_gram = transpose @ centered_features + ridge_lambda * identity
    operator = torch.linalg.solve(regularized_gram, transpose @ centered_actions)
    condition_number = torch.linalg.cond(regularized_gram)
    return operator, condition_number


def bounded_translation_correction(
    query_geometry: torch.Tensor,
    demo_geometry_sequence: torch.Tensor,
    operator: torch.Tensor,
    *,
    position_scale_m: float,
    correction_limit: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """计算 query--Demo 局部平移修正并施加 L2 上限。"""
    if position_scale_m <= 0 or correction_limit <= 0:
        raise ValueError("position scale 与 correction limit 必须为正")
    delta = (
        query_geometry[:, :POSITION_FEATURE_DIM]
        - demo_geometry_sequence[:, 0, :POSITION_FEATURE_DIM]
    ) / position_scale_m
    raw = torch.bmm(delta[:, None, :], operator).squeeze(1)
    raw_norm = torch.linalg.vector_norm(raw, dim=1)
    multiplier = torch.clamp(
        correction_limit / raw_norm.clamp_min(1e-12),
        max=1.0,
    )
    return raw * multiplier[:, None], raw_norm, raw_norm > correction_limit


def demo_radius_gate(
    query_geometry: torch.Tensor,
    demo_geometry_sequence: torch.Tensor,
    *,
    position_scale_m: float,
    minimum_radius: float,
    transition_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """只用 Demo observed path radius 判断 query 是否位于局部支持域。"""
    if position_scale_m <= 0 or minimum_radius <= 0:
        raise ValueError("position scale 与 minimum radius 必须为正")
    batch, state_count = demo_geometry_sequence.shape[:2]
    dummy_actions = demo_geometry_sequence.new_empty((batch, state_count - 1, 0))
    mask = _transition_mask(dummy_actions, transition_mask)
    sequence = (
        demo_geometry_sequence[..., :POSITION_FEATURE_DIM] / position_scale_m
    )
    initial = sequence[:, :1]
    normalization = math.sqrt(POSITION_FEATURE_DIM)
    path_distance = torch.linalg.vector_norm(sequence - initial, dim=2)
    path_distance = path_distance / normalization
    state_mask = torch.cat(
        (
            torch.ones((batch, 1), dtype=torch.bool, device=mask.device),
            mask,
        ),
        dim=1,
    )
    path_distance = path_distance.masked_fill(~state_mask, -torch.inf)
    radius = path_distance.max(dim=1).values.clamp_min(minimum_radius)
    query = query_geometry[:, :POSITION_FEATURE_DIM] / position_scale_m
    query_distance = torch.linalg.vector_norm(query - initial[:, 0], dim=1)
    query_distance = query_distance / normalization
    return query_distance <= radius, query_distance, radius

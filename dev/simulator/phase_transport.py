"""Phase-conditioned sparse Demo action transport primitives."""

from __future__ import annotations

import torch


PHASE_NAMES = ("open_pre_grasp", "closed_post_grasp")


def phase_ids(geometry: torch.Tensor) -> torch.Tensor:
    """仅用当前可观测 gripper state 区分 pre/post-grasp phase。"""
    if geometry.ndim != 2 or geometry.shape[1] <= 15:
        raise ValueError("geometry 必须为 [B,D] 且包含 gripper state")
    return (geometry[:, 15] < 0.5).long()


def phase_matched_nearest(
    *,
    distances: torch.Tensor,
    query_phase: torch.Tensor,
    candidate_phase: torch.Tensor,
    query_episodes: torch.Tensor | None = None,
    candidate_episodes: torch.Tensor | None = None,
) -> torch.Tensor:
    """返回每个 query 的同 phase 最近邻，可选地排除同 episode。"""
    if distances.ndim != 2:
        raise ValueError("distances 必须为 [Q,C]")
    valid = query_phase[:, None] == candidate_phase[None, :]
    if valid.shape != distances.shape:
        raise ValueError("phase 与 distance shape 不一致")
    if query_episodes is not None or candidate_episodes is not None:
        if query_episodes is None or candidate_episodes is None:
            raise ValueError("query/candidate episode IDs 必须同时提供")
        valid &= query_episodes[:, None] != candidate_episodes[None, :]
    if not bool(valid.any(dim=1).all()):
        raise ValueError("至少一个 query 没有可用的同 phase 候选")
    return distances.masked_fill(~valid, torch.inf).argmin(dim=1)


def fit_phase_weights(
    *,
    query_geometry: torch.Tensor,
    demo_geometry: torch.Tensor,
    target_actions: torch.Tensor,
    demo_actions: torch.Tensor,
    phase: torch.Tensor,
    position_scale_m: float,
    ridge_lambda: float,
) -> tuple[torch.Tensor, dict[str, int]]:
    """拟合两个无偏置 3D→H×3 ridge maps，保留 identity-at-zero。"""
    if position_scale_m <= 0 or ridge_lambda <= 0:
        raise ValueError("position scale 与 ridge lambda 必须为正")
    horizon = int(target_actions.shape[1])
    weights = torch.zeros(2, 3, horizon * 3, dtype=torch.float64)
    sample_counts: dict[str, int] = {}
    for phase_id, phase_name in enumerate(PHASE_NAMES):
        mask = phase == phase_id
        count = int(mask.sum())
        if count < 3:
            raise ValueError(f"{phase_name} train pairs 不足：{count}")
        feature_slice = slice(0, 3) if phase_id == 0 else slice(3, 6)
        x = (
            query_geometry[mask, feature_slice]
            - demo_geometry[mask, feature_slice]
        ).double() / position_scale_m
        y = (
            target_actions[mask, :, :3] - demo_actions[mask, :, :3]
        ).reshape(count, horizon * 3).double()
        gram = x.T @ x + ridge_lambda * torch.eye(3, dtype=torch.float64)
        weights[phase_id] = torch.linalg.solve(gram, x.T @ y)
        sample_counts[phase_name] = count
    return weights.float(), sample_counts


def transport_actions(
    *,
    query_geometry: torch.Tensor,
    demo_geometry: torch.Tensor,
    demo_actions: torch.Tensor,
    phase: torch.Tensor,
    weights: torch.Tensor,
    position_scale_m: float,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """只迁移 translation，保持检索 Demo 的 rotation 与 gripper action。"""
    if position_scale_m <= 0:
        raise ValueError("position scale 必须为正")
    horizon = int(demo_actions.shape[1])
    delta = torch.empty(
        len(query_geometry),
        3,
        dtype=query_geometry.dtype,
        device=query_geometry.device,
    )
    open_mask = phase == 0
    delta[open_mask] = (
        query_geometry[open_mask, 0:3] - demo_geometry[open_mask, 0:3]
    ) / position_scale_m
    delta[~open_mask] = (
        query_geometry[~open_mask, 3:6] - demo_geometry[~open_mask, 3:6]
    ) / position_scale_m
    selected_weights = weights[phase]
    residual = torch.einsum("bi,bij->bj", delta, selected_weights).reshape(
        -1, horizon, 3
    )
    prediction = demo_actions.clone()
    translation_unclipped = prediction[:, :, :3] + residual
    prediction[:, :, :3] = translation_unclipped.clamp(-1.0, 1.0)
    clipped = translation_unclipped != prediction[:, :, :3]
    clipped_count = int(clipped.sum())
    component_count = int(clipped.numel())
    return prediction, {
        "translation_components_clipped": clipped_count,
        "translation_components_total": component_count,
        "translation_component_clip_rate": clipped_count / component_count,
        "chunks_with_translation_clip_rate": float(
            clipped.flatten(start_dim=1).any(dim=1).float().mean()
        ),
        "normalized_residual_abs_mean": float(residual.abs().mean()),
        "normalized_residual_abs_max": float(residual.abs().max()),
    }

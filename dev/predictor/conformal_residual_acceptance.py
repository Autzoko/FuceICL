"""以 episode-level conformal risk control 选择性启用 Demo residual。"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class ConformalAcceptanceCalibration:
    """一个冻结风险分数的 CRC 阈值与审计统计。"""

    alpha: float
    threshold: float
    episodes: int
    empirical_risk: float
    corrected_risk: float
    calibration_acceptance: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def residual_risk_scores(
    *,
    retrieval_distance: torch.Tensor,
    residual_norm: torch.Tensor,
    base_gate: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """构造预注册的 distance、applied-residual 与乘积风险分数。"""
    expected = retrieval_distance.shape
    if expected != residual_norm.shape or expected != base_gate.shape:
        raise ValueError("distance、residual 与 gate shape 必须一致")
    if retrieval_distance.ndim != 1:
        raise ValueError("风险分数输入必须是一维张量")
    if not bool(
        torch.isfinite(retrieval_distance).all()
        and torch.isfinite(residual_norm).all()
        and torch.isfinite(base_gate).all()
    ):
        raise ValueError("风险分数输入包含非有限值")
    if bool((retrieval_distance < 0.0).any()):
        raise ValueError("retrieval distance 不能为负")
    if bool((residual_norm < 0.0).any()):
        raise ValueError("residual norm 不能为负")
    if bool(((base_gate < 0.0) | (base_gate > 1.0)).any()):
        raise ValueError("base gate 必须位于 [0,1]")
    applied_residual = base_gate * residual_norm
    return {
        "distance": retrieval_distance,
        "applied_residual": applied_residual,
        "codra": retrieval_distance * applied_residual,
    }


def episode_risk(
    *,
    accepted: torch.Tensor,
    harmful: torch.Tensor,
    episode_ids: torch.Tensor,
) -> torch.Tensor:
    """返回每个 episode 中启用且有害 residual 的 chunk 比例。"""
    if accepted.shape != harmful.shape or accepted.shape != episode_ids.shape:
        raise ValueError("accepted、harmful 与 episode_ids shape 必须一致")
    if accepted.ndim != 1:
        raise ValueError("episode risk 输入必须是一维张量")
    values = []
    for episode in torch.unique(episode_ids, sorted=True):
        member = episode_ids == episode
        values.append((accepted[member] & harmful[member]).float().mean())
    if not values:
        raise ValueError("至少需要一个 calibration episode")
    return torch.stack(values)


def calibrate_acceptance_threshold(
    *,
    scores: torch.Tensor,
    harmful: torch.Tensor,
    episode_ids: torch.Tensor,
    alpha: float,
) -> ConformalAcceptanceCalibration:
    """选择满足有限样本 CRC 修正的最大 residual 接受阈值。"""
    if scores.shape != harmful.shape or scores.shape != episode_ids.shape:
        raise ValueError("scores、harmful 与 episode_ids shape 必须一致")
    if scores.ndim != 1 or len(scores) == 0:
        raise ValueError("scores 必须是非空一维张量")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("scores 包含非有限值")
    episodes = int(torch.unique(episode_ids).numel())
    if episodes < 2:
        raise ValueError("CRC 至少需要两个 calibration episodes")
    finite_sample_floor = 1.0 / (episodes + 1)
    if not finite_sample_floor <= alpha < 1.0:
        raise ValueError(
            f"alpha 必须位于 [{finite_sample_floor:.6f},1)"
        )
    minimum = float(scores.min())
    reject_all = minimum - max(1.0, abs(minimum))
    candidates = [reject_all, *torch.unique(scores, sorted=True).tolist()]
    selected = reject_all
    selected_empirical = 0.0
    selected_corrected = finite_sample_floor
    selected_acceptance = 0.0
    for threshold in candidates:
        accepted = scores <= threshold
        empirical = float(
            episode_risk(
                accepted=accepted,
                harmful=harmful.bool(),
                episode_ids=episode_ids,
            ).mean()
        )
        corrected = (
            episodes / (episodes + 1) * empirical + finite_sample_floor
        )
        if corrected <= alpha + 1e-12:
            selected = float(threshold)
            selected_empirical = empirical
            selected_corrected = corrected
            selected_acceptance = float(accepted.float().mean())
    return ConformalAcceptanceCalibration(
        alpha=alpha,
        threshold=selected,
        episodes=episodes,
        empirical_risk=selected_empirical,
        corrected_risk=selected_corrected,
        calibration_acceptance=selected_acceptance,
    )


def selective_gate(
    *,
    base_gate: torch.Tensor,
    scores: torch.Tensor,
    calibration: ConformalAcceptanceCalibration,
) -> tuple[torch.Tensor, torch.Tensor]:
    """应用冻结 CRC 阈值；拒绝样本的 gate 精确为零。"""
    if base_gate.shape != scores.shape:
        raise ValueError("base_gate 与 scores shape 必须一致")
    if bool(((base_gate < 0.0) | (base_gate > 1.0)).any()):
        raise ValueError("base_gate 必须位于 [0,1]")
    accepted = scores <= calibration.threshold
    return torch.where(accepted, base_gate, torch.zeros_like(base_gate)), accepted

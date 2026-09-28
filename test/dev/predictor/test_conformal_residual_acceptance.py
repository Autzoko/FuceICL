"""CoDRA episode-level conformal calibration 的不变量测试。"""

from __future__ import annotations

import torch

from dev.predictor.conformal_residual_acceptance import (
    calibrate_acceptance_threshold,
    episode_risk,
    residual_risk_scores,
    selective_gate,
)


def test_scores_use_applied_residual() -> None:
    values = residual_risk_scores(
        retrieval_distance=torch.tensor([0.5, 2.0]),
        residual_norm=torch.tensor([2.0, 0.5]),
        base_gate=torch.tensor([0.25, 0.5]),
    )
    assert torch.allclose(values["distance"], torch.tensor([0.5, 2.0]))
    assert torch.allclose(values["applied_residual"], torch.tensor([0.5, 0.25]))
    assert torch.allclose(values["codra"], torch.tensor([0.25, 0.5]))


def test_crc_selects_largest_valid_threshold() -> None:
    scores = torch.tensor([0.1, 0.9, 0.2, 1.0, 0.3, 1.1, 0.4, 1.2])
    harmful = torch.tensor([False, True] * 4)
    episodes = torch.arange(4).repeat_interleave(2)
    calibration = calibrate_acceptance_threshold(
        scores=scores,
        harmful=harmful,
        episode_ids=episodes,
        alpha=0.3,
    )
    gate, accepted = selective_gate(
        base_gate=torch.full((8,), 0.7),
        scores=scores,
        calibration=calibration,
    )

    assert abs(calibration.threshold - 0.9) < 1e-6
    assert calibration.corrected_risk <= 0.3 + 1e-12
    assert torch.equal(accepted, scores <= 0.9)
    assert torch.equal(gate[~accepted], torch.zeros_like(gate[~accepted]))


def test_episode_risk_is_monotone_with_acceptance_threshold() -> None:
    scores = torch.tensor([0.1, 0.2, 0.3, 0.4])
    harmful = torch.tensor([True, False, True, False])
    episodes = torch.tensor([0, 0, 1, 1])
    risks = []
    for threshold in scores:
        risks.append(
            episode_risk(
                accepted=scores <= threshold,
                harmful=harmful,
                episode_ids=episodes,
            ).mean()
        )
    assert bool((torch.diff(torch.stack(risks)) >= 0.0).all())

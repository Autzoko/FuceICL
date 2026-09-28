"""Causal-history feature 与 gate 不变量测试。"""

from __future__ import annotations

import torch

from dev.predictor.causal_history_gate import (
    CausalHistoryBenefitGate,
    CausalHistoryGateConfig,
    HISTORY_FEATURE_DIM,
    causal_history_features,
    first_order_history,
    shuffled_history_within_episode,
)
from dev.predictor.canonical_geometry import GEOMETRY_DIM


def test_history_is_causal_and_resets_at_episode_boundary() -> None:
    records = [
        {"episode": 0, "frame": 0},
        {"episode": 0, "frame": 1},
        {"episode": 0, "frame": 2},
        {"episode": 1, "frame": 0},
    ]
    geometry = torch.arange(4 * GEOMETRY_DIM).reshape(4, GEOMETRY_DIM).float()
    history = causal_history_features(records, geometry)
    assert history.shape == (4, HISTORY_FEATURE_DIM)
    assert torch.equal(history[0], torch.zeros_like(history[0]))
    assert history[1, -2:].tolist() == [1.0, 0.0]
    assert history[2, -2:].tolist() == [1.0, 1.0]
    assert torch.equal(history[3], torch.zeros_like(history[3]))
    assert first_order_history(history).shape == (4, 11)


def test_shuffle_preserves_episode_marginals() -> None:
    records = [{"episode": value // 3, "frame": value % 3} for value in range(6)]
    features = torch.arange(6 * HISTORY_FEATURE_DIM).reshape(6, -1).float()
    shuffled = shuffled_history_within_episode(records, features, seed=7)
    for start in (0, 3):
        assert torch.equal(
            features[start : start + 3].sort(dim=0).values,
            shuffled[start : start + 3].sort(dim=0).values,
        )


def test_history_gate_shape_and_range() -> None:
    config = CausalHistoryGateConfig(input_dim=62, hidden_dim=8)
    gate = CausalHistoryBenefitGate(
        config,
        feature_mean=torch.zeros(62),
        feature_std=torch.ones(62),
    )
    output = gate(torch.zeros(5, 62))
    assert output.shape == (5,)
    assert bool(((output >= 0.0) & (output <= 1.0)).all())

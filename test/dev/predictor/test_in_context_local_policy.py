"""In-context local policy 的 masked ridge 与 phase-local context 测试。"""

from __future__ import annotations

import torch

from dev.predictor.in_context_local_policy import (
    bounded_translation_correction,
    demo_radius_gate,
    fit_local_translation_operator,
)
from dev.simulator.phase_local_context import build_phase_local_context_bank


def test_masked_operator_identity_and_radius() -> None:
    generator = torch.Generator().manual_seed(29)
    sequence = torch.randn(3, 9, 17, generator=generator) * 0.01
    actions = torch.randn(3, 8, 7, generator=generator) * 0.05
    mask = torch.tensor(
        [
            [True] * 8,
            [True] * 5 + [False] * 3,
            [True] * 3 + [False] * 5,
        ]
    )
    operator, condition = fit_local_translation_operator(
        sequence,
        actions,
        position_scale_m=0.1,
        ridge_lambda=0.1,
        transition_mask=mask,
    )
    correction, _, _ = bounded_translation_correction(
        sequence[:, 0],
        sequence,
        operator,
        position_scale_m=0.1,
        correction_limit=0.25,
    )
    accepted, distance, radius = demo_radius_gate(
        sequence[:, 0],
        sequence,
        position_scale_m=0.1,
        minimum_radius=0.05,
        transition_mask=mask,
    )
    assert torch.equal(correction, torch.zeros_like(correction))
    assert accepted.all()
    assert torch.equal(distance, torch.zeros_like(distance))
    assert torch.isfinite(operator).all() and torch.isfinite(condition).all()
    assert torch.all(radius >= 0.05)


def test_phase_local_context_stops_at_phase_change() -> None:
    geometry = torch.zeros(4, 17)
    geometry[:, 0] = torch.arange(4).float()
    geometry[:, 15] = torch.tensor([1.0, 1.0, 0.0, 0.0])
    actions = torch.zeros(4, 2, 7)
    sequence = geometry[:, None].repeat(1, 3, 1)
    sequence[0, 1] = geometry[1]
    sequence[0, 2] = geometry[2]
    sequence[1, 1] = geometry[2]
    sequence[1, 2] = geometry[3]
    sequence[2, 1] = geometry[3]
    sequence[2, 2] = geometry[3]
    sequence[3, 1:] = geometry[3]
    records = [
        {"episode": 0, "frame": frame} for frame in range(4)
    ]

    bank = build_phase_local_context_bank(
        records=records,
        geometry=geometry,
        actions=actions,
        stored_geometry_sequence=sequence,
        max_transitions=3,
    )

    assert bank.lengths.tolist() == [1, 0, 2, 2]
    assert bank.transition_mask.sum(dim=1).tolist() == [1, 0, 2, 2]


def test_phase_local_context_extends_past_stored_horizon() -> None:
    geometry = torch.zeros(5, 17)
    geometry[:, 0] = torch.arange(5).float()
    actions = torch.zeros(5, 2, 7)
    sequence = geometry[:, None].repeat(1, 3, 1)
    for frame in range(5):
        sequence[frame, 1] = geometry[min(frame + 1, 4)]
        sequence[frame, 2] = geometry[min(frame + 2, 4)]
    records = [{"episode": 0, "frame": frame} for frame in range(5)]

    bank = build_phase_local_context_bank(
        records=records,
        geometry=geometry,
        actions=actions,
        stored_geometry_sequence=sequence,
        max_transitions=4,
    )

    assert int(bank.lengths[0]) == 4
    assert torch.equal(bank.geometry_sequence[0, 4], geometry[4])

"""Low-Rank Demo Action Transport 的结构不变量测试。"""

from __future__ import annotations

import torch

from dev.predictor.low_rank_demo_transport import (
    LowRankDemoActionTransport,
    LowRankTransportConfig,
)


def _inputs() -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(17)
    query = torch.randn(3, 17, generator=generator)
    demo = torch.randn(3, 17, generator=generator)
    actions = torch.randn(3, 6, 7, generator=generator)
    actions[..., 6] = torch.randint(
        0,
        2,
        (3, 6),
        generator=generator,
    ).float()
    mask = torch.ones(3, 6, dtype=torch.bool)
    return query, demo, actions, mask


def test_identity_no_demo_and_gripper_prior() -> None:
    model = LowRankDemoActionTransport().eval()
    query, demo, actions, mask = _inputs()

    identity = model(demo, demo, actions, mask)
    assert torch.equal(identity, actions)

    prediction = model(query, demo, actions, mask)
    assert torch.equal(prediction[..., 6], actions[..., 6])

    no_demo = model(query, demo, actions, torch.zeros_like(mask))
    assert torch.equal(no_demo, torch.zeros_like(actions))


def test_rank_capacity_and_operator_diagnostics() -> None:
    config = LowRankTransportConfig(rank=4, hidden_dim=96)
    model = LowRankDemoActionTransport(config).eval()
    _, demo, actions, mask = _inputs()

    left, right = model.transport_factors(demo, actions, mask)
    bounds = model.operator_frobenius_upper_bound(demo, actions, mask)

    assert left.shape == (3, 6, 6, 4)
    assert right.shape == (3, 6, 4, 17)
    assert bounds.shape == (3, 6)
    assert torch.all(bounds >= 0)
    assert sum(parameter.numel() for parameter in model.parameters()) < 1_000_000


def test_transition_context_preserves_structural_constraints() -> None:
    config = LowRankTransportConfig(transition_context=True)
    model = LowRankDemoActionTransport(config).eval()
    query, demo, actions, mask = _inputs()
    future = torch.randn(3, 6, 17, generator=torch.Generator().manual_seed(23))
    sequence = torch.cat((demo[:, None, :], future), dim=1)

    identity = model(demo, demo, actions, mask, sequence)
    assert torch.equal(identity, actions)

    no_demo = model(query, demo, actions, torch.zeros_like(mask), sequence)
    assert torch.equal(no_demo, torch.zeros_like(actions))
    assert sum(parameter.numel() for parameter in model.parameters()) < 1_000_000

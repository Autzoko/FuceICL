"""Query-aligned low-rank Demo transport 的结构不变量测试。"""

from __future__ import annotations

import torch

from dev.predictor.query_aligned_transport import (
    QueryAlignedLowRankDemoTransport,
)


def _inputs() -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(43)
    query = torch.randn(4, 17, generator=generator)
    demo = torch.randn(4, 17, generator=generator)
    actions = torch.randn(4, 6, 7, generator=generator)
    actions[..., 6] = torch.randint(
        0,
        2,
        (4, 6),
        generator=generator,
    ).float()
    mask = torch.ones(4, 6, dtype=torch.bool)
    return query, demo, actions, mask


def test_identity_no_demo_and_discrete_prior() -> None:
    model = QueryAlignedLowRankDemoTransport().eval()
    query, demo, actions, mask = _inputs()

    identity = model(demo, demo, actions, mask)
    assert torch.equal(identity, actions)

    prediction = model(query, demo, actions, mask)
    assert torch.equal(prediction[..., 3:], actions[..., 3:])

    no_demo = model(query, demo, actions, torch.zeros_like(mask))
    assert torch.equal(no_demo, torch.zeros_like(actions))


def test_rank_attention_and_parameter_budget() -> None:
    model = QueryAlignedLowRankDemoTransport().eval()
    query, demo, actions, mask = _inputs()

    left, right, delta, attention = model.transport_factors(
        query,
        demo,
        actions,
        mask,
    )

    assert left.shape == (4, 6, 3, 4)
    assert right.shape == (4, 6, 4, 17)
    assert delta.shape == (4, 17)
    assert attention.shape == (4, 4, 6, 6)
    assert torch.allclose(attention.sum(dim=-1), torch.ones(4, 4, 6))
    assert sum(parameter.numel() for parameter in model.parameters()) < 1_000_000

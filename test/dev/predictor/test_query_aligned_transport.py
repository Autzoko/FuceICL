"""Query-aligned low-rank Demo transport 的结构不变量测试。"""

from __future__ import annotations

import torch

from dev.predictor.benefit_calibrated_shrinkage import (
    BenefitCalibratedGate,
    apply_shrinkage,
    frozen_transport_features,
    optimal_shrinkage_target,
)
from dev.predictor.query_aligned_transport import (
    QueryAlignedLowRankDemoTransport,
)
from dev.predictor.train_query_aligned_transport import (
    _primary_selection_indices,
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

    diagnostic_prediction, delta, attention = model.forward_with_diagnostics(
        query,
        demo,
        actions,
        mask,
    )
    assert torch.equal(diagnostic_prediction, prediction)
    assert delta.shape == (4, 17)
    assert attention.shape == (4, 4, 6, 6)


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


def test_primary_selection_does_not_require_hard_negatives() -> None:
    records = [
        {"chunk_id": "query", "task": "task", "episode": 0},
        {"chunk_id": "demo", "task": "task", "episode": 1},
    ]
    pair_rows = [
        {
            "query_id": "query",
            "positive_ids": ["demo"],
            "hard_negatives": {},
        }
    ]
    action_masks = torch.ones(2, 6, dtype=torch.bool)
    scores = torch.tensor([[0.0, 0.9], [0.9, 0.0]])
    text_mask = torch.ones(2, 2, dtype=torch.bool)

    query_indices, selections = _primary_selection_indices(
        records=records,
        pair_rows=pair_rows,
        action_masks=action_masks,
        retrieval_scores=scores,
        text_mask=text_mask,
    )

    assert query_indices == [0]
    assert selections == {
        "oracle": [(1, 1)],
        "retrieved": [(1, 1)],
    }

    restricted_indices, restricted = _primary_selection_indices(
        records=[
            *records,
            {"chunk_id": "held", "task": "held", "episode": 2},
        ],
        pair_rows=pair_rows,
        action_masks=torch.ones(3, 6, dtype=torch.bool),
        retrieval_scores=torch.tensor(
            [
                [0.0, 0.8, 0.99],
                [0.8, 0.0, 0.0],
                [0.99, 0.0, 0.0],
            ]
        ),
        text_mask=torch.ones(3, 3, dtype=torch.bool),
        candidate_tasks={"task"},
    )
    assert restricted_indices == [0]
    assert restricted["retrieved"] == [(1, 1)]


def test_benefit_gate_features_and_projection_invariants() -> None:
    transport = QueryAlignedLowRankDemoTransport().eval()
    query, demo, actions, mask = _inputs()
    transported, features = frozen_transport_features(
        transport,
        query,
        demo,
        actions,
        mask,
    )
    gate_model = BenefitCalibratedGate().eval()
    gate = gate_model(features)
    prediction = apply_shrinkage(actions, transported, gate, mask)

    assert features.shape == (4, 40)
    assert bool(((gate >= 0.0) & (gate <= 1.0)).all())
    assert torch.equal(prediction[..., 3:], actions[..., 3:])
    assert torch.equal(
        apply_shrinkage(actions, transported, torch.zeros(4), mask),
        actions,
    )
    assert torch.equal(
        apply_shrinkage(actions, transported, torch.ones(4), mask),
        transported,
    )
    optimal, energy = optimal_shrinkage_target(
        actions,
        transported,
        transported,
        mask,
    )
    active = energy > 1e-8
    assert torch.allclose(optimal[active], torch.ones_like(optimal)[active])
    assert sum(parameter.numel() for parameter in gate_model.parameters()) < 5_000

    target = torch.randn_like(actions)
    oracle_gate, _ = optimal_shrinkage_target(
        actions,
        transported,
        target,
        mask,
    )
    oracle = apply_shrinkage(actions, transported, oracle_gate, mask)

    def translation_loss(value: torch.Tensor) -> torch.Tensor:
        error = (value[..., :3] - target[..., :3]).square()
        return (error * mask[..., None]).sum(dim=(1, 2))

    oracle_loss = translation_loss(oracle)
    assert bool((oracle_loss <= translation_loss(actions) + 1e-6).all())
    assert bool((oracle_loss <= translation_loss(transported) + 1e-6).all())

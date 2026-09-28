"""Retriever-anchored 多 Demo mixture 的结构不变量测试。"""

from __future__ import annotations

import torch

from dev.predictor.retriever_anchored_demo_mixture import (
    CANDIDATE_FEATURE_DIM,
    RetrieverAnchoredDemoMixture,
    mix_demo_hypotheses,
    retrieval_prior,
)
from dev.simulator.train_retriever_anchored_demo_mixture import _oracle_convex


def _inputs() -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(59)
    features = torch.randn(3, 4, CANDIDATE_FEATURE_DIM, generator=generator)
    distances = torch.rand(3, 4, generator=generator)
    mask = torch.tensor(
        [
            [True, True, True, True],
            [True, True, False, False],
            [False, False, False, False],
        ]
    )
    hypotheses = torch.randn(3, 4, 6, 7, generator=generator)
    return features, distances, mask, hypotheses


def test_posterior_respects_odds_bound_and_mask() -> None:
    features, distances, mask, _ = _inputs()
    model = RetrieverAnchoredDemoMixture().eval()
    prior = retrieval_prior(distances, mask)
    posterior = model(features, prior, mask)

    assert torch.allclose(posterior[:2].sum(dim=1), torch.ones(2))
    assert torch.equal(posterior[2], torch.zeros(4))
    assert torch.equal(posterior[~mask], torch.zeros_like(posterior[~mask]))
    for batch in range(2):
        valid = torch.where(mask[batch])[0]
        for left in valid:
            for right in valid:
                distortion = (
                    posterior[batch, left]
                    / posterior[batch, right]
                    / (prior[batch, left] / prior[batch, right])
                )
                value = float(distortion.detach())
                assert value <= 4.0 + 1e-6
                assert value >= 0.25 - 1e-6


def test_scorer_is_candidate_permutation_equivariant() -> None:
    features, distances, mask, _ = _inputs()
    model = RetrieverAnchoredDemoMixture().eval()
    prior = retrieval_prior(distances, mask)
    expected = model(features, prior, mask)
    permutation = torch.tensor([2, 0, 3, 1])
    actual = model(
        features[:, permutation],
        prior[:, permutation],
        mask[:, permutation],
    )
    assert torch.allclose(actual, expected[:, permutation])


def test_mixture_convex_translation_and_discrete_prior() -> None:
    _, distances, mask, hypotheses = _inputs()
    weights = retrieval_prior(distances, mask)
    output = mix_demo_hypotheses(hypotheses, weights, mask)

    minimum = hypotheses[:2, :, :, :3].amin(dim=1)
    maximum = hypotheses[:2, :, :, :3].amax(dim=1)
    assert bool((output[:2, :, :3] >= minimum - 1e-6).all())
    assert bool((output[:2, :, :3] <= maximum + 1e-6).all())
    assert torch.equal(output[:2, :, 3:], hypotheses[:2, 0, :, 3:])
    assert torch.equal(output[2], torch.zeros_like(output[2]))


def test_parameter_budget() -> None:
    model = RetrieverAnchoredDemoMixture()
    assert sum(parameter.numel() for parameter in model.parameters()) < 5_000


def test_analysis_oracle_projects_into_candidate_convex_hull() -> None:
    hypotheses = torch.zeros(1, 4, 2, 7)
    hypotheses[0, :, :, 0] = torch.tensor([0.0, 2.0, 4.0, 6.0])[:, None]
    target = torch.zeros(1, 2, 7)
    target[0, :, 0] = 1.0

    prediction = _oracle_convex(hypotheses, target)

    assert torch.allclose(prediction[..., :3], target[..., :3], atol=1e-5)
    assert torch.equal(prediction[..., 3:], hypotheses[:, 0, :, 3:])

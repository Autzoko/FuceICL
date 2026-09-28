"""Retrieval-cell 支持域证书的数学与回退不变量测试。"""

from __future__ import annotations

import torch

from dev.predictor.retrieval_cell_certified_shrinkage import (
    RetrievalCellCertificate,
    empirical_cell_lipschitz_constant,
    support_certified_gate,
)


def _certificate() -> RetrievalCellCertificate:
    geometry = torch.tensor([[0.0], [1.0], [0.0]])
    benefit = torch.tensor([0.6, 0.4, 0.9])
    cell = torch.tensor([4, 4, 8])
    slope, audit = empirical_cell_lipschitz_constant(
        geometry,
        benefit,
        cell,
        geometry_mean=torch.zeros(1),
        geometry_std=torch.ones(1),
    )
    assert abs(slope - 0.2) < 1e-6
    assert audit["same_cell_pairs"] == 1
    return RetrievalCellCertificate(
        support_geometry=geometry,
        support_benefit=benefit,
        support_cell=cell,
        geometry_mean=torch.zeros(1),
        geometry_std=torch.ones(1),
        lipschitz_constant=slope,
    )


def test_cell_lower_bound_and_unseen_cell_fallback() -> None:
    certificate = _certificate()
    lower, distance, supported = certificate.lower_bound(
        torch.tensor([[0.5], [0.2], [0.0]]),
        torch.tensor([4, 8, 7]),
    )

    assert torch.allclose(lower, torch.tensor([0.5, 0.86, 0.0]))
    assert torch.allclose(distance[:2], torch.tensor([0.5, 0.2]))
    assert torch.isinf(distance[2])
    assert torch.equal(supported, torch.tensor([True, True, False]))
    gate = support_certified_gate(torch.full((3,), 0.8), lower)
    assert torch.allclose(gate, torch.tensor([0.8, 0.8, 0.0]))


def test_certificate_preserves_copy_squared_loss_bound() -> None:
    certificate = _certificate()
    query = torch.linspace(0.0, 1.0, 21)[:, None]
    cell = torch.full((len(query),), 4)
    lower, _, _ = certificate.lower_bound(query, cell)
    gate = support_certified_gate(torch.ones(len(query)), lower)

    # 构造满足 0.2-Lipschitz 假设的真实最优 residual 坐标。
    true_benefit = 0.6 - 0.2 * query[:, 0]
    residual = torch.ones(len(query))
    target_offset = true_benefit * residual
    copy_loss = target_offset.square()
    certified_loss = (gate * residual - target_offset).square()

    assert bool((lower <= true_benefit + 1e-7).all())
    assert bool((certified_loss <= copy_loss + 1e-7).all())


def test_exact_support_interpolation_with_empirical_slope() -> None:
    certificate = _certificate()
    lower, _, _ = certificate.lower_bound(
        certificate.support_geometry,
        certificate.support_cell,
    )
    assert torch.allclose(lower, certificate.support_benefit)


def test_near_duplicate_aliasing_is_reported() -> None:
    slope, audit = empirical_cell_lipschitz_constant(
        torch.tensor([[0.0], [0.0]]),
        torch.tensor([0.1, 0.9]),
        torch.tensor([3, 3]),
        geometry_mean=torch.zeros(1),
        geometry_std=torch.ones(1),
    )

    assert slope == 0.0
    assert audit["near_duplicate_conflicts"] == 1
    assert abs(float(audit["near_duplicate_max_benefit_gap"]) - 0.8) < 1e-6

"""visibility-conditioned belief 的公共接口测试。"""

import numpy as np
import pytest

from src.components.retriever.belief_tracker import (
    VisibilityConditionedBeliefTracker,
    belief_risk_accepted,
)


def test_visible_update_and_missing_tcp_propagation() -> None:
    tracker = VisibilityConditionedBeliefTracker(minimum_visible_points=2)
    visible = np.asarray([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]])

    first = tracker.update(visible, np.asarray([0.0, 0.0, 0.0]))
    missing = tracker.update(
        np.empty((0, 3)),
        np.asarray([0.004, -0.003, 0.0]),
    )

    assert first.visible
    assert first.age_steps == 0
    assert np.allclose(missing.center_world, [0.014, -0.003, 0.0])
    assert not missing.visible
    assert missing.age_steps == 1
    assert missing.observed_point_count == 0
    assert missing.tcp_net_displacement_since_visible_m == pytest.approx(0.005)
    assert missing.tcp_path_length_since_visible_m == pytest.approx(0.005)


def test_new_measurement_resets_risk_statistics() -> None:
    tracker = VisibilityConditionedBeliefTracker(minimum_visible_points=1)
    tracker.update(np.asarray([[0.0, 0.0, 0.0]]), np.zeros(3))
    tracker.update(np.empty((0, 3)), np.asarray([0.02, 0.0, 0.0]))

    estimate = tracker.update(
        np.asarray([[0.03, 0.0, 0.0]]),
        np.asarray([0.02, 0.0, 0.0]),
    )

    assert estimate.visible
    assert estimate.age_steps == 0
    assert estimate.tcp_net_displacement_since_visible_m == 0.0
    assert estimate.tcp_path_length_since_visible_m == 0.0


def test_missing_before_initial_measurement_is_rejected() -> None:
    tracker = VisibilityConditionedBeliefTracker(minimum_visible_points=1)

    with pytest.raises(RuntimeError, match="首次有效可见测量"):
        tracker.update(np.empty((0, 3)), np.zeros(3))


def test_analytic_risk_gate_boundary() -> None:
    assert belief_risk_accepted(
        visible=False,
        tcp_net_displacement_m=0.01,
        maximum_tcp_net_displacement_m=0.01,
    )
    assert not belief_risk_accepted(
        visible=False,
        tcp_net_displacement_m=0.010001,
        maximum_tcp_net_displacement_m=0.01,
    )
    assert belief_risk_accepted(
        visible=True,
        tcp_net_displacement_m=1.0,
        maximum_tcp_net_displacement_m=0.01,
    )

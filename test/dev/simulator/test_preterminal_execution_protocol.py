"""终止前 progress schedule 与 coverage 的最小契约测试。"""

from types import SimpleNamespace

from dev.simulator.evaluate_demo_relative_preterminal_execution import _coverage
from dev.simulator.preprocess_rotated_layout_slide_belief_chunks import (
    _belief_frames,
)


def test_belief_frames_respect_frozen_preterminal_fraction() -> None:
    config = SimpleNamespace(
        action_horizon=6,
        progress_samples=8,
        progress_fraction_maximum=0.7,
    )

    frames = _belief_frames(branch=10, action_steps=116, config=config)

    assert len(frames) == len(set(frames)) == 8
    assert frames[0] == 10
    assert frames[-1] == 80


def test_coverage_reports_progress_and_pair_minima() -> None:
    audits = [
        {
            "valid": not (pair_id == 1 and progress == 1),
            "pair_id": pair_id,
            "progress_index": progress,
        }
        for pair_id in range(2)
        for progress in range(2)
    ]

    result = _coverage(audits, expected_progress_samples=2)

    assert result["fraction"] == 0.75
    assert result["minimum_progress_fraction"] == 0.5
    assert result["minimum_pair_fraction"] == 0.5

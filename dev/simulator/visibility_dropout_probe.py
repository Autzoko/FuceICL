"""确定性 visibility-dropout 风险 probe 的轻量选择逻辑。"""

from __future__ import annotations

from typing import Any

import numpy as np


def select_visibility_dropout_probes(
    tcp_positions: np.ndarray,
    *,
    branch: int,
    last_h6_start: int,
    accepted_limit_m: float,
    minimum_positive_m: float,
) -> dict[str, Any]:
    """选择首次拒绝前最接近阈值的 accepted probe 与首个 rejected probe。"""
    values = np.asarray(tcp_positions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("tcp_positions 必须为 [T,3]")
    if not 0 <= branch < last_h6_start < len(values):
        raise ValueError("branch/last_h6_start 非法")
    frames = np.arange(branch + 1, last_h6_start + 1, dtype=np.int64)
    displacement = np.linalg.norm(values[frames] - values[branch], axis=1)
    accepted_candidates = np.flatnonzero(
        (displacement >= minimum_positive_m)
        & (displacement <= accepted_limit_m)
    )
    rejected = np.flatnonzero(displacement > accepted_limit_m)

    def probe(index: int | None) -> dict[str, Any] | None:
        if index is None:
            return None
        return {
            "frame": int(frames[index]),
            "tcp_net_displacement_m": float(displacement[index]),
        }

    rejected_index = int(rejected[0]) if len(rejected) else None
    accepted = accepted_candidates
    if rejected_index is not None:
        accepted = accepted[accepted < rejected_index]
    accepted_index = (
        int(accepted[np.argmax(displacement[accepted])])
        if len(accepted)
        else None
    )

    return {
        "accepted_probe": probe(accepted_index),
        "rejected_probe": probe(rejected_index),
        "candidate_frames": len(frames),
        "maximum_tcp_net_displacement_m": float(displacement.max()),
    }

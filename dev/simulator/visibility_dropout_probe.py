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
    """选择首个非零 accepted probe 与首个 rejected probe。"""
    values = np.asarray(tcp_positions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("tcp_positions 必须为 [T,3]")
    if not 0 <= branch < last_h6_start < len(values):
        raise ValueError("branch/last_h6_start 非法")
    frames = np.arange(branch + 1, last_h6_start + 1, dtype=np.int64)
    displacement = np.linalg.norm(values[frames] - values[branch], axis=1)
    accepted = np.flatnonzero(
        (displacement >= minimum_positive_m)
        & (displacement <= accepted_limit_m)
    )
    rejected = np.flatnonzero(displacement > accepted_limit_m)

    def probe(indices: np.ndarray) -> dict[str, Any] | None:
        if not len(indices):
            return None
        index = int(indices[0])
        return {
            "frame": int(frames[index]),
            "tcp_net_displacement_m": float(displacement[index]),
        }

    return {
        "accepted_probe": probe(accepted),
        "rejected_probe": probe(rejected),
        "candidate_frames": len(frames),
        "maximum_tcp_net_displacement_m": float(displacement.max()),
    }

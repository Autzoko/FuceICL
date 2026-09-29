"""Forced-dropout risk stratum 的轻量、可单测 support 门控。"""

from __future__ import annotations

import math
from typing import Any


def evaluate_dropout_risk_support(
    report: dict[str, Any],
    *,
    required: bool,
    minimum_accepted: int,
    minimum_rejected: int,
    minimum_both: int,
    minimum_accepted_displacement_m: float,
    maximum_accepted_displacement_m: float,
    minimum_rejected_displacement_m: float,
    maximum_accepted_belief_error_m: float,
) -> dict[str, Any]:
    """只按预注册数量、因果 provenance 与物理边界判定 support。"""
    if not required:
        return {
            "required": False,
            "passed": True,
            "criteria": {},
        }
    if not isinstance(report, dict):
        raise TypeError("dropout risk report 必须是 dict")
    counts = {
        "accepted": int(report.get("accepted_probe_trajectories", -1)),
        "rejected": int(report.get("rejected_probe_trajectories", -1)),
        "both": int(report.get("both_probe_trajectories", -1)),
    }
    accepted_displacement = report.get("accepted_tcp_net_displacement_m")
    rejected_displacement = report.get("rejected_tcp_net_displacement_m")
    accepted_error = report.get("accepted_belief_error_m")
    summaries_valid = all(
        isinstance(value, dict)
        for value in (
            accepted_displacement,
            rejected_displacement,
            accepted_error,
        )
    )

    def finite(source: Any, key: str) -> float:
        if not isinstance(source, dict):
            return math.nan
        value = source.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return math.nan
        return float(value) if math.isfinite(value) else math.nan

    accepted_minimum = finite(accepted_displacement, "minimum")
    accepted_maximum = finite(accepted_displacement, "maximum")
    rejected_minimum = finite(rejected_displacement, "minimum")
    accepted_error_maximum = finite(accepted_error, "maximum")
    criteria = {
        "D1_enabled": report.get("enabled") is True,
        "D2_causal_selection": (
            report.get("selection_uses_actor_state") is False
        ),
        "D3_causal_gate": report.get("gate_uses_actor_error") is False,
        "D4_non_vacuous_counts": (
            counts["accepted"] >= minimum_accepted
            and counts["rejected"] >= minimum_rejected
            and counts["both"] >= minimum_both
        ),
        "D5_boundary_bracketing": (
            summaries_valid
            and accepted_minimum >= minimum_accepted_displacement_m
            and accepted_maximum <= maximum_accepted_displacement_m
            and rejected_minimum > minimum_rejected_displacement_m
        ),
        "D6_accepted_belief_bound": (
            summaries_valid
            and accepted_error_maximum
            <= maximum_accepted_belief_error_m
        ),
    }
    return {
        "required": True,
        "passed": all(criteria.values()),
        "criteria": criteria,
        "counts": counts,
        "accepted_tcp_net_displacement_m": accepted_displacement,
        "rejected_tcp_net_displacement_m": rejected_displacement,
        "accepted_belief_error_m": accepted_error,
    }

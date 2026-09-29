"""Forced-dropout risk support 门控测试。"""

import unittest

from dev.simulator.dropout_risk_support import (
    evaluate_dropout_risk_support,
)


class DropoutRiskSupportTest(unittest.TestCase):
    def _report(self) -> dict:
        return {
            "enabled": True,
            "selection_uses_actor_state": False,
            "gate_uses_actor_error": False,
            "accepted_probe_trajectories": 96,
            "rejected_probe_trajectories": 96,
            "both_probe_trajectories": 96,
            "accepted_tcp_net_displacement_m": {
                "minimum": 0.006,
                "median": 0.008,
                "maximum": 0.0099,
            },
            "rejected_tcp_net_displacement_m": {
                "minimum": 0.0101,
                "median": 0.012,
                "maximum": 0.015,
            },
            "accepted_belief_error_m": {
                "minimum": 0.01,
                "median": 0.02,
                "maximum": 0.03,
            },
        }

    def _evaluate(self, report: dict) -> dict:
        return evaluate_dropout_risk_support(
            report,
            required=True,
            minimum_accepted=90,
            minimum_rejected=90,
            minimum_both=90,
            minimum_accepted_displacement_m=0.005,
            maximum_accepted_displacement_m=0.01,
            minimum_rejected_displacement_m=0.01,
            maximum_accepted_belief_error_m=0.035,
        )

    def test_valid_boundary_support_passes(self) -> None:
        result = self._evaluate(self._report())

        self.assertTrue(result["passed"])
        self.assertTrue(all(result["criteria"].values()))

    def test_privileged_selection_or_weak_counts_fail(self) -> None:
        report = self._report()
        report["selection_uses_actor_state"] = True
        report["accepted_probe_trajectories"] = 89

        result = self._evaluate(report)

        self.assertFalse(result["passed"])
        self.assertFalse(result["criteria"]["D2_causal_selection"])
        self.assertFalse(result["criteria"]["D4_non_vacuous_counts"])

    def test_out_of_bound_accepted_error_fails(self) -> None:
        report = self._report()
        report["accepted_belief_error_m"]["maximum"] = 0.036

        result = self._evaluate(report)

        self.assertFalse(result["passed"])
        self.assertFalse(result["criteria"]["D6_accepted_belief_bound"])


if __name__ == "__main__":
    unittest.main()

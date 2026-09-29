"""Visibility-dropout risk probe selection 的单元测试。"""

import unittest

import numpy as np

from dev.simulator.visibility_dropout_probe import (
    select_visibility_dropout_probes,
)


class VisibilityDropoutProbeSupportTest(unittest.TestCase):
    def test_selects_first_accepted_and_rejected_probe(self) -> None:
        positions = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.004, 0.0, 0.0],
                [0.010, 0.0, 0.0],
                [0.011, 0.0, 0.0],
                [0.020, 0.0, 0.0],
            ]
        )

        result = select_visibility_dropout_probes(
            positions,
            branch=1,
            last_h6_start=5,
            accepted_limit_m=0.01,
            minimum_positive_m=1e-6,
        )

        self.assertEqual(result["accepted_probe"]["frame"], 3)
        self.assertEqual(result["rejected_probe"]["frame"], 4)

    def test_reports_missing_support_without_fabricating_probe(self) -> None:
        positions = np.zeros((5, 3), dtype=np.float64)

        result = select_visibility_dropout_probes(
            positions,
            branch=0,
            last_h6_start=4,
            accepted_limit_m=0.01,
            minimum_positive_m=1e-6,
        )

        self.assertIsNone(result["accepted_probe"])
        self.assertIsNone(result["rejected_probe"])


if __name__ == "__main__":
    unittest.main()

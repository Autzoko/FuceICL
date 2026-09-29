from __future__ import annotations

from copy import deepcopy
import unittest

import numpy as np

from dev.simulator.evaluate_dual_view_observability import (
    DualViewObservabilityConfig,
)
from dev.simulator.evaluate_visibility_conditioned_belief import (
    _paired_bootstrap,
)
from dev.simulator.prepare_rotated_layout_multiview_replay import (
    CAMERA_VARIANT,
    _patched_metadata,
)


def _metadata() -> dict:
    return {
        "env_info": {
            "env_id": "RotatedLayoutSlideToward-v0",
            "env_kwargs": {"operation": "toward", "obs_mode": "none"},
        }
    }


class DualViewProtocolTest(unittest.TestCase):
    def test_patch_records_camera_without_mutating_template(self) -> None:
        template = _metadata()
        result = _patched_metadata(deepcopy(template), operation="toward")

        self.assertNotIn(
            "camera_variant", template["env_info"]["env_kwargs"]
        )
        self.assertEqual(
            result["env_info"]["env_kwargs"]["camera_variant"],
            CAMERA_VARIANT,
        )

    def test_patch_rejects_second_patch(self) -> None:
        metadata = _patched_metadata(_metadata(), operation="toward")

        with self.assertRaisesRegex(ValueError, "二次修改"):
            _patched_metadata(metadata, operation="toward")

    def test_config_rejects_unknown_camera(self) -> None:
        with self.assertRaisesRegex(ValueError, "camera variant"):
            DualViewObservabilityConfig(
                schema_version="dual-view-observability-v1",
                expected_camera_variant="searched-camera",
                expected_query_frames=1,
                expected_baseline_report_sha256="hash",
                maximum_below_threshold_fraction=0.01,
                maximum_missing_run_frames=2,
                minimum_fixed_progress_visible_fraction=1.0,
                maximum_centroid_error_m=0.025,
            )

    def test_paired_bootstrap_detects_uniform_improvement(self) -> None:
        primary = np.asarray([0.01, 0.02, 0.03])
        baseline = primary + 0.01

        result = _paired_bootstrap(
            primary,
            baseline,
            samples=100,
            seed=7,
        )

        self.assertAlmostEqual(result["mean_m"], -0.01)
        self.assertLess(result["ci95_high_m"], 0.0)


if __name__ == "__main__":
    unittest.main()

"""RLBench archive adapter 与评估标签的单元测试。"""

from __future__ import annotations

import unittest

import numpy as np

from dev.retriever.evaluate_rlbench import (
    EvaluationConfig,
    _category,
    _evaluate_rankings,
)
from dev.retriever.rlbench_adapter import (
    DEPTH_SCALE,
    decode_depth,
    decode_mask,
    pointcloud_from_depth,
    quaternion_xyzw_to_rotation_6d,
)


def _record(
    identifier: str,
    *,
    task: str = "place",
    phase: str = "contact",
    gripper: str = "closed",
    center_x: float = 0.0,
    eef_x: float = 0.02,
) -> dict:
    return {
        "chunk_id": identifier,
        "task": task,
        "phase": phase,
        "gripper_state": gripper,
        "active_center": [center_x, 0.0, 0.8],
        "eef_relative_active": [
            eef_x,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ],
    }


class RLBenchAdapterTest(unittest.TestCase):
    def test_official_depth_and_mask_decoding(self) -> None:
        depth_rgb = np.array([[[0, 1, 0], [255, 255, 255]]], dtype=np.uint8)
        decoded = decode_depth(depth_rgb, near=0.1, far=1.1)
        self.assertAlmostEqual(decoded[0, 0], 0.1 + 256.0 / DEPTH_SCALE)
        self.assertAlmostEqual(decoded[0, 1], 1.1)

        mask_rgb = np.array([[[1, 2, 3], [255, 0, 0]]], dtype=np.uint8)
        mask = decode_mask(mask_rgb)
        self.assertEqual(mask.tolist(), [[1 + 2 * 256 + 3 * 65536, 255]])

    def test_identity_camera_pointcloud(self) -> None:
        depth = np.array([[1.0, 2.0], [3.0, 4.0]])
        points = pointcloud_from_depth(
            depth,
            np.eye(4),
            np.eye(3),
        )
        np.testing.assert_allclose(points[0, 1], [2.0, 0.0, 2.0])
        np.testing.assert_allclose(points[1, 0], [0.0, 3.0, 3.0])

    def test_quaternion_identity_to_rotation_6d(self) -> None:
        rotation = quaternion_xyzw_to_rotation_6d([0.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(rotation, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])

    def test_hard_negative_categories_are_mutually_exclusive(self) -> None:
        config = EvaluationConfig()
        query = _record("query")
        values = {
            "positive": _record("positive"),
            "wrong_task": _record("wrong_task", task="open"),
            "wrong_phase": _record("wrong_phase", phase="finish"),
            "wrong_gripper": _record("wrong_gripper", gripper="open"),
            "wrong_layout": _record("wrong_layout", center_x=0.4),
        }
        for expected, candidate in values.items():
            expected_category = "strict_positive" if expected == "positive" else expected
            self.assertEqual(_category(query, candidate, config), expected_category)

    def test_recall_and_intrusion_metrics(self) -> None:
        config = EvaluationConfig(recall_k=(1, 4))
        query = _record("query")
        candidates = {
            "wrong_phase": _record("wrong_phase", phase="finish"),
            "positive": _record("positive"),
            "wrong_task": _record("wrong_task", task="open"),
            "wrong_layout": _record("wrong_layout", center_x=0.4),
        }
        report = _evaluate_rankings(
            [query],
            candidates,
            {"query": list(candidates)},
            config,
        )
        self.assertEqual(report["recall"]["strict"]["recall@1"], 0.0)
        self.assertEqual(report["recall"]["strict"]["recall@4"], 1.0)
        self.assertEqual(
            report["hard_negative_intrusion"]["top@1"]["wrong_phase"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()

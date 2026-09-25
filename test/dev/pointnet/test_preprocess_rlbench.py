"""RLBench PointNet++ 事件标签与 pair 构造测试。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from dev.pointnet.preprocess_rlbench import (
    PointNetPreprocessConfig,
    RLBenchPointNetPreprocessor,
    build_pair_labels,
    detect_event_anchors,
)


def _record(
    identifier: str,
    *,
    episode: int,
    task: str = "place",
    phase: str = "contact",
    gripper: str = "closed",
    action_x: float = 0.05,
    center_x: float = 0.0,
) -> dict:
    return {
        "chunk_id": identifier,
        "episode": episode,
        "task": task,
        "phase": phase,
        "gripper_state": gripper,
        "future_translation": [action_x, 0.0, 0.0],
        "future_rotation_axis_angle": [0.0, 0.0, 0.0],
        "effect_translation": [action_x, 0.0, 0.0],
        "effect_valid": True,
        "active_center": [center_x, 0.0, 0.8],
        "active_extent": [0.04, 0.04, 0.08],
        "eef_relative_position": [0.02, 0.0, 0.0],
    }


class EventAnchorTest(unittest.TestCase):
    def test_gripper_events_define_contact_and_release(self) -> None:
        anchors = detect_event_anchors([1.0] * 10 + [0.0] * 10 + [1.0] * 5)
        values = {anchor.phase: anchor for anchor in anchors}
        self.assertEqual(values["contact"].frame, 10)
        self.assertEqual(values["finish"].frame, 22)
        self.assertEqual(values["contact"].source, "gripper_event")

    def test_no_event_uses_explicit_fallback(self) -> None:
        anchors = detect_event_anchors([1.0] * 21)
        self.assertEqual([anchor.frame for anchor in anchors], [5, 10, 15, 18])
        self.assertTrue(all(anchor.source == "time_fallback" for anchor in anchors))


class RobotHandleFilterTest(unittest.TestCase):
    def test_rigid_eef_track_is_robot_but_static_object_is_not(self) -> None:
        config = PointNetPreprocessConfig.from_json(
            Path("dev/pointnet/config/rlbench_smoke.json")
        )
        preprocessor = RLBenchPointNetPreprocessor(config)
        frames = [0, 1, 2, 3]
        eef_positions = [
            np.asarray([0.00, 0.0, 0.8]),
            np.asarray([0.05, 0.0, 0.8]),
            np.asarray([0.10, 0.0, 0.8]),
            np.asarray([0.15, 0.0, 0.8]),
        ]
        demo = [
            SimpleNamespace(
                gripper_pose=np.concatenate((position, [0.0, 0.0, 0.0, 1.0]))
            )
            for position in eef_positions
        ]
        frame_segments = {
            frame: {
                40: SimpleNamespace(center=eef_positions[frame] + [0.02, 0.0, 0.0]),
                99: SimpleNamespace(center=np.asarray([0.20, 0.0, 0.8])),
            }
            for frame in frames
        }

        handles = preprocessor._robot_handles(demo, frames, frame_segments)

        self.assertIn(40, handles)
        self.assertNotIn(99, handles)


class PairLabelTest(unittest.TestCase):
    def test_positive_must_be_cross_episode_and_action_compatible(self) -> None:
        rows = [
            _record("query", episode=0),
            _record("positive", episode=1, action_x=0.06),
            _record("same_episode", episode=0, action_x=0.05),
            _record("wrong_phase", episode=2, phase="finish"),
            _record("wrong_gripper", episode=3, gripper="open"),
            _record("wrong_layout", episode=4, center_x=0.5),
            _record("collision", episode=5, task="open_drawer"),
        ]
        labels = {row["query_id"]: row for row in build_pair_labels(rows)}
        query = labels["query"]

        self.assertIn("positive", query["positive_ids"])
        self.assertNotIn("same_episode", query["positive_ids"])
        self.assertIn("wrong_phase", query["hard_negatives"]["wrong_phase"])
        self.assertIn("wrong_gripper", query["hard_negatives"]["wrong_gripper"])
        self.assertIn("wrong_layout", query["hard_negatives"]["wrong_layout"])
        self.assertIn("collision", query["hard_negatives"]["geometry_collision"])


if __name__ == "__main__":
    unittest.main()

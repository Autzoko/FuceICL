"""Point-derived layout action transport 的几何不变量测试。"""

from __future__ import annotations

import math
import unittest

import torch

from src.components.predictor import transport_planar_layout_action


def _rotation_z(angle: float) -> torch.Tensor:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return torch.tensor(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )


class PlanarLayoutTransportTest(unittest.TestCase):
    def test_quarter_turn_rotates_pose_action_and_preserves_gripper(self) -> None:
        action = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0]],
            dtype=torch.float64,
        )
        output = transport_planar_layout_action(
            action,
            demo_tcp_rotation_world=torch.eye(3, dtype=torch.float64),
            query_tcp_rotation_world=torch.eye(3, dtype=torch.float64),
            demo_layout_axis_world=torch.tensor([1.0, 0.0, 0.0]),
            query_layout_axis_world=torch.tensor([0.0, 1.0, 0.0]),
        )

        expected = torch.tensor(
            [[0.0, 1.0, 0.0, -1.0, 0.0, 0.0, -1.0]],
            dtype=torch.float64,
        )
        self.assertTrue(torch.allclose(output, expected, atol=1e-12))
        self.assertTrue(torch.equal(output[:, 6], action[:, 6]))

    def test_identity_is_exact_and_common_world_yaw_is_equivariant(self) -> None:
        torch.manual_seed(29)
        action = torch.randn(6, 7, dtype=torch.float64)
        identity = torch.eye(3, dtype=torch.float64)
        demo_axis = torch.tensor([0.8, -0.3, 0.0], dtype=torch.float64)
        query_axis = torch.tensor([0.2, 0.9, 0.0], dtype=torch.float64)
        base = transport_planar_layout_action(
            action,
            demo_tcp_rotation_world=identity,
            query_tcp_rotation_world=identity,
            demo_layout_axis_world=demo_axis,
            query_layout_axis_world=query_axis,
        )
        common = _rotation_z(0.73)
        transformed = transport_planar_layout_action(
            action,
            demo_tcp_rotation_world=common,
            query_tcp_rotation_world=common,
            demo_layout_axis_world=common @ demo_axis,
            query_layout_axis_world=common @ query_axis,
        )
        exact_identity = transport_planar_layout_action(
            action,
            demo_tcp_rotation_world=identity,
            query_tcp_rotation_world=identity,
            demo_layout_axis_world=demo_axis,
            query_layout_axis_world=demo_axis,
        )

        self.assertTrue(torch.allclose(base, transformed, atol=1e-12))
        self.assertTrue(torch.equal(exact_identity, action))

    def test_invalid_geometry_is_rejected(self) -> None:
        action = torch.zeros(6, 7)
        identity = torch.eye(3)
        with self.assertRaisesRegex(ValueError, "平面投影退化"):
            transport_planar_layout_action(
                action,
                demo_tcp_rotation_world=identity,
                query_tcp_rotation_world=identity,
                demo_layout_axis_world=torch.tensor([0.0, 0.0, 1.0]),
                query_layout_axis_world=torch.tensor([1.0, 0.0, 0.0]),
            )
        with self.assertRaisesRegex(ValueError, "正交旋转矩阵"):
            transport_planar_layout_action(
                action,
                demo_tcp_rotation_world=torch.ones(3, 3),
                query_tcp_rotation_world=identity,
                demo_layout_axis_world=torch.tensor([1.0, 0.0, 0.0]),
                query_layout_axis_world=torch.tensor([1.0, 0.0, 0.0]),
            )


if __name__ == "__main__":
    unittest.main()

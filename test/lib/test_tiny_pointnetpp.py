"""Tiny PointNet++ 的接口、mask 与梯度测试。"""

from __future__ import annotations

import unittest

import torch

from lib.tiny_pointnetpp import TinyPointNetPPConfig, TinyPointNetPPEncoder


class TinyPointNetPPTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = TinyPointNetPPEncoder(
            TinyPointNetPPConfig(
                output_dim=16,
                sa1_points=8,
                sa1_neighbors=4,
                sa2_points=4,
                sa2_neighbors=4,
            )
        )

    def test_output_is_finite_normalized_and_differentiable(self) -> None:
        points = torch.randn(2, 12, 3, requires_grad=True)
        mask = torch.ones(2, 12, dtype=torch.bool)
        mask[1, 8:] = False

        embeddings = self.model(points, mask)

        self.assertEqual(tuple(embeddings.shape), (2, 16))
        self.assertTrue(torch.isfinite(embeddings).all())
        self.assertTrue(
            torch.allclose(
                torch.linalg.vector_norm(embeddings, dim=1),
                torch.ones(2),
                atol=1e-5,
            )
        )
        embeddings.sum().backward()
        self.assertIsNotNone(points.grad)
        self.assertTrue(torch.isfinite(points.grad).all())

    def test_masked_padding_does_not_change_embedding(self) -> None:
        self.model.eval()
        points = torch.randn(1, 12, 3)
        mask = torch.zeros(1, 12, dtype=torch.bool)
        mask[:, :7] = True
        modified = points.clone()
        modified[:, 7:] = 1e4

        with torch.inference_mode():
            first = self.model(points, mask)
            second = self.model(modified, mask)

        self.assertTrue(torch.allclose(first, second, atol=1e-6))

    def test_point_permutation_does_not_change_embedding(self) -> None:
        self.model.eval()
        points = torch.randn(1, 12, 3)
        permutation = torch.randperm(12)

        with torch.inference_mode():
            first = self.model(points)
            second = self.model(points[:, permutation])

        self.assertTrue(torch.allclose(first, second, atol=1e-5))

    def test_all_invalid_cloud_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "至少需要一个有效点"):
            self.model(torch.randn(1, 4, 3), torch.zeros(1, 4, dtype=torch.bool))


if __name__ == "__main__":
    unittest.main()

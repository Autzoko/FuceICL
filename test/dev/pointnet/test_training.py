"""PointNet Siamese Retriever 模型与训练损失测试。"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from dev.pointnet.dataset import PointNetPairDataset, STATE_DIM
from dev.pointnet.retriever_model import GeometricSiameseRetriever
from dev.pointnet.train import TrainConfig, _losses


class RetrieverModelTest(unittest.TestCase):
    def test_forward_shapes_and_finite_gradient(self) -> None:
        torch.manual_seed(7)
        model = GeometricSiameseRetriever()
        context = {
            "active_points": torch.randn(2, 128, 3),
            "target_points": torch.randn(2, 128, 3),
            "state": torch.randn(2, STATE_DIM),
            "target_valid": torch.tensor([True, False]),
        }

        outputs = model(context)
        outputs["embedding"].sum().backward()

        self.assertEqual(outputs["embedding"].shape, (2, 128))
        self.assertEqual(outputs["phase_logits"].shape, (2, 4))
        self.assertEqual(outputs["future_prediction"].shape, (2, 10))
        self.assertTrue(torch.isfinite(outputs["embedding"]).all())
        self.assertTrue(
            all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
        )

    def test_triplet_loss_prefers_positive_similarity(self) -> None:
        config = TrainConfig(
            seed=1,
            epochs=1,
            batch_size=2,
            num_workers=0,
            shard_cache_size=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            triplet_margin=0.15,
            alignment_weight=0.05,
            phase_weight=0.1,
            future_weight=0.2,
            gradient_clip_norm=1.0,
            point_jitter_std_m=0.0,
            point_jitter_clip_m=0.0,
            point_dropout_probability=0.0,
            amp=False,
            eval_batch_size=2,
            recall_k=(1, 4),
        )
        anchor_embedding = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        outputs = {
            "embedding": torch.cat(
                (
                    anchor_embedding,
                    anchor_embedding,
                    -anchor_embedding,
                )
            ),
            "phase_logits": torch.zeros(6, 4, requires_grad=True),
            "future_prediction": torch.zeros(6, 10, requires_grad=True),
        }
        anchor = {
            "phase_id": torch.tensor([0, 1]),
            "future_target": torch.zeros(2, 10),
            "effect_valid": torch.tensor([True, False]),
        }

        losses = _losses(outputs, anchor, 2, config)

        self.assertEqual(float(losses["triplet"]), 0.0)
        self.assertEqual(float(losses["triplet_accuracy"]), 1.0)


class PairDatasetTest(unittest.TestCase):
    def test_loads_context_without_future_leakage_into_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.joinpath("shards").mkdir()
            count, points = 3, 32
            arrays = {
                "active_points": np.zeros((count, points, 3), np.float32),
                "target_points": np.zeros((count, points, 3), np.float32),
                "active_center": np.zeros((count, 3), np.float32),
                "active_extent": np.ones((count, 3), np.float32),
                "target_center": np.ones((count, 3), np.float32),
                "target_extent": np.ones((count, 3), np.float32),
                "eef_relative_active": np.zeros((count, 9), np.float32),
                "eef_velocity": np.zeros((count, 6), np.float32),
                "gripper_width": np.zeros(count, np.float32),
                "target_valid": np.asarray([True, False, True]),
                "phase_id": np.asarray([0, 1, 2], np.int8),
                "future_translation": np.ones((count, 3), np.float32),
                "future_rotation_axis_angle": np.ones((count, 3), np.float32),
                "future_gripper_delta": np.ones(count, np.float32),
                "effect_translation": np.ones((count, 3), np.float32),
                "effect_valid": np.ones(count, np.bool_),
            }
            np.savez(root / "shards/train-00000.npz", **arrays)
            manifest = [
                {
                    "chunk_id": identifier,
                    "shard": "shards/train-00000.npz",
                    "row": index,
                    "task": "place",
                    "episode": index,
                }
                for index, identifier in enumerate(("query", "positive", "negative"))
            ]
            root.joinpath("manifest-train.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in manifest)
            )
            pair = {
                "query_id": "query",
                "positive_ids": ["positive"],
                "hard_negatives": {
                    "wrong_phase": ["negative"],
                    "wrong_gripper": [],
                    "wrong_layout": [],
                    "geometry_collision": [],
                },
            }
            root.joinpath("pairs-train.jsonl").write_text(json.dumps(pair) + "\n")

            sample = PointNetPairDataset(root, "train", seed=7)[0]

            self.assertEqual(sample["anchor"]["state"].shape, (STATE_DIM,))
            self.assertEqual(sample["anchor"]["future_target"].shape, (10,))
            self.assertFalse(
                np.shares_memory(
                    sample["anchor"]["state"],
                    sample["anchor"]["future_target"],
                )
            )


if __name__ == "__main__":
    unittest.main()

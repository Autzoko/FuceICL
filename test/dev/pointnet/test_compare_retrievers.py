"""统一 Retriever 对比评测的单元测试。"""

from __future__ import annotations

import unittest

import torch

from dev.pointnet.compare_retrievers import (
    ComparisonConfig,
    GeometryScales,
    evaluate_scores,
    geometry_similarity,
    paired_bootstrap_delta,
)
from dev.pointnet.dataset import STATE_DIM


class GeometryBaselineTest(unittest.TestCase):
    def test_identical_context_has_highest_similarity(self) -> None:
        states = torch.zeros(3, STATE_DIM)
        states[:, 3:6] = torch.tensor([0.1, 0.1, 0.1])
        states[:, 15:21] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        states[1] = states[0]
        states[2] = states[0]
        states[2, 0] = 0.5
        states[2, 12] = 0.3
        scales = GeometryScales(*(0.1 for _ in range(8)))

        scores = geometry_similarity(states, scales)

        self.assertEqual(scores.shape, (3, 3))
        self.assertAlmostEqual(float(scores[0, 0]), 1.0, places=5)
        self.assertAlmostEqual(float(scores[0, 1]), 1.0, places=5)
        self.assertGreater(float(scores[0, 1]), float(scores[0, 2]))


class RankingEvaluationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            {
                "chunk_id": f"chunk-{index}",
                "task": "place",
                "episode": index,
            }
            for index in range(4)
        ]
        self.pairs = [
            {
                "query_id": "chunk-0",
                "positive_ids": ["chunk-1"],
                "hard_negatives": {
                    "wrong_phase": ["chunk-2"],
                    "wrong_gripper": [],
                    "wrong_layout": ["chunk-3"],
                    "geometry_collision": [],
                },
            },
            *[
                {
                    "query_id": f"chunk-{index}",
                    "positive_ids": [],
                    "hard_negatives": {
                        "wrong_phase": [],
                        "wrong_gripper": [],
                        "wrong_layout": [],
                        "geometry_collision": [],
                    },
                }
                for index in range(1, 4)
            ],
        ]
        self.config = ComparisonConfig(
            recall_k=(1, 2),
            fusion_weights=(0.5,),
            batch_size=2,
            num_workers=0,
            bootstrap_samples=100,
            seed=7,
        )

    def test_recall_and_hard_negative_intrusion(self) -> None:
        scores = torch.tensor(
            [
                [1.0, 0.9, 0.8, 0.1],
                [0.9, 1.0, 0.0, 0.0],
                [0.8, 0.0, 1.0, 0.0],
                [0.1, 0.0, 0.0, 1.0],
            ]
        )

        report, outcomes = evaluate_scores(
            scores, self.records, self.pairs, self.config, "same_task"
        )

        self.assertEqual(report["eligible_queries"], 1)
        self.assertEqual(report["recall"]["recall@1"], 1.0)
        self.assertEqual(
            report["hard_negative_intrusion"]["wrong_phase"]["intrusion@2"],
            1.0,
        )
        self.assertEqual(outcomes["chunk-0"]["rank"], 1)

    def test_paired_bootstrap_delta(self) -> None:
        first = {
            "a": {"hits": {"4": 0}},
            "b": {"hits": {"4": 1}},
        }
        second = {
            "a": {"hits": {"4": 1}},
            "b": {"hits": {"4": 1}},
        }

        result = paired_bootstrap_delta(
            first, second, recall_k=4, samples=100, seed=7
        )

        self.assertEqual(result["queries"], 2)
        self.assertEqual(result["delta"], 0.5)


if __name__ == "__main__":
    unittest.main()

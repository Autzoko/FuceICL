"""规范化任务键分桶器的隔离单元测试。"""

from __future__ import annotations

import unittest

from src.components.retriever import (
    TaskBucketCandidate,
    TaskBucketRouter,
    TaskKey,
)


class TaskBucketRouterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.toward = TaskKey(
            operation=" Push ",
            active_object="Red   Cube",
            reference_object="Green Marker",
            relation_effect="Approach",
        )
        self.away = TaskKey(
            operation="push",
            active_object="red cube",
            reference_object="green marker",
            relation_effect="separate",
        )
        self.router = TaskBucketRouter()
        self.router.build_index(
            [
                TaskBucketCandidate("toward-1", self.toward),
                TaskBucketCandidate("toward-2", self.toward),
                TaskBucketCandidate("away-1", self.away),
            ]
        )

    def test_normalization_and_stable_exact_bucket(self) -> None:
        query = TaskKey(
            operation="PUSH",
            active_object="red cube",
            reference_object=" green marker ",
            relation_effect="approach",
        )

        result = self.router.route(query)

        self.assertTrue(result.accepted)
        self.assertIsNone(result.reason)
        self.assertEqual(
            [item.candidate_id for item in result.candidates],
            ["toward-1", "toward-2"],
        )

    def test_opposite_relation_never_falls_back(self) -> None:
        query = TaskKey(
            operation="push",
            active_object="red cube",
            reference_object="green marker",
            relation_effect="withdraw",
        )

        result = self.router.route(query)

        self.assertFalse(result.accepted)
        self.assertEqual(result.candidates, ())
        self.assertEqual(result.reason, "no_exact_task_key_match")

    def test_qualifiers_are_canonical_and_part_of_key(self) -> None:
        left = TaskKey(
            operation="open",
            active_object="drawer",
            qualifiers=("Left", " TOP ", "left"),
        )
        right = TaskKey(
            operation="open",
            active_object="drawer",
            qualifiers=("top", "right"),
        )
        self.assertEqual(left.qualifiers, ("left", "top"))
        self.assertNotEqual(left, right)

    def test_duplicate_identifier_and_unindexed_route_are_rejected(self) -> None:
        empty = TaskBucketRouter()
        with self.assertRaisesRegex(RuntimeError, "build_index"):
            empty.route(self.toward)
        with self.assertRaisesRegex(ValueError, "candidate_id 必须唯一"):
            empty.build_index(
                [
                    TaskBucketCandidate("same", self.toward),
                    TaskBucketCandidate("same", self.away),
                ]
            )


if __name__ == "__main__":
    unittest.main()

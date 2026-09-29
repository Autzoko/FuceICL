"""轻量 task-bucket embedding Retriever 的接口与拒绝行为测试。"""

from __future__ import annotations

import unittest

import torch

from src.components.retriever import (
    EmbeddingCandidate,
    EmbeddingQuery,
    EmbeddingRetrieverConfig,
    ExactEmbeddingRetriever,
    TaskKey,
)


PUSH = TaskKey("push", "cube", "anchor", "approach")
PULL = TaskKey("push", "cube", "anchor", "separate")


def _candidate(
    identifier: str,
    embedding: tuple[float, ...],
    *,
    task_key: TaskKey = PUSH,
    episode: str | None = None,
    payload: object = None,
) -> EmbeddingCandidate:
    return EmbeddingCandidate(
        candidate_id=identifier,
        task_key=task_key,
        embedding=torch.tensor(embedding),
        episode_id=episode,
        payload=payload,
    )


class ExactEmbeddingRetrieverTest(unittest.TestCase):
    def test_exact_bucket_ranking_episode_cap_and_payload(self) -> None:
        payload = {"action": [1, 2, 3]}
        retriever = ExactEmbeddingRetriever(
            EmbeddingRetrieverConfig(default_top_k=3, max_per_episode=1)
        )
        retriever.build_index(
            [
                _candidate("a0", (1.0, 0.0), episode="a"),
                _candidate("a1", (0.99, 0.01), episode="a"),
                _candidate(
                    "b0",
                    (0.8, 0.2),
                    episode="b",
                    payload=payload,
                ),
                _candidate("wrong", (1.0, 0.0), task_key=PULL),
            ]
        )

        result = retriever.retrieve(
            EmbeddingQuery(PUSH, torch.tensor([1.0, 0.0]))
        )

        self.assertTrue(result.accepted)
        self.assertIsNone(result.reason)
        self.assertEqual(
            [hit.candidate.candidate_id for hit in result.hits],
            ["a0", "b0"],
        )
        self.assertIs(result.hits[1].candidate.payload, payload)
        self.assertNotIn(
            "wrong",
            [hit.candidate.candidate_id for hit in result.hits],
        )

    def test_unknown_bucket_and_threshold_are_explicit_rejections(self) -> None:
        retriever = ExactEmbeddingRetriever(
            EmbeddingRetrieverConfig(minimum_score=0.9)
        )
        retriever.build_index([_candidate("a", (1.0, 0.0))])

        unknown = retriever.retrieve(
            EmbeddingQuery(PULL, torch.tensor([1.0, 0.0]))
        )
        low_score = retriever.retrieve(
            EmbeddingQuery(PUSH, torch.tensor([0.0, 1.0]))
        )

        self.assertFalse(unknown.accepted)
        self.assertEqual(unknown.reason, "no_exact_task_key_match")
        self.assertFalse(low_score.accepted)
        self.assertEqual(low_score.reason, "below_score_threshold")

    def test_failed_rebuild_preserves_previous_index(self) -> None:
        retriever = ExactEmbeddingRetriever()
        retriever.build_index([_candidate("valid", (1.0, 0.0))])

        with self.assertRaisesRegex(ValueError, "维度必须一致"):
            retriever.build_index(
                [
                    _candidate("x", (1.0, 0.0)),
                    _candidate("y", (1.0, 0.0, 0.0)),
                ]
            )

        result = retriever.retrieve(
            EmbeddingQuery(PUSH, torch.tensor([1.0, 0.0]))
        )
        self.assertEqual(result.hits[0].candidate.candidate_id, "valid")
        self.assertEqual(retriever.embedding_dim, 2)

    def test_query_dimension_and_zero_embedding_are_rejected(self) -> None:
        retriever = ExactEmbeddingRetriever()
        retriever.build_index([_candidate("valid", (1.0, 0.0))])
        with self.assertRaisesRegex(ValueError, "维度与索引不一致"):
            retriever.retrieve(
                EmbeddingQuery(PUSH, torch.tensor([1.0, 0.0, 0.0]))
            )
        with self.assertRaisesRegex(ValueError, "零向量"):
            EmbeddingQuery(PUSH, torch.zeros(2))


if __name__ == "__main__":
    unittest.main()

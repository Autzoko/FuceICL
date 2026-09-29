"""belief-state Demo 精确检索公共契约测试。"""

import unittest

import torch

from src.components.retriever import (
    BeliefDemoCandidate,
    BeliefDemoQuery,
    ExactBeliefDemoRetriever,
)


class ExactBeliefDemoRetrieverTest(unittest.TestCase):
    def setUp(self) -> None:
        self.retriever = ExactBeliefDemoRetriever(
            torch.tensor([1.0, 2.0])
        )
        self.toward_action = torch.ones(6, 7)
        self.away_action = -torch.ones(6, 7)
        self.retriever.build_index(
            [
                BeliefDemoCandidate(
                    "toward-near",
                    "toward",
                    torch.tensor([0.0, 0.0]),
                    self.toward_action,
                ),
                BeliefDemoCandidate(
                    "toward-far",
                    "toward",
                    torch.tensor([2.0, 4.0]),
                    self.toward_action,
                ),
                BeliefDemoCandidate(
                    "away-identical",
                    "away",
                    torch.tensor([0.1, 0.2]),
                    self.away_action,
                ),
            ]
        )
        self.assertEqual(self.retriever.action_shape, (6, 7))

    def test_retrieval_matches_frozen_normalized_mse(self) -> None:
        result = self.retriever.retrieve(
            BeliefDemoQuery("toward", torch.tensor([0.1, 0.2])),
            top_k=2,
        )

        self.assertTrue(result.accepted)
        self.assertEqual(
            [hit.candidate.candidate_id for hit in result.hits],
            ["toward-near", "toward-far"],
        )
        expected = ((0.1 / 1.0) ** 2 + (0.2 / 2.0) ** 2) / 2
        self.assertAlmostEqual(
            result.hits[0].normalized_mse_distance,
            expected,
        )

    def test_typed_operation_prevents_cross_operation_match(self) -> None:
        result = self.retriever.retrieve(
            BeliefDemoQuery("away", torch.tensor([0.1, 0.2]))
        )

        self.assertEqual(result.hits[0].candidate.candidate_id, "away-identical")
        self.assertEqual(result.hits[0].normalized_mse_distance, 0.0)

    def test_unknown_operation_is_explicit_refusal(self) -> None:
        result = self.retriever.retrieve(
            BeliefDemoQuery("unknown", torch.zeros(2))
        )

        self.assertFalse(result.accepted)
        self.assertEqual(result.hits, ())
        self.assertEqual(result.reason, "no_typed_operation_bucket")

    def test_candidate_owns_frozen_tensor_copies(self) -> None:
        state = torch.tensor([1.0, 2.0])
        action = torch.ones(6, 7)
        candidate = BeliefDemoCandidate(
            "copy-check",
            "toward",
            state,
            action,
        )

        state.zero_()
        action.zero_()

        torch.testing.assert_close(candidate.state, torch.tensor([1.0, 2.0]))
        torch.testing.assert_close(candidate.raw_action, torch.ones(6, 7))

    def test_equal_distances_keep_index_order(self) -> None:
        action = torch.zeros(6, 7)
        retriever = ExactBeliefDemoRetriever(torch.ones(2))
        retriever.build_index(
            [
                BeliefDemoCandidate("first", "slide", torch.ones(2), action),
                BeliefDemoCandidate("second", "slide", torch.ones(2), action),
            ]
        )

        result = retriever.retrieve(
            BeliefDemoQuery("slide", torch.zeros(2)),
            top_k=2,
        )

        self.assertEqual(
            [hit.candidate.candidate_id for hit in result.hits],
            ["first", "second"],
        )


if __name__ == "__main__":
    unittest.main()

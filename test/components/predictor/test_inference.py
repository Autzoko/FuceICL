"""Retriever→Predictor 最小输入契约测试。"""

from __future__ import annotations

import unittest

import torch

from src.components.predictor import (
    ActionChunkRequest,
    LayoutEquivariantDemoPolicy,
    PreparedDemoContext,
    RetrievalAugmentedActionPredictor,
)


class RetrievalAugmentedActionPredictorTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(23)
        self.policy = LayoutEquivariantDemoPolicy()
        self.predictor = RetrievalAugmentedActionPredictor(self.policy)
        self.state = torch.randn(21, dtype=torch.float64)
        self.action = torch.randn(6, 7, dtype=torch.float64) * 0.01

    def test_identity_demo_preserves_action_and_provenance(self) -> None:
        result = self.predictor.predict(
            ActionChunkRequest(
                query_state=self.state,
                demo=PreparedDemoContext(
                    candidate_id="episode-1:chunk-2",
                    state=self.state.clone(),
                    transported_action=self.action,
                    retrieval_score=0.93,
                ),
            )
        )

        self.assertTrue(result.used_demo)
        self.assertEqual(result.candidate_id, "episode-1:chunk-2")
        self.assertEqual(result.retrieval_score, 0.93)
        self.assertEqual(result.reason, "demo_conditioned")
        self.assertTrue(torch.equal(result.action, self.action.float()))
        self.assertFalse(result.action.requires_grad)

    def test_no_demo_is_exact_zero_despite_query_state(self) -> None:
        result = self.predictor.predict(
            ActionChunkRequest(query_state=self.state, demo=None)
        )

        self.assertFalse(result.used_demo)
        self.assertIsNone(result.candidate_id)
        self.assertEqual(result.reason, "no_demo")
        self.assertTrue(torch.equal(result.action, torch.zeros(6, 7)))

    def test_request_rejects_redundant_or_invalid_shapes(self) -> None:
        bad_demo = PreparedDemoContext(
            candidate_id="bad",
            state=torch.zeros(20),
            transported_action=torch.zeros(6, 7),
        )
        with self.assertRaisesRegex(ValueError, "Demo state dimension"):
            self.predictor.predict(
                ActionChunkRequest(torch.zeros(21), bad_demo)
            )
        with self.assertRaisesRegex(ValueError, "transported action shape"):
            self.predictor.predict(
                ActionChunkRequest(
                    torch.zeros(21),
                    PreparedDemoContext(
                        "bad-action",
                        torch.zeros(21),
                        torch.zeros(5, 7),
                    ),
                )
            )
        with self.assertRaisesRegex(ValueError, r"\[-1,1\]"):
            PreparedDemoContext(
                "bad-score",
                torch.zeros(21),
                torch.zeros(6, 7),
                retrieval_score=1.1,
            )


if __name__ == "__main__":
    unittest.main()

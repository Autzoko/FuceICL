"""Selective belief retrieval-to-action pipeline 契约测试。"""

import unittest

import torch

from src.components.pipeline import (
    SelectiveBeliefActionPipeline,
    SelectiveBeliefActionRequest,
)
from src.components.predictor import (
    DemoRelativeActionPredictor,
    DemoRelativePolicy,
)
from src.components.retriever import (
    BeliefDemoCandidate,
    BeliefDemoQuery,
    ExactBeliefDemoRetriever,
)


class SelectiveBeliefActionPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(29)
        policy = DemoRelativePolicy()
        action = torch.randn(6, 7) * 0.01
        self.state = torch.randn(26)
        self.action = action
        retriever = ExactBeliefDemoRetriever(policy.state_std)
        retriever.build_index(
            [
                BeliefDemoCandidate(
                    "episode-2:chunk-3",
                    "slide toward",
                    self.state,
                    action,
                )
            ]
        )
        self.pipeline = SelectiveBeliefActionPipeline(
            retriever,
            DemoRelativeActionPredictor(policy),
        )

    def test_matching_identity_demo_preserves_action(self) -> None:
        result = self.pipeline.predict(
            SelectiveBeliefActionRequest(
                BeliefDemoQuery("slide toward", self.state),
                belief_risk_accepted=True,
            )
        )

        self.assertTrue(result.retrieval.accepted)
        self.assertTrue(result.prediction.used_demo)
        self.assertEqual(
            result.prediction.candidate_id,
            "episode-2:chunk-3",
        )
        self.assertTrue(torch.equal(result.prediction.action, self.action))

    def test_risk_rejection_never_uses_demo(self) -> None:
        result = self.pipeline.predict(
            SelectiveBeliefActionRequest(
                BeliefDemoQuery("slide toward", self.state),
                belief_risk_accepted=False,
            )
        )

        self.assertFalse(result.retrieval.accepted)
        self.assertEqual(result.retrieval.reason, "belief_risk_rejected")
        self.assertFalse(result.prediction.used_demo)
        self.assertTrue(torch.equal(result.prediction.action, torch.zeros(6, 7)))

    def test_unknown_operation_never_uses_demo(self) -> None:
        result = self.pipeline.predict(
            SelectiveBeliefActionRequest(
                BeliefDemoQuery("unknown", self.state),
                belief_risk_accepted=True,
            )
        )

        self.assertFalse(result.retrieval.accepted)
        self.assertEqual(
            result.retrieval.reason,
            "no_typed_operation_bucket",
        )
        self.assertFalse(result.prediction.used_demo)


if __name__ == "__main__":
    unittest.main()

"""Demo-relative action-prior 公共推理契约测试。"""

import unittest

import torch

from src.components.predictor import (
    DemoRelativeActionChunkRequest,
    DemoRelativeActionPredictor,
    DemoRelativePolicy,
    PreparedRawDemoContext,
)


class DemoRelativeActionPredictorTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(29)
        self.policy = DemoRelativePolicy()
        self.predictor = DemoRelativeActionPredictor(self.policy)
        self.query = torch.randn(26, dtype=torch.float64)
        self.demo = torch.randn(26, dtype=torch.float64)
        self.action = torch.randn(6, 7, dtype=torch.float64) * 0.01

    def _context(
        self,
        *,
        query: torch.Tensor | None = None,
        demo: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, PreparedRawDemoContext]:
        resolved_query = self.query if query is None else query
        resolved_demo = self.demo if demo is None else demo
        return resolved_query, PreparedRawDemoContext(
            candidate_id="episode-2:chunk-3",
            state=resolved_demo,
            raw_action=self.action,
            retrieval_distance=0.17,
        )

    def test_identity_context_preserves_raw_demo(self) -> None:
        query, context = self._context(query=self.demo, demo=self.demo)

        result = self.predictor.predict(
            DemoRelativeActionChunkRequest(query, context)
        )

        self.assertTrue(result.used_demo)
        self.assertEqual(result.candidate_id, "episode-2:chunk-3")
        self.assertEqual(result.retrieval_distance, 0.17)
        self.assertTrue(torch.equal(result.action, self.action.float()))

    def test_common_state_shift_does_not_change_prediction(self) -> None:
        query, context = self._context()
        baseline = self.predictor.predict(
            DemoRelativeActionChunkRequest(query, context)
        )
        shift = torch.randn(26, dtype=torch.float64)
        shifted_query, shifted_context = self._context(
            query=query + shift,
            demo=self.demo + shift,
        )

        shifted = self.predictor.predict(
            DemoRelativeActionChunkRequest(
                shifted_query,
                shifted_context,
            )
        )

        self.assertTrue(torch.allclose(baseline.action, shifted.action))

    def test_no_demo_is_exact_zero(self) -> None:
        result = self.predictor.predict(
            DemoRelativeActionChunkRequest(self.query, None)
        )

        self.assertFalse(result.used_demo)
        self.assertEqual(result.reason, "no_demo")
        self.assertTrue(torch.equal(result.action, torch.zeros(6, 7)))

    def test_policy_remains_lightweight(self) -> None:
        self.assertEqual(self.policy.parameter_count, 10_916)

    def test_request_owns_tensor_copies(self) -> None:
        query = self.query.clone()
        demo = self.demo.clone()
        action = self.action.clone()
        context = PreparedRawDemoContext("copy", demo, action)
        request = DemoRelativeActionChunkRequest(query, context)

        query.zero_()
        demo.zero_()
        action.zero_()

        torch.testing.assert_close(request.query_state, self.query)
        torch.testing.assert_close(request.demo.state, self.demo)
        torch.testing.assert_close(request.demo.raw_action, self.action)


if __name__ == "__main__":
    unittest.main()

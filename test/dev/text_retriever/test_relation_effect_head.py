"""Relation-effect head 结构与 conformal 工具测试。"""

from __future__ import annotations

import unittest

import torch

from dev.text_retriever.relation_effect_head import (
    RelationEffectHead,
    RelationEffectHeadConfig,
    TokenRelationEffectHead,
    TokenRelationEffectHeadConfig,
    conformal_quantile,
    conformal_sets,
    singleton_predictions,
)
from lib.GLiNER2_Base import ParsedInstruction, TextSpan
from dev.text_retriever.relation_effect_head import mask_object_spans


class RelationEffectHeadTest(unittest.TestCase):
    def test_forward_shape_and_parameter_budget(self) -> None:
        model = RelationEffectHead(RelationEffectHeadConfig(hidden_dim=64))
        logits = model(torch.zeros(5, 384), torch.ones(5, 384))

        self.assertEqual(logits.shape, (5, 3))
        self.assertLessEqual(
            sum(parameter.numel() for parameter in model.parameters()),
            50_000,
        )

    def test_object_mask_preserves_relation_words(self) -> None:
        text = "Move the red cube toward the green marker."
        parsed = ParsedInstruction(
            text=text,
            goal_operation="move",
            goal_operation_confidence=1.0,
            operations=(),
            objects=(
                TextSpan("red cube", 1.0, 9, 17),
                TextSpan("green marker", 1.0, 29, 41),
            ),
        )

        masked = mask_object_spans(parsed)

        self.assertEqual(masked, "Move the [OBJECT] toward the [OBJECT] .")

    def test_token_head_masks_padding_and_normalizes_attention(self) -> None:
        model = TokenRelationEffectHead(
            TokenRelationEffectHeadConfig(attention_dim=32)
        )
        tokens = torch.randn(3, 7, 384)
        mask = torch.tensor(
            [
                [1, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1, 1],
            ],
            dtype=torch.bool,
        )

        logits = model(tokens, mask)
        _, weights = model.pool(tokens, mask)

        self.assertEqual(logits.shape, (3, 3))
        self.assertTrue(torch.allclose(weights.sum(dim=1), torch.ones(3)))
        self.assertEqual(float(weights[~mask].abs().max().detach()), 0.0)
        self.assertLess(
            sum(parameter.numel() for parameter in model.parameters()),
            30_000,
        )

    def test_split_conformal_sets_cover_calibration_labels(self) -> None:
        probabilities = torch.tensor(
            [
                [0.90, 0.05, 0.05],
                [0.10, 0.80, 0.10],
                [0.05, 0.05, 0.90],
                [0.70, 0.20, 0.10],
            ]
        )
        labels = torch.tensor([0, 1, 2, 0])
        quantile = conformal_quantile(probabilities, labels, alpha=0.25)
        prediction_sets = conformal_sets(probabilities, quantile=quantile)

        covered = prediction_sets[torch.arange(4), labels]
        self.assertTrue(bool(covered.all()))
        self.assertEqual(
            singleton_predictions(prediction_sets),
            ["approach", "separate", "other", "approach"],
        )


if __name__ == "__main__":
    unittest.main()

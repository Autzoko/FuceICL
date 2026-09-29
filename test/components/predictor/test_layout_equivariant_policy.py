"""Demo-anchored action-chunk Predictor 的结构不变量测试。"""

from __future__ import annotations

import unittest

import torch

from dev.predictor.layout_equivariant_demo_policy import (
    LayoutEquivariantDemoPolicy as LegacyPolicy,
)
from src.components.predictor import (
    LayoutEquivariantDemoPolicy,
    LayoutEquivariantDemoPolicyConfig,
)


class LayoutEquivariantDemoPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)
        self.model = LayoutEquivariantDemoPolicy()
        self.query = torch.randn(3, 21)
        self.demo = torch.randn(3, 21)
        self.actions = torch.randn(3, 6, 7) * 0.01
        self.mask = torch.ones(3)

    def test_capacity_identity_no_demo_and_gripper_prior(self) -> None:
        self.assertEqual(self.model.parameter_count, 11_940)
        identity = self.model(
            self.demo,
            self.demo,
            self.actions,
            self.mask,
        )
        no_demo = self.model(
            self.query,
            self.demo,
            self.actions,
            torch.zeros_like(self.mask),
        )
        prediction = self.model(
            self.query,
            self.demo,
            self.actions,
            self.mask,
        )

        self.assertTrue(torch.equal(identity, self.actions))
        self.assertTrue(torch.equal(no_demo, torch.zeros_like(no_demo)))
        self.assertTrue(torch.equal(prediction[..., 6], self.actions[..., 6]))

    def test_residual_is_physically_bounded(self) -> None:
        prediction = self.model(
            self.query,
            self.demo,
            self.actions,
            self.mask,
        )
        residual = prediction[..., :6] - self.actions[..., :6]
        self.assertLessEqual(
            float(residual[..., :3].detach().abs().max()),
            self.model.config.maximum_translation_residual_m,
        )
        self.assertLessEqual(
            float(residual[..., 3:6].detach().abs().max()),
            self.model.config.maximum_rotation_residual_rad,
        )

    def test_legacy_import_and_state_dict_are_compatible(self) -> None:
        self.assertIs(LegacyPolicy, LayoutEquivariantDemoPolicy)
        restored = LegacyPolicy()
        restored.load_state_dict(self.model.state_dict(), strict=True)
        expected = self.model(
            self.query,
            self.demo,
            self.actions,
            self.mask,
        )
        actual = restored(
            self.query,
            self.demo,
            self.actions,
            self.mask,
        )
        self.assertTrue(torch.equal(actual, expected))

    def test_invalid_inputs_and_normalizer_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "严格为正"):
            LayoutEquivariantDemoPolicy(state_std=torch.zeros(21))
        with self.assertRaisesRegex(TypeError, "浮点"):
            self.model(
                self.query.to(torch.int64),
                self.demo,
                self.actions,
                self.mask,
            )
        with self.assertRaisesRegex(ValueError, r"\[0,1\]"):
            self.model(
                self.query,
                self.demo,
                self.actions,
                torch.full_like(self.mask, 2.0),
            )

    def test_config_rejects_non_finite_and_boolean_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "正整数"):
            LayoutEquivariantDemoPolicyConfig(hidden_dim=True)
        with self.assertRaisesRegex(ValueError, "有限正数"):
            LayoutEquivariantDemoPolicyConfig(
                maximum_translation_residual_m=float("nan")
            )


if __name__ == "__main__":
    unittest.main()

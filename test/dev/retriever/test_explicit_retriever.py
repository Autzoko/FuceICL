"""显式语义—几何 Retriever 强基线测试。"""

from __future__ import annotations

from typing import Sequence
import unittest

import torch

from dev.retriever import (
    ExplicitFeatureRetriever,
    ExplicitRetrieverConfig,
    GeometricContext,
    RetrieverCandidate,
    RetrieverQuery,
)
from lib.GLiNER2_Base import ParsedInstruction, TextSpan
from src.components.retriever import TextRetriever


class _FakeParser:
    _OBJECTS = {
        "query": "bowl",
        "bad text": "bowl",
        "good text": "bowl",
        "unrelated": "drawer",
    }

    def parse_many(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 8,
    ) -> list[ParsedInstruction]:
        del batch_size
        return [self.parse(text) for text in texts]

    def parse(self, text: str) -> ParsedInstruction:
        object_name = self._OBJECTS[text]
        return ParsedInstruction(
            text=text,
            goal_operation="place",
            goal_operation_confidence=1.0,
            operations=(),
            objects=(TextSpan(object_name, 1.0, None, None),),
        )


class _FakeEncoder:
    @staticmethod
    def _axis(index: int, sign: float = 1.0) -> torch.Tensor:
        vector = torch.zeros(384, dtype=torch.float32)
        vector[index] = sign
        return vector

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 64,
    ) -> torch.Tensor:
        del batch_size
        vectors = {
            "query": self._axis(0),
            "bad text": self._axis(0),
            "good text": self._axis(0),
            "unrelated": self._axis(1),
            "bowl": self._axis(2),
            "drawer": self._axis(2, -1.0),
            "cup": self._axis(3),
            "mug": self._axis(3),
        }
        return torch.stack([vectors[text] for text in texts])


IDENTITY_ROTATION_6D = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
ROTATION_Z_PI_6D = torch.tensor([-1.0, 0.0, 0.0, 0.0, -1.0, 0.0])


def _context(
    *,
    active_x: float,
    target_offset: float,
    eef_x: float,
    gripper_width: float = 0.04,
    active_rotation: torch.Tensor = IDENTITY_ROTATION_6D,
    target_rotation: torch.Tensor = IDENTITY_ROTATION_6D,
) -> GeometricContext:
    active_center = torch.tensor([active_x, 0.0, 0.0])
    target_center = active_center + torch.tensor([target_offset, 0.0, 0.0])
    cloud = torch.tensor(
        [
            [-0.02, -0.02, 0.0],
            [0.02, -0.02, 0.0],
            [0.02, 0.02, 0.0],
            [-0.02, 0.02, 0.0],
        ],
        dtype=torch.float32,
    )
    return GeometricContext(
        active_points=cloud,
        active_center=active_center,
        active_extent=torch.tensor([0.04, 0.04, 0.08]),
        target_points=cloud * 2.0,
        target_center=target_center,
        target_extent=torch.tensor([0.08, 0.08, 0.04]),
        eef_relative_active=torch.tensor(
            [eef_x, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ),
        eef_velocity=torch.zeros(6),
        gripper_width=gripper_width,
        active_rotation_6d=active_rotation,
        target_rotation_6d=target_rotation,
    )


class ExplicitFeatureRetrieverTest(unittest.TestCase):
    def setUp(self) -> None:
        encoder = _FakeEncoder()
        text_retriever = TextRetriever(_FakeParser(), encoder)
        self.retriever = ExplicitFeatureRetriever(
            text_retriever,
            semantic_encoder=encoder,
            config=ExplicitRetrieverConfig(text_top_k=3, default_top_k=3),
        )

    def test_explicit_features_rerank_and_preserve_payload(self) -> None:
        payload = {"actions": [1, 2, 3]}
        candidates = [
            RetrieverCandidate(
                "bad",
                "bad text",
                _context(
                    active_x=0.5,
                    target_offset=-0.2,
                    eef_x=0.4,
                    gripper_width=0.08,
                    active_rotation=ROTATION_Z_PI_6D,
                ),
                active_object_semantic="mug",
                target_object_semantic="bowl",
            ),
            RetrieverCandidate(
                "good",
                "good text",
                _context(active_x=0.0, target_offset=0.2, eef_x=0.02),
                payload=payload,
                active_object_semantic="mug",
                target_object_semantic="bowl",
            ),
            RetrieverCandidate(
                "unrelated",
                "unrelated",
                _context(active_x=0.0, target_offset=0.2, eef_x=0.02),
                active_object_semantic="mug",
                target_object_semantic="bowl",
            ),
        ]
        self.retriever.build_index(candidates)

        result = self.retriever.retrieve(
            RetrieverQuery(
                "query",
                _context(active_x=0.0, target_offset=0.2, eef_x=0.02),
                active_object_semantic="cup",
                target_object_semantic="bowl",
            )
        )

        self.assertEqual(
            [candidate.candidate.candidate_id for candidate in result.hits],
            ["good", "bad", "unrelated"],
        )
        hit = result.hits[0]
        self.assertEqual(hit.candidate.candidate_id, "good")
        self.assertIs(hit.candidate.payload, payload)
        self.assertGreater(hit.semantic_score, 0.99)
        self.assertGreater(hit.compatibility_score, 0.99)
        self.assertGreater(hit.object_pose_score, 0.99)
        self.assertGreater(hit.layout_score, 0.99)
        self.assertGreater(hit.eef_pose_score, 0.99)
        self.assertAlmostEqual(
            result.text_result.hits[0].score,
            result.text_result.hits[1].score,
        )

    def test_low_orientation_confidence_moves_score_toward_neutral(self) -> None:
        query = _context(active_x=0.0, target_offset=0.2, eef_x=0.02)
        certain_bad = _context(
            active_x=0.0,
            target_offset=0.2,
            eef_x=0.02,
            active_rotation=ROTATION_Z_PI_6D,
        )
        uncertain_bad = GeometricContext(
            **{
                **certain_bad.__dict__,
                "active_orientation_confidence": 0.0,
            }
        )

        certain_score = self.retriever._object_pose_score(query, certain_bad)
        uncertain_score = self.retriever._object_pose_score(query, uncertain_bad)

        self.assertGreater(uncertain_score, certain_score)
        self.assertLess(uncertain_score, 1.0)

    def test_retrieve_requires_index(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "build_index"):
            self.retriever.retrieve(
                RetrieverQuery(
                    "query",
                    _context(active_x=0.0, target_offset=0.2, eef_x=0.02),
                )
            )


if __name__ == "__main__":
    unittest.main()

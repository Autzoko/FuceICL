"""第一版多阶段 Retriever 的隔离排序测试。"""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as functional

from dev.retriever import (
    ActionSemantics,
    ContextEncoderConfig,
    GeometricContext,
    GeometricContextEncoder,
    MultistageRetriever,
    MultistageRetrieverConfig,
    RetrieverCandidate,
    RetrieverQuery,
)
from lib.GLiNER2_Base import ParsedInstruction, TextSpan
from src.components.retriever import TextRetriever


class _FakeParser:
    _OBJECTS = {
        "query": "bowl",
        "wrong context": "bowl",
        "good context": "bowl",
        "unrelated": "drawer",
    }

    def parse_many(
        self,
        texts: list[str],
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


class _FakeTextEncoder:
    @staticmethod
    def _vector(first: float, second: float) -> torch.Tensor:
        vector = torch.zeros(384, dtype=torch.float32)
        vector[0] = first
        vector[1] = second
        return vector

    def encode(self, texts: list[str], *, batch_size: int = 64) -> torch.Tensor:
        del batch_size
        vectors = {
            "query": self._vector(1.0, 0.0),
            "wrong context": self._vector(1.0, 0.0),
            "good context": self._vector(1.0, 0.0),
            "unrelated": self._vector(0.0, 1.0),
            "bowl": self._vector(0.0, 1.0),
            "drawer": self._vector(0.0, -1.0),
        }
        return torch.stack([vectors[text] for text in texts])


class _FakeContextEncoder:
    """使用 active object x 位置构造可控的单位向量。"""

    def encode(
        self,
        contexts: list[GeometricContext],
        *,
        batch_size: int = 32,
    ) -> torch.Tensor:
        del batch_size
        values = torch.stack(
            [
                torch.tensor(
                    [1.0, float(context.active_center[0])],
                    dtype=torch.float32,
                )
                for context in contexts
            ]
        )
        return functional.normalize(values, dim=1)


def _context(
    *,
    active_x: float,
    target_offset: float,
    eef_x: float,
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
        gripper_width=0.04,
    )


class MultistageRetrieverTest(unittest.TestCase):
    def setUp(self) -> None:
        text_retriever = TextRetriever(_FakeParser(), _FakeTextEncoder())
        self.retriever = MultistageRetriever(
            text_retriever,
            _FakeContextEncoder(),
            config=MultistageRetrieverConfig(
                text_top_k=2,
                default_top_k=1,
                text_weight=0.10,
                geometry_weight=0.35,
                layout_weight=0.25,
                state_weight=0.20,
                action_weight=0.10,
            ),
        )

    def test_context_reranks_text_candidates_and_preserves_payload(self) -> None:
        payload = {"actions": [1, 2, 3]}
        candidates = [
            RetrieverCandidate(
                "wrong",
                "wrong context",
                _context(active_x=1.0, target_offset=-0.5, eef_x=0.8),
                ActionSemantics(operation="place", phase="pre_grasp"),
            ),
            RetrieverCandidate(
                "good",
                "good context",
                _context(active_x=0.0, target_offset=0.2, eef_x=0.02),
                ActionSemantics(operation="place", phase="pre_grasp"),
                payload,
            ),
            RetrieverCandidate(
                "unrelated",
                "unrelated",
                _context(active_x=0.0, target_offset=0.2, eef_x=0.02),
            ),
        ]
        self.retriever.build_index(candidates)

        result = self.retriever.retrieve(
            RetrieverQuery(
                "query",
                _context(active_x=0.0, target_offset=0.2, eef_x=0.02),
                ActionSemantics(operation="place", phase="pre_grasp"),
            )
        )

        self.assertEqual(result.hits[0].candidate.candidate_id, "good")
        self.assertIs(result.hits[0].candidate.payload, payload)
        self.assertNotIn(
            "unrelated",
            [hit.candidate.candidate_id for hit in result.hits],
        )
        self.assertGreater(result.hits[0].layout_score, 0.99)
        self.assertGreater(result.hits[0].state_score, 0.99)

    def test_retrieve_requires_index(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "build_index"):
            self.retriever.retrieve(
                RetrieverQuery(
                    "query",
                    _context(active_x=0.0, target_offset=0.2, eef_x=0.02),
                )
            )

    def test_real_context_encoder_handles_optional_target(self) -> None:
        encoder = GeometricContextEncoder(
            ContextEncoderConfig(
                point_embedding_dim=16,
                numeric_embedding_dim=8,
                output_dim=24,
            )
        )
        with_target = _context(active_x=0.0, target_offset=0.2, eef_x=0.02)
        without_target = GeometricContext(
            active_points=with_target.active_points[:3],
            active_center=with_target.active_center,
            active_extent=with_target.active_extent,
            eef_relative_active=with_target.eef_relative_active,
            eef_velocity=with_target.eef_velocity,
            gripper_width=with_target.gripper_width,
        )

        embeddings = encoder.encode([with_target, without_target], batch_size=1)

        self.assertEqual(tuple(embeddings.shape), (2, 24))
        self.assertTrue(torch.isfinite(embeddings).all())
        self.assertTrue(
            torch.allclose(
                torch.linalg.vector_norm(embeddings, dim=1),
                torch.ones(2),
                atol=1e-5,
            )
        )


if __name__ == "__main__":
    unittest.main()

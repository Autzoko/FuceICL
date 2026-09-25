"""文本初筛器的隔离单元测试。"""

from __future__ import annotations

import math
import unittest

import torch

from lib.GLiNER2_Base import ParsedInstruction, TextSpan
from src.components.retriever import TextCandidate, TextRetriever


class _FakeParser:
    """按测试词表返回结构化结果，避免单元测试加载大模型。"""

    _OBJECTS = {
        "query": ("bowl",),
        "raw winner": ("drawer",),
        "object winner": ("bowl",),
        "plain query": (),
        "plain candidate": (),
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
        objects = tuple(
            TextSpan(value, 1.0, None, None)
            for value in self._OBJECTS.get(text, ())
        )
        return ParsedInstruction(text, "place", 1.0, (), objects)


class _FakeEncoder:
    """返回可控的 384 维单位向量。"""

    def __init__(self) -> None:
        self._vectors = {
            "query": self._vector(1.0, 0.0),
            "raw winner": self._vector(1.0, 0.0),
            "object winner": self._vector(0.9, math.sqrt(0.19)),
            "bowl": self._vector(0.0, 1.0),
            "drawer": self._vector(0.0, -1.0),
            "plain query": self._vector(1.0, 0.0),
            "plain candidate": self._vector(0.8, 0.6),
        }

    @staticmethod
    def _vector(first: float, second: float) -> torch.Tensor:
        value = torch.zeros(384, dtype=torch.float32)
        value[0] = first
        value[1] = second
        return value

    def encode(self, texts: list[str], *, batch_size: int = 64) -> torch.Tensor:
        del batch_size
        return torch.stack([self._vectors[text] for text in texts])


class TextRetrieverTest(unittest.TestCase):
    def setUp(self) -> None:
        self.retriever = TextRetriever(_FakeParser(), _FakeEncoder())

    def test_object_similarity_can_improve_ranking(self) -> None:
        self.retriever.build_index(
            [
                TextCandidate("raw", "raw winner"),
                TextCandidate("object", "object winner"),
            ]
        )

        result = self.retriever.retrieve("query", top_k=2)

        self.assertEqual(
            [hit.candidate.candidate_id for hit in result.hits],
            ["object", "raw"],
        )
        self.assertAlmostEqual(result.hits[0].score, 0.92, places=5)
        self.assertAlmostEqual(result.hits[0].object_score, 1.0, places=5)

    def test_missing_objects_fall_back_to_raw_text_score(self) -> None:
        self.retriever.build_index(
            [TextCandidate("plain", "plain candidate")]
        )

        hit = self.retriever.retrieve("plain query").hits[0]

        self.assertAlmostEqual(hit.raw_text_score, 0.8, places=5)
        self.assertEqual(hit.object_score, 0.0)
        self.assertAlmostEqual(hit.score, 0.64, places=5)

    def test_duplicate_candidate_id_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "candidate_id 必须唯一"):
            self.retriever.build_index(
                [
                    TextCandidate("same", "raw winner"),
                    TextCandidate("same", "object winner"),
                ]
            )

    def test_retrieve_requires_index_and_positive_top_k(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "build_index"):
            self.retriever.retrieve("query")

        self.retriever.build_index([TextCandidate("one", "raw winner")])
        with self.assertRaisesRegex(ValueError, "top_k"):
            self.retriever.retrieve("query", top_k=0)


if __name__ == "__main__":
    unittest.main()

"""GLiNER + MiniLM 文本初筛器。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

import torch
from torch.nn.utils.rnn import pad_sequence

from lib.GLiNER2_Base import (
    DEFAULT_MODEL_DIR as DEFAULT_GLINER_MODEL_DIR,
    GLiNERTextParser,
    ParsedInstruction,
)
from lib.all_MiniLM_L6_v2 import (
    DEFAULT_MODEL_DIR as DEFAULT_MINILM_MODEL_DIR,
    EMBEDDING_DIM,
    MiniLMTextEncoder,
)


@dataclass(frozen=True)
class TextCandidate:
    """待建立索引的一条任务文本。"""

    candidate_id: str
    text: str


@dataclass(frozen=True)
class TextRetrieverConfig:
    """文本初筛配置；默认权重来自冻结的 DROID 500-episode 实验。"""

    raw_text_weight: float = 0.8
    object_weight: float = 0.2
    default_top_k: int = 10
    parse_batch_size: int = 8
    encode_batch_size: int = 64
    score_batch_size: int = 1024

    def __post_init__(self) -> None:
        weights = (self.raw_text_weight, self.object_weight)
        if any(not math.isfinite(weight) or weight < 0 for weight in weights):
            raise ValueError("检索权重不能为负数")
        if abs(self.raw_text_weight + self.object_weight - 1.0) > 1e-6:
            raise ValueError("raw_text_weight 与 object_weight 之和必须为 1")
        sizes = (
            self.default_top_k,
            self.parse_batch_size,
            self.encode_batch_size,
            self.score_batch_size,
        )
        if any(isinstance(size, bool) or not isinstance(size, int) for size in sizes):
            raise TypeError("batch size 和 top_k 必须是整数")
        if min(sizes) <= 0:
            raise ValueError("batch size 和 top_k 必须大于 0")


@dataclass(frozen=True)
class TextRetrievalHit:
    """单个候选的排序结果及可审计分数。"""

    candidate: TextCandidate
    parsed: ParsedInstruction
    score: float
    raw_text_score: float
    object_score: float


@dataclass(frozen=True)
class TextRetrievalResult:
    """一次查询的解析结果和 Top-K 候选。"""

    query: ParsedInstruction
    hits: tuple[TextRetrievalHit, ...]


@dataclass(frozen=True)
class _IndexedCandidate:
    """内部索引记录，不作为公共 API 导出。"""

    candidate: TextCandidate
    parsed: ParsedInstruction
    object_embeddings: torch.Tensor


class TextRetriever:
    """使用原始文本语义和无角色物体集合完成任务文本初筛。

    当前冻结的排序不使用 operation 分数。GLiNER 仍会解析并保存 operation，便于
    后续增加独立 operation bucket，而无需重新定义索引数据格式。
    """

    def __init__(
        self,
        parser: GLiNERTextParser,
        encoder: MiniLMTextEncoder,
        *,
        config: TextRetrieverConfig | None = None,
    ) -> None:
        self.parser = parser
        self.encoder = encoder
        self.config = config or TextRetrieverConfig()
        self._records: tuple[_IndexedCandidate, ...] = ()
        self._raw_embeddings = torch.empty(
            (0, EMBEDDING_DIM),
            dtype=torch.float32,
        )

    @classmethod
    def from_local_models(
        cls,
        *,
        gliner_model_dir: str | Path = DEFAULT_GLINER_MODEL_DIR,
        minilm_model_dir: str | Path = DEFAULT_MINILM_MODEL_DIR,
        device: str = "cpu",
        config: TextRetrieverConfig | None = None,
    ) -> "TextRetriever":
        """从本仓库的本地 checkpoint 组装完整文本初筛器。"""
        return cls(
            GLiNERTextParser(gliner_model_dir, device=device),
            MiniLMTextEncoder(minilm_model_dir, device=device),
            config=config,
        )

    @property
    def is_indexed(self) -> bool:
        """是否已经建立至少一条候选索引。"""
        return bool(self._records)

    def __len__(self) -> int:
        return len(self._records)

    @staticmethod
    def _validate_candidates(
        candidates: Sequence[TextCandidate],
    ) -> list[TextCandidate]:
        if isinstance(candidates, (str, bytes)):
            raise TypeError("candidates 必须是 TextCandidate 序列")
        values = list(candidates)
        if not values:
            raise ValueError("至少需要一条候选文本")
        if any(not isinstance(item, TextCandidate) for item in values):
            raise TypeError("candidates 中的元素必须是 TextCandidate")
        if any(
            not isinstance(item.candidate_id, str) or not item.candidate_id.strip()
            for item in values
        ):
            raise ValueError("candidate_id 不能为空")
        if any(
            not isinstance(item.text, str) or not item.text.strip()
            for item in values
        ):
            raise ValueError("候选文本不能为空")
        identifiers = [item.candidate_id for item in values]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("candidate_id 必须唯一")
        return values

    def _encode_object_sets(
        self,
        parsed: Sequence[ParsedInstruction],
    ) -> list[torch.Tensor]:
        # 对相同 object span 只编码一次，避免大规模 episode 索引中的重复计算。
        object_sets = [
            tuple(dict.fromkeys(span.text for span in item.objects))
            for item in parsed
        ]
        vocabulary = sorted({text for values in object_sets for text in values})
        if not vocabulary:
            return [
                torch.empty((0, EMBEDDING_DIM), dtype=torch.float32)
                for _ in parsed
            ]

        vectors = self.encoder.encode(
            vocabulary,
            batch_size=self.config.encode_batch_size,
        )
        lookup = dict(zip(vocabulary, vectors, strict=True))
        return [
            torch.stack([lookup[text] for text in values])
            if values
            else torch.empty((0, EMBEDDING_DIM), dtype=torch.float32)
            for values in object_sets
        ]

    def build_index(self, candidates: Sequence[TextCandidate]) -> None:
        """解析并编码候选文本；重复调用会原子替换当前内存索引。"""
        values = self._validate_candidates(candidates)
        parsed = self.parser.parse_many(
            [item.text for item in values],
            batch_size=self.config.parse_batch_size,
        )
        raw_embeddings = self.encoder.encode(
            [item.text for item in values],
            batch_size=self.config.encode_batch_size,
        )
        object_embeddings = self._encode_object_sets(parsed)

        # 全部中间结果成功生成后再替换，避免异常留下半成品索引。
        records = tuple(
            _IndexedCandidate(item, structure, object_vectors)
            for item, structure, object_vectors in zip(
                values,
                parsed,
                object_embeddings,
                strict=True,
            )
        )
        self._records = records
        self._raw_embeddings = raw_embeddings

    def _score_object_sets(self, query_objects: torch.Tensor) -> torch.Tensor:
        scores = torch.zeros(len(self._records), dtype=torch.float32)
        if not len(query_objects):
            return scores

        batch_size = self.config.score_batch_size
        for start in range(0, len(self._records), batch_size):
            batch = self._records[start : start + batch_size]
            valid = [
                (offset, record.object_embeddings)
                for offset, record in enumerate(batch)
                if len(record.object_embeddings)
            ]
            if not valid:
                continue

            offsets, object_sets = zip(*valid, strict=True)
            lengths = torch.tensor([len(values) for values in object_sets])
            padded = pad_sequence(object_sets, batch_first=True)
            mask = torch.arange(padded.shape[1])[None, :] < lengths[:, None]

            pairwise = torch.einsum("qd,bmd->bqm", query_objects, padded)
            pairwise = pairwise.masked_fill(~mask[:, None, :], float("-inf"))
            query_to_candidate = pairwise.max(dim=2).values.mean(dim=1)
            candidate_to_query = pairwise.max(dim=1).values
            candidate_to_query = (
                candidate_to_query.masked_fill(~mask, 0.0).sum(dim=1)
                / lengths.to(torch.float32)
            )
            batch_scores = 0.5 * (query_to_candidate + candidate_to_query)
            for offset, value in zip(offsets, batch_scores, strict=True):
                scores[start + offset] = value
        return scores

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
    ) -> TextRetrievalResult:
        """返回按冻结分数排序的 Top-K 候选。"""
        if not self.is_indexed:
            raise RuntimeError("请先调用 build_index() 建立候选索引")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query 必须是非空字符串")

        requested_top_k = self.config.default_top_k if top_k is None else top_k
        if isinstance(requested_top_k, bool) or not isinstance(requested_top_k, int):
            raise TypeError("top_k 必须是整数")
        if requested_top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        parsed_query = self.parser.parse(query)
        query_embedding = self.encoder.encode(
            [query],
            batch_size=1,
        )[0]
        raw_scores = self._raw_embeddings @ query_embedding
        query_objects = self._encode_object_sets([parsed_query])[0]
        object_scores = self._score_object_sets(query_objects)
        scores = (
            self.config.raw_text_weight * raw_scores
            + self.config.object_weight * object_scores
        )

        result_size = min(requested_top_k, len(self._records))
        ranking = torch.argsort(scores, descending=True, stable=True)[:result_size]
        hits = tuple(
            TextRetrievalHit(
                candidate=self._records[index].candidate,
                parsed=self._records[index].parsed,
                score=float(scores[index]),
                raw_text_score=float(raw_scores[index]),
                object_score=float(object_scores[index]),
            )
            for index in ranking.tolist()
        )
        return TextRetrievalResult(query=parsed_query, hits=hits)

"""任务分桶后的轻量 embedding 精确检索。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import torch

from .task_router import TaskKey


def _identifier(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} 不能为空")
    return value.strip()


def _normalized_embedding(value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError("embedding 必须是 torch.Tensor")
    if value.ndim != 1 or value.numel() == 0:
        raise ValueError("embedding shape 必须为 [D] 且 D > 0")
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError("embedding 必须是有限浮点 Tensor")
    result = value.detach().float().cpu()
    norm = torch.linalg.vector_norm(result)
    if float(norm) <= 1e-12:
        raise ValueError("embedding 不能是零向量")
    return result / norm


@dataclass(frozen=True)
class EmbeddingRetrieverConfig:
    """精确检索配置；阈值为空时不隐式拒绝。"""

    default_top_k: int = 4
    max_per_episode: int | None = 1
    minimum_score: float | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.default_top_k, bool)
            or not isinstance(self.default_top_k, int)
            or self.default_top_k <= 0
        ):
            raise ValueError("default_top_k 必须是正整数")
        if self.max_per_episode is not None and (
            isinstance(self.max_per_episode, bool)
            or not isinstance(self.max_per_episode, int)
            or self.max_per_episode <= 0
        ):
            raise ValueError("max_per_episode 必须是正整数或 None")
        if self.minimum_score is not None and (
            isinstance(self.minimum_score, bool)
            or not isinstance(self.minimum_score, (int, float))
            or not math.isfinite(self.minimum_score)
            or not -1.0 <= self.minimum_score <= 1.0
        ):
            raise ValueError("minimum_score 必须位于 [-1, 1] 或为 None")


@dataclass(frozen=True)
class EmbeddingCandidate:
    """候选；embedding 由冻结的几何/PointNet encoder 离线生成。"""

    candidate_id: str
    task_key: TaskKey
    embedding: torch.Tensor
    episode_id: str | None = None
    payload: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_id",
            _identifier(self.candidate_id, field="candidate_id"),
        )
        if not isinstance(self.task_key, TaskKey):
            raise TypeError("task_key 必须是 TaskKey")
        if self.episode_id is not None:
            object.__setattr__(
                self,
                "episode_id",
                _identifier(self.episode_id, field="episode_id"),
            )
        _normalized_embedding(self.embedding)


@dataclass(frozen=True)
class EmbeddingQuery:
    """在线 query；只携带已解析 task key 与当前 context embedding。"""

    task_key: TaskKey
    embedding: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.task_key, TaskKey):
            raise TypeError("task_key 必须是 TaskKey")
        _normalized_embedding(self.embedding)


@dataclass(frozen=True)
class EmbeddingRetrievalHit:
    """候选及其余弦相似度。"""

    candidate: EmbeddingCandidate
    score: float


@dataclass(frozen=True)
class EmbeddingRetrievalResult:
    """检索结果；无安全候选时显式给出拒绝原因。"""

    query: EmbeddingQuery
    hits: tuple[EmbeddingRetrievalHit, ...]
    accepted: bool
    reason: str | None


@dataclass(frozen=True)
class _Bucket:
    candidates: tuple[EmbeddingCandidate, ...]
    embeddings: torch.Tensor


class ExactEmbeddingRetriever:
    """在完整 task key bucket 内执行 CPU 精确内积检索。

    本类不加载文本或点云模型，也不修改 Demo payload。候选 embedding 应在
    离线构库时计算，query embedding 由同一冻结 encoder 在线生成。
    """

    def __init__(
        self,
        config: EmbeddingRetrieverConfig | None = None,
    ) -> None:
        self.config = config or EmbeddingRetrieverConfig()
        self._buckets: dict[TaskKey, _Bucket] = {}
        self._size = 0
        self._embedding_dim: int | None = None

    @property
    def is_indexed(self) -> bool:
        return self._size > 0

    @property
    def embedding_dim(self) -> int | None:
        return self._embedding_dim

    def __len__(self) -> int:
        return self._size

    def build_index(
        self,
        candidates: Sequence[EmbeddingCandidate],
    ) -> None:
        """完整校验后原子替换旧索引。"""
        if isinstance(candidates, (str, bytes)):
            raise TypeError("candidates 必须是 EmbeddingCandidate 序列")
        values = list(candidates)
        if not values:
            raise ValueError("至少需要一个 EmbeddingCandidate")
        if any(not isinstance(value, EmbeddingCandidate) for value in values):
            raise TypeError("candidates 元素必须是 EmbeddingCandidate")
        identifiers = [value.candidate_id for value in values]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("candidate_id 必须全局唯一")

        normalized = [_normalized_embedding(value.embedding) for value in values]
        dimensions = {int(value.numel()) for value in normalized}
        if len(dimensions) != 1:
            raise ValueError("所有 embedding 维度必须一致")
        grouped: dict[TaskKey, list[tuple[EmbeddingCandidate, torch.Tensor]]] = {}
        for candidate, embedding in zip(values, normalized, strict=True):
            grouped.setdefault(candidate.task_key, []).append(
                (candidate, embedding)
            )
        buckets = {
            key: _Bucket(
                candidates=tuple(value[0] for value in rows),
                embeddings=torch.stack([value[1] for value in rows]),
            )
            for key, rows in grouped.items()
        }
        self._buckets = buckets
        self._size = len(values)
        self._embedding_dim = dimensions.pop()

    def retrieve(
        self,
        query: EmbeddingQuery,
        *,
        top_k: int | None = None,
    ) -> EmbeddingRetrievalResult:
        """返回稳定排序的 top-K；未知 task key 与低分均显式拒绝。"""
        if not self.is_indexed:
            raise RuntimeError("请先调用 build_index() 建立索引")
        if not isinstance(query, EmbeddingQuery):
            raise TypeError("query 必须是 EmbeddingQuery")
        requested = self.config.default_top_k if top_k is None else top_k
        if (
            isinstance(requested, bool)
            or not isinstance(requested, int)
            or requested <= 0
        ):
            raise ValueError("top_k 必须是正整数")
        bucket = self._buckets.get(query.task_key)
        if bucket is None:
            return EmbeddingRetrievalResult(
                query=query,
                hits=(),
                accepted=False,
                reason="no_exact_task_key_match",
            )
        embedding = _normalized_embedding(query.embedding)
        if embedding.numel() != self._embedding_dim:
            raise ValueError("query embedding 维度与索引不一致")
        scores = torch.mv(bucket.embeddings, embedding)
        order = sorted(
            range(len(bucket.candidates)),
            key=lambda index: (-float(scores[index]), index),
        )
        hits = []
        episode_counts: dict[str, int] = {}
        for index in order:
            score = max(-1.0, min(1.0, float(scores[index])))
            if (
                self.config.minimum_score is not None
                and score < self.config.minimum_score
            ):
                continue
            candidate = bucket.candidates[index]
            episode = candidate.episode_id or candidate.candidate_id
            if self.config.max_per_episode is not None:
                count = episode_counts.get(episode, 0)
                if count >= self.config.max_per_episode:
                    continue
                episode_counts[episode] = count + 1
            hits.append(EmbeddingRetrievalHit(candidate, score))
            if len(hits) == requested:
                break
        return EmbeddingRetrievalResult(
            query=query,
            hits=tuple(hits),
            accepted=bool(hits),
            reason=None if hits else "below_score_threshold",
        )

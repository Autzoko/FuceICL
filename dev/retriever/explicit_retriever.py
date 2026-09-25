"""无需训练的显式语义—几何强基线 Retriever。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol, Sequence

import torch
import torch.nn.functional as functional

from src.components.retriever.text_retriever import (
    TextCandidate,
    TextRetrievalResult,
    TextRetriever,
)

from .scoring import (
    extent_log_distance,
    mean_score,
    rbf_score,
    rotation_6d_to_matrix,
    rotation_distance,
    rotation_matrix_distance,
    tensor_distance,
)
from .types import GeometricContext, RetrieverCandidate, RetrieverQuery


class SemanticEncoder(Protocol):
    """MiniLM 类文本编码器的最小接口。"""

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
    ) -> torch.Tensor: ...


@dataclass(frozen=True)
class ExplicitRetrieverConfig:
    """显式基线配置；最终分项权重之和必须为 1。"""

    text_top_k: int = 50
    default_top_k: int = 4
    encode_batch_size: int = 64
    text_weight: float = 0.15
    semantic_weight: float = 0.25
    object_pose_weight: float = 0.15
    layout_weight: float = 0.20
    eef_pose_weight: float = 0.20
    gripper_weight: float = 0.05
    object_position_sigma_m: float = 0.20
    layout_position_sigma_m: float = 0.15
    extent_log_sigma: float = 0.50
    object_rotation_sigma_rad: float = 0.75
    eef_position_sigma_m: float = 0.10
    eef_rotation_sigma_rad: float = 0.75
    gripper_sigma_m: float = 0.03

    def __post_init__(self) -> None:
        sizes = (self.text_top_k, self.default_top_k, self.encode_batch_size)
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in sizes
        ):
            raise TypeError("top_k 和 batch_size 必须是整数")
        if min(sizes) <= 0:
            raise ValueError("top_k 和 batch_size 必须大于 0")

        if any(
            not math.isfinite(value) or value < 0
            for value in self.score_weights
        ):
            raise ValueError("打分权重必须是有限非负数")
        if abs(sum(self.score_weights) - 1.0) > 1e-6:
            raise ValueError("最终打分权重之和必须为 1")

        scales = (
            self.object_position_sigma_m,
            self.layout_position_sigma_m,
            self.extent_log_sigma,
            self.object_rotation_sigma_rad,
            self.eef_position_sigma_m,
            self.eef_rotation_sigma_rad,
            self.gripper_sigma_m,
        )
        if any(not math.isfinite(value) or value <= 0 for value in scales):
            raise ValueError("相似度尺度必须是有限正数")

    @property
    def score_weights(self) -> tuple[float, ...]:
        return (
            self.text_weight,
            self.semantic_weight,
            self.object_pose_weight,
            self.layout_weight,
            self.eef_pose_weight,
            self.gripper_weight,
        )


@dataclass(frozen=True)
class ExplicitRetrievalHit:
    """显式基线的候选与可审计分项分数。"""

    candidate: RetrieverCandidate
    score: float
    text_score: float
    semantic_score: float
    compatibility_score: float
    object_pose_score: float
    layout_score: float
    eef_pose_score: float
    gripper_score: float


@dataclass(frozen=True)
class ExplicitRetrievalResult:
    """一次显式基线检索结果。"""

    text_result: TextRetrievalResult
    hits: tuple[ExplicitRetrievalHit, ...]


@dataclass(frozen=True)
class _SemanticPair:
    active: torch.Tensor | None
    target: torch.Tensor | None


def _normalize_label(label: str | None) -> str | None:
    if label is None:
        return None
    if not isinstance(label, str) or not label.strip():
        raise ValueError("object semantic 必须是非空字符串或 None")
    return " ".join(label.casefold().split())


class ExplicitFeatureRetriever:
    """文本 Top-K 后使用显式物体语义、位姿和 EEF 连续性重排。"""

    def __init__(
        self,
        text_retriever: TextRetriever,
        *,
        semantic_encoder: SemanticEncoder | None = None,
        config: ExplicitRetrieverConfig | None = None,
    ) -> None:
        self.text_retriever = text_retriever
        self.semantic_encoder = semantic_encoder or text_retriever.encoder
        self.config = config or ExplicitRetrieverConfig()
        self._candidates: dict[str, RetrieverCandidate] = {}
        self._semantic_pairs: dict[str, _SemanticPair] = {}
        self._semantic_vocabulary: dict[str, torch.Tensor] = {}

    @property
    def is_indexed(self) -> bool:
        return bool(self._candidates)

    def __len__(self) -> int:
        return len(self._candidates)

    @staticmethod
    def _validate_candidates(
        candidates: Sequence[RetrieverCandidate],
    ) -> list[RetrieverCandidate]:
        if isinstance(candidates, (str, bytes)):
            raise TypeError("candidates 必须是 RetrieverCandidate 序列")
        values = list(candidates)
        if not values:
            raise ValueError("至少需要一条候选 Demo")
        if any(not isinstance(item, RetrieverCandidate) for item in values):
            raise TypeError("candidates 中的元素必须是 RetrieverCandidate")
        identifiers = [item.candidate_id for item in values]
        if any(not isinstance(value, str) or not value.strip() for value in identifiers):
            raise ValueError("candidate_id 不能为空")
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("candidate_id 必须唯一")
        if any(not isinstance(item.text, str) or not item.text.strip() for item in values):
            raise ValueError("候选文本不能为空")
        if any(
            item.target_object_semantic is not None and not item.context.has_target
            for item in values
        ):
            raise ValueError("没有 target context 时不能提供 target object semantic")
        return values

    def _encode_vocabulary(
        self,
        candidates: Sequence[RetrieverCandidate],
    ) -> dict[str, torch.Tensor]:
        labels = sorted(
            {
                normalized
                for item in candidates
                for normalized in (
                    _normalize_label(item.active_object_semantic),
                    _normalize_label(item.target_object_semantic),
                )
                if normalized is not None
            }
        )
        if not labels:
            return {}
        embeddings = self.semantic_encoder.encode(
            labels,
            batch_size=self.config.encode_batch_size,
        )
        if embeddings.ndim != 2 or embeddings.shape[0] != len(labels):
            raise ValueError("Semantic Encoder 返回 shape 必须为 [B, D]")
        if embeddings.shape[1] == 0 or not torch.isfinite(embeddings).all():
            raise ValueError("Semantic Encoder 返回了空维度、NaN 或 Inf")
        embeddings = functional.normalize(embeddings.float().cpu(), p=2, dim=-1)
        return dict(zip(labels, embeddings, strict=True))

    def build_index(self, candidates: Sequence[RetrieverCandidate]) -> None:
        """建立文本及角色化物体语义索引。"""
        values = self._validate_candidates(candidates)
        vocabulary = self._encode_vocabulary(values)
        semantic_pairs = {
            item.candidate_id: _SemanticPair(
                active=vocabulary.get(_normalize_label(item.active_object_semantic)),
                target=vocabulary.get(_normalize_label(item.target_object_semantic)),
            )
            for item in values
        }
        text_candidates = [
            TextCandidate(item.candidate_id, item.text) for item in values
        ]

        self.text_retriever.build_index(text_candidates)
        self._candidates = {item.candidate_id: item for item in values}
        self._semantic_pairs = semantic_pairs
        self._semantic_vocabulary = vocabulary

    def _query_semantic(self, label: str | None) -> torch.Tensor | None:
        normalized = _normalize_label(label)
        if normalized is None:
            return None
        cached = self._semantic_vocabulary.get(normalized)
        if cached is not None:
            return cached
        embedding = self.semantic_encoder.encode([normalized], batch_size=1)
        if embedding.ndim != 2 or embedding.shape[0] != 1:
            raise ValueError("Semantic Encoder 的 query 输出 shape 必须为 [1, D]")
        if embedding.shape[1] == 0 or not torch.isfinite(embedding).all():
            raise ValueError("Semantic Encoder 返回了空维度、NaN 或 Inf")
        return functional.normalize(embedding[0].float().cpu(), p=2, dim=0)

    @staticmethod
    def _embedding_score(
        query: torch.Tensor | None,
        candidate: torch.Tensor | None,
    ) -> float:
        if query is None or candidate is None:
            return 0.5
        cosine = torch.dot(query, candidate)
        return float(torch.clamp(0.5 * (cosine + 1.0), 0.0, 1.0))

    def _semantic_scores(
        self,
        query: _SemanticPair,
        candidate: _SemanticPair,
        *,
        compare_target: bool,
    ) -> tuple[float, float]:
        active_score = self._embedding_score(query.active, candidate.active)
        scores = [active_score]
        if compare_target:
            scores.append(
                self._embedding_score(
                    query.target,
                    candidate.target,
                )
            )
        return mean_score(scores), active_score

    def _orientation_score(
        self,
        query_rotation: torch.Tensor | None,
        query_confidence: float,
        candidate_rotation: torch.Tensor | None,
        candidate_confidence: float,
    ) -> float | None:
        if query_rotation is None:
            return None
        if candidate_rotation is None:
            return 0.5
        confidence = min(query_confidence, candidate_confidence)
        measured = rbf_score(
            rotation_distance(query_rotation, candidate_rotation),
            self.config.object_rotation_sigma_rad,
        )
        return confidence * measured + (1.0 - confidence) * 0.5

    def _object_pose_score(
        self,
        query: GeometricContext,
        candidate: GeometricContext,
    ) -> float:
        scores = [
            rbf_score(
                tensor_distance(query.active_center, candidate.active_center),
                self.config.object_position_sigma_m,
            )
        ]
        active_orientation = self._orientation_score(
            query.active_rotation_6d,
            query.active_orientation_confidence,
            candidate.active_rotation_6d,
            candidate.active_orientation_confidence,
        )
        if active_orientation is not None:
            scores.append(active_orientation)

        if query.has_target != candidate.has_target:
            scores.append(0.0)
        elif query.has_target:
            scores.append(
                rbf_score(
                    tensor_distance(query.target_center, candidate.target_center),
                    self.config.object_position_sigma_m,
                )
            )
            target_orientation = self._orientation_score(
                query.target_rotation_6d,
                query.target_orientation_confidence,
                candidate.target_rotation_6d,
                candidate.target_orientation_confidence,
            )
            if target_orientation is not None:
                scores.append(target_orientation)
        return mean_score(scores)

    def _relative_orientation_score(
        self,
        query: GeometricContext,
        candidate: GeometricContext,
    ) -> float | None:
        query_rotations = (query.active_rotation_6d, query.target_rotation_6d)
        if any(rotation is None for rotation in query_rotations):
            return None
        candidate_rotations = (
            candidate.active_rotation_6d,
            candidate.target_rotation_6d,
        )
        if any(rotation is None for rotation in candidate_rotations):
            return 0.5

        query_relative = rotation_6d_to_matrix(query.active_rotation_6d).T
        query_relative = query_relative @ rotation_6d_to_matrix(
            query.target_rotation_6d
        )
        candidate_relative = rotation_6d_to_matrix(candidate.active_rotation_6d).T
        candidate_relative = candidate_relative @ rotation_6d_to_matrix(
            candidate.target_rotation_6d
        )
        confidence = min(
            query.active_orientation_confidence,
            query.target_orientation_confidence,
            candidate.active_orientation_confidence,
            candidate.target_orientation_confidence,
        )
        measured = rbf_score(
            rotation_matrix_distance(query_relative, candidate_relative),
            self.config.object_rotation_sigma_rad,
        )
        return confidence * measured + (1.0 - confidence) * 0.5

    def _layout_score(
        self,
        query: GeometricContext,
        candidate: GeometricContext,
    ) -> float:
        scores = [
            rbf_score(
                extent_log_distance(query.active_extent, candidate.active_extent),
                self.config.extent_log_sigma,
            )
        ]
        if query.has_target != candidate.has_target:
            scores.append(0.0)
        elif query.has_target:
            query_relative = query.target_center - query.active_center
            candidate_relative = candidate.target_center - candidate.active_center
            scores.extend(
                (
                    rbf_score(
                        tensor_distance(query_relative, candidate_relative),
                        self.config.layout_position_sigma_m,
                    ),
                    rbf_score(
                        extent_log_distance(
                            query.target_extent,
                            candidate.target_extent,
                        ),
                        self.config.extent_log_sigma,
                    ),
                )
            )
            relative_orientation = self._relative_orientation_score(query, candidate)
            if relative_orientation is not None:
                scores.append(relative_orientation)
        return mean_score(scores)

    def _eef_pose_score(
        self,
        query: GeometricContext,
        candidate: GeometricContext,
    ) -> float:
        return mean_score(
            (
                rbf_score(
                    tensor_distance(
                        query.eef_relative_active[:3],
                        candidate.eef_relative_active[:3],
                    ),
                    self.config.eef_position_sigma_m,
                ),
                rbf_score(
                    rotation_distance(
                        query.eef_relative_active[3:9],
                        candidate.eef_relative_active[3:9],
                    ),
                    self.config.eef_rotation_sigma_rad,
                ),
            )
        )

    def retrieve(
        self,
        query: RetrieverQuery,
        *,
        top_k: int | None = None,
    ) -> ExplicitRetrievalResult:
        """对文本候选进行显式重排并返回原始 Demo。"""
        if not self.is_indexed:
            raise RuntimeError("请先调用 build_index() 建立候选索引")
        if not isinstance(query, RetrieverQuery):
            raise TypeError("query 必须是 RetrieverQuery")
        if not isinstance(query.text, str) or not query.text.strip():
            raise ValueError("query text 不能为空")
        if query.target_object_semantic is not None and not query.context.has_target:
            raise ValueError("没有 target context 时不能提供 target object semantic")

        requested_top_k = self.config.default_top_k if top_k is None else top_k
        if isinstance(requested_top_k, bool) or not isinstance(requested_top_k, int):
            raise TypeError("top_k 必须是整数")
        if requested_top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        text_result = self.text_retriever.retrieve(
            query.text,
            top_k=min(self.config.text_top_k, len(self)),
        )
        query_semantics = _SemanticPair(
            active=self._query_semantic(query.active_object_semantic),
            target=self._query_semantic(query.target_object_semantic),
        )
        scored: list[tuple[int, ExplicitRetrievalHit]] = []
        for rank, text_hit in enumerate(text_result.hits):
            identifier = text_hit.candidate.candidate_id
            candidate = self._candidates[identifier]
            text_score = max(0.0, min(1.0, 0.5 * (text_hit.score + 1.0)))
            semantic_score, active_semantic_score = self._semantic_scores(
                query_semantics,
                self._semantic_pairs[identifier],
                compare_target=(
                    query.context.has_target
                    or query.target_object_semantic is not None
                ),
            )
            object_pose_score = self._object_pose_score(
                query.context,
                candidate.context,
            )
            layout_score = self._layout_score(query.context, candidate.context)
            eef_pose_score = self._eef_pose_score(query.context, candidate.context)
            gripper_score = rbf_score(
                abs(query.context.gripper_width - candidate.context.gripper_width),
                self.config.gripper_sigma_m,
            )
            # 任务文本和 active-object 语义共同门控几何证据，避免“几何位置
            # 恰好相似但任务/物体无关”的候选反超语义兼容候选。
            compatibility_score = text_score * active_semantic_score
            score = (
                self.config.text_weight * text_score
                + self.config.semantic_weight * semantic_score
                + compatibility_score
                * (
                    self.config.object_pose_weight * object_pose_score
                    + self.config.layout_weight * layout_score
                    + self.config.eef_pose_weight * eef_pose_score
                    + self.config.gripper_weight * gripper_score
                )
            )
            scored.append(
                (
                    rank,
                    ExplicitRetrievalHit(
                        candidate=candidate,
                        score=score,
                        text_score=text_score,
                        semantic_score=semantic_score,
                        compatibility_score=compatibility_score,
                        object_pose_score=object_pose_score,
                        layout_score=layout_score,
                        eef_pose_score=eef_pose_score,
                        gripper_score=gripper_score,
                    ),
                )
            )

        scored.sort(key=lambda item: (-item[1].score, item[0]))
        result_size = min(requested_top_k, len(scored))
        return ExplicitRetrievalResult(
            text_result=text_result,
            hits=tuple(item[1] for item in scored[:result_size]),
        )

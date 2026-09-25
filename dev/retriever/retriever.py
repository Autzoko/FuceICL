"""文本召回后进行几何、状态与动作语义重排的第一版 Retriever。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
import torch.nn.functional as functional

from src.components.retriever.text_retriever import (
    TextCandidate,
    TextRetrievalResult,
    TextRetriever,
)

from .context_encoder import GeometricContextEncoder
from .scoring import (
    extent_log_distance,
    mean_score,
    rbf_score,
    rotation_distance,
    tensor_distance,
)
from .types import ActionSemantics, GeometricContext, RetrieverCandidate, RetrieverQuery


@dataclass(frozen=True)
class MultistageRetrieverConfig:
    """两阶段检索配置；所有最终打分权重之和必须为 1。"""

    text_top_k: int = 50
    default_top_k: int = 4
    encode_batch_size: int = 32
    text_weight: float = 0.25
    geometry_weight: float = 0.30
    layout_weight: float = 0.20
    state_weight: float = 0.15
    action_weight: float = 0.10
    layout_position_sigma_m: float = 0.20
    extent_log_sigma: float = 0.50
    eef_position_sigma_m: float = 0.10
    eef_rotation_sigma_rad: float = 0.75
    velocity_sigma: float = 0.25
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

        weights = self.score_weights
        if any(not math.isfinite(value) or value < 0 for value in weights):
            raise ValueError("打分权重必须是有限非负数")
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("最终打分权重之和必须为 1")

        scales = (
            self.layout_position_sigma_m,
            self.extent_log_sigma,
            self.eef_position_sigma_m,
            self.eef_rotation_sigma_rad,
            self.velocity_sigma,
            self.gripper_sigma_m,
        )
        if any(not math.isfinite(value) or value <= 0 for value in scales):
            raise ValueError("相似度尺度必须是有限正数")

    @property
    def score_weights(self) -> tuple[float, ...]:
        return (
            self.text_weight,
            self.geometry_weight,
            self.layout_weight,
            self.state_weight,
            self.action_weight,
        )


@dataclass(frozen=True)
class RetrievalHit:
    """最终候选以及可审计的分项分数。"""

    candidate: RetrieverCandidate
    score: float
    text_score: float
    geometry_score: float
    layout_score: float
    state_score: float
    action_score: float


@dataclass(frozen=True)
class RetrievalResult:
    """一次多阶段检索结果。"""

    text_result: TextRetrievalResult
    hits: tuple[RetrievalHit, ...]


class MultistageRetriever:
    """文本 Top-K 召回 + domain-invariant context 重排。

    检索器只返回数据库中原始 ``RetrieverCandidate``，不会重定位、缩放或
    改写轨迹。当前分项权重是待验证的初始值，不应被视为训练后的最终参数。
    """

    def __init__(
        self,
        text_retriever: TextRetriever,
        context_encoder: GeometricContextEncoder,
        *,
        config: MultistageRetrieverConfig | None = None,
    ) -> None:
        self.text_retriever = text_retriever
        self.context_encoder = context_encoder
        self.config = config or MultistageRetrieverConfig()
        self._candidates: dict[str, RetrieverCandidate] = {}
        self._context_embeddings: dict[str, torch.Tensor] = {}

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

    def build_index(self, candidates: Sequence[RetrieverCandidate]) -> None:
        """建立文本和 context 索引；全部成功后再替换旧索引。"""
        values = self._validate_candidates(candidates)
        text_candidates = [
            TextCandidate(candidate_id=item.candidate_id, text=item.text)
            for item in values
        ]
        embeddings = self.context_encoder.encode(
            [item.context for item in values],
            batch_size=self.config.encode_batch_size,
        )
        if embeddings.ndim != 2 or embeddings.shape[0] != len(values):
            raise ValueError("Context Encoder 返回 shape 必须为 [B, D]")
        if embeddings.shape[1] == 0 or not torch.isfinite(embeddings).all():
            raise ValueError("Context Encoder 返回了空维度、NaN 或 Inf")
        embeddings = functional.normalize(embeddings.float().cpu(), p=2, dim=-1)

        # 文本索引最后构建；失败时本类的 context 索引不会留下半成品。
        self.text_retriever.build_index(text_candidates)
        self._candidates = {item.candidate_id: item for item in values}
        self._context_embeddings = {
            item.candidate_id: embedding
            for item, embedding in zip(values, embeddings, strict=True)
        }

    def _layout_score(
        self,
        query: GeometricContext,
        candidate: GeometricContext,
    ) -> float:
        scores = [
            rbf_score(
                tensor_distance(query.active_center, candidate.active_center),
                self.config.layout_position_sigma_m,
            ),
            rbf_score(
                extent_log_distance(query.active_extent, candidate.active_extent),
                self.config.extent_log_sigma,
            ),
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
        else:
            scores.append(1.0)
        return mean_score(scores)

    def _state_score(
        self,
        query: GeometricContext,
        candidate: GeometricContext,
    ) -> float:
        query_eef = query.eef_relative_active
        candidate_eef = candidate.eef_relative_active
        return mean_score(
            (
                rbf_score(
                    tensor_distance(query_eef[:3], candidate_eef[:3]),
                    self.config.eef_position_sigma_m,
                ),
                rbf_score(
                    rotation_distance(query_eef[3:9], candidate_eef[3:9]),
                    self.config.eef_rotation_sigma_rad,
                ),
                rbf_score(
                    tensor_distance(query.eef_velocity, candidate.eef_velocity),
                    self.config.velocity_sigma,
                ),
                rbf_score(
                    abs(query.gripper_width - candidate.gripper_width),
                    self.config.gripper_sigma_m,
                ),
            )
        )

    @staticmethod
    def _label_score(query: str | None, candidate: str | None) -> float | None:
        if query is None:
            return None
        if candidate is None:
            return 0.5
        return float(query.strip().casefold() == candidate.strip().casefold())

    @staticmethod
    def _direction_score(
        query: torch.Tensor | None,
        candidate: torch.Tensor | None,
    ) -> float | None:
        if query is None:
            return None
        if candidate is None:
            return 0.5
        cosine = functional.cosine_similarity(
            query.float().cpu().unsqueeze(0),
            candidate.float().cpu().unsqueeze(0),
        )[0]
        return float(torch.clamp(0.5 * (cosine + 1.0), 0.0, 1.0))

    def _action_score(
        self,
        query: ActionSemantics,
        candidate: ActionSemantics,
        *,
        parsed_query_operation: str | None,
        parsed_candidate_operation: str | None,
    ) -> float:
        query_operation = query.operation or parsed_query_operation
        candidate_operation = candidate.operation or parsed_candidate_operation
        scores = (
            self._label_score(query_operation, candidate_operation),
            self._label_score(query.phase, candidate.phase),
            self._label_score(query.gripper_event, candidate.gripper_event),
            self._direction_score(
                query.translation_direction,
                candidate.translation_direction,
            ),
            self._direction_score(query.rotation_axis, candidate.rotation_axis),
        )
        available = [score for score in scores if score is not None]
        return mean_score(available) if available else 0.5

    def retrieve(
        self,
        query: RetrieverQuery,
        *,
        top_k: int | None = None,
    ) -> RetrievalResult:
        """返回文本候选中的 Top-K；不修改任何 candidate payload。"""
        if not self.is_indexed:
            raise RuntimeError("请先调用 build_index() 建立候选索引")
        if not isinstance(query, RetrieverQuery):
            raise TypeError("query 必须是 RetrieverQuery")
        if not isinstance(query.text, str) or not query.text.strip():
            raise ValueError("query text 不能为空")

        requested_top_k = self.config.default_top_k if top_k is None else top_k
        if isinstance(requested_top_k, bool) or not isinstance(requested_top_k, int):
            raise TypeError("top_k 必须是整数")
        if requested_top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        text_result = self.text_retriever.retrieve(
            query.text,
            top_k=min(self.config.text_top_k, len(self)),
        )
        query_embedding = self.context_encoder.encode(
            [query.context],
            batch_size=1,
        )
        if query_embedding.ndim != 2 or query_embedding.shape[0] != 1:
            raise ValueError("Context Encoder 的 query 输出 shape 必须为 [1, D]")
        query_embedding = functional.normalize(
            query_embedding[0].float().cpu(),
            p=2,
            dim=0,
        )

        scored: list[tuple[int, RetrievalHit]] = []
        for rank, text_hit in enumerate(text_result.hits):
            identifier = text_hit.candidate.candidate_id
            candidate = self._candidates[identifier]
            geometry_cosine = torch.dot(
                query_embedding,
                self._context_embeddings[identifier],
            )
            geometry_score = float(
                torch.clamp(0.5 * (geometry_cosine + 1.0), 0.0, 1.0)
            )
            text_score = max(0.0, min(1.0, 0.5 * (text_hit.score + 1.0)))
            layout_score = self._layout_score(query.context, candidate.context)
            state_score = self._state_score(query.context, candidate.context)
            action_score = self._action_score(
                query.action,
                candidate.action,
                parsed_query_operation=text_result.query.goal_operation,
                parsed_candidate_operation=text_hit.parsed.goal_operation,
            )
            score = (
                self.config.text_weight * text_score
                + self.config.geometry_weight * geometry_score
                + self.config.layout_weight * layout_score
                + self.config.state_weight * state_score
                + self.config.action_weight * action_score
            )
            scored.append(
                (
                    rank,
                    RetrievalHit(
                        candidate=candidate,
                        score=score,
                        text_score=text_score,
                        geometry_score=geometry_score,
                        layout_score=layout_score,
                        state_score=state_score,
                        action_score=action_score,
                    ),
                )
            )

        scored.sort(key=lambda item: (-item[1].score, item[0]))
        result_size = min(requested_top_k, len(scored))
        return RetrievalResult(
            text_result=text_result,
            hits=tuple(item[1] for item in scored[:result_size]),
        )

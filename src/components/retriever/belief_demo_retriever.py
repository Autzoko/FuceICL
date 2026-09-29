"""按 typed operation 分桶的轻量 belief-state Demo 精确检索。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch


def _operation(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("operation 不能为空")
    return " ".join(value.casefold().split())


def _vector(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} 必须是浮点 Tensor")
    if value.ndim != 1 or value.numel() == 0:
        raise ValueError(f"{name} shape 必须为 [D] 且 D>0")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} 含 NaN/Inf")
    return value.detach().float().cpu().clone()


def _action(value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError("raw_action 必须是浮点 Tensor")
    if value.ndim != 2 or value.shape[1] != 7 or value.shape[0] == 0:
        raise ValueError("raw_action shape 必须为 [H,7]")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("raw_action 含 NaN/Inf")
    return value.detach().float().cpu().clone()


@dataclass(frozen=True)
class BeliefDemoCandidate:
    """一个可检索 Demo chunk；输入的 state/action 会被防御性复制。"""

    candidate_id: str
    operation: str
    state: torch.Tensor
    raw_action: torch.Tensor
    episode_id: str | None = None
    payload: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ValueError("candidate_id 不能为空")
        object.__setattr__(self, "candidate_id", self.candidate_id.strip())
        object.__setattr__(self, "operation", _operation(self.operation))
        if self.episode_id is not None:
            if not isinstance(self.episode_id, str) or not self.episode_id.strip():
                raise ValueError("episode_id 必须为非空字符串或 None")
            object.__setattr__(self, "episode_id", self.episode_id.strip())
        object.__setattr__(self, "state", _vector(self.state, name="state"))
        object.__setattr__(self, "raw_action", _action(self.raw_action))


@dataclass(frozen=True)
class BeliefDemoQuery:
    """线上 typed operation 与当前 26D causal belief state。"""

    operation: str
    state: torch.Tensor

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation", _operation(self.operation))
        object.__setattr__(self, "state", _vector(self.state, name="state"))


@dataclass(frozen=True)
class BeliefDemoHit:
    candidate: BeliefDemoCandidate
    normalized_mse_distance: float


@dataclass(frozen=True)
class BeliefDemoRetrievalResult:
    query: BeliefDemoQuery
    hits: tuple[BeliefDemoHit, ...]
    accepted: bool
    reason: str | None


@dataclass(frozen=True)
class _Bucket:
    candidates: tuple[BeliefDemoCandidate, ...]
    normalized_states: torch.Tensor


class ExactBeliefDemoRetriever:
    """在 operation bucket 内执行 state-normalized MSE top-K。"""

    def __init__(self, state_std: torch.Tensor) -> None:
        standard_deviation = _vector(state_std, name="state_std")
        if bool((standard_deviation <= 0).any()):
            raise ValueError("state_std 必须严格为正")
        self.state_std = standard_deviation.clamp_min(1e-5)
        self._buckets: dict[str, _Bucket] = {}
        self._size = 0
        self._action_shape: tuple[int, int] | None = None

    @property
    def is_indexed(self) -> bool:
        return self._size > 0

    def __len__(self) -> int:
        return self._size

    @property
    def action_shape(self) -> tuple[int, int] | None:
        """索引中统一的 ``[H, 7]`` action chunk shape。"""
        return self._action_shape

    def build_index(self, candidates: Sequence[BeliefDemoCandidate]) -> None:
        """完整校验后原子替换旧索引。"""
        if isinstance(candidates, (str, bytes)):
            raise TypeError("candidates 必须是 BeliefDemoCandidate 序列")
        values = list(candidates)
        if not values:
            raise ValueError("至少需要一个 BeliefDemoCandidate")
        if any(not isinstance(value, BeliefDemoCandidate) for value in values):
            raise TypeError("candidates 元素必须是 BeliefDemoCandidate")
        identifiers = [value.candidate_id for value in values]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("candidate_id 必须全局唯一")
        state_dim = self.state_std.numel()
        states = [_vector(value.state, name="state") for value in values]
        if any(state.numel() != state_dim for state in states):
            raise ValueError("candidate state dimension 与 state_std 不一致")
        actions = [_action(value.raw_action) for value in values]
        action_shapes = {tuple(value.shape) for value in actions}
        if len(action_shapes) != 1:
            raise ValueError("所有 raw_action shape 必须一致")

        grouped: dict[str, list[tuple[BeliefDemoCandidate, torch.Tensor]]] = {}
        for candidate, state in zip(values, states, strict=True):
            grouped.setdefault(candidate.operation, []).append(
                (candidate, state / self.state_std)
            )
        self._buckets = {
            operation: _Bucket(
                candidates=tuple(row[0] for row in rows),
                normalized_states=torch.stack([row[1] for row in rows]),
            )
            for operation, rows in grouped.items()
        }
        self._size = len(values)
        self._action_shape = action_shapes.pop()

    def retrieve(
        self,
        query: BeliefDemoQuery,
        *,
        top_k: int = 1,
    ) -> BeliefDemoRetrievalResult:
        """返回稳定排序的 top-K；未知 operation 显式拒绝。"""
        if not self.is_indexed:
            raise RuntimeError("请先调用 build_index() 建立索引")
        if not isinstance(query, BeliefDemoQuery):
            raise TypeError("query 必须是 BeliefDemoQuery")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k 必须是正整数")
        bucket = self._buckets.get(query.operation)
        if bucket is None:
            return BeliefDemoRetrievalResult(
                query=query,
                hits=(),
                accepted=False,
                reason="no_typed_operation_bucket",
            )
        state = _vector(query.state, name="state")
        if state.numel() != self.state_std.numel():
            raise ValueError("query state dimension 与 state_std 不一致")
        delta = bucket.normalized_states - state / self.state_std
        distances = torch.mean(delta * delta, dim=1)
        if top_k == 1:
            order = [int(torch.argmin(distances))]
        else:
            # stable=True 在距离相同时保持离线索引顺序，避免 Python 标量循环。
            order = torch.argsort(distances, stable=True)[:top_k].tolist()
        hits = tuple(
            BeliefDemoHit(
                candidate=bucket.candidates[index],
                normalized_mse_distance=float(distances[index]),
            )
            for index in order
        )
        return BeliefDemoRetrievalResult(
            query=query,
            hits=hits,
            accepted=True,
            reason=None,
        )

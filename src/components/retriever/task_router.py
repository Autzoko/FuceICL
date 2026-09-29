"""规范化任务键的确定性分桶；匹配失败时显式拒绝。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


def _canonical(value: str, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} 必须是字符串")
    normalized = " ".join(value.casefold().split())
    if not normalized:
        raise ValueError(f"{field} 不能为空")
    return normalized


def _optional_canonical(value: str | None, *, field: str) -> str | None:
    return None if value is None else _canonical(value, field=field)


@dataclass(frozen=True)
class TaskKey:
    """数据适配器提供的 canonical task contract，不负责解析自由文本。"""

    operation: str
    active_object: str
    reference_object: str | None = None
    relation_effect: str | None = None
    qualifiers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "operation",
            _canonical(self.operation, field="operation"),
        )
        object.__setattr__(
            self,
            "active_object",
            _canonical(self.active_object, field="active_object"),
        )
        object.__setattr__(
            self,
            "reference_object",
            _optional_canonical(
                self.reference_object,
                field="reference_object",
            ),
        )
        object.__setattr__(
            self,
            "relation_effect",
            _optional_canonical(
                self.relation_effect,
                field="relation_effect",
            ),
        )
        if isinstance(self.qualifiers, (str, bytes)):
            raise TypeError("qualifiers 必须是字符串 tuple")
        normalized = tuple(
            sorted(
                {
                    _canonical(value, field="qualifier")
                    for value in self.qualifiers
                }
            )
        )
        object.__setattr__(self, "qualifiers", normalized)


@dataclass(frozen=True)
class TaskBucketCandidate:
    """带规范化任务键的 episode/chunk bucket 候选。"""

    candidate_id: str
    key: TaskKey

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ValueError("candidate_id 不能为空")
        if not isinstance(self.key, TaskKey):
            raise TypeError("key 必须是 TaskKey")


@dataclass(frozen=True)
class TaskRouteResult:
    """确定性分桶结果；空 bucket 是可审计的拒绝。"""

    query: TaskKey
    candidates: tuple[TaskBucketCandidate, ...]
    accepted: bool
    reason: str | None


class TaskBucketRouter:
    """按完整 canonical key 精确分桶，不做不可靠的部分匹配回退。"""

    def __init__(self) -> None:
        self._buckets: dict[TaskKey, tuple[TaskBucketCandidate, ...]] = {}
        self._size = 0

    @property
    def is_indexed(self) -> bool:
        return self._size > 0

    def __len__(self) -> int:
        return self._size

    def build_index(
        self,
        candidates: Sequence[TaskBucketCandidate],
    ) -> None:
        """原子替换索引，并保留输入顺序作为稳定的 bucket 顺序。"""
        if isinstance(candidates, (str, bytes)):
            raise TypeError("candidates 必须是 TaskBucketCandidate 序列")
        values = list(candidates)
        if not values:
            raise ValueError("至少需要一个 task bucket candidate")
        if any(not isinstance(item, TaskBucketCandidate) for item in values):
            raise TypeError("candidates 元素必须是 TaskBucketCandidate")
        identifiers = [item.candidate_id for item in values]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("candidate_id 必须唯一")
        mutable: dict[TaskKey, list[TaskBucketCandidate]] = {}
        for item in values:
            mutable.setdefault(item.key, []).append(item)
        buckets = {
            key: tuple(items) for key, items in mutable.items()
        }
        self._buckets = buckets
        self._size = len(values)

    def route(self, query: TaskKey) -> TaskRouteResult:
        """返回完整 key 相同的 bucket；未知 key 不进行近似执行。"""
        if not self.is_indexed:
            raise RuntimeError("请先调用 build_index() 建立 task bucket 索引")
        if not isinstance(query, TaskKey):
            raise TypeError("query 必须是 TaskKey")
        candidates = self._buckets.get(query, ())
        return TaskRouteResult(
            query=query,
            candidates=candidates,
            accepted=bool(candidates),
            reason=None if candidates else "no_exact_task_key_match",
        )

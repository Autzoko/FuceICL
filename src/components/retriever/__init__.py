"""Retriever 组件的稳定公共接口；文本模型依赖按需加载。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .task_router import (
    TaskBucketCandidate,
    TaskBucketRouter,
    TaskKey,
    TaskRouteResult,
)

if TYPE_CHECKING:
    from .text_retriever import (
        TextCandidate,
        TextRetrievalHit,
        TextRetrievalResult,
        TextRetriever,
        TextRetrieverConfig,
    )


_TEXT_EXPORTS = frozenset(
    {
        "TextCandidate",
        "TextRetrievalHit",
        "TextRetrievalResult",
        "TextRetriever",
        "TextRetrieverConfig",
    }
)


def __getattr__(name: str) -> Any:
    """只在访问文本接口时加载 GLiNER/MiniLM 依赖。"""
    if name not in _TEXT_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import text_retriever

    value = getattr(text_retriever, name)
    globals()[name] = value
    return value

__all__ = [
    "TextCandidate",
    "TextRetrievalHit",
    "TextRetrievalResult",
    "TextRetriever",
    "TextRetrieverConfig",
    "TaskBucketCandidate",
    "TaskBucketRouter",
    "TaskKey",
    "TaskRouteResult",
]

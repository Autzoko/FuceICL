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
    from .embedding_retriever import (
        EmbeddingCandidate,
        EmbeddingQuery,
        EmbeddingRetrievalHit,
        EmbeddingRetrievalResult,
        EmbeddingRetrieverConfig,
        ExactEmbeddingRetriever,
    )
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

_EMBEDDING_EXPORTS = frozenset(
    {
        "EmbeddingCandidate",
        "EmbeddingQuery",
        "EmbeddingRetrievalHit",
        "EmbeddingRetrievalResult",
        "EmbeddingRetrieverConfig",
        "ExactEmbeddingRetriever",
    }
)


def __getattr__(name: str) -> Any:
    """按需加载文本模型或 embedding 检索依赖。"""
    if name in _TEXT_EXPORTS:
        from . import text_retriever as module
    elif name in _EMBEDDING_EXPORTS:
        from . import embedding_retriever as module
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(module, name)
    globals()[name] = value
    return value

__all__ = [
    "EmbeddingCandidate",
    "EmbeddingQuery",
    "EmbeddingRetrievalHit",
    "EmbeddingRetrievalResult",
    "EmbeddingRetrieverConfig",
    "ExactEmbeddingRetriever",
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

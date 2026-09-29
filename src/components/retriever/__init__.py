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
    from .belief_demo_retriever import (
        BeliefDemoCandidate,
        BeliefDemoHit,
        BeliefDemoQuery,
        BeliefDemoRetrievalResult,
        ExactBeliefDemoRetriever,
    )
    from .belief_tracker import (
        ObjectBeliefEstimate,
        VisibilityConditionedBeliefTracker,
        belief_risk_accepted,
    )
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

_BELIEF_EXPORTS = frozenset(
    {
        "ObjectBeliefEstimate",
        "VisibilityConditionedBeliefTracker",
        "belief_risk_accepted",
    }
)

_BELIEF_RETRIEVER_EXPORTS = frozenset(
    {
        "BeliefDemoCandidate",
        "BeliefDemoHit",
        "BeliefDemoQuery",
        "BeliefDemoRetrievalResult",
        "ExactBeliefDemoRetriever",
    }
)


def __getattr__(name: str) -> Any:
    """按需加载文本模型或 embedding 检索依赖。"""
    if name in _TEXT_EXPORTS:
        from . import text_retriever as module
    elif name in _EMBEDDING_EXPORTS:
        from . import embedding_retriever as module
    elif name in _BELIEF_EXPORTS:
        from . import belief_tracker as module
    elif name in _BELIEF_RETRIEVER_EXPORTS:
        from . import belief_demo_retriever as module
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(module, name)
    globals()[name] = value
    return value

__all__ = [
    "BeliefDemoCandidate",
    "BeliefDemoHit",
    "BeliefDemoQuery",
    "BeliefDemoRetrievalResult",
    "EmbeddingCandidate",
    "EmbeddingQuery",
    "EmbeddingRetrievalHit",
    "EmbeddingRetrievalResult",
    "EmbeddingRetrieverConfig",
    "ExactEmbeddingRetriever",
    "ExactBeliefDemoRetriever",
    "ObjectBeliefEstimate",
    "TextCandidate",
    "TextRetrievalHit",
    "TextRetrievalResult",
    "TextRetriever",
    "TextRetrieverConfig",
    "TaskBucketCandidate",
    "TaskBucketRouter",
    "TaskKey",
    "TaskRouteResult",
    "VisibilityConditionedBeliefTracker",
    "belief_risk_accepted",
]

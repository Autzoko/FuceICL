"""Retriever 组件的稳定公共接口。"""

from .text_retriever import (
    TextCandidate,
    TextRetrievalHit,
    TextRetrievalResult,
    TextRetriever,
    TextRetrieverConfig,
)

__all__ = [
    "TextCandidate",
    "TextRetrievalHit",
    "TextRetrievalResult",
    "TextRetriever",
    "TextRetrieverConfig",
]

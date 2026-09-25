"""第一版多阶段 Demo Retriever 实验实现。

公共符号使用惰性导入，避免只运行 RLBench adapter 时被迫安装文本模型依赖。
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "ActionSemantics": (".types", "ActionSemantics"),
    "AdapterConfig": (".rlbench_adapter", "AdapterConfig"),
    "ContextEncoderConfig": (".context_encoder", "ContextEncoderConfig"),
    "ExplicitFeatureRetriever": (".explicit_retriever", "ExplicitFeatureRetriever"),
    "ExplicitRetrievalHit": (".explicit_retriever", "ExplicitRetrievalHit"),
    "ExplicitRetrievalResult": (".explicit_retriever", "ExplicitRetrievalResult"),
    "ExplicitRetrieverConfig": (".explicit_retriever", "ExplicitRetrieverConfig"),
    "GeometricContext": (".types", "GeometricContext"),
    "GeometricContextEncoder": (".context_encoder", "GeometricContextEncoder"),
    "MultistageRetriever": (".retriever", "MultistageRetriever"),
    "MultistageRetrieverConfig": (".retriever", "MultistageRetrieverConfig"),
    "RetrievalHit": (".retriever", "RetrievalHit"),
    "RetrievalResult": (".retriever", "RetrievalResult"),
    "RetrieverCandidate": (".types", "RetrieverCandidate"),
    "RetrieverQuery": (".types", "RetrieverQuery"),
    "RLBenchArchiveAdapter": (".rlbench_adapter", "RLBenchArchiveAdapter"),
    "RLBenchChunkRecord": (".rlbench_adapter", "RLBenchChunkRecord"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

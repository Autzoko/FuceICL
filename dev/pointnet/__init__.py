"""Tiny PointNet++ 的 RLBench 数据准备与训练实验。

公共符号使用惰性导入，保证 ``python -m`` 入口不会被包初始化提前加载。
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "EventAnchor",
    "PointNetPreprocessConfig",
    "build_pair_labels",
    "detect_event_anchors",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(".preprocess_rlbench", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

"""轻量、严格依赖 Demo 的 action-chunk Predictor 公共接口。"""

from .layout_equivariant_policy import (
    LayoutEquivariantDemoPolicy,
    LayoutEquivariantDemoPolicyConfig,
)

__all__ = [
    "LayoutEquivariantDemoPolicy",
    "LayoutEquivariantDemoPolicyConfig",
]

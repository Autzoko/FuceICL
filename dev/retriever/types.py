"""第一版多阶段 Retriever 的数据契约。"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import torch


def _validate_vector(name: str, value: torch.Tensor, size: int) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} 必须是 torch.Tensor")
    if value.shape != (size,):
        raise ValueError(f"{name} shape 必须为 [{size}]")
    if not value.is_floating_point():
        raise TypeError(f"{name} 必须是浮点 Tensor")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} 包含 NaN 或 Inf")


def _validate_points(
    name: str,
    points: torch.Tensor,
    point_mask: torch.Tensor | None,
) -> None:
    if not isinstance(points, torch.Tensor):
        raise TypeError(f"{name} 必须是 torch.Tensor")
    if points.ndim != 2 or points.shape[-1] != 3 or not len(points):
        raise ValueError(f"{name} shape 必须为 [N, 3] 且 N > 0")
    if not points.is_floating_point() or not torch.isfinite(points).all():
        raise ValueError(f"{name} 必须是有限浮点 Tensor")
    if point_mask is not None:
        if point_mask.shape != (len(points),):
            raise ValueError(f"{name}_mask shape 必须为 [N]")
        if not point_mask.to(torch.bool).any():
            raise ValueError(f"{name}_mask 至少包含一个有效点")


def _validate_rotation_6d(name: str, value: torch.Tensor) -> None:
    _validate_vector(name, value, 6)
    first = value[:3]
    second = value[3:6]
    first_norm = torch.linalg.vector_norm(first)
    if first_norm <= 1e-6:
        raise ValueError(f"{name} 中的 rotation_6d 退化")
    second_orthogonal = second - torch.dot(second, first) * first / first_norm.square()
    if torch.linalg.vector_norm(second_orthogonal) <= 1e-6:
        raise ValueError(f"{name} 中的 rotation_6d 退化")


def _validate_confidence(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} 必须是数值")
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} 必须位于 [0, 1]")


@dataclass(frozen=True)
class GeometricContext:
    """Query 与 candidate chunk 起点共享的几何和机器人状态。"""

    active_points: torch.Tensor
    active_center: torch.Tensor
    active_extent: torch.Tensor
    eef_relative_active: torch.Tensor
    eef_velocity: torch.Tensor
    gripper_width: float
    active_point_mask: torch.Tensor | None = None
    target_points: torch.Tensor | None = None
    target_center: torch.Tensor | None = None
    target_extent: torch.Tensor | None = None
    target_point_mask: torch.Tensor | None = None
    active_rotation_6d: torch.Tensor | None = None
    target_rotation_6d: torch.Tensor | None = None
    active_orientation_confidence: float = 1.0
    target_orientation_confidence: float = 1.0

    def __post_init__(self) -> None:
        _validate_points(
            "active_points",
            self.active_points,
            self.active_point_mask,
        )
        _validate_vector("active_center", self.active_center, 3)
        _validate_vector("active_extent", self.active_extent, 3)
        _validate_vector("eef_relative_active", self.eef_relative_active, 9)
        _validate_rotation_6d(
            "eef_relative_active.rotation_6d",
            self.eef_relative_active[3:9],
        )
        _validate_vector("eef_velocity", self.eef_velocity, 6)
        _validate_confidence(
            "active_orientation_confidence",
            self.active_orientation_confidence,
        )
        _validate_confidence(
            "target_orientation_confidence",
            self.target_orientation_confidence,
        )
        if self.active_rotation_6d is not None:
            _validate_rotation_6d(
                "active_rotation_6d",
                self.active_rotation_6d,
            )
        if isinstance(self.gripper_width, bool) or not isinstance(
            self.gripper_width,
            (int, float),
        ):
            raise TypeError("gripper_width 必须是数值")
        if not math.isfinite(self.gripper_width) or self.gripper_width < 0:
            raise ValueError("gripper_width 不能为负数")
        if (self.active_extent <= 0).any():
            raise ValueError("active_extent 必须大于 0")

        target_values = (
            self.target_points,
            self.target_center,
            self.target_extent,
        )
        if any(value is None for value in target_values) and any(
            value is not None for value in target_values
        ):
            raise ValueError(
                "target_points/center/extent 必须同时提供或同时省略"
            )
        if self.target_points is None:
            if self.target_point_mask is not None or self.target_rotation_6d is not None:
                raise ValueError(
                    "没有 target_points 时不能提供 target mask/rotation"
                )
            return

        _validate_points(
            "target_points",
            self.target_points,
            self.target_point_mask,
        )
        _validate_vector("target_center", self.target_center, 3)
        _validate_vector("target_extent", self.target_extent, 3)
        if self.target_rotation_6d is not None:
            _validate_rotation_6d(
                "target_rotation_6d",
                self.target_rotation_6d,
            )
        if (self.target_extent <= 0).any():
            raise ValueError("target_extent 必须大于 0")

    @property
    def has_target(self) -> bool:
        return self.target_points is not None


@dataclass(frozen=True)
class ActionSemantics:
    """用于匹配的动作语义；不包含需要修改 Demo 的轨迹。"""

    operation: str | None = None
    phase: str | None = None
    gripper_event: str | None = None
    translation_direction: torch.Tensor | None = None
    rotation_axis: torch.Tensor | None = None

    def __post_init__(self) -> None:
        for name in ("operation", "phase", "gripper_event"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} 必须是非空字符串或 None")
        for name in ("translation_direction", "rotation_axis"):
            value = getattr(self, name)
            if value is not None:
                _validate_vector(name, value, 3)
                if torch.linalg.vector_norm(value) <= 1e-6:
                    raise ValueError(f"{name} 不能是零向量")


@dataclass(frozen=True)
class RetrieverCandidate:
    """数据库中的原始 Demo chunk 及其检索 key。"""

    candidate_id: str
    text: str
    context: GeometricContext
    action: ActionSemantics = field(default_factory=ActionSemantics)
    payload: Any = None
    active_object_semantic: str | None = None
    target_object_semantic: str | None = None


@dataclass(frozen=True)
class RetrieverQuery:
    """在线查询；只包含当前与过去可观测信息。"""

    text: str
    context: GeometricContext
    action: ActionSemantics = field(default_factory=ActionSemantics)
    active_object_semantic: str | None = None
    target_object_semantic: str | None = None

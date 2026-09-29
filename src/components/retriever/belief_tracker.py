"""基于分割点云与 TCP 运动的轻量因果物体 belief。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _points(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("observed_points_world shape 必须为 [N,3]")
    if not np.isfinite(array).all():
        raise ValueError("observed_points_world 必须全部有限")
    return array


def _position(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError("tcp_position_world shape 必须为 [3] 且全部有限")
    return array


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).copy()
    result.setflags(write=False)
    return result


def belief_risk_accepted(
    *,
    visible: bool,
    tcp_net_displacement_m: float,
    maximum_tcp_net_displacement_m: float | None,
) -> bool:
    """执行零参数风险门控；不读取估计误差或仿真真值。"""
    if not np.isfinite(tcp_net_displacement_m) or tcp_net_displacement_m < 0:
        raise ValueError("tcp_net_displacement_m 必须是非负有限数")
    if maximum_tcp_net_displacement_m is not None and (
        not np.isfinite(maximum_tcp_net_displacement_m)
        or maximum_tcp_net_displacement_m <= 0
    ):
        raise ValueError("maximum_tcp_net_displacement_m 必须为正或 None")
    return bool(
        visible
        or maximum_tcp_net_displacement_m is None
        or tcp_net_displacement_m <= maximum_tcp_net_displacement_m
    )


@dataclass(frozen=True)
class ObjectBeliefEstimate:
    """一个时刻的不可变 object belief 与因果风险统计。"""

    points_world: np.ndarray
    center_world: np.ndarray
    visible: bool
    age_steps: int
    observed_point_count: int
    tcp_net_displacement_since_visible_m: float
    tcp_path_length_since_visible_m: float

    def accepts(self, maximum_tcp_net_displacement_m: float | None) -> bool:
        """判断当前 belief 是否位于解析运动包络预算内。"""
        return belief_risk_accepted(
            visible=self.visible,
            tcp_net_displacement_m=(
                self.tcp_net_displacement_since_visible_m
            ),
            maximum_tcp_net_displacement_m=(
                maximum_tcp_net_displacement_m
            ),
        )


class VisibilityConditionedBeliefTracker:
    """可见时测量更新，不可见时累加已发生的 TCP 位移。"""

    def __init__(self, minimum_visible_points: int = 6) -> None:
        if (
            isinstance(minimum_visible_points, bool)
            or not isinstance(minimum_visible_points, int)
            or minimum_visible_points <= 0
        ):
            raise ValueError("minimum_visible_points 必须是正整数")
        self.minimum_visible_points = minimum_visible_points
        self.reset()

    @property
    def initialized(self) -> bool:
        return self._points_world is not None

    def reset(self) -> None:
        """清除 episode 状态；新 episode 必须先获得有效可见测量。"""
        self._points_world: np.ndarray | None = None
        self._center_world: np.ndarray | None = None
        self._previous_tcp: np.ndarray | None = None
        self._last_visible_tcp: np.ndarray | None = None
        self._age_steps = 0
        self._tcp_path_length_m = 0.0

    def update(
        self,
        observed_points_world: np.ndarray,
        tcp_position_world: np.ndarray,
    ) -> ObjectBeliefEstimate:
        """按时间顺序消费一次观测并返回当前 belief。"""
        observed = _points(observed_points_world)
        tcp = _position(tcp_position_world)
        visible = len(observed) >= self.minimum_visible_points
        if visible:
            self._points_world = observed.copy()
            self._center_world = observed.mean(axis=0)
            self._last_visible_tcp = tcp.copy()
            self._age_steps = 0
            self._tcp_path_length_m = 0.0
        else:
            if (
                self._points_world is None
                or self._center_world is None
                or self._previous_tcp is None
                or self._last_visible_tcp is None
            ):
                raise RuntimeError(
                    "首次有效可见测量前不能传播 object belief"
                )
            delta = tcp - self._previous_tcp
            self._points_world = self._points_world + delta
            self._center_world = self._center_world + delta
            self._age_steps += 1
            self._tcp_path_length_m += float(np.linalg.norm(delta))
        self._previous_tcp = tcp.copy()
        if (
            self._points_world is None
            or self._center_world is None
            or self._last_visible_tcp is None
        ):
            raise RuntimeError("object belief 内部状态不完整")
        return ObjectBeliefEstimate(
            points_world=_readonly(self._points_world),
            center_world=_readonly(self._center_world),
            visible=visible,
            age_steps=self._age_steps,
            observed_point_count=len(observed),
            tcp_net_displacement_since_visible_m=float(
                np.linalg.norm(tcp - self._last_visible_tcp)
            ),
            tcp_path_length_since_visible_m=self._tcp_path_length_m,
        )

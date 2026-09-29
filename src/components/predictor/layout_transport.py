"""基于可观测有向 layout axis 的平面 action transport。"""

from __future__ import annotations

import torch


def _finite_vector(value: torch.Tensor, *, name: str) -> None:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} 必须是浮点 Tensor")
    if value.shape != (3,):
        raise ValueError(f"{name} shape 必须为 [3]")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} 含 NaN/Inf")


def _rotation_matrix(value: torch.Tensor, *, name: str) -> None:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} 必须是浮点 Tensor")
    if value.shape != (3, 3):
        raise ValueError(f"{name} shape 必须为 [3,3]")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} 含 NaN/Inf")
    matrix = value.detach().double().cpu()
    identity = torch.eye(3, dtype=torch.float64)
    if not bool(torch.allclose(matrix.T @ matrix, identity, atol=1e-4, rtol=0.0)):
        raise ValueError(f"{name} 不是正交旋转矩阵")
    if abs(float(torch.linalg.det(matrix)) - 1.0) > 1e-4:
        raise ValueError(f"{name} determinant 必须为 +1")


def _planar_yaw(axis: torch.Tensor, *, name: str) -> torch.Tensor:
    _finite_vector(axis, name=name)
    if float(torch.linalg.vector_norm(axis[:2])) <= 1e-9:
        raise ValueError(f"{name} 的平面投影退化")
    return torch.atan2(axis[1], axis[0])


def transport_planar_layout_action(
    action: torch.Tensor,
    *,
    demo_tcp_rotation_world: torch.Tensor,
    query_tcp_rotation_world: torch.Tensor,
    demo_layout_axis_world: torch.Tensor,
    query_layout_axis_world: torch.Tensor,
) -> torch.Tensor:
    """把 `[H,7]` Demo EEF action 映射到 query EEF frame。

    layout axis 必须来自当前或缓存的感知几何。函数只改变 translation 与
    axis-angle rotation vector，gripper target 保持逐元素不变。
    """
    if not isinstance(action, torch.Tensor) or not action.is_floating_point():
        raise TypeError("action 必须是浮点 Tensor")
    if action.ndim != 2 or action.shape[1] != 7 or not len(action):
        raise ValueError("action shape 必须为 [H,7] 且 H > 0")
    if not bool(torch.isfinite(action).all()):
        raise ValueError("action 含 NaN/Inf")
    _rotation_matrix(
        demo_tcp_rotation_world,
        name="demo_tcp_rotation_world",
    )
    _rotation_matrix(
        query_tcp_rotation_world,
        name="query_tcp_rotation_world",
    )
    demo_axis = demo_layout_axis_world.to(
        device=action.device,
        dtype=action.dtype,
    )
    query_axis = query_layout_axis_world.to(
        device=action.device,
        dtype=action.dtype,
    )
    demo_yaw = _planar_yaw(demo_axis, name="demo_layout_axis_world")
    query_yaw = _planar_yaw(query_axis, name="query_layout_axis_world")
    yaw_delta = query_yaw - demo_yaw
    cosine = torch.cos(yaw_delta)
    sine = torch.sin(yaw_delta)
    zero = torch.zeros((), device=action.device, dtype=action.dtype)
    one = torch.ones((), device=action.device, dtype=action.dtype)
    layout_rotation = torch.stack(
        (
            torch.stack((cosine, -sine, zero)),
            torch.stack((sine, cosine, zero)),
            torch.stack((zero, zero, one)),
        )
    )
    demo_rotation = demo_tcp_rotation_world.to(
        device=action.device,
        dtype=action.dtype,
    )
    query_rotation = query_tcp_rotation_world.to(
        device=action.device,
        dtype=action.dtype,
    )
    frame = query_rotation.T @ layout_rotation @ demo_rotation
    output = action.clone()
    output[:, :3] = action[:, :3] @ frame.T
    output[:, 3:6] = action[:, 3:6] @ frame.T
    return output

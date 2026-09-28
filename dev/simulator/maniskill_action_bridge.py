"""在 canonical EEF action 与 ManiSkill Panda controller 之间转换。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from mani_skill.utils.geometry.rotation_conversions import (
    axis_angle_to_matrix,
    euler_angles_to_matrix,
    matrix_to_axis_angle,
    matrix_to_euler_angles,
)

from dev.simulator.preprocess_maniskill_chunks import (
    _quaternion_wxyz_to_matrix,
)


@dataclass(frozen=True)
class ControllerAction:
    """转换后的归一化 controller action 及其 clipping 诊断。"""

    value: np.ndarray
    translation_clipped: bool
    rotation_clipped: bool
    unscaled_translation_norm_m: float
    unscaled_rotation_norm_rad: float


def preprocess_normalized_controller_actions(actions: np.ndarray) -> np.ndarray:
    """复现 ManiSkill 对归一化 action 的逐维及旋转范数裁剪。"""
    processed = np.asarray(actions, dtype=np.float64).copy()
    if processed.ndim != 2 or processed.shape[1] != 7:
        raise ValueError("normalized controller actions 必须为 [T,7]")
    if not np.isfinite(processed).all():
        raise ValueError("normalized controller actions 含 NaN 或 Inf")
    processed[:, :3] = np.clip(processed[:, :3], -1.0, 1.0)
    rotation_norm = np.linalg.norm(processed[:, 3:6], axis=1)
    clipped = rotation_norm > 1.0
    processed[clipped, 3:6] /= rotation_norm[clipped, None]
    processed[:, 6] = np.clip(processed[:, 6], -1.0, 1.0)
    return processed


def _rotation_tensor(value: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float64)


def canonical_to_controller(
    canonical_action: np.ndarray,
    tcp_pose_wxyz: np.ndarray,
    *,
    position_limit_m: float,
    rotation_scale_rad: float,
) -> ControllerAction:
    """把 body-frame axis-angle action 转成 ManiSkill 3.0.1 controller action。

    Panda ``pd_ee_delta_pose`` 默认使用 root-frame translation、root-aligned
    body rotation 和归一化 action。ManiSkill 3.0.1 的 rotation 实现用
    ``normalized * rot_lower``，因此这里显式保留负的 rotation scale。
    """
    action = np.asarray(canonical_action, dtype=np.float64)
    pose = np.asarray(tcp_pose_wxyz, dtype=np.float64)
    if action.shape != (7,) or pose.shape != (7,):
        raise ValueError("canonical action 与 TCP pose 都必须是一维 7D")
    if position_limit_m <= 0 or rotation_scale_rad == 0:
        raise ValueError("controller position/rotation scale 非法")
    if not np.isfinite(action).all() or not np.isfinite(pose).all():
        raise ValueError("action/TCP pose 含 NaN 或 Inf")

    current_rotation = _rotation_tensor(_quaternion_wxyz_to_matrix(pose[3:7]))
    body_delta = axis_angle_to_matrix(_rotation_tensor(action[3:6]))
    root_delta = current_rotation @ body_delta @ current_rotation.T
    root_euler = matrix_to_euler_angles(root_delta, "XYZ").numpy()
    root_translation = current_rotation.numpy() @ action[:3]

    translation_normalized = root_translation / position_limit_m
    rotation_normalized = root_euler / rotation_scale_rad
    translation_clipped = bool(np.any(np.abs(translation_normalized) > 1.0))
    rotation_norm = float(np.linalg.norm(rotation_normalized))
    rotation_clipped = rotation_norm > 1.0
    translation_normalized = np.clip(translation_normalized, -1.0, 1.0)
    if rotation_clipped:
        rotation_normalized = rotation_normalized / rotation_norm
    # canonical gripper 使用 [0,1]，线性映射可保留 motion planner 的连续命令。
    gripper = float(np.clip(2.0 * action[6] - 1.0, -1.0, 1.0))
    controller = np.concatenate(
        (translation_normalized, rotation_normalized, [gripper])
    ).astype(np.float32)
    return ControllerAction(
        value=controller,
        translation_clipped=translation_clipped,
        rotation_clipped=rotation_clipped,
        unscaled_translation_norm_m=float(np.linalg.norm(root_translation)),
        unscaled_rotation_norm_rad=float(np.linalg.norm(root_euler)),
    )


def controller_to_canonical_unclipped(
    controller_action: np.ndarray,
    tcp_pose_wxyz: np.ndarray,
    *,
    position_limit_m: float,
    rotation_scale_rad: float,
) -> np.ndarray:
    """仅用于数值 round-trip；调用方必须保证 action 未经历 clipping。"""
    action = np.asarray(controller_action, dtype=np.float64)
    pose = np.asarray(tcp_pose_wxyz, dtype=np.float64)
    if action.shape != (7,) or pose.shape != (7,):
        raise ValueError("controller action 与 TCP pose 都必须是一维 7D")
    current_rotation = _rotation_tensor(_quaternion_wxyz_to_matrix(pose[3:7]))
    root_translation = action[:3] * position_limit_m
    root_euler = _rotation_tensor(action[3:6] * rotation_scale_rad)
    root_delta = euler_angles_to_matrix(root_euler, "XYZ")
    body_translation = current_rotation.T.numpy() @ root_translation
    body_delta = current_rotation.T @ root_delta @ current_rotation
    body_axis_angle = matrix_to_axis_angle(body_delta).numpy()
    gripper = float(np.clip((action[6] + 1.0) / 2.0, 0.0, 1.0))
    return np.concatenate((body_translation, body_axis_angle, [gripper])).astype(
        np.float32
    )


def audit_round_trip(
    *,
    position_limit_m: float,
    rotation_scale_rad: float,
) -> dict[str, float | int]:
    """用多组非单位 TCP rotation 验证 frame/sign/rotation convention。"""
    tcp_rotations = (
        np.asarray([1.0, 0.0, 0.0, 0.0]),
        np.asarray([0.9238795, 0.0, 0.3826834, 0.0]),
        np.asarray([0.8660254, 0.2886751, -0.2886751, 0.2886751]),
    )
    actions = (
        np.asarray([0.01, -0.02, 0.015, 0.02, -0.01, 0.015, 1.0]),
        np.asarray([-0.025, 0.01, 0.005, -0.015, 0.02, -0.01, 0.0]),
    )
    errors = []
    for quaternion in tcp_rotations:
        pose = np.concatenate((np.asarray([0.4, 0.0, 0.3]), quaternion))
        for action in actions:
            converted = canonical_to_controller(
                action,
                pose,
                position_limit_m=position_limit_m,
                rotation_scale_rad=rotation_scale_rad,
            )
            if converted.translation_clipped or converted.rotation_clipped:
                raise RuntimeError("round-trip 测试 action 不应触发 clipping")
            recovered = controller_to_canonical_unclipped(
                converted.value,
                pose,
                position_limit_m=position_limit_m,
                rotation_scale_rad=rotation_scale_rad,
            )
            errors.append(np.abs(recovered - action))
    stacked = np.stack(errors)
    maximum = float(stacked.max())
    if maximum > 1e-5:
        raise RuntimeError(f"canonical/controller round-trip error={maximum:.3e}")
    return {
        "cases": len(errors),
        "maximum_absolute_error": maximum,
        "translation_maximum_absolute_error": float(stacked[:, :3].max()),
        "rotation_maximum_absolute_error": float(stacked[:, 3:6].max()),
        "gripper_maximum_absolute_error": float(stacked[:, 6].max()),
    }

"""带可见 layout axis、隐藏 operation 的旋转滑块任务。"""

from __future__ import annotations

import math

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks.tabletop.push_cube import PushCubeEnv
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose


class RotatedLayoutSlideEnv(PushCubeEnv):
    """operation 决定沿可见 object→anchor axis 的正向或反向推动。"""

    anchor_half_size = 0.012
    anchor_offset_m = 0.08
    maximum_axis_yaw_rad = math.pi / 3.0

    def __init__(self, *args, operation: str, **kwargs) -> None:
        if operation not in {"toward", "away"}:
            raise ValueError("operation 必须为 toward 或 away")
        self.operation = operation
        super().__init__(*args, **kwargs)

    @property
    def operation_direction(self) -> float:
        """toward anchor 为 +axis，away 为 -axis。"""
        return 1.0 if self.operation == "toward" else -1.0

    def _load_scene(self, options: dict) -> None:
        super()._load_scene(options)
        self.anchor = actors.build_cube(
            self.scene,
            half_size=self.anchor_half_size,
            color=np.asarray([0.15, 0.8, 0.2, 1.0]),
            name="layout_anchor",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(p=[0.08, 0.0, self.anchor_half_size]),
        )
        # goal 仅用于 success，不进入 RGB-D/pointcloud observation。
        self._hidden_objects.append(self.goal_region)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict) -> None:
        """同 seed/axis 的两种 operation 共享全部可见状态。"""
        options = options or {}
        with torch.device(self.device):
            batch = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((batch, 3))
            xyz[..., :2] = torch.rand((batch, 2)) * 0.1 - 0.05
            xyz[..., 2] = self.cube_half_size
            self.obj.set_pose(Pose.create_from_pq(p=xyz, q=[1, 0, 0, 0]))

            requested_yaw = options.get("axis_yaw_rad")
            if requested_yaw is None:
                yaw = (
                    torch.rand((batch,)) * 2.0 - 1.0
                ) * self.maximum_axis_yaw_rad
            else:
                yaw = torch.full((batch,), float(requested_yaw))
            axis = torch.stack(
                (torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)),
                dim=-1,
            )
            self.layout_axis = axis
            self.axis_yaw = yaw

            anchor_position = xyz + axis * self.anchor_offset_m
            anchor_position[..., 2] = self.anchor_half_size
            self.anchor.set_pose(
                Pose.create_from_pq(p=anchor_position, q=[1, 0, 0, 0])
            )

            goal_offset = self.operation_direction * (0.1 + self.goal_radius)
            goal_position = xyz + axis * goal_offset
            goal_position[..., 2] = 1e-3
            self.goal_region.set_pose(
                Pose.create_from_pq(
                    p=goal_position,
                    q=euler2quat(0, np.pi / 2, 0),
                )
            )

    def _get_obs_extra(self, info: dict) -> dict:
        """视觉模式不暴露 task/goal；state 模式只暴露可见物体。"""
        observation = {"tcp_pose": self.agent.tcp.pose.raw_pose}
        if self.obs_mode_struct.use_state:
            observation.update(
                obj_pose=self.obj.pose.raw_pose,
                anchor_pose=self.anchor.pose.raw_pose,
            )
        return observation


@register_env(
    "RotatedLayoutSlideToward-v0",
    max_episode_steps=150,
    operation="toward",
)
class RotatedLayoutSlideTowardEnv(RotatedLayoutSlideEnv):
    """隐藏 operation 为沿 object→anchor axis 推动。"""


@register_env(
    "RotatedLayoutSlideAway-v0",
    max_episode_steps=150,
    operation="away",
)
class RotatedLayoutSlideAwayEnv(RotatedLayoutSlideEnv):
    """隐藏 operation 为沿 anchor 反方向推动。"""

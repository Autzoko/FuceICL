"""共享观测、隐藏 operation 的 ManiSkill 双向滑块任务。"""

from __future__ import annotations

import numpy as np
import torch
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks.tabletop.push_cube import PushCubeEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose


class BidirectionalSlideCubeEnv(PushCubeEnv):
    """只改变隐藏目标方向。

    reset state 与可见 observation 不随 operation 改变。
    """

    def __init__(self, *args, operation: str, **kwargs) -> None:
        if operation not in {"left", "right"}:
            raise ValueError("operation 必须为 left 或 right")
        self.operation = operation
        super().__init__(*args, **kwargs)

    @property
    def operation_direction(self) -> float:
        """right 为 +x，left 为 -x。"""
        return 1.0 if self.operation == "right" else -1.0

    def _load_scene(self, options: dict) -> None:
        super()._load_scene(options)
        # goal 只用于 success/evaluation，不进入 RGB-D/pointcloud observation。
        self._hidden_objects.append(self.goal_region)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict) -> None:
        """执行相同随机调用，只在隐藏 goal pose 上分叉。"""
        with torch.device(self.device):
            batch = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((batch, 3))
            xyz[..., :2] = torch.rand((batch, 2)) * 0.2 - 0.1
            xyz[..., 2] = self.cube_half_size
            self.obj.set_pose(Pose.create_from_pq(p=xyz, q=[1, 0, 0, 0]))
            offset = self.operation_direction * (0.1 + self.goal_radius)
            goal = xyz + torch.tensor([offset, 0.0, 0.0])
            goal[..., 2] = 1e-3
            self.goal_region.set_pose(
                Pose.create_from_pq(
                    p=goal,
                    q=euler2quat(0, np.pi / 2, 0),
                )
            )

    def _get_obs_extra(self, info: dict) -> dict:
        """移除 goal/task 信息；视觉模式仅保留当前可观测状态。"""
        observation = {"tcp_pose": self.agent.tcp.pose.raw_pose}
        if self.obs_mode_struct.use_state:
            observation["obj_pose"] = self.obj.pose.raw_pose
        return observation


@register_env(
    "BidirectionalSlideLeft-v0",
    max_episode_steps=100,
    operation="left",
)
class BidirectionalSlideLeftEnv(BidirectionalSlideCubeEnv):
    """隐藏 operation 为向左滑动。"""


@register_env(
    "BidirectionalSlideRight-v0",
    max_episode_steps=100,
    operation="right",
)
class BidirectionalSlideRightEnv(BidirectionalSlideCubeEnv):
    """隐藏 operation 为向右滑动。"""

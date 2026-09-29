"""为自定义 BidirectionalSlide 环境启用 ManiSkill 官方 trajectory replay。"""

from __future__ import annotations

import multiprocessing as mp

# replay 根据 metadata 中的 env_id 创建环境，因此必须先完成注册。
from dev.simulator import bidirectional_slide_env  # noqa: F401
from mani_skill.trajectory.replay_trajectory import main, parse_args


if __name__ == "__main__":
    # 与官方入口保持一致，避免 Vulkan/Warp fork 后状态不安全。
    mp.set_start_method("spawn")
    main(parse_args())

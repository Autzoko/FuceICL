"""为 RotatedLayoutSlide 注册环境并调用 ManiSkill 官方 trajectory replay。"""

from __future__ import annotations

import multiprocessing as mp

from dev.simulator import rotated_layout_slide_env  # noqa: F401
from mani_skill.trajectory.replay_trajectory import main, parse_args


if __name__ == "__main__":
    # Vulkan/Warp 在 fork 后不安全，与 ManiSkill 官方入口保持 spawn。
    mp.set_start_method("spawn")
    main(parse_args())

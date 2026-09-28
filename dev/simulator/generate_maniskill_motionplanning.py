"""用显式 start seed 调用 ManiSkill 官方 motion-planning solver。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from mani_skill.examples.motionplanning.panda.run import (
    MP_SOLUTIONS,
    _main,
)


def run(
    *,
    env_id: str,
    output_root: Path,
    num_trajectories: int,
    start_seed: int,
    sim_backend: str,
    trajectory_name: str,
) -> Path:
    if env_id not in MP_SOLUTIONS:
        raise ValueError(f"ManiSkill 没有 {env_id} 的官方 motion planner")
    if num_trajectories <= 0 or start_seed < 0:
        raise ValueError("trajectory 数量必须为正，start seed 不能为负")
    if output_root.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output_root}")
    # 上游 parse_args 当前忽略其 args 参数，因此显式构造 _main 所需字段。
    arguments = SimpleNamespace(
        env_id=env_id,
        obs_mode="none",
        num_traj=num_trajectories,
        only_count_success=True,
        reward_mode=None,
        sim_backend=sim_backend,
        render_mode="rgb_array",
        vis=False,
        save_video=False,
        traj_name=trajectory_name,
        shader="default",
        record_dir=str(output_root),
        num_procs=1,
    )
    output = Path(_main(arguments, proc_id=0, start_seed=start_seed))
    print(
        json.dumps(
            {
                "env_id": env_id,
                "num_successful_trajectories": num_trajectories,
                "start_seed": start_seed,
                "trajectory_path": str(output),
            }
        ),
        flush=True,
    )
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-trajectories", type=int, required=True)
    parser.add_argument("--start-seed", type=int, required=True)
    parser.add_argument("--sim-backend", default="physx_cpu")
    parser.add_argument("--trajectory-name", default="trajectory")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        env_id=args.env_id,
        output_root=args.output_root.resolve(),
        num_trajectories=args.num_trajectories,
        start_seed=args.start_seed,
        sim_backend=args.sim_backend,
        trajectory_name=args.trajectory_name,
    )

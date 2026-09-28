"""在分配到的计算节点上验证 ManiSkill 环境与控制接口。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Any

import gymnasium as gym
import mani_skill  # noqa: F401  # 导入时注册 ManiSkill environments。
import numpy as np


def _git_commit(project_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _tree_signature(value: Any) -> Any:
    """只记录 observation 的结构、shape 和 dtype，不写入大数组。"""
    if isinstance(value, dict):
        return {key: _tree_signature(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_tree_signature(item) for item in value]
    if hasattr(value, "shape"):
        return {
            "shape": list(value.shape),
            "dtype": str(getattr(value, "dtype", "unknown")),
        }
    return {"type": type(value).__name__}


def _scalar_summary(value: Any) -> Any:
    """把 info 中的小型标量转为 JSON；数组只记录范围，避免日志膨胀。"""
    if isinstance(value, dict):
        return {key: _scalar_summary(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_scalar_summary(item) for item in value]
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.size <= 8:
            return value.tolist()
        return {
            "shape": list(value.shape),
            "min": float(np.min(value)),
            "max": float(np.max(value)),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def run(
    *,
    project_root: Path,
    output_path: Path,
    env_id: str,
    obs_mode: str,
    control_mode: str,
    sim_backend: str,
    seed: int,
    num_steps: int,
) -> None:
    """执行确定性的零动作 reset/step，并保存可审计报告。"""
    if output_path.exists():
        raise FileExistsError(f"输出已存在，拒绝覆盖：{output_path}")
    if num_steps <= 0:
        raise ValueError("num_steps 必须为正")

    started = time.perf_counter()
    environment = gym.make(
        env_id,
        obs_mode=obs_mode,
        control_mode=control_mode,
        sim_backend=sim_backend,
        num_envs=1,
    )
    try:
        observation, reset_info = environment.reset(seed=seed)
        initial_signature = _tree_signature(observation)
        action = np.zeros(environment.action_space.shape, dtype=np.float32)
        step_records: list[dict[str, Any]] = []
        for step in range(num_steps):
            observation, reward, terminated, truncated, info = environment.step(action)
            step_records.append(
                {
                    "step": step,
                    "reward": _scalar_summary(reward),
                    "terminated": _scalar_summary(terminated),
                    "truncated": _scalar_summary(truncated),
                    "info": _scalar_summary(info),
                }
            )
        final_signature = _tree_signature(observation)
    finally:
        environment.close()

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_commit": _git_commit(project_root),
        "hostname": platform.node(),
        "pid": os.getpid(),
        "versions": {
            "mani_skill": mani_skill.__version__,
            "gymnasium": gym.__version__,
            "numpy": np.__version__,
        },
        "environment": {
            "env_id": env_id,
            "obs_mode": obs_mode,
            "control_mode": control_mode,
            "sim_backend": sim_backend,
            "seed": seed,
            "num_steps": num_steps,
            "action_space": str(environment.action_space),
            "observation_space": str(environment.observation_space),
        },
        "reset_info": _scalar_summary(reset_info),
        "initial_observation": initial_signature,
        "final_observation": final_signature,
        "steps": step_records,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(f"{output_path.suffix}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(output_path)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-id", default="PickCube-v1")
    parser.add_argument("--obs-mode", default="state")
    parser.add_argument("--control-mode", default="pd_ee_delta_pose")
    parser.add_argument("--sim-backend", default="physx_cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    run(
        project_root=arguments.project_root.resolve(),
        output_path=arguments.output.resolve(),
        env_id=arguments.env_id,
        obs_mode=arguments.obs_mode,
        control_mode=arguments.control_mode,
        sim_backend=arguments.sim_backend,
        seed=arguments.seed,
        num_steps=arguments.num_steps,
    )


if __name__ == "__main__":
    main()

"""审计 observation-derived action 与 ManiSkill controller command 的保真度。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any, Mapping, Sequence

import gymnasium as gym
import h5py
import mani_skill
import numpy as np
import sapien
from mani_skill.trajectory import utils as trajectory_utils

from dev.predictor.action_chunk_data import matrix_to_axis_angle
from dev.simulator.evaluate_maniskill_closed_loop import (
    _audit_runtime_controller,
)
from dev.simulator.maniskill_action_bridge import (
    audit_round_trip,
    canonical_to_controller,
)
from dev.simulator.preprocess_maniskill_chunks import (
    _gripper_open,
    _quaternion_wxyz_to_matrix,
)


POLICIES = (
    "raw_controller_action",
    "observed_delta_raw_gripper",
    "observed_delta_state_gripper",
    "online_pose_tracking_raw_gripper",
)


@dataclass(frozen=True)
class FidelityConfig:
    """控制保真度审计的固定协议。"""

    schema_version: str
    episode_ids: tuple[int, ...]
    policies: tuple[str, ...]
    position_limit_m: float
    rotation_scale_rad: float

    @classmethod
    def from_json(cls, path: Path) -> "FidelityConfig":
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["episode_ids"] = tuple(payload["episode_ids"])
        payload["policies"] = tuple(payload["policies"])
        return cls(**payload)

    def __post_init__(self) -> None:
        if not self.schema_version.strip() or not self.episode_ids:
            raise ValueError("schema version 和 episode IDs 不能为空")
        if len(set(self.episode_ids)) != len(self.episode_ids):
            raise ValueError("episode IDs 不能重复")
        if self.policies != POLICIES:
            raise ValueError(f"action fidelity policies 必须固定为 {POLICIES}")
        if self.position_limit_m <= 0 or self.rotation_scale_rad == 0:
            raise ValueError("controller scale 非法")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _single(value: Any) -> np.ndarray:
    array = _to_numpy(value)
    return array[0] if array.ndim > 1 and array.shape[0] == 1 else array


def _scalar_bool(value: Any) -> bool:
    array = _to_numpy(value)
    return bool(array.reshape(-1)[0])


def _tcp_pose(observation: Mapping[str, Any]) -> np.ndarray:
    pose = _single(observation["extra"]["tcp_pose"]).astype(np.float64)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError(f"online TCP pose 非法：{pose.shape}")
    return pose


def _canonical_to_target(
    current_pose: np.ndarray,
    target_pose: np.ndarray,
    gripper_open: bool,
) -> np.ndarray:
    """把当前 TCP 到目标 TCP 的 SE(3) 差写到当前 EEF 坐标系。"""
    current = np.asarray(current_pose, dtype=np.float64)
    target = np.asarray(target_pose, dtype=np.float64)
    current_rotation = _quaternion_wxyz_to_matrix(current[3:7])
    target_rotation = _quaternion_wxyz_to_matrix(target[3:7])
    action = np.empty(7, dtype=np.float32)
    action[:3] = current_rotation.T @ (target[:3] - current[:3])
    action[3:6] = matrix_to_axis_angle(
        current_rotation.T @ target_rotation
    )
    action[6] = float(gripper_open)
    return action


def _pose_error(current_pose: np.ndarray, target_pose: np.ndarray) -> tuple[float, float]:
    current = np.asarray(current_pose, dtype=np.float64)
    target = np.asarray(target_pose, dtype=np.float64)
    translation = float(np.linalg.norm(current[:3] - target[:3]))
    current_rotation = _quaternion_wxyz_to_matrix(current[3:7])
    target_rotation = _quaternion_wxyz_to_matrix(target[3:7])
    rotation = float(
        np.linalg.norm(matrix_to_axis_angle(current_rotation.T @ target_rotation))
    )
    return translation, rotation


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> list[float]:
    denominator_norm = np.linalg.norm(denominator, axis=1)
    valid = denominator_norm > 1e-6
    return (
        np.linalg.norm(numerator[valid], axis=1) / denominator_norm[valid]
    ).tolist()


def _preprocess_raw_controller_actions(actions: np.ndarray) -> np.ndarray:
    """复现 ManiSkill normalized controller 对 source action 的裁剪。"""
    processed = np.asarray(actions, dtype=np.float64).copy()
    processed[:, :3] = np.clip(processed[:, :3], -1.0, 1.0)
    rotation_norm = np.linalg.norm(processed[:, 3:6], axis=1)
    clipped = rotation_norm > 1.0
    processed[clipped, 3:6] /= rotation_norm[clipped, None]
    processed[:, 6] = np.clip(processed[:, 6], -1.0, 1.0)
    return processed


def _stats(values: Sequence[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(array),
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def _offline_command_audit(
    actions: np.ndarray,
    tcp_poses: np.ndarray,
    qpos: np.ndarray,
    config: FidelityConfig,
) -> dict[str, Any]:
    reconstructed = []
    raw = []
    state_gripper = []
    raw_gripper = []
    limit = min(len(actions), len(tcp_poses) - 1)
    for step in range(limit):
        gripper = bool(actions[step, 6] >= 0.0)
        canonical = _canonical_to_target(
            tcp_poses[step], tcp_poses[step + 1], gripper
        )
        converted = canonical_to_controller(
            canonical,
            tcp_poses[step],
            position_limit_m=config.position_limit_m,
            rotation_scale_rad=config.rotation_scale_rad,
        )
        reconstructed.append(converted.value)
        raw.append(actions[step])
        state_gripper.append(_gripper_open(qpos[step + 1]) >= 0.5)
        raw_gripper.append(gripper)
    observed = np.asarray(reconstructed, dtype=np.float64)
    raw_controller = np.asarray(raw, dtype=np.float64)
    controller = _preprocess_raw_controller_actions(raw_controller)
    return {
        "steps": limit,
        "raw_action_min": raw_controller.min(axis=0).tolist(),
        "raw_action_max": raw_controller.max(axis=0).tolist(),
        "raw_translation_clip_rate": float(
            np.mean(np.any(np.abs(raw_controller[:, :3]) > 1.0, axis=1))
        ),
        "raw_rotation_clip_rate": float(
            np.mean(np.linalg.norm(raw_controller[:, 3:6], axis=1) > 1.0)
        ),
        "preprocessed_controller_mae": (
            np.abs(observed - controller).mean(axis=0).tolist()
        ),
        "translation_norm_ratio": _stats(
            _ratio(observed[:, :3], controller[:, :3])
        ),
        "rotation_norm_ratio": _stats(
            _ratio(observed[:, 3:6], controller[:, 3:6])
        ),
        "state_gripper_matches_raw": float(
            np.mean(np.asarray(state_gripper) == np.asarray(raw_gripper))
        ),
    }


def _policy_action(
    policy: str,
    step: int,
    online_pose: np.ndarray,
    raw_actions: np.ndarray,
    logged_tcp_poses: np.ndarray,
    logged_qpos: np.ndarray,
    config: FidelityConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    raw = np.asarray(raw_actions[step], dtype=np.float32)
    if policy == "raw_controller_action":
        return raw, {
            "translation_clipped": bool(np.any(np.abs(raw[:3]) > 1.0)),
            "rotation_clipped": bool(np.linalg.norm(raw[3:6]) > 1.0),
            "canonical_action": None,
        }
    target_index = min(step + 1, len(logged_tcp_poses) - 1)
    raw_gripper = bool(raw[6] >= 0.0)
    if policy == "observed_delta_state_gripper":
        gripper = _gripper_open(logged_qpos[target_index]) >= 0.5
    else:
        gripper = raw_gripper
    if policy in (
        "observed_delta_raw_gripper",
        "observed_delta_state_gripper",
    ):
        canonical = _canonical_to_target(
            logged_tcp_poses[step],
            logged_tcp_poses[target_index],
            gripper,
        )
    elif policy == "online_pose_tracking_raw_gripper":
        canonical = _canonical_to_target(
            online_pose,
            logged_tcp_poses[target_index],
            gripper,
        )
    else:
        raise ValueError(f"未知 fidelity policy：{policy}")
    converted = canonical_to_controller(
        canonical,
        online_pose,
        position_limit_m=config.position_limit_m,
        rotation_scale_rad=config.rotation_scale_rad,
    )
    return converted.value, {
        "translation_clipped": converted.translation_clipped,
        "rotation_clipped": converted.rotation_clipped,
        "canonical_action": canonical.tolist(),
    }


def _summarize(rollouts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    steps = [step for rollout in rollouts for step in rollout["step_records"]]
    return {
        "episodes": len(rollouts),
        "successes": sum(bool(row["success"]) for row in rollouts),
        "success_rate": float(np.mean([bool(row["success"]) for row in rollouts])),
        "translation_clip_rate": float(
            np.mean([bool(row["translation_clipped"]) for row in steps])
        ),
        "rotation_clip_rate": float(
            np.mean([bool(row["rotation_clipped"]) for row in steps])
        ),
        "tcp_translation_error_m": _stats(
            [float(row["tcp_translation_error_m"]) for row in steps]
        ),
        "tcp_rotation_error_rad": _stats(
            [float(row["tcp_rotation_error_rad"]) for row in steps]
        ),
        "max_reward": _stats([float(row["reward"]) for row in steps]),
    }


def run(
    *,
    project_root: Path,
    h5_path: Path,
    metadata_path: Path,
    config_path: Path,
    output_path: Path,
    config: FidelityConfig,
) -> None:
    if output_path.exists():
        raise FileExistsError(f"输出已存在，拒绝覆盖：{output_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    episodes = {int(row["episode_id"]): row for row in metadata["episodes"]}
    missing = set(config.episode_ids) - set(episodes)
    if missing:
        raise ValueError(f"metadata 缺少 episodes：{sorted(missing)}")
    for episode_id in config.episode_ids:
        episode = episodes[episode_id]
        if not episode.get("success", False):
            raise ValueError(f"episode {episode_id} 不是成功 Demo")
        if episode.get("control_mode") != "pd_ee_delta_pose":
            raise ValueError(f"episode {episode_id} control mode 不匹配")

    environment = gym.make(
        "PickCube-v1",
        obs_mode="pointcloud",
        control_mode="pd_ee_delta_pose",
        sim_backend="physx_cpu",
        render_backend="gpu",
        num_envs=1,
        reconfiguration_freq=1,
    )
    rollouts: dict[str, list[dict[str, Any]]] = {
        policy: [] for policy in config.policies
    }
    offline_audits = []
    initial_pose_max_abs_difference = 0.0
    try:
        runtime_controller = _audit_runtime_controller(environment, config)
        round_trip = audit_round_trip(
            position_limit_m=config.position_limit_m,
            rotation_scale_rad=config.rotation_scale_rad,
        )
        with h5py.File(h5_path, "r") as handle:
            for episode_id in config.episode_ids:
                group = handle[f"traj_{episode_id}"]
                raw_actions = np.asarray(group["actions"], dtype=np.float32)
                logged_tcp = np.asarray(group["obs/extra/tcp_pose"], dtype=np.float64)
                logged_qpos = np.asarray(group["obs/agent/qpos"], dtype=np.float64)
                initial_env_state = trajectory_utils.index_dict(
                    group["env_states"], 0
                )
                if raw_actions.ndim != 2 or raw_actions.shape[1] != 7:
                    raise ValueError(f"episode {episode_id} action shape 非法")
                if not np.isfinite(raw_actions).all():
                    raise ValueError(f"episode {episode_id} controller action 非法")
                if len(logged_tcp) != len(logged_qpos):
                    raise ValueError(f"episode {episode_id} TCP/qpos 长度不一致")
                if len(raw_actions) not in (len(logged_tcp), len(logged_tcp) - 1):
                    raise ValueError(f"episode {episode_id} action/obs 长度不一致")
                offline_audits.append(
                    {
                        "episode_id": episode_id,
                        **_offline_command_audit(
                            raw_actions, logged_tcp, logged_qpos, config
                        ),
                    }
                )
                episode = episodes[episode_id]
                seed = int(episode["episode_seed"])
                for policy in config.policies:
                    environment.reset(seed=seed)
                    environment.unwrapped.set_state_dict(initial_env_state)
                    observation = environment.unwrapped.get_obs()
                    initial_pose = _tcp_pose(observation)
                    initial_difference = float(
                        np.max(np.abs(initial_pose - logged_tcp[0]))
                    )
                    initial_pose_max_abs_difference = max(
                        initial_pose_max_abs_difference, initial_difference
                    )
                    if initial_difference > 5e-3:
                        raise ValueError(
                            f"episode {episode_id} 初始 TCP 与 replay 差异 "
                            f"{initial_difference:.4g}"
                        )
                    initial_info = environment.unwrapped.evaluate()
                    success = _scalar_bool(initial_info.get("success", False))
                    success_step = 0 if success else None
                    records = []
                    for step in range(len(raw_actions)):
                        online_pose = _tcp_pose(observation)
                        action, diagnostic = _policy_action(
                            policy,
                            step,
                            online_pose,
                            raw_actions,
                            logged_tcp,
                            logged_qpos,
                            config,
                        )
                        observation, reward, terminated, truncated, info = (
                            environment.step(action)
                        )
                        target_index = min(step + 1, len(logged_tcp) - 1)
                        translation_error, rotation_error = _pose_error(
                            _tcp_pose(observation), logged_tcp[target_index]
                        )
                        step_success = _scalar_bool(info.get("success", False))
                        records.append(
                            {
                                "step": step + 1,
                                "reward": float(_to_numpy(reward).reshape(-1)[0]),
                                "success": step_success,
                                "tcp_translation_error_m": translation_error,
                                "tcp_rotation_error_rad": rotation_error,
                                "controller_action": action.tolist(),
                                **diagnostic,
                            }
                        )
                        if step_success:
                            success = True
                            success_step = step + 1
                            break
                        if _scalar_bool(terminated) or _scalar_bool(truncated):
                            break
                    row = {
                        "episode_id": episode_id,
                        "seed": seed,
                        "policy": policy,
                        "success": success,
                        "success_step": success_step,
                        "steps": len(records),
                        "step_records": records,
                    }
                    rollouts[policy].append(row)
                    print(
                        json.dumps(
                            {
                                key: row[key]
                                for key in (
                                    "episode_id",
                                    "seed",
                                    "policy",
                                    "success",
                                    "success_step",
                                    "steps",
                                )
                            }
                        ),
                        flush=True,
                    )
    finally:
        environment.close()

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "purpose": "separate controller/action abstraction error from retrieval error",
            "raw": "execute source HDF5 controller actions",
            "observed_delta": "execute adjacent logged TCP pose delta",
            "tracking": "track next logged TCP pose from current online pose",
            "selection": "first four successful source episodes; fixed before execution",
            "initialization": "seed reset followed by exact HDF5 env_states[0]",
        },
        "config": asdict(config),
        "git_commit": _git_commit(project_root),
        "hostname": platform.node(),
        "versions": {
            "mani_skill": mani_skill.__version__,
            "gymnasium": gym.__version__,
            "numpy": np.__version__,
            "sapien": sapien.__version__,
        },
        "h5_sha256": _sha256(h5_path),
        "metadata_sha256": _sha256(metadata_path),
        "config_sha256": _sha256(config_path),
        "runtime_controller": runtime_controller,
        "controller_round_trip": round_trip,
        "initial_tcp_pose_max_abs_difference": initial_pose_max_abs_difference,
        "offline_command_audits": offline_audits,
        "policy_summaries": {
            policy: _summarize(rows) for policy, rows in rollouts.items()
        },
        "rollouts": rollouts,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(f"{output_path.suffix}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    print(
        json.dumps(
            {
                "policy_summaries": report["policy_summaries"],
                "output": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    run(
        project_root=arguments.project_root.resolve(),
        h5_path=arguments.h5.resolve(),
        metadata_path=arguments.metadata.resolve(),
        config_path=arguments.config.resolve(),
        output_path=arguments.output.resolve(),
        config=FidelityConfig.from_json(arguments.config.resolve()),
    )


if __name__ == "__main__":
    main()

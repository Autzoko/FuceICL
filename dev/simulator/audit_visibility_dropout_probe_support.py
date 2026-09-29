"""审计 branch 后确定性 visibility dropout 能否产生非空风险 probes。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import h5py
import numpy as np

from dev.simulator.preprocess_bidirectional_slide_predictor import (
    _metadata_by_seed,
    _trajectory,
)
from dev.simulator.preprocess_rotated_layout_slide_predictor import (
    OPERATIONS,
    _sha256,
)
from dev.simulator.preprocess_rotated_layout_slide_progress_chunks import (
    _paths,
)
from dev.simulator.visibility_dropout_probe import (
    select_visibility_dropout_probes,
)


@dataclass(frozen=True)
class VisibilityDropoutSupportConfig:
    """不读取 actor truth 的 visibility-dropout support 诊断协议。"""

    schema_version: str
    expected_pairs: int
    action_horizon: int
    maximum_accepted_tcp_net_displacement_m: float
    minimum_positive_tcp_net_displacement_m: float

    @classmethod
    def from_json(cls, path: Path) -> "VisibilityDropoutSupportConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def __post_init__(self) -> None:
        if self.schema_version != "visibility-dropout-support-audit-v1":
            raise ValueError("未知 visibility dropout support schema")
        if self.expected_pairs <= 0 or self.action_horizon <= 0:
            raise ValueError("pair 数量与 action horizon 必须为正")
        if self.maximum_accepted_tcp_net_displacement_m <= 0.0:
            raise ValueError("risk threshold 必须为正")
        if not (
            0.0 < self.minimum_positive_tcp_net_displacement_m
            < self.maximum_accepted_tcp_net_displacement_m
        ):
            raise ValueError("minimum positive displacement 非法")


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def run(
    *,
    project_root: Path,
    config_path: Path,
    replay_root: Path,
    output_path: Path,
    config: VisibilityDropoutSupportConfig,
) -> None:
    if output_path.exists():
        raise FileExistsError(f"输出已存在，拒绝覆盖：{output_path}")
    audit_path = replay_root / "replay_audit_report.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("summary", {}).get("all_criteria_passed", False):
        raise ValueError("replay audit 未通过")
    pairs = audit.get("pairs", [])
    if len(pairs) != config.expected_pairs:
        raise ValueError("replay pair 数量与配置不一致")
    paths = _paths(replay_root)
    episodes = {
        operation: _metadata_by_seed(paths[operation]["json"])
        for operation in OPERATIONS
    }
    handles = {
        operation: h5py.File(paths[operation]["h5"], "r")
        for operation in OPERATIONS
    }
    rows = []
    try:
        for pair in pairs:
            for operation in OPERATIONS:
                seed = int(pair["seed"])
                trajectory = _trajectory(
                    handles[operation], episodes[operation][seed]
                )
                tcp_positions = np.asarray(
                    trajectory["obs/extra/tcp_pose"][:, :3],
                    dtype=np.float64,
                )
                action_steps = len(trajectory["actions"])
                branch = int(pair["replay_branch_frame"])
                probes = select_visibility_dropout_probes(
                    tcp_positions,
                    branch=branch,
                    last_h6_start=action_steps - config.action_horizon,
                    accepted_limit_m=(
                        config.maximum_accepted_tcp_net_displacement_m
                    ),
                    minimum_positive_m=(
                        config.minimum_positive_tcp_net_displacement_m
                    ),
                )
                rows.append(
                    {
                        "pair_id": int(pair["pair_id"]),
                        "seed": seed,
                        "operation": operation,
                        "branch_frame": branch,
                        **probes,
                    }
                )
    finally:
        for handle in handles.values():
            handle.close()

    accepted = sum(row["accepted_probe"] is not None for row in rows)
    rejected = sum(row["rejected_probe"] is not None for row in rows)
    both = sum(
        row["accepted_probe"] is not None
        and row["rejected_probe"] is not None
        for row in rows
    )
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(project_root),
        "config": asdict(config),
        "config_sha256": _sha256(config_path),
        "replay_audit_sha256": _sha256(audit_path),
        "protocol": {
            "dropout_start": "first frame after replay branch",
            "accepted_probe": (
                "largest TCP net displacement <= limit before first rejection"
            ),
            "rejected_probe": "first TCP net displacement > limit",
            "actor_state_used": False,
            "segmentation_or_visibility_used": False,
            "future_action_or_reward_used": False,
        },
        "summary": {
            "trajectories": len(rows),
            "accepted_probe_trajectories": accepted,
            "rejected_probe_trajectories": rejected,
            "both_probe_trajectories": both,
            "accepted_probe_fraction": accepted / len(rows),
            "rejected_probe_fraction": rejected / len(rows),
            "both_probe_fraction": both / len(rows),
        },
        "rows": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(f"{output_path.suffix}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    resolved_config = arguments.config.resolve()
    run(
        project_root=arguments.project_root.resolve(),
        config_path=resolved_config,
        replay_root=arguments.replay_root.resolve(),
        output_path=arguments.output.resolve(),
        config=VisibilityDropoutSupportConfig.from_json(resolved_config),
    )

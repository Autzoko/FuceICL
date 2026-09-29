"""审计冻结 progress schedule 的 belief 误差与因果风险特征。"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np

from dev.simulator.evaluate_dual_view_observability import _camera_variants
from dev.simulator.preprocess_bidirectional_slide_predictor import (
    _metadata_by_seed,
    _trajectory,
)
from dev.simulator.preprocess_maniskill_chunks import _infer_actor_label
from dev.simulator.preprocess_rotated_layout_slide_belief_chunks import (
    BeliefChunkConfig,
    _active_beliefs,
    _belief_frames,
    _git_commit,
)
from dev.simulator.preprocess_rotated_layout_slide_predictor import (
    OPERATIONS,
    _sha256,
)
from dev.simulator.preprocess_rotated_layout_slide_progress_chunks import _paths


def _summary(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "maximum": None,
        }
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": float(np.max(array)),
    }


def _group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    visible = [row for row in rows if row["active_visible"]]
    missing = [row for row in rows if not row["active_visible"]]
    return {
        "rows": len(rows),
        "visible_rows": len(visible),
        "missing_rows": len(missing),
        "belief_error_m": _summary(row["belief_error_m"] for row in rows),
        "visible_error_m": _summary(
            row["belief_error_m"] for row in visible
        ),
        "missing_error_m": _summary(
            row["belief_error_m"] for row in missing
        ),
        "missing_age_steps": _summary(row["belief_age_steps"] for row in missing),
        "missing_tcp_net_displacement_m": _summary(
            row["tcp_net_displacement_since_visible_m"] for row in missing
        ),
        "missing_tcp_path_length_m": _summary(
            row["tcp_path_length_since_visible_m"] for row in missing
        ),
    }


def run(
    *,
    project_root: Path,
    config_path: Path,
    replay_root: Path,
    output_path: Path,
    config: BeliefChunkConfig,
) -> None:
    if output_path.exists():
        raise FileExistsError(f"输出已存在，拒绝覆盖：{output_path}")
    audit_path = replay_root / "replay_audit_report.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("summary", {}).get("all_criteria_passed", False):
        raise ValueError("replay audit 未通过")
    pair_rows = audit.get("pairs", [])
    if len(pair_rows) != config.expected_pairs:
        raise ValueError("replay audit pair 数量不匹配")
    camera_variants = _camera_variants(replay_root)
    if not all(
        value == config.expected_camera_variant
        for operations in camera_variants.values()
        for value in operations.values()
    ):
        raise ValueError("replay camera metadata 不匹配")

    paths = _paths(replay_root)
    episodes = {
        operation: _metadata_by_seed(paths[operation]["json"])
        for operation in OPERATIONS
    }
    handles = {
        operation: h5py.File(paths[operation]["h5"], "r")
        for operation in OPERATIONS
    }
    rows: list[dict[str, Any]] = []
    try:
        for pair in pair_rows:
            pair_id = int(pair["pair_id"])
            seed = int(pair["seed"])
            branch = int(pair["replay_branch_frame"])
            for operation in OPERATIONS:
                trajectory = _trajectory(handles[operation], episodes[operation][seed])
                xyzw = np.asarray(trajectory["obs/pointcloud/xyzw"])
                segmentation = np.asarray(
                    trajectory["obs/pointcloud/segmentation"]
                )
                actor_positions = np.asarray(
                    trajectory["env_states/actors/cube"][:, :3],
                    dtype=np.float64,
                )
                tcp_positions = np.asarray(
                    trajectory["obs/extra/tcp_pose"][:, :3],
                    dtype=np.float64,
                )
                actions = np.asarray(trajectory["actions"])
                label, _ = _infer_actor_label(
                    xyzw=xyzw,
                    segmentation=segmentation,
                    actor_positions=actor_positions,
                    config=config,
                    role="active",
                )
                frames = _belief_frames(branch, len(actions), config)
                beliefs = _active_beliefs(
                    xyzw=xyzw,
                    segmentation=segmentation,
                    label=label,
                    tcp_positions=tcp_positions,
                    branch=branch,
                    frames=frames,
                    minimum_points=config.minimum_label_points,
                )
                for progress_index, frame in enumerate(frames):
                    belief = beliefs[frame]
                    error = float(
                        np.linalg.norm(
                            belief.center_world - actor_positions[frame]
                        )
                    )
                    limit = (
                        config.maximum_centroid_error_m
                        if belief.visible
                        else config.maximum_belief_error_m
                    )
                    rows.append(
                        {
                            "pair_id": pair_id,
                            "seed": seed,
                            "operation": operation,
                            "progress_index": progress_index,
                            "frame": frame,
                            "active_visible": belief.visible,
                            "belief_age_steps": belief.age_steps,
                            "observed_active_points": (
                                belief.observed_point_count
                            ),
                            "tcp_net_displacement_since_visible_m": (
                                belief.tcp_net_displacement_since_visible_m
                            ),
                            "tcp_path_length_since_visible_m": (
                                belief.tcp_path_length_since_visible_m
                            ),
                            "belief_error_m": error,
                            "error_limit_m": limit,
                            "within_limit": error <= limit,
                        }
                    )
    finally:
        for handle in handles.values():
            handle.close()

    expected = config.expected_pairs * len(OPERATIONS) * config.progress_samples
    if len(rows) != expected:
        raise ValueError("belief schedule audit rows 数量不匹配")
    violations = [row for row in rows if not row["within_limit"]]
    by_progress = {
        str(index): _group(
            [row for row in rows if row["progress_index"] == index]
        )
        for index in range(config.progress_samples)
    }
    by_age: dict[str, Any] = {}
    grouped_age: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not row["active_visible"]:
            grouped_age[int(row["belief_age_steps"])].append(row)
    for age, selected in sorted(grouped_age.items()):
        by_age[str(age)] = _group(selected)

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(project_root),
        "config": asdict(config),
        "config_sha256": _sha256(config_path),
        "replay_audit_sha256": _sha256(audit_path),
        "camera_variants": camera_variants,
        "protocol": {
            "privileged_fields_audit_only": ["active actor position"],
            "causal_risk_features": [
                "visibility",
                "belief age",
                "TCP net displacement since visible",
                "TCP path length since visible",
            ],
            "threshold_selection_performed": False,
        },
        "summary": _group(rows),
        "violations": violations,
        "violation_count": len(violations),
        "violation_fraction": len(violations) / len(rows),
        "by_progress": by_progress,
        "by_missing_age": by_age,
        "rows": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(f"{output_path.suffix}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    print(
        json.dumps(
            {
                "summary": report["summary"],
                "violation_count": len(violations),
                "violations": violations,
                "by_missing_age": by_age,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


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
        config=BeliefChunkConfig.from_json(resolved_config),
    )

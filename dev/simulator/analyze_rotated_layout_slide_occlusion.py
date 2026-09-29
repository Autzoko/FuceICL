"""审计 RotatedLayoutSlide 的单相机遮挡与最小因果状态估计。"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Iterable

import h5py
import numpy as np

from dev.simulator.preprocess_bidirectional_slide_predictor import (
    _metadata_by_seed,
    _trajectory,
)
from dev.simulator.preprocess_maniskill_chunks import (
    _infer_actor_label,
    _segmented_points,
)
from dev.simulator.preprocess_rotated_layout_slide_predictor import (
    OPERATIONS,
    _sha256,
)
from dev.simulator.preprocess_rotated_layout_slide_progress_chunks import (
    _paths,
)


@dataclass(frozen=True)
class OcclusionAuditConfig:
    """冻结的遮挡诊断协议；actor pose 只用于离线误差审计。"""

    schema_version: str
    expected_pairs: int
    action_horizon: int
    progress_samples: int
    label_probe_frames: int
    minimum_label_points: int
    maximum_centroid_error_m: float
    audit_motion_threshold_m: float
    causal_history_frames: int | None = None

    @classmethod
    def from_json(cls, path: Path) -> "OcclusionAuditConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def __post_init__(self) -> None:
        schemas = {
            "rotated-layout-slide-occlusion-audit-v1",
            "rotated-layout-slide-causal-switch-audit-v1",
        }
        if self.schema_version not in schemas:
            raise ValueError("未知 occlusion audit schema")
        positive = (
            self.expected_pairs,
            self.action_horizon,
            self.progress_samples,
            self.label_probe_frames,
            self.minimum_label_points,
            self.maximum_centroid_error_m,
            self.audit_motion_threshold_m,
        )
        if min(positive) <= 0 or self.progress_samples < 2:
            raise ValueError("occlusion audit 配置非法")
        if self.schema_version.endswith("causal-switch-audit-v1"):
            if self.causal_history_frames is None:
                raise ValueError("causal switch 缺少 history frames")
        if (
            self.causal_history_frames is not None
            and self.causal_history_frames <= 0
        ):
            raise ValueError("causal history frames 必须为正数")


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _summary(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if len(array) == 0:
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


def _missing_runs(visible: np.ndarray) -> list[tuple[int, int]]:
    """返回局部索引下闭区间的连续不可见区段。"""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(visible.tolist()):
        if not value and start is None:
            start = index
        elif value and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(visible) - 1))
    return runs


def _fixed_frames(branch: int, last: int, count: int) -> np.ndarray:
    frames = np.rint(np.linspace(branch, last, count)).astype(np.int64)
    if len(np.unique(frames)) != count:
        raise ValueError("trajectory 太短，固定进度帧出现重复")
    return frames


def _tracker_audit(
    *,
    centers: list[np.ndarray | None],
    actor_positions: np.ndarray,
    tcp_positions: np.ndarray,
    motion_threshold_m: float,
) -> dict[str, list[float]]:
    """比较零参数 hold 与 TCP delta 传播；真值只参与误差计算。"""
    metrics: dict[str, list[float]] = defaultdict(list)
    hold: np.ndarray | None = None
    propagated: np.ndarray | None = None
    last_visible: int | None = None
    for index, center in enumerate(centers):
        if center is not None:
            hold = center.copy()
            propagated = center.copy()
            last_visible = index
            continue
        if hold is None or propagated is None or last_visible is None:
            raise ValueError("因果 tracker 在首次有效观测前遇到遮挡")
        propagated += tcp_positions[index] - tcp_positions[index - 1]
        truth = actor_positions[index]
        hold_error = float(np.linalg.norm(hold - truth))
        propagated_error = float(np.linalg.norm(propagated - truth))
        actor_delta = truth - actor_positions[last_visible]
        tcp_delta = tcp_positions[index] - tcp_positions[last_visible]
        actor_motion = float(np.linalg.norm(actor_delta))
        tcp_motion = float(np.linalg.norm(tcp_delta))
        metrics["hold_error_m"].append(hold_error)
        metrics["tcp_delta_error_m"].append(propagated_error)
        metrics["tcp_minus_hold_error_m"].append(
            propagated_error - hold_error
        )
        metrics["actor_motion_since_visible_m"].append(actor_motion)
        metrics["tcp_motion_since_visible_m"].append(tcp_motion)
        if actor_motion >= motion_threshold_m:
            metrics["moving_hold_error_m"].append(hold_error)
            metrics["moving_tcp_delta_error_m"].append(propagated_error)
        if actor_motion > 1e-12 and tcp_motion > 1e-12:
            metrics["actor_tcp_displacement_cosine"].append(
                float(np.dot(actor_delta, tcp_delta) / (actor_motion * tcp_motion))
            )
    return dict(metrics)


def _causal_switch_audit(
    *,
    centers: list[np.ndarray | None],
    actor_positions: np.ndarray,
    tcp_positions: np.ndarray,
    history_frames: int,
) -> dict[str, list[float]]:
    """用遮挡前的反事实预测误差选择 hold 或 TCP propagation。"""
    metrics: dict[str, list[float]] = defaultdict(list)
    visible = np.asarray([center is not None for center in centers])
    for start, end in _missing_runs(visible):
        if start == 0:
            raise ValueError("因果 switch 在首次有效观测前遇到遮挡")
        last_visible = start - 1
        history_start = last_visible - history_frames
        history_available = history_start >= 0 and all(
            center is not None
            for center in centers[history_start : last_visible + 1]
        )
        use_tcp = False
        if history_available:
            first_center = centers[history_start]
            last_center = centers[last_visible]
            if first_center is None or last_center is None:
                raise AssertionError("已通过 history availability 检查")
            hold_history_error = float(
                np.linalg.norm(last_center - first_center)
            )
            tcp_history_error = float(
                np.linalg.norm(
                    first_center
                    + tcp_positions[last_visible]
                    - tcp_positions[history_start]
                    - last_center
                )
            )
            use_tcp = tcp_history_error < hold_history_error
            metrics["history_tcp_minus_hold_error_m"].append(
                tcp_history_error - hold_history_error
            )
        metrics["causal_tcp_selected"].append(float(use_tcp))
        metrics["causal_history_available"].append(float(history_available))

        last_center = centers[last_visible]
        if last_center is None:
            raise AssertionError("missing run 前一帧必须可见")
        run_candidates: list[tuple[float, float]] = []
        for index in range(start, end + 1):
            truth = actor_positions[index]
            hold = last_center
            propagated = (
                last_center
                + tcp_positions[index]
                - tcp_positions[last_visible]
            )
            hold_error = float(np.linalg.norm(hold - truth))
            tcp_error = float(np.linalg.norm(propagated - truth))
            run_candidates.append((hold_error, tcp_error))
            causal_error = tcp_error if use_tcp else hold_error
            metrics["causal_switch_error_m"].append(causal_error)
            metrics["causal_minus_hold_error_m"].append(
                causal_error - hold_error
            )
            metrics["causal_minus_tcp_error_m"].append(
                causal_error - tcp_error
            )
        oracle_index = int(
            np.mean([value[1] for value in run_candidates])
            < np.mean([value[0] for value in run_candidates])
        )
        metrics["oracle_tcp_selected"].append(float(oracle_index))
        metrics["causal_mode_correct"].append(float(use_tcp == bool(oracle_index)))
        metrics["oracle_run_mean_error_m"].append(
            float(np.mean([value[oracle_index] for value in run_candidates]))
        )
        metrics["oracle_switch_error_m"].extend(
            value[oracle_index] for value in run_candidates
        )
    return dict(metrics)


def _flatten(
    rows: Iterable[dict[str, list[float]]], name: str
) -> list[float]:
    return [value for row in rows for value in row.get(name, [])]


def run(
    *,
    project_root: Path,
    config_path: Path,
    replay_root: Path,
    output_path: Path,
    config: OcclusionAuditConfig,
) -> None:
    if output_path.exists():
        raise FileExistsError(f"输出文件已存在，拒绝覆盖：{output_path}")
    audit_path = replay_root / "replay_audit_report.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("summary", {}).get("all_criteria_passed", False):
        raise ValueError("replay audit 未通过")
    pairs = audit.get("pairs", [])
    if len(pairs) != config.expected_pairs:
        raise ValueError("replay audit pair 数量不匹配")

    paths = _paths(replay_root)
    for operation in OPERATIONS:
        expected = audit["files"][operation]["replay"]
        if _sha256(paths[operation]["h5"]) != expected["h5_sha256"]:
            raise ValueError(f"{operation} replay HDF5 hash 不匹配")
        if _sha256(paths[operation]["json"]) != expected["json_sha256"]:
            raise ValueError(f"{operation} replay JSON hash 不匹配")
    episodes = {
        operation: _metadata_by_seed(paths[operation]["json"])
        for operation in OPERATIONS
    }
    handles = {
        operation: h5py.File(paths[operation]["h5"], "r")
        for operation in OPERATIONS
    }

    trajectory_rows: list[dict[str, Any]] = []
    tracker_rows: list[dict[str, list[float]]] = []
    progress_counts = {
        role: np.zeros((config.progress_samples, 2), dtype=np.int64)
        for role in ("active", "anchor")
    }
    all_counts: dict[str, list[int]] = defaultdict(list)
    visible_centroid_errors: list[float] = []
    try:
        for pair in pairs:
            seed = int(pair["seed"])
            branch = int(pair["replay_branch_frame"])
            for operation in OPERATIONS:
                trajectory = _trajectory(
                    handles[operation], episodes[operation][seed]
                )
                xyzw = trajectory["obs/pointcloud/xyzw"]
                segmentation = trajectory["obs/pointcloud/segmentation"]
                actor_positions = {
                    role: np.asarray(
                        trajectory[f"env_states/actors/{actor}"][:, :3],
                        dtype=np.float64,
                    )
                    for role, actor in (
                        ("active", "cube"),
                        ("anchor", "layout_anchor"),
                    )
                }
                labels = {
                    role: _infer_actor_label(
                        xyzw=xyzw,
                        segmentation=segmentation,
                        actor_positions=actor_positions[role],
                        config=config,
                        role=role,
                    )[0]
                    for role in ("active", "anchor")
                }
                if labels["active"] == labels["anchor"]:
                    raise ValueError(f"seed={seed} segmentation label 冲突")
                actions = trajectory["actions"]
                last = len(actions) - config.action_horizon
                if last < branch:
                    raise ValueError("branch 后没有完整 action chunk")
                frames = np.arange(branch, last + 1, dtype=np.int64)
                fixed = _fixed_frames(
                    branch, last, config.progress_samples
                )
                tcp_positions = np.asarray(
                    trajectory["obs/extra/tcp_pose"][frames, :3],
                    dtype=np.float64,
                )
                centers: list[np.ndarray | None] = []
                visibility: dict[str, list[bool]] = {
                    "active": [],
                    "anchor": [],
                }
                counts_by_frame: dict[str, dict[int, int]] = {
                    "active": {},
                    "anchor": {},
                }
                for frame in frames.tolist():
                    role_points = {
                        role: _segmented_points(
                            np.asarray(xyzw[frame]),
                            np.asarray(segmentation[frame]),
                            labels[role],
                        )
                        for role in ("active", "anchor")
                    }
                    for role, points in role_points.items():
                        count = len(points)
                        counts_by_frame[role][frame] = count
                        all_counts[role].append(count)
                        visibility[role].append(
                            count >= config.minimum_label_points
                        )
                    active_points = role_points["active"]
                    center = (
                        active_points.mean(axis=0)
                        if len(active_points) >= config.minimum_label_points
                        else None
                    )
                    centers.append(center)
                    if center is not None:
                        visible_centroid_errors.append(
                            float(
                                np.linalg.norm(
                                    center - actor_positions["active"][frame]
                                )
                            )
                        )
                if centers[0] is None:
                    raise ValueError(
                        f"seed={seed} operation={operation} branch active 不可见"
                    )

                fixed_visibility: dict[str, list[bool]] = {}
                for role in ("active", "anchor"):
                    fixed_visibility[role] = []
                    for progress_index, frame in enumerate(fixed.tolist()):
                        visible = (
                            counts_by_frame[role][frame]
                            >= config.minimum_label_points
                        )
                        fixed_visibility[role].append(visible)
                        progress_counts[role][progress_index, int(visible)] += 1

                tracker = _tracker_audit(
                    centers=centers,
                    actor_positions=actor_positions["active"][frames],
                    tcp_positions=tcp_positions,
                    motion_threshold_m=config.audit_motion_threshold_m,
                )
                if config.causal_history_frames is not None:
                    switch = _causal_switch_audit(
                        centers=centers,
                        actor_positions=actor_positions["active"][frames],
                        tcp_positions=tcp_positions,
                        history_frames=config.causal_history_frames,
                    )
                    tracker.update(switch)
                tracker_rows.append(tracker)
                row: dict[str, Any] = {
                    "pair_id": int(pair["pair_id"]),
                    "seed": seed,
                    "split": str(pair["split"]),
                    "operation": operation,
                    "branch_frame": branch,
                    "last_query_frame": last,
                    "query_frames": int(len(frames)),
                    "fixed_frames": fixed.tolist(),
                }
                for role in ("active", "anchor"):
                    visible = np.asarray(visibility[role], dtype=np.bool_)
                    runs = _missing_runs(visible)
                    counts = np.asarray(
                        [counts_by_frame[role][frame] for frame in frames],
                        dtype=np.int64,
                    )
                    row[role] = {
                        "visible_frames": int(visible.sum()),
                        "missing_frames": int((~visible).sum()),
                        "zero_point_frames": int((counts == 0).sum()),
                        "missing_run_count": len(runs),
                        "maximum_missing_run_frames": max(
                            (end - start + 1 for start, end in runs),
                            default=0,
                        ),
                        "fixed_progress_visible": fixed_visibility[role],
                    }
                row["tracker"] = {
                    name: _summary(values) for name, values in tracker.items()
                }
                trajectory_rows.append(row)
    finally:
        for handle in handles.values():
            handle.close()

    expected_trajectories = config.expected_pairs * len(OPERATIONS)
    if len(trajectory_rows) != expected_trajectories:
        raise ValueError("trajectory 数量不匹配")
    if max(visible_centroid_errors) > config.maximum_centroid_error_m:
        raise ValueError("可见 active centroid error 超限")

    total_frames = sum(row["query_frames"] for row in trajectory_rows)
    visibility_report: dict[str, Any] = {}
    for role in ("active", "anchor"):
        counts = np.asarray(all_counts[role], dtype=np.int64)
        missing = counts < config.minimum_label_points
        by_progress = []
        for index in range(config.progress_samples):
            invisible, visible = progress_counts[role][index].tolist()
            by_progress.append(
                {
                    "progress_index": index,
                    "visible": visible,
                    "missing": invisible,
                    "visible_fraction": visible / (visible + invisible),
                }
            )
        visibility_report[role] = {
            "frames": int(len(counts)),
            "visible_fraction": float(np.mean(~missing)),
            "below_threshold_fraction": float(np.mean(missing)),
            "zero_point_fraction": float(np.mean(counts == 0)),
            "point_count": _summary(counts.tolist()),
            "maximum_missing_run_frames": max(
                row[role]["maximum_missing_run_frames"]
                for row in trajectory_rows
            ),
            "trajectories_with_missing_frames": sum(
                row[role]["missing_frames"] > 0
                for row in trajectory_rows
            ),
            "trajectories_with_missing_fixed_progress": sum(
                not all(row[role]["fixed_progress_visible"])
                for row in trajectory_rows
            ),
            "fixed_progress": by_progress,
        }

    metric_names = (
        "hold_error_m",
        "tcp_delta_error_m",
        "tcp_minus_hold_error_m",
        "moving_hold_error_m",
        "moving_tcp_delta_error_m",
        "actor_motion_since_visible_m",
        "tcp_motion_since_visible_m",
        "actor_tcp_displacement_cosine",
        "history_tcp_minus_hold_error_m",
        "causal_tcp_selected",
        "causal_history_available",
        "causal_switch_error_m",
        "causal_minus_hold_error_m",
        "causal_minus_tcp_error_m",
        "oracle_tcp_selected",
        "causal_mode_correct",
        "oracle_run_mean_error_m",
        "oracle_switch_error_m",
    )
    tracker_report = {
        name: _summary(_flatten(tracker_rows, name)) for name in metric_names
    }
    paired_trajectory_differences = [
        float(np.mean(row["tcp_minus_hold_error_m"]))
        for row in tracker_rows
        if row.get("tcp_minus_hold_error_m")
    ]
    tracker_report["per_trajectory_tcp_minus_hold_mean_m"] = _summary(
        paired_trajectory_differences
    )
    causal_trajectory_differences = [
        float(np.mean(row["causal_minus_hold_error_m"]))
        for row in tracker_rows
        if row.get("causal_minus_hold_error_m")
    ]
    tracker_report["per_trajectory_causal_minus_hold_mean_m"] = _summary(
        causal_trajectory_differences
    )
    def group_report(indices: list[int]) -> dict[str, Any]:
        group_frames = sum(
            trajectory_rows[index]["query_frames"] for index in indices
        )
        return {
            "trajectories": len(indices),
            "query_frames": group_frames,
            "active_missing_fraction": sum(
                trajectory_rows[index]["active"]["missing_frames"]
                for index in indices
            )
            / group_frames,
            "anchor_missing_fraction": sum(
                trajectory_rows[index]["anchor"]["missing_frames"]
                for index in indices
            )
            / group_frames,
            "tracker_audit": {
                name: _summary(
                    value
                    for index in indices
                    for value in tracker_rows[index].get(name, [])
                )
                for name in metric_names
            },
        }
    operation_report = {
        operation: group_report(
            [
                index
                for index, row in enumerate(trajectory_rows)
                if row["operation"] == operation
            ]
        )
        for operation in OPERATIONS
    }
    split_names = sorted({row["split"] for row in trajectory_rows})
    split_report = {
        split: group_report(
            [
                index
                for index, row in enumerate(trajectory_rows)
                if row["split"] == split
            ]
        )
        for split in split_names
    }
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(project_root),
        "config": asdict(config),
        "config_sha256": _sha256(config_path),
        "replay_audit_sha256": _sha256(audit_path),
        "protocol": {
            "online_inputs": [
                "segmented point cloud",
                "TCP pose history",
            ],
            "privileged_audit_only": [
                "active actor position",
                "anchor actor position",
            ],
            "tracker_candidates": [
                "last-observation hold",
                "unconditional TCP-delta propagation during occlusion",
            ]
            + (
                ["causal retrospective-error switch"]
                if config.causal_history_frames is not None
                else []
            ),
            "parameter_tuning_performed": False,
        },
        "summary": {
            "pairs": config.expected_pairs,
            "trajectories": expected_trajectories,
            "query_frames": total_frames,
            "visible_active_centroid_error_m": _summary(
                visible_centroid_errors
            ),
            "visibility": visibility_report,
            "tracker_audit": tracker_report,
            "by_operation": operation_report,
            "by_split": split_report,
        },
        "trajectories": trajectory_rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        f".{output_path.name}.incomplete-{os.getpid()}"
    )
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    print(json.dumps(report["summary"], indent=2, sort_keys=True), flush=True)


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
        config=OcclusionAuditConfig.from_json(resolved_config),
    )

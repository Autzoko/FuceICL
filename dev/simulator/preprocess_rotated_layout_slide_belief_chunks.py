"""构建 dual-view visibility-conditioned belief 多进度 H6 chunks。"""

from __future__ import annotations

import argparse
import atexit
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

import h5py
import numpy as np

from dev.simulator.evaluate_dual_view_observability import _camera_variants
from dev.simulator.preprocess_bidirectional_slide_predictor import (
    _metadata_by_seed,
    _sample_points,
    _trajectory,
)
from dev.simulator.preprocess_maniskill_chunks import (
    _infer_actor_label,
    _quaternion_wxyz_to_matrix,
    _segmented_points,
)
from dev.simulator.preprocess_rotated_layout_slide_causal_anchor_chunks import (
    STATE_FEATURES as ANCHOR_STATE_FEATURES,
    _sample_seed,
    _state as _anchor_state,
)
from dev.simulator.preprocess_rotated_layout_slide_predictor import (
    OPERATIONS,
    SPLIT_IDS,
    _actions,
    _angle_degrees,
    _sha256,
)
from dev.simulator.preprocess_rotated_layout_slide_progress_chunks import (
    _paths,
)
from src.components.retriever.belief_tracker import (
    ObjectBeliefEstimate as ActiveBelief,
    VisibilityConditionedBeliefTracker,
    belief_risk_accepted as _belief_risk_accepted,
)
from dev.simulator.visibility_dropout_probe import (
    select_visibility_dropout_probes,
)


STATE_FEATURES = ANCHOR_STATE_FEATURES + (
    "active_visible",
    "belief_age_steps",
)


@dataclass(frozen=True)
class BeliefChunkConfig:
    """冻结的 visibility-conditioned belief chunk 数据协议。"""

    schema_version: str
    seed: int
    expected_pairs: int
    expected_camera_variant: str
    action_horizon: int
    progress_samples: int
    active_points: int
    anchor_points: int
    label_probe_frames: int
    minimum_label_points: int
    maximum_centroid_error_m: float
    maximum_belief_error_m: float
    position_limit_m: float
    rotation_scale_rad: float
    maximum_layout_axis_error_degrees: float
    progress_fraction_maximum: float = 1.0
    belief_risk_gate_maximum_tcp_net_displacement_m: float | None = None
    dropout_risk_probe_enabled: bool = False
    dropout_risk_probe_minimum_positive_displacement_m: float = 0.000001

    @classmethod
    def from_json(cls, path: Path) -> "BeliefChunkConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def __post_init__(self) -> None:
        if self.schema_version != "rotated-layout-slide-belief-chunks-v1":
            raise ValueError("未知 belief chunk schema")
        if self.expected_camera_variant != "dual-fixed-v1":
            raise ValueError("belief chunk camera variant 不匹配")
        positive = (
            self.expected_pairs,
            self.action_horizon,
            self.progress_samples,
            self.active_points,
            self.anchor_points,
            self.label_probe_frames,
            self.minimum_label_points,
            self.maximum_centroid_error_m,
            self.maximum_belief_error_m,
            self.position_limit_m,
            self.maximum_layout_axis_error_degrees,
        )
        if min(positive) <= 0 or self.seed < 0:
            raise ValueError("belief chunk 配置非法")
        if self.rotation_scale_rad == 0 or self.action_horizon != 6:
            raise ValueError("controller scale 非法或 horizon 不是 6")
        if self.progress_samples < 2:
            raise ValueError("progress_samples 至少为 2")
        if self.maximum_belief_error_m < self.maximum_centroid_error_m:
            raise ValueError("belief error 门槛不得小于 measurement 门槛")
        if not 0.0 < self.progress_fraction_maximum <= 1.0:
            raise ValueError("progress_fraction_maximum 必须位于 (0,1]")
        if (
            self.belief_risk_gate_maximum_tcp_net_displacement_m is not None
            and self.belief_risk_gate_maximum_tcp_net_displacement_m <= 0.0
        ):
            raise ValueError("belief risk gate 位移门槛必须为正")
        if not isinstance(self.dropout_risk_probe_enabled, bool):
            raise TypeError("dropout_risk_probe_enabled 必须是 bool")
        if self.dropout_risk_probe_minimum_positive_displacement_m <= 0.0:
            raise ValueError("dropout risk probe 最小正位移必须为正")
        if (
            self.dropout_risk_probe_enabled
            and self.belief_risk_gate_maximum_tcp_net_displacement_m is None
        ):
            raise ValueError("dropout risk probe 必须启用 belief risk gate")
        if (
            self.belief_risk_gate_maximum_tcp_net_displacement_m is not None
            and self.dropout_risk_probe_minimum_positive_displacement_m
            >= self.belief_risk_gate_maximum_tcp_net_displacement_m
        ):
            raise ValueError("dropout risk probe 最小位移必须小于 gate 阈值")


def _belief_frames(
    branch: int,
    action_steps: int,
    config: BeliefChunkConfig,
) -> list[int]:
    """在完整 H6 可取区间的冻结前缀内均匀采样。"""
    last = action_steps - config.action_horizon
    if last < branch:
        raise ValueError("branch 后没有完整 action chunk")
    safe_last = branch + int(
        np.floor(config.progress_fraction_maximum * (last - branch))
    )
    frames = np.rint(
        np.linspace(branch, safe_last, config.progress_samples)
    ).astype(np.int64)
    if len(np.unique(frames)) != config.progress_samples:
        raise ValueError("trajectory 太短，belief progress frames 出现重复")
    return frames.tolist()


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _active_beliefs(
    *,
    xyzw: np.ndarray,
    segmentation: np.ndarray,
    label: int,
    tcp_positions: np.ndarray,
    branch: int,
    frames: list[int],
    minimum_points: int,
) -> dict[int, ActiveBelief]:
    """逐帧测量更新；不可见时只累计已发生的 TCP 位移。"""
    requested = set(frames)
    beliefs: dict[int, ActiveBelief] = {}
    tracker = VisibilityConditionedBeliefTracker(
        minimum_visible_points=minimum_points
    )
    for frame in range(branch, max(frames) + 1):
        observed = _segmented_points(xyzw[frame], segmentation[frame], label)
        try:
            belief = tracker.update(observed, tcp_positions[frame])
        except RuntimeError as error:
            raise ValueError("active belief 缺少可传播的因果测量") from error
        if frame in requested:
            beliefs[frame] = belief
    if set(beliefs) != requested:
        raise ValueError("belief 固定帧不完整")
    return beliefs


def _state(
    *,
    belief: ActiveBelief,
    initial_active_center: np.ndarray,
    initial_layout_axis: np.ndarray,
    tcp_pose: np.ndarray,
    qpos: np.ndarray,
    qvel: np.ndarray,
) -> np.ndarray:
    base = _anchor_state(
        active_center=belief.center_world,
        initial_active_center=initial_active_center,
        initial_layout_axis=initial_layout_axis,
        tcp_pose=tcp_pose,
        qpos=qpos,
        qvel=qvel,
    )
    value = np.concatenate(
        (
            base,
            np.asarray(
                [float(belief.visible), float(belief.age_steps)],
                dtype=np.float32,
            ),
        )
    )
    if value.shape != (len(STATE_FEATURES),) or not np.isfinite(value).all():
        raise ValueError("belief state shape/value 非法")
    return value


def run(
    *,
    project_root: Path,
    config_path: Path,
    replay_root: Path,
    output_root: Path,
    config: BeliefChunkConfig,
) -> None:
    if output_root.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output_root}")
    audit_path = replay_root / "replay_audit_report.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("summary", {}).get("all_criteria_passed", False):
        raise ValueError("replay audit 未通过")
    pair_rows = audit.get("pairs", [])
    if len(pair_rows) != config.expected_pairs:
        raise ValueError("replay audit pair 数量不匹配")
    camera_variants = _camera_variants(replay_root)
    if not all(
        variant == config.expected_camera_variant
        for values in camera_variants.values()
        for variant in values.values()
    ):
        raise ValueError("replay camera metadata 不匹配")
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
    fields = (
        "pair_id",
        "seed",
        "split_id",
        "operation_id",
        "progress_index",
        "progress_fraction",
        "frame",
        "active_visible",
        "belief_age_steps",
        "belief_risk_accepted",
        "belief_risk_score_m",
        "active_points",
        "anchor_points",
        "state",
        "tcp_pose",
        "action",
    )
    arrays: dict[str, list[Any]] = {name: [] for name in fields}
    index_rows = []
    initial_axis_errors = []
    belief_errors = {True: [], False: []}
    memory_point_counts = []
    label_probe_errors = []
    risk_accepted_count = 0
    risk_rejected_rows: list[dict[str, Any]] = []
    dropout_probe_rows: list[dict[str, Any]] = []
    dropout_support_rows: list[dict[str, Any]] = []
    try:
        for pair in pair_rows:
            pair_id = int(pair["pair_id"])
            seed = int(pair["seed"])
            split = str(pair["split"])
            branch = int(pair["replay_branch_frame"])
            if split not in SPLIT_IDS:
                raise ValueError(f"未知 split：{split}")
            for operation_id, operation in enumerate(OPERATIONS):
                trajectory = _trajectory(
                    handles[operation], episodes[operation][seed]
                )
                xyzw = np.asarray(trajectory["obs/pointcloud/xyzw"])
                segmentation = np.asarray(
                    trajectory["obs/pointcloud/segmentation"]
                )
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
                labels = {}
                for role in ("active", "anchor"):
                    labels[role], diagnostic = _infer_actor_label(
                        xyzw=xyzw,
                        segmentation=segmentation,
                        actor_positions=actor_positions[role],
                        config=config,
                        role=role,
                    )
                    label_probe_errors.append(
                        float(diagnostic["selected_probe_error_max_m"])
                    )
                if labels["active"] == labels["anchor"]:
                    raise ValueError(f"seed={seed} segmentation label 冲突")
                tcp_poses = np.asarray(
                    trajectory["obs/extra/tcp_pose"], dtype=np.float64
                )
                qpos = np.asarray(
                    trajectory["obs/agent/qpos"], dtype=np.float64
                )
                qvel = np.asarray(
                    trajectory["obs/agent/qvel"], dtype=np.float64
                )
                actions = np.asarray(trajectory["actions"], dtype=np.float64)
                memory_points = {
                    role: _segmented_points(
                        xyzw[branch], segmentation[branch], labels[role]
                    )
                    for role in ("active", "anchor")
                }
                memory_counts = {
                    role: len(points)
                    for role, points in memory_points.items()
                }
                memory_point_counts.extend(memory_counts.values())
                if min(memory_counts.values()) < config.minimum_label_points:
                    raise ValueError(
                        f"seed={seed} branch memory points 不足：{memory_counts}"
                    )
                memory_centers = {
                    role: points.mean(axis=0)
                    for role, points in memory_points.items()
                }
                initial_axis = (
                    memory_centers["anchor"] - memory_centers["active"]
                )
                initial_axis[2] = 0.0
                privileged_axis = (
                    actor_positions["anchor"][branch]
                    - actor_positions["active"][branch]
                )
                privileged_axis[2] = 0.0
                axis_error = _angle_degrees(initial_axis, privileged_axis)
                initial_axis_errors.append(axis_error)
                branch_rotation = _quaternion_wxyz_to_matrix(
                    tcp_poses[branch, 3:7]
                )
                anchor_points = _sample_points(
                    memory_points["anchor"],
                    memory_centers["anchor"],
                    branch_rotation,
                    count=config.anchor_points,
                    seed=_sample_seed(
                        config.seed, pair_id, operation, 0, "initial_anchor"
                    ),
                )
                frames = _belief_frames(branch, len(actions), config)
                beliefs = _active_beliefs(
                    xyzw=xyzw,
                    segmentation=segmentation,
                    label=labels["active"],
                    tcp_positions=tcp_poses[:, :3],
                    branch=branch,
                    frames=frames,
                    minimum_points=config.minimum_label_points,
                )
                if config.dropout_risk_probe_enabled:
                    risk_threshold = (
                        config.belief_risk_gate_maximum_tcp_net_displacement_m
                    )
                    minimum_probe_displacement = (
                        config.dropout_risk_probe_minimum_positive_displacement_m
                    )
                    selection = select_visibility_dropout_probes(
                        tcp_poses[:, :3],
                        branch=branch,
                        last_h6_start=len(actions) - config.action_horizon,
                        accepted_limit_m=risk_threshold,
                        minimum_positive_m=minimum_probe_displacement,
                    )
                    dropout_support_rows.append(
                        {
                            "pair_id": pair_id,
                            "seed": seed,
                            "operation": operation,
                            **selection,
                        }
                    )
                    selected = {
                        int(value["frame"]): role
                        for role, value in (
                            ("accepted", selection["accepted_probe"]),
                            ("rejected", selection["rejected_probe"]),
                        )
                        if value is not None
                    }
                    if selected:
                        dropout_tracker = VisibilityConditionedBeliefTracker(
                            minimum_visible_points=config.minimum_label_points
                        )
                        dropout_tracker.update(
                            memory_points["active"], tcp_poses[branch, :3]
                        )
                        for dropout_frame in range(
                            branch + 1,
                            max(selected) + 1,
                        ):
                            dropout_belief = dropout_tracker.update(
                                np.empty((0, 3), dtype=np.float64),
                                tcp_poses[dropout_frame, :3],
                            )
                            role = selected.get(dropout_frame)
                            if role is None:
                                continue
                            dropout_displacement = (
                                dropout_belief.tcp_net_displacement_since_visible_m
                            )
                            accepted = _belief_risk_accepted(
                                visible=dropout_belief.visible,
                                tcp_net_displacement_m=dropout_displacement,
                                maximum_tcp_net_displacement_m=risk_threshold,
                            )
                            if accepted != (role == "accepted"):
                                raise RuntimeError(
                                    "dropout probe role 与解析 gate 不一致"
                                )
                            error = float(
                                np.linalg.norm(
                                    dropout_belief.center_world
                                    - actor_positions["active"][dropout_frame]
                                )
                            )
                            if (
                                accepted
                                and error > config.maximum_belief_error_m
                            ):
                                raise ValueError(
                                    f"seed={seed} dropout accepted belief "
                                    f"error={error:.4f} m 超限"
                                )
                            dropout_probe_rows.append(
                                {
                                    "pair_id": pair_id,
                                    "seed": seed,
                                    "operation": operation,
                                    "role": role,
                                    "frame": dropout_frame,
                                    "belief_age_steps": (
                                        dropout_belief.age_steps
                                    ),
                                    "tcp_net_displacement_m": (
                                        dropout_displacement
                                    ),
                                    "belief_error_m": error,
                                    "risk_accepted": accepted,
                                }
                            )
                for progress_index, frame in enumerate(frames):
                    belief = beliefs[frame]
                    belief_error = float(
                        np.linalg.norm(
                            belief.center_world
                            - actor_positions["active"][frame]
                        )
                    )
                    belief_errors[belief.visible].append(belief_error)
                    limit = (
                        config.maximum_centroid_error_m
                        if belief.visible
                        else config.maximum_belief_error_m
                    )
                    risk_threshold = (
                        config.belief_risk_gate_maximum_tcp_net_displacement_m
                    )
                    risk_accepted = _belief_risk_accepted(
                        visible=belief.visible,
                        tcp_net_displacement_m=(
                            belief.tcp_net_displacement_since_visible_m
                        ),
                        maximum_tcp_net_displacement_m=risk_threshold,
                    )
                    risk_accepted_count += int(risk_accepted)
                    if belief_error > limit and risk_accepted:
                        raise ValueError(
                            f"seed={seed} frame={frame} belief error="
                            f"{belief_error:.4f} m 超限"
                        )
                    if not risk_accepted:
                        risk_rejected_rows.append(
                            {
                                "pair_id": pair_id,
                                "seed": seed,
                                "operation": operation,
                                "progress_index": progress_index,
                                "frame": frame,
                                "belief_error_m": belief_error,
                                "belief_age_steps": belief.age_steps,
                                "tcp_net_displacement_since_visible_m": (
                                    belief.tcp_net_displacement_since_visible_m
                                ),
                            }
                        )
                    rotation = _quaternion_wxyz_to_matrix(
                        tcp_poses[frame, 3:7]
                    )
                    active_points = _sample_points(
                        belief.points_world,
                        belief.center_world,
                        rotation,
                        count=config.active_points,
                        seed=_sample_seed(
                            config.seed,
                            pair_id,
                            operation,
                            progress_index,
                            "belief_active",
                        ),
                    )
                    state = _state(
                        belief=belief,
                        initial_active_center=memory_centers["active"],
                        initial_layout_axis=initial_axis,
                        tcp_pose=tcp_poses[frame],
                        qpos=qpos[frame],
                        qvel=qvel[frame],
                    )
                    chunk = _actions(
                        normalized=actions,
                        tcp_poses=tcp_poses,
                        branch=frame,
                        config=config,
                    )
                    progress_fraction = progress_index / (
                        config.progress_samples - 1
                    )
                    scalars = {
                        "pair_id": pair_id,
                        "seed": seed,
                        "split_id": SPLIT_IDS[split],
                        "operation_id": operation_id,
                        "progress_index": progress_index,
                        "progress_fraction": progress_fraction,
                        "frame": frame,
                        "active_visible": int(belief.visible),
                        "belief_age_steps": belief.age_steps,
                        "belief_risk_accepted": int(risk_accepted),
                        "belief_risk_score_m": (
                            belief.tcp_net_displacement_since_visible_m
                        ),
                    }
                    for name, value in scalars.items():
                        arrays[name].append(value)
                    arrays["active_points"].append(active_points)
                    arrays["anchor_points"].append(anchor_points)
                    arrays["state"].append(state)
                    arrays["tcp_pose"].append(
                        tcp_poses[frame].astype(np.float32)
                    )
                    arrays["action"].append(chunk)
                    index_rows.append(
                        {
                            "row": len(index_rows),
                            "pair_id": pair_id,
                            "seed": seed,
                            "split": split,
                            "operation": operation,
                            "progress_index": progress_index,
                            "progress_fraction": progress_fraction,
                            "frame": frame,
                            "active_visible": belief.visible,
                            "belief_age_steps": belief.age_steps,
                            "tcp_net_displacement_since_visible_m": (
                                belief.tcp_net_displacement_since_visible_m
                            ),
                            "tcp_path_length_since_visible_m": (
                                belief.tcp_path_length_since_visible_m
                            ),
                            "observed_active_points": (
                                belief.observed_point_count
                            ),
                            "belief_error_m": belief_error,
                            "belief_risk_accepted": risk_accepted,
                            "belief_risk_score_m": (
                                belief.tcp_net_displacement_since_visible_m
                            ),
                            "initial_layout_axis_error_degrees": axis_error,
                        }
                    )
    finally:
        for handle in handles.values():
            handle.close()

    expected_rows = (
        config.expected_pairs * len(OPERATIONS) * config.progress_samples
    )
    if len(index_rows) != expected_rows:
        raise ValueError("belief chunk rows 数量不匹配")
    maximum_axis_error = max(initial_axis_errors)
    if maximum_axis_error > config.maximum_layout_axis_error_degrees:
        raise ValueError(
            f"initial layout axis max error={maximum_axis_error:.3f} deg"
        )
    integer_names = {
        "pair_id",
        "seed",
        "split_id",
        "operation_id",
        "progress_index",
        "frame",
        "active_visible",
        "belief_age_steps",
        "belief_risk_accepted",
    }
    stacked = {
        name: np.asarray(values, dtype=np.int64)
        if name in integer_names
        else np.asarray(values, dtype=np.float32)
        if name in {"progress_fraction", "belief_risk_score_m"}
        else np.stack(values)
        for name, values in arrays.items()
    }
    temporary = output_root.with_name(
        f".{output_root.name}.incomplete-{os.getpid()}"
    )
    temporary.mkdir(parents=True)

    def cleanup() -> None:
        shutil.rmtree(temporary, ignore_errors=True)

    atexit.register(cleanup)
    data_path = temporary / "progress_chunks.npz"
    with data_path.open("wb") as stream:
        np.savez_compressed(stream, **stacked)
    index_path = temporary / "index.jsonl"
    with index_path.open("w", encoding="utf-8") as stream:
        for row in index_rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    observed_errors = belief_errors[True]
    propagated_errors = belief_errors[False]
    accepted_dropout = [
        row for row in dropout_probe_rows if row["role"] == "accepted"
    ]
    rejected_dropout = [
        row for row in dropout_probe_rows if row["role"] == "rejected"
    ]

    def dropout_stats(
        rows: list[dict[str, Any]],
        field: str,
    ) -> dict[str, float] | None:
        if not rows:
            return None
        values = np.asarray([row[field] for row in rows], dtype=np.float64)
        return {
            "minimum": float(values.min()),
            "median": float(np.median(values)),
            "maximum": float(values.max()),
        }

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(project_root),
        "config": asdict(config),
        "config_sha256": _sha256(config_path),
        "replay_audit_sha256": _sha256(audit_path),
        "camera_variants": camera_variants,
        "state_features": STATE_FEATURES,
        "audit_only_fields": [
            "operation_id",
            "split_id",
            "progress_index",
            "progress_fraction",
        ],
        "protocol": {
            "active_belief": (
                "visible centroid update, otherwise accumulated TCP delta"
            ),
            "anchor_memory": "branch segmented pointcloud",
            "future_query_fields_used": False,
            "actor_state_used_as_input": False,
            "progress_feature_used_as_input": False,
            "test_split_used": False,
            "rows_per_trajectory": config.progress_samples,
            "progress_fraction_maximum": (
                config.progress_fraction_maximum
            ),
            "progress_schedule": (
                "uniform in [branch, floor(branch + maximum * "
                "(last_h6_start - branch))]"
            ),
            "belief_risk_gate": (
                "accept visible; when missing require TCP net displacement "
                "since last visible within configured bound"
                if config.belief_risk_gate_maximum_tcp_net_displacement_m
                is not None
                else None
            ),
            "dropout_risk_probe": (
                "force missing after branch; select boundary accepted and first "
                "rejected frame using only TCP net displacement"
                if config.dropout_risk_probe_enabled
                else None
            ),
        },
        "rows": expected_rows,
        "pairs": config.expected_pairs,
        "split_row_counts": {
            split: int(np.sum(stacked["split_id"] == split_id))
            for split, split_id in SPLIT_IDS.items()
        },
        "belief": {
            "observed_rows": len(observed_errors),
            "propagated_rows": len(propagated_errors),
            "maximum_age_steps": int(stacked["belief_age_steps"].max()),
            "observed_error_mean_m": float(np.mean(observed_errors)),
            "observed_error_maximum_m": float(np.max(observed_errors)),
            "propagated_error_mean_m": float(np.mean(propagated_errors)),
            "propagated_error_maximum_m": float(
                np.max(propagated_errors)
            ),
        },
        "branch_memory_minimum_points": min(memory_point_counts),
        "label_probe_error_max_m": max(label_probe_errors),
        "initial_layout_axis_error_degrees": {
            "median": float(np.median(initial_axis_errors)),
            "maximum": maximum_axis_error,
        },
        "belief_risk_gate": {
            "enabled": (
                config.belief_risk_gate_maximum_tcp_net_displacement_m
                is not None
            ),
            "maximum_tcp_net_displacement_m": (
                config.belief_risk_gate_maximum_tcp_net_displacement_m
            ),
            "accepted_rows": risk_accepted_count,
            "rejected_rows": len(risk_rejected_rows),
            "coverage": risk_accepted_count / expected_rows,
            "rejected": risk_rejected_rows,
            "privileged_error_used_for_gate": False,
        },
        "dropout_risk_probe": {
            "enabled": config.dropout_risk_probe_enabled,
            "dropout_start": "first frame after replay branch",
            "selection_uses_actor_state": False,
            "gate_uses_actor_error": False,
            "expected_trajectories": (
                config.expected_pairs * len(OPERATIONS)
            ),
            "accepted_probe_trajectories": len(accepted_dropout),
            "rejected_probe_trajectories": len(rejected_dropout),
            "both_probe_trajectories": sum(
                row["accepted_probe"] is not None
                and row["rejected_probe"] is not None
                for row in dropout_support_rows
            ),
            "accepted_tcp_net_displacement_m": dropout_stats(
                accepted_dropout, "tcp_net_displacement_m"
            ),
            "rejected_tcp_net_displacement_m": dropout_stats(
                rejected_dropout, "tcp_net_displacement_m"
            ),
            "accepted_belief_error_m": dropout_stats(
                accepted_dropout, "belief_error_m"
            ),
            "rejected_belief_error_m": dropout_stats(
                rejected_dropout, "belief_error_m"
            ),
            "support": dropout_support_rows,
            "probes": dropout_probe_rows,
        },
        "files": {},
    }
    report["files"] = {
        "progress_chunks.npz": _sha256(data_path),
        "index.jsonl": _sha256(index_path),
    }
    report_path = temporary / "preprocess_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_root)
    atexit.unregister(cleanup)
    print(
        json.dumps(
            {
                "rows": expected_rows,
                "split_row_counts": report["split_row_counts"],
                "belief": report["belief"],
                "initial_layout_axis_error_degrees": report[
                    "initial_layout_axis_error_degrees"
                ],
                "files": report["files"],
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
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    resolved_config = arguments.config.resolve()
    run(
        project_root=arguments.project_root.resolve(),
        config_path=resolved_config,
        replay_root=arguments.replay_root.resolve(),
        output_root=arguments.output_root.resolve(),
        config=BeliefChunkConfig.from_json(resolved_config),
    )

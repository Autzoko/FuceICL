"""评估 retrieved Demo 对遮挡期 object motion mode 的 oracle 迁移上界。"""

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

from dev.simulator.analyze_rotated_layout_slide_occlusion import (
    _missing_runs,
    _paths,
    _summary,
)
from dev.simulator.preprocess_bidirectional_slide_predictor import (
    _metadata_by_seed,
    _trajectory,
)
from dev.simulator.preprocess_maniskill_chunks import (
    _infer_actor_label,
    _quaternion_wxyz_to_matrix,
    _segmented_points,
)
from dev.simulator.preprocess_rotated_layout_slide_predictor import (
    OPERATIONS,
    _sha256,
)


@dataclass(frozen=True)
class DemoModeTransferConfig:
    """冻结的 Demo motion-mode oracle transfer 协议。"""

    schema_version: str
    seed: int
    expected_pairs: int
    expected_train_pairs: int
    action_horizon: int
    label_probe_frames: int
    minimum_label_points: int
    maximum_centroid_error_m: float
    top_k: int
    bootstrap_samples: int
    minimum_recoverable_gap_fraction: float

    @classmethod
    def from_json(cls, path: Path) -> "DemoModeTransferConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def __post_init__(self) -> None:
        if self.schema_version != "demo-mode-transfer-oracle-v1":
            raise ValueError("未知 Demo mode transfer schema")
        positive = (
            self.expected_pairs,
            self.expected_train_pairs,
            self.action_horizon,
            self.label_probe_frames,
            self.minimum_label_points,
            self.maximum_centroid_error_m,
            self.top_k,
            self.bootstrap_samples,
            self.minimum_recoverable_gap_fraction,
        )
        if min(positive) <= 0 or self.seed < 0 or self.top_k % 2 != 1:
            raise ValueError("Demo mode transfer 配置非法")


@dataclass
class RunRecord:
    """一个 active-object 遮挡段及两个固定动力学候选的审计误差。"""

    pair_id: int
    seed: int
    split: str
    operation: str
    run_index: int
    onset_step: int
    onset_progress: float
    initial_key: np.ndarray
    hold_errors: np.ndarray
    tcp_errors: np.ndarray

    @property
    def oracle_tcp(self) -> bool:
        return bool(np.mean(self.tcp_errors) < np.mean(self.hold_errors))

    @property
    def key(self) -> np.ndarray:
        return np.concatenate((self.initial_key, [self.onset_progress]))


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _extract_runs(
    *,
    replay_root: Path,
    audit: dict[str, Any],
    config: DemoModeTransferConfig,
) -> tuple[list[RunRecord], dict[str, int]]:
    paths = _paths(replay_root)
    episodes = {
        operation: _metadata_by_seed(paths[operation]["json"])
        for operation in OPERATIONS
    }
    handles = {
        operation: h5py.File(paths[operation]["h5"], "r")
        for operation in OPERATIONS
    }
    runs: list[RunRecord] = []
    trajectory_counts: dict[str, int] = {}
    try:
        for pair in audit["pairs"]:
            pair_id = int(pair["pair_id"])
            seed = int(pair["seed"])
            split = str(pair["split"])
            branch = int(pair["replay_branch_frame"])
            trajectory_counts[split] = trajectory_counts.get(split, 0) + 1
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
                initial_points = {
                    role: _segmented_points(
                        np.asarray(xyzw[branch]),
                        np.asarray(segmentation[branch]),
                        labels[role],
                    )
                    for role in ("active", "anchor")
                }
                if min(len(value) for value in initial_points.values()) < (
                    config.minimum_label_points
                ):
                    raise ValueError(f"seed={seed} 初始 layout 不可见")
                initial_centers = {
                    role: points.mean(axis=0)
                    for role, points in initial_points.items()
                }
                for role in ("active", "anchor"):
                    error = np.linalg.norm(
                        initial_centers[role] - actor_positions[role][branch]
                    )
                    if error > config.maximum_centroid_error_m:
                        raise ValueError(
                            f"seed={seed} initial {role} centroid error 超限"
                        )
                tcp_poses = np.asarray(
                    trajectory["obs/extra/tcp_pose"], dtype=np.float64
                )
                rotation = _quaternion_wxyz_to_matrix(
                    tcp_poses[branch, 3:7]
                )
                initial_key = np.concatenate(
                    (
                        rotation.T
                        @ (
                            initial_centers["active"]
                            - tcp_poses[branch, :3]
                        ),
                        rotation.T
                        @ (
                            initial_centers["anchor"]
                            - initial_centers["active"]
                        ),
                    )
                )
                last = len(trajectory["actions"]) - config.action_horizon
                frames = np.arange(branch, last + 1, dtype=np.int64)
                centers: list[np.ndarray | None] = []
                for frame in frames.tolist():
                    points = _segmented_points(
                        np.asarray(xyzw[frame]),
                        np.asarray(segmentation[frame]),
                        labels["active"],
                    )
                    center = (
                        points.mean(axis=0)
                        if len(points) >= config.minimum_label_points
                        else None
                    )
                    if center is not None:
                        error = np.linalg.norm(
                            center - actor_positions["active"][frame]
                        )
                        if error > config.maximum_centroid_error_m:
                            raise ValueError(
                                f"seed={seed} frame={frame} active "
                                "centroid error 超限"
                            )
                    centers.append(center)
                visible = np.asarray(
                    [center is not None for center in centers], dtype=np.bool_
                )
                tcp_positions = tcp_poses[frames, :3]
                active_truth = actor_positions["active"][frames]
                for run_index, (start, end) in enumerate(
                    _missing_runs(visible)
                ):
                    if start == 0 or centers[start - 1] is None:
                        raise ValueError("遮挡段缺少因果初始观测")
                    last_center = np.asarray(centers[start - 1])
                    hold_errors = []
                    tcp_errors = []
                    for index in range(start, end + 1):
                        propagated = (
                            last_center
                            + tcp_positions[index]
                            - tcp_positions[start - 1]
                        )
                        hold_errors.append(
                            np.linalg.norm(last_center - active_truth[index])
                        )
                        tcp_errors.append(
                            np.linalg.norm(propagated - active_truth[index])
                        )
                    runs.append(
                        RunRecord(
                            pair_id=pair_id,
                            seed=seed,
                            split=split,
                            operation=operation,
                            run_index=run_index,
                            onset_step=start,
                            onset_progress=start / (len(frames) - 1),
                            initial_key=initial_key,
                            hold_errors=np.asarray(hold_errors),
                            tcp_errors=np.asarray(tcp_errors),
                        )
                    )
    finally:
        for handle in handles.values():
            handle.close()
    return runs, trajectory_counts


def _errors(record: RunRecord, use_tcp: bool) -> np.ndarray:
    return record.tcp_errors if use_tcp else record.hold_errors


def _bootstrap_difference(
    primary: np.ndarray,
    baseline: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    difference = primary - baseline
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        draw = rng.integers(0, len(difference), size=len(difference))
        means[index] = float(np.mean(difference[draw]))
    return {
        "mean_m": float(np.mean(difference)),
        "ci95_low_m": float(np.quantile(means, 0.025)),
        "ci95_high_m": float(np.quantile(means, 0.975)),
    }


def _evaluate_split(
    *,
    queries: list[RunRecord],
    pool: list[RunRecord],
    mean: np.ndarray,
    scale: np.ndarray,
    config: DemoModeTransferConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pool_keys = np.stack([(record.key - mean) / scale for record in pool])
    pool_tcp_fraction = float(np.mean([record.oracle_tcp for record in pool]))
    majority_tcp = pool_tcp_fraction >= 0.5
    methods: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "hold",
            "tcp",
            "pool_random_expectation",
            "pool_majority",
            "progress_top1",
            "geometry_progress_top1",
            "geometry_progress_top5",
            "query_oracle",
        )
    }
    run_means: dict[str, list[float]] = {name: [] for name in methods}
    query_rows = []
    primary_modes = []
    progress_modes = []
    top5_modes = []
    for query in queries:
        query_key = (query.key - mean) / scale
        distances = np.linalg.norm(pool_keys - query_key[None, :], axis=1)
        order = sorted(
            range(len(pool)),
            key=lambda index: (
                float(distances[index]),
                pool[index].pair_id,
                pool[index].run_index,
            ),
        )
        progress_order = sorted(
            range(len(pool)),
            key=lambda index: (
                abs(pool[index].onset_progress - query.onset_progress),
                pool[index].pair_id,
                pool[index].run_index,
            ),
        )
        primary_tcp = pool[order[0]].oracle_tcp
        progress_tcp = pool[progress_order[0]].oracle_tcp
        top5_tcp = sum(
            pool[index].oracle_tcp for index in order[: config.top_k]
        ) > config.top_k // 2
        oracle_tcp = query.oracle_tcp
        primary_modes.append(primary_tcp == oracle_tcp)
        progress_modes.append(progress_tcp == oracle_tcp)
        top5_modes.append(top5_tcp == oracle_tcp)
        selected = {
            "hold": query.hold_errors,
            "tcp": query.tcp_errors,
            "pool_random_expectation": (
                (1.0 - pool_tcp_fraction) * query.hold_errors
                + pool_tcp_fraction * query.tcp_errors
            ),
            "pool_majority": _errors(query, majority_tcp),
            "progress_top1": _errors(query, progress_tcp),
            "geometry_progress_top1": _errors(query, primary_tcp),
            "geometry_progress_top5": _errors(query, top5_tcp),
            "query_oracle": _errors(query, oracle_tcp),
        }
        for name, values in selected.items():
            methods[name].append(values)
            run_means[name].append(float(np.mean(values)))
        query_rows.append(
            {
                "pair_id": query.pair_id,
                "seed": query.seed,
                "split": query.split,
                "operation": query.operation,
                "run_index": query.run_index,
                "onset_progress": query.onset_progress,
                "frames": len(query.hold_errors),
                "query_oracle_tcp": oracle_tcp,
                "selected_demo_pair_id": pool[order[0]].pair_id,
                "selected_demo_run_index": pool[order[0]].run_index,
                "selected_demo_distance": float(distances[order[0]]),
                "selected_demo_tcp": primary_tcp,
                "method_run_mean_error_m": {
                    name: float(np.mean(values))
                    for name, values in selected.items()
                },
            }
        )

    summaries = {
        name: _summary(np.concatenate(values).tolist())
        for name, values in methods.items()
    }
    hold_mean = float(summaries["hold"]["mean"])
    tcp_mean = float(summaries["tcp"]["mean"])
    best_fixed_name = "hold" if hold_mean <= tcp_mean else "tcp"
    best_fixed_mean = min(hold_mean, tcp_mean)
    oracle_mean = float(summaries["query_oracle"]["mean"])
    primary_mean = float(summaries["geometry_progress_top1"]["mean"])
    gap = best_fixed_mean - oracle_mean
    recovered = (
        (best_fixed_mean - primary_mean) / gap if gap > 0 else None
    )
    best_fixed_p95 = float(summaries[best_fixed_name]["p95"])
    criteria = {
        "mean_below_best_fixed": primary_mean < best_fixed_mean,
        "mean_below_pool_majority": primary_mean
        < float(summaries["pool_majority"]["mean"]),
        "mean_below_progress_top1": primary_mean
        < float(summaries["progress_top1"]["mean"]),
        "p95_not_above_best_fixed": float(
            summaries["geometry_progress_top1"]["p95"]
        )
        <= best_fixed_p95,
        "recoverable_gap_fraction_at_least_minimum": recovered is not None
        and recovered >= config.minimum_recoverable_gap_fraction,
    }
    report = {
        "runs": len(queries),
        "frames": sum(len(record.hold_errors) for record in queries),
        "pool_tcp_mode_fraction": pool_tcp_fraction,
        "method_error_m": summaries,
        "mode_accuracy": {
            "progress_top1": float(np.mean(progress_modes)),
            "geometry_progress_top1": float(np.mean(primary_modes)),
            "geometry_progress_top5": float(np.mean(top5_modes)),
        },
        "best_fixed": best_fixed_name,
        "recoverable_gap_fraction": recovered,
        "paired_run_bootstrap_primary_minus_best_fixed": (
            _bootstrap_difference(
                np.asarray(run_means["geometry_progress_top1"]),
                np.asarray(run_means[best_fixed_name]),
                samples=config.bootstrap_samples,
                seed=config.seed + sum(record.pair_id for record in queries),
            )
        ),
        "criteria": criteria,
        "all_criteria_passed": all(criteria.values()),
    }
    return report, query_rows


def run(
    *,
    project_root: Path,
    config_path: Path,
    replay_root: Path,
    output_path: Path,
    config: DemoModeTransferConfig,
) -> None:
    if output_path.exists():
        raise FileExistsError(f"输出文件已存在，拒绝覆盖：{output_path}")
    audit_path = replay_root / "replay_audit_report.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("summary", {}).get("all_criteria_passed", False):
        raise ValueError("replay audit 未通过")
    if len(audit.get("pairs", [])) != config.expected_pairs:
        raise ValueError("replay pair 数量不匹配")
    paths = _paths(replay_root)
    for operation in OPERATIONS:
        expected = audit["files"][operation]["replay"]
        if _sha256(paths[operation]["h5"]) != expected["h5_sha256"]:
            raise ValueError(f"{operation} replay HDF5 hash 不匹配")
        if _sha256(paths[operation]["json"]) != expected["json_sha256"]:
            raise ValueError(f"{operation} replay JSON hash 不匹配")

    records, trajectory_counts = _extract_runs(
        replay_root=replay_root,
        audit=audit,
        config=config,
    )
    if trajectory_counts.get("train") != config.expected_train_pairs:
        raise ValueError("train pair 数量不匹配")
    pool = [record for record in records if record.split == "train"]
    if len(pool) < config.top_k:
        raise ValueError("Demo run pool 太小")
    pool_keys = np.stack([record.key for record in pool])
    mean = pool_keys.mean(axis=0)
    scale = pool_keys.std(axis=0)
    scale[scale < 1e-6] = 1.0

    split_reports = {}
    query_rows = []
    for split in ("val", "test"):
        queries = [record for record in records if record.split == split]
        if not queries:
            raise ValueError(f"{split} 没有 active 遮挡段")
        split_report, rows = _evaluate_split(
            queries=queries,
            pool=pool,
            mean=mean,
            scale=scale,
            config=config,
        )
        split_reports[split] = split_report
        query_rows.extend(rows)
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(project_root),
        "config": asdict(config),
        "config_sha256": _sha256(config_path),
        "replay_audit_sha256": _sha256(audit_path),
        "protocol": {
            "online_retrieval_key": [
                "initial active-from-TCP in EEF (3D)",
                "initial anchor-from-active in EEF (3D)",
                "occlusion onset progress (1D)",
            ],
            "pool_split": "train only",
            "query_splits": ["val", "test"],
            "privileged_demo_field": "per-run oracle motion mode",
            "privileged_query_fields": "tracking-error audit only",
            "confirmation_result": False,
        },
        "trajectory_pair_counts": trajectory_counts,
        "run_counts": {
            split: sum(record.split == split for record in records)
            for split in sorted(trajectory_counts)
        },
        "train_key_normalization": {
            "mean": mean.tolist(),
            "scale": scale.tolist(),
        },
        "splits": split_reports,
        "advance_to_demo_mode_encoder": all(
            split_reports[split]["all_criteria_passed"]
            for split in ("val", "test")
        ),
        "queries": query_rows,
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
    print(
        json.dumps(
            {
                "run_counts": report["run_counts"],
                "splits": split_reports,
                "advance_to_demo_mode_encoder": report[
                    "advance_to_demo_mode_encoder"
                ],
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
        config=DemoModeTransferConfig.from_json(resolved_config),
    )

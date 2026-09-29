"""用在线可得 elapsed step 复验 Demo motion-mode oracle transfer。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from dev.simulator.analyze_rotated_layout_slide_occlusion import _summary
from dev.simulator.evaluate_demo_mode_transfer import (
    RunRecord,
    _bootstrap_difference,
    _errors,
    _extract_runs,
)
from dev.simulator.preprocess_rotated_layout_slide_predictor import (
    OPERATIONS,
    _sha256,
)
from dev.simulator.preprocess_rotated_layout_slide_progress_chunks import (
    _paths,
)


@dataclass(frozen=True)
class CausalDemoModeKeyConfig:
    """冻结的 geometry + elapsed-step 因果 key 协议。"""

    schema_version: str
    seed: int
    expected_pairs: int
    expected_train_pairs: int
    action_horizon: int
    label_probe_frames: int
    minimum_label_points: int
    maximum_centroid_error_m: float
    bootstrap_samples: int
    minimum_recoverable_gap_fraction: float

    @classmethod
    def from_json(cls, path: Path) -> "CausalDemoModeKeyConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def __post_init__(self) -> None:
        if self.schema_version != "causal-demo-mode-key-v1":
            raise ValueError("未知 causal Demo mode key schema")
        positive = (
            self.expected_pairs,
            self.expected_train_pairs,
            self.action_horizon,
            self.label_probe_frames,
            self.minimum_label_points,
            self.maximum_centroid_error_m,
            self.bootstrap_samples,
            self.minimum_recoverable_gap_fraction,
        )
        if min(positive) <= 0 or self.seed < 0:
            raise ValueError("causal Demo mode key 配置非法")
        if self.minimum_recoverable_gap_fraction > 1:
            raise ValueError("minimum recoverable gap fraction 不能超过 1")


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _normalizer(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.asarray(scale)
    scale[scale < 1e-6] = 1.0
    return mean, scale


def _stable_nearest(
    distances: np.ndarray,
    pool: list[RunRecord],
    eligible: list[int],
) -> int:
    return min(
        eligible,
        key=lambda index: (
            float(distances[index]),
            pool[index].pair_id,
            pool[index].run_index,
        ),
    )


def _evaluate_split(
    *,
    queries: list[RunRecord],
    pool: list[RunRecord],
    geometry_mean: np.ndarray,
    geometry_scale: np.ndarray,
    elapsed_mean: np.ndarray,
    elapsed_scale: np.ndarray,
    config: CausalDemoModeKeyConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pool_geometry = np.stack(
        [
            (record.initial_key - geometry_mean) / geometry_scale
            for record in pool
        ]
    )
    pool_elapsed = np.asarray(
        [[record.onset_step] for record in pool], dtype=np.float64
    )
    pool_elapsed = (pool_elapsed - elapsed_mean) / elapsed_scale
    pool_joint = np.concatenate((pool_geometry, pool_elapsed), axis=1)
    methods: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "hold",
            "tcp",
            "pool_majority",
            "elapsed_top1",
            "geometry_top1",
            "geometry_elapsed_top1",
            "query_oracle",
        )
    }
    run_means: dict[str, list[float]] = {name: [] for name in methods}
    accuracies = {
        "elapsed_top1": [],
        "geometry_top1": [],
        "geometry_elapsed_top1": [],
    }
    majority_matches: list[bool] = []
    query_rows = []
    for query in queries:
        eligible = [
            index
            for index, candidate in enumerate(pool)
            if candidate.operation == query.operation
        ]
        if not eligible:
            raise ValueError(f"operation={query.operation} 没有 Demo run")
        pool_tcp_fraction = float(
            np.mean([pool[index].oracle_tcp for index in eligible])
        )
        majority_tcp = pool_tcp_fraction >= 0.5
        query_geometry = (
            query.initial_key - geometry_mean
        ) / geometry_scale
        query_elapsed = (
            np.asarray([query.onset_step], dtype=np.float64) - elapsed_mean
        ) / elapsed_scale
        query_joint = np.concatenate((query_geometry, query_elapsed))
        geometry_distances = np.linalg.norm(
            pool_geometry - query_geometry[None, :], axis=1
        )
        elapsed_distances = np.abs(pool_elapsed[:, 0] - query_elapsed[0])
        joint_distances = np.linalg.norm(
            pool_joint - query_joint[None, :], axis=1
        )
        geometry_index = _stable_nearest(
            geometry_distances, pool, eligible
        )
        elapsed_index = _stable_nearest(elapsed_distances, pool, eligible)
        joint_index = _stable_nearest(joint_distances, pool, eligible)
        modes = {
            "pool_majority": majority_tcp,
            "elapsed_top1": pool[elapsed_index].oracle_tcp,
            "geometry_top1": pool[geometry_index].oracle_tcp,
            "geometry_elapsed_top1": pool[joint_index].oracle_tcp,
            "query_oracle": query.oracle_tcp,
        }
        majority_matches.append(majority_tcp == query.oracle_tcp)
        selected = {
            "hold": query.hold_errors,
            "tcp": query.tcp_errors,
            **{
                name: _errors(query, mode)
                for name, mode in modes.items()
            },
        }
        for name, values in selected.items():
            methods[name].append(values)
            run_means[name].append(float(np.mean(values)))
        for name in accuracies:
            accuracies[name].append(modes[name] == query.oracle_tcp)
        query_rows.append(
            {
                "pair_id": query.pair_id,
                "seed": query.seed,
                "split": query.split,
                "operation": query.operation,
                "run_index": query.run_index,
                "onset_step": query.onset_step,
                "frames": len(query.hold_errors),
                "query_oracle_tcp": query.oracle_tcp,
                "joint_demo_pair_id": pool[joint_index].pair_id,
                "joint_demo_run_index": pool[joint_index].run_index,
                "joint_demo_tcp": pool[joint_index].oracle_tcp,
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
    primary_name = "geometry_elapsed_top1"
    primary_mean = float(summaries[primary_name]["mean"])
    oracle_mean = float(summaries["query_oracle"]["mean"])
    gap = best_fixed_mean - oracle_mean
    recovered = (
        (best_fixed_mean - primary_mean) / gap if gap > 0 else None
    )
    operations = sorted({record.operation for record in pool})
    pool_tcp_fraction = {
        operation: float(
            np.mean(
                [
                    record.oracle_tcp
                    for record in pool
                    if record.operation == operation
                ]
            )
        )
        for operation in operations
    }
    majority_accuracy = float(np.mean(majority_matches))
    mode_accuracy = {
        name: float(np.mean(values)) for name, values in accuracies.items()
    }
    criteria = {
        "mean_below_best_fixed": primary_mean < best_fixed_mean,
        "mean_below_pool_majority": primary_mean
        < float(summaries["pool_majority"]["mean"]),
        "mean_below_elapsed_only": primary_mean
        < float(summaries["elapsed_top1"]["mean"]),
        "mean_below_geometry_only": primary_mean
        < float(summaries["geometry_top1"]["mean"]),
        "p95_not_above_best_fixed": float(summaries[primary_name]["p95"])
        <= float(summaries[best_fixed_name]["p95"]),
        "mode_accuracy_above_pool_majority": mode_accuracy[primary_name]
        > majority_accuracy,
        "recoverable_gap_fraction_at_least_minimum": recovered is not None
        and recovered >= config.minimum_recoverable_gap_fraction,
    }
    report = {
        "runs": len(queries),
        "frames": sum(len(record.hold_errors) for record in queries),
        "pool_tcp_mode_fraction_by_operation": pool_tcp_fraction,
        "pool_majority_accuracy": majority_accuracy,
        "method_error_m": summaries,
        "mode_accuracy": mode_accuracy,
        "best_fixed": best_fixed_name,
        "recoverable_gap_fraction": recovered,
        "paired_run_bootstrap_primary_minus_best_fixed": (
            _bootstrap_difference(
                np.asarray(run_means[primary_name]),
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
    config: CausalDemoModeKeyConfig,
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
    records, pair_counts = _extract_runs(
        replay_root=replay_root,
        audit=audit,
        config=config,
    )
    if pair_counts.get("train") != config.expected_train_pairs:
        raise ValueError("train pair 数量不匹配")
    pool = [record for record in records if record.split == "train"]
    geometry_mean, geometry_scale = _normalizer(
        np.stack([record.initial_key for record in pool])
    )
    elapsed_mean, elapsed_scale = _normalizer(
        np.asarray([[record.onset_step] for record in pool], dtype=np.float64)
    )
    split_reports = {}
    query_rows = []
    for split in ("val", "test"):
        queries = [record for record in records if record.split == split]
        split_report, rows = _evaluate_split(
            queries=queries,
            pool=pool,
            geometry_mean=geometry_mean,
            geometry_scale=geometry_scale,
            elapsed_mean=elapsed_mean,
            elapsed_scale=elapsed_scale,
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
            "pool_split": "train only",
            "query_splits": ["val", "test"],
            "primary_key": "initial observed geometry + elapsed steps",
            "query_future_length_used": False,
            "query_future_action_used": False,
            "privileged_demo_field": "per-run oracle motion mode",
            "privileged_query_fields": "tracking-error audit only",
            "confirmation_result": False,
        },
        "pair_counts": pair_counts,
        "run_counts": {
            split: sum(record.split == split for record in records)
            for split in sorted(pair_counts)
        },
        "train_normalization": {
            "geometry_mean": geometry_mean.tolist(),
            "geometry_scale": geometry_scale.tolist(),
            "elapsed_mean": elapsed_mean.tolist(),
            "elapsed_scale": elapsed_scale.tolist(),
        },
        "splits": split_reports,
        "advance_to_demo_mode_extractor": all(
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
                "advance_to_demo_mode_extractor": report[
                    "advance_to_demo_mode_extractor"
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
        config=CausalDemoModeKeyConfig.from_json(resolved_config),
    )

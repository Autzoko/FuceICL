"""确认 dual-view 缺失时 TCP propagation 构成可靠因果物体 belief。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from dev.simulator.evaluate_dual_view_observability import _camera_variants


@dataclass(frozen=True)
class VisibilityConditionedBeliefConfig:
    """冻结的外部确认数量、统计与误差门槛。"""

    schema_version: str
    seed: int
    expected_pairs: int
    expected_camera_variant: str
    expected_source_report_sha256: str
    bootstrap_samples: int
    minimum_missing_trajectories: int
    maximum_tcp_oracle_gap_m: float
    maximum_combined_mean_error_m: float
    maximum_visible_centroid_error_m: float

    @classmethod
    def from_json(
        cls, path: Path
    ) -> "VisibilityConditionedBeliefConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def __post_init__(self) -> None:
        if self.schema_version != "visibility-conditioned-belief-v1":
            raise ValueError("未知 visibility-conditioned belief schema")
        if self.expected_camera_variant != "dual-fixed-v1":
            raise ValueError("confirmation camera variant 不匹配")
        positive = (
            self.expected_pairs,
            self.bootstrap_samples,
            self.minimum_missing_trajectories,
            self.maximum_tcp_oracle_gap_m,
            self.maximum_combined_mean_error_m,
            self.maximum_visible_centroid_error_m,
        )
        if min(positive) <= 0 or self.seed < 0:
            raise ValueError("belief confirmation 配置非法")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
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


def _paired_bootstrap(
    primary: np.ndarray,
    baseline: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    if primary.shape != baseline.shape or primary.ndim != 1:
        raise ValueError("paired bootstrap 输入 shape 不匹配")
    if len(primary) == 0:
        raise ValueError("paired bootstrap 输入为空")
    difference = primary - baseline
    generator = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        draw = generator.integers(0, len(difference), size=len(difference))
        means[index] = float(np.mean(difference[draw]))
    return {
        "mean_m": float(np.mean(difference)),
        "ci95_low_m": float(np.quantile(means, 0.025)),
        "ci95_high_m": float(np.quantile(means, 0.975)),
    }


def _tracker_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for trajectory in report.get("trajectories", []):
        tracker = trajectory.get("tracker", {})
        if tracker.get("tcp_delta_error_m", {}).get("count", 0) <= 0:
            continue
        rows.append(
            {
                "pair_id": int(trajectory["pair_id"]),
                "seed": int(trajectory["seed"]),
                "operation": str(trajectory["operation"]),
                "split": str(trajectory["split"]),
                "missing_frames": int(
                    tracker["tcp_delta_error_m"]["count"]
                ),
                "hold_mean_error_m": float(
                    tracker["hold_error_m"]["mean"]
                ),
                "tcp_mean_error_m": float(
                    tracker["tcp_delta_error_m"]["mean"]
                ),
                "oracle_mean_error_m": float(
                    tracker["oracle_switch_error_m"]["mean"]
                ),
            }
        )
    return rows


def run(
    *,
    project_root: Path,
    config_path: Path,
    source_report_path: Path,
    replay_root: Path,
    audit_path: Path,
    output_path: Path,
    config: VisibilityConditionedBeliefConfig,
) -> None:
    if output_path.exists():
        raise FileExistsError(f"输出文件已存在，拒绝覆盖：{output_path}")
    if _sha256(source_report_path) != config.expected_source_report_sha256:
        raise ValueError("confirmation source report hash 不匹配")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    replay_audit_path = replay_root / "replay_audit_report.json"
    replay_audit = json.loads(replay_audit_path.read_text(encoding="utf-8"))
    if audit.get("config", {}).get("expected_pairs") != config.expected_pairs:
        raise ValueError("confirmation audit pair 数量不匹配")
    summary = audit.get("summary", {})
    tracker = summary.get("tracker_audit", {})
    visible = summary.get("visible_active_centroid_error_m", {})
    rows = _tracker_rows(audit)
    hold = np.asarray(
        [row["hold_mean_error_m"] for row in rows], dtype=np.float64
    )
    tcp = np.asarray(
        [row["tcp_mean_error_m"] for row in rows], dtype=np.float64
    )
    bootstrap = _paired_bootstrap(
        tcp,
        hold,
        samples=config.bootstrap_samples,
        seed=config.seed,
    )
    missing = tracker["tcp_delta_error_m"]
    oracle = tracker["oracle_switch_error_m"]
    hold_summary = tracker["hold_error_m"]
    visible_count = int(visible["count"])
    missing_count = int(missing["count"])
    combined_count = visible_count + missing_count
    combined_mean = (
        visible_count * float(visible["mean"])
        + missing_count * float(missing["mean"])
    ) / combined_count
    camera_variants = _camera_variants(replay_root)
    criteria = {
        "replay_audit_passed": bool(
            replay_audit.get("summary", {}).get(
                "all_criteria_passed", False
            )
        ),
        "minimum_missing_trajectories_met": len(rows)
        >= config.minimum_missing_trajectories,
        "camera_metadata_valid": all(
            variant == config.expected_camera_variant
            for values in camera_variants.values()
            for variant in values.values()
        ),
        "combined_frame_count_valid": combined_count
        == int(summary.get("query_frames", -1)),
        "tcp_mean_below_hold": float(missing["mean"])
        < float(hold_summary["mean"]),
        "paired_bootstrap_ci_below_zero": bootstrap["ci95_high_m"] < 0,
        "tcp_p95_not_above_hold": float(missing["p95"])
        <= float(hold_summary["p95"]),
        "tcp_oracle_gap_within_limit": (
            float(missing["mean"]) - float(oracle["mean"])
        )
        <= config.maximum_tcp_oracle_gap_m,
        "combined_mean_within_limit": combined_mean
        <= config.maximum_combined_mean_error_m,
        "visible_centroid_maximum_within_limit": float(visible["maximum"])
        <= config.maximum_visible_centroid_error_m,
    }
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(project_root),
        "config": asdict(config),
        "config_sha256": _sha256(config_path),
        "source_report_sha256": _sha256(source_report_path),
        "replay_audit_sha256": _sha256(replay_audit_path),
        "occlusion_audit_sha256": _sha256(audit_path),
        "camera_variants": camera_variants,
        "missing_trajectories": len(rows),
        "missing_frames": missing_count,
        "visible_frames": visible_count,
        "hold_error_m": hold_summary,
        "tcp_error_m": missing,
        "oracle_error_m": oracle,
        "visible_centroid_error_m": visible,
        "combined_belief_mean_error_m": combined_mean,
        "paired_trajectory_bootstrap_tcp_minus_hold": bootstrap,
        "criteria": criteria,
        "all_criteria_passed": all(criteria.values()),
        "trajectories": rows,
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
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not report["all_criteria_passed"]:
        raise RuntimeError("visibility-conditioned belief 未通过确认门槛")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    resolved_config = arguments.config.resolve()
    run(
        project_root=arguments.project_root.resolve(),
        config_path=resolved_config,
        source_report_path=arguments.source_report.resolve(),
        replay_root=arguments.replay_root.resolve(),
        audit_path=arguments.audit.resolve(),
        output_path=arguments.output.resolve(),
        config=VisibilityConditionedBeliefConfig.from_json(resolved_config),
    )

"""对冻结单/双视角遮挡报告执行可观测性门控。"""

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


OPERATIONS = ("toward", "away")
REPLAY_STEM = "trajectory.pointcloud.pd_ee_delta_pose.physx_cpu"


@dataclass(frozen=True)
class DualViewObservabilityConfig:
    """固定双视角 intervention 的数据与可见性门槛。"""

    schema_version: str
    expected_camera_variant: str
    expected_query_frames: int
    expected_baseline_report_sha256: str
    maximum_below_threshold_fraction: float
    maximum_missing_run_frames: int
    minimum_fixed_progress_visible_fraction: float
    maximum_centroid_error_m: float

    @classmethod
    def from_json(cls, path: Path) -> "DualViewObservabilityConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def __post_init__(self) -> None:
        if self.schema_version != "dual-view-observability-v1":
            raise ValueError("未知 dual-view observability schema")
        if self.expected_camera_variant != "dual-fixed-v1":
            raise ValueError("camera variant 不匹配")
        if self.expected_query_frames <= 0 or self.maximum_missing_run_frames < 0:
            raise ValueError("frame 数量或 missing-run 门槛非法")
        probabilities = (
            self.maximum_below_threshold_fraction,
            self.minimum_fixed_progress_visible_fraction,
        )
        if any(not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("visibility fraction 必须位于 [0,1]")
        if self.maximum_centroid_error_m <= 0:
            raise ValueError("centroid error 门槛必须为正")


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


def _camera_variants(replay_root: Path) -> dict[str, dict[str, str | None]]:
    variants = {}
    for operation in OPERATIONS:
        variants[operation] = {}
        for kind, name in (
            ("source_copy", "trajectory.json"),
            ("replay", f"{REPLAY_STEM}.json"),
        ):
            metadata = json.loads(
                (replay_root / operation / name).read_text(encoding="utf-8")
            )
            variants[operation][kind] = (
                metadata.get("env_info", {})
                .get("env_kwargs", {})
                .get("camera_variant")
            )
    return variants


def _visibility_metrics(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    visibility = report.get("summary", {}).get("visibility", {})
    result = {}
    for role in ("active", "anchor"):
        metrics = visibility.get(role)
        if not isinstance(metrics, dict):
            raise ValueError(f"candidate report 缺少 {role} visibility")
        progress = metrics.get("fixed_progress", [])
        if not progress:
            raise ValueError(f"candidate report 缺少 {role} fixed progress")
        result[role] = {
            "below_threshold_fraction": float(
                metrics["below_threshold_fraction"]
            ),
            "maximum_missing_run_frames": int(
                metrics["maximum_missing_run_frames"]
            ),
            "minimum_fixed_progress_visible_fraction": min(
                float(row["visible_fraction"]) for row in progress
            ),
        }
    return result


def run(
    *,
    project_root: Path,
    config_path: Path,
    baseline_path: Path,
    candidate_path: Path,
    replay_root: Path,
    output_path: Path,
    config: DualViewObservabilityConfig,
) -> None:
    if output_path.exists():
        raise FileExistsError(f"输出文件已存在，拒绝覆盖：{output_path}")
    if _sha256(baseline_path) != config.expected_baseline_report_sha256:
        raise ValueError("单视角 baseline report hash 不匹配")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    replay_audit_path = replay_root / "replay_audit_report.json"
    replay_audit = json.loads(replay_audit_path.read_text(encoding="utf-8"))
    camera_variants = _camera_variants(replay_root)
    candidate_metrics = _visibility_metrics(candidate)
    baseline_metrics = _visibility_metrics(baseline)
    query_frames = int(candidate.get("summary", {}).get("query_frames", -1))
    centroid_maximum = candidate.get(
        "summary", {}
    ).get("visible_active_centroid_error_m", {}).get("maximum")
    if centroid_maximum is None:
        raise ValueError("candidate report 缺少 visible centroid error")
    metadata_valid = all(
        variant == config.expected_camera_variant
        for values in camera_variants.values()
        for variant in values.values()
    )
    criteria = {
        "replay_audit_passed": bool(
            replay_audit.get("summary", {}).get(
                "all_criteria_passed", False
            )
        ),
        "camera_metadata_valid": metadata_valid,
        "query_frame_count_valid": query_frames
        == config.expected_query_frames,
        "below_threshold_fraction_valid": all(
            values["below_threshold_fraction"]
            <= config.maximum_below_threshold_fraction
            for values in candidate_metrics.values()
        ),
        "missing_run_valid": all(
            values["maximum_missing_run_frames"]
            <= config.maximum_missing_run_frames
            for values in candidate_metrics.values()
        ),
        "fixed_progress_coverage_valid": all(
            values["minimum_fixed_progress_visible_fraction"]
            >= config.minimum_fixed_progress_visible_fraction
            for values in candidate_metrics.values()
        ),
        "centroid_error_valid": float(centroid_maximum)
        <= config.maximum_centroid_error_m,
        "strictly_better_than_single_view": all(
            candidate_metrics[role]["below_threshold_fraction"]
            < baseline_metrics[role]["below_threshold_fraction"]
            for role in candidate_metrics
        ),
    }
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(project_root),
        "config": asdict(config),
        "config_sha256": _sha256(config_path),
        "baseline_report_sha256": _sha256(baseline_path),
        "candidate_report_sha256": _sha256(candidate_path),
        "replay_audit_sha256": _sha256(replay_audit_path),
        "camera_variants": camera_variants,
        "query_frames": query_frames,
        "baseline_visibility": baseline_metrics,
        "candidate_visibility": candidate_metrics,
        "visible_active_centroid_maximum_error_m": float(centroid_maximum),
        "criteria": criteria,
        "all_criteria_passed": all(criteria.values()),
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
        raise RuntimeError("dual-view observability 未通过预注册门槛")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    resolved_config = arguments.config.resolve()
    run(
        project_root=arguments.project_root.resolve(),
        config_path=resolved_config,
        baseline_path=arguments.baseline.resolve(),
        candidate_path=arguments.candidate.resolve(),
        replay_root=arguments.replay_root.resolve(),
        output_path=arguments.output.resolve(),
        config=DualViewObservabilityConfig.from_json(resolved_config),
    )

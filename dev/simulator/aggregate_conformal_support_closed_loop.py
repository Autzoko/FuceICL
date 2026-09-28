"""汇总三任务 RSCC closed-loop 预注册结果。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from dev.pointnet.compare_retrievers import _sha256
from dev.simulator.aggregate_query_aligned_closed_loop import (
    EXPECTED_TASKS,
    _git_commit,
    _paired_bootstrap,
    _stats,
    _successes,
)


CONFORMAL_POLICIES = (
    "phase_matched_copy_h6",
    "bcsg_h6",
    "codra_h6",
    "rscc_h6",
)


def _pooled_acceptance_quartiles(
    reports: Sequence[dict[str, Any]],
    policy: str,
) -> list[dict[str, float | int | None]]:
    values: list[list[bool]] = [[] for _ in range(4)]
    for report in reports:
        maximum = int(report["config"]["max_episode_steps"])
        for rollout in report["rollouts"][policy]:
            for step in rollout["step_records"]:
                accepted = step.get("residual_accepted")
                if accepted is None:
                    continue
                quartile = min(3, (int(step["step"]) - 1) * 4 // maximum)
                values[quartile].append(bool(accepted))
    return [
        {
            "quartile": index,
            "replans": len(rows),
            "acceptance": None if not rows else float(np.mean(rows)),
        }
        for index, rows in enumerate(values)
    ]


def run(
    *,
    project_root: Path,
    report_paths: Sequence[Path],
    output_path: Path,
) -> None:
    if output_path.exists():
        raise FileExistsError(f"输出已存在，拒绝覆盖：{output_path}")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in report_paths]
    tasks = [str(report["task"]) for report in reports]
    if len(reports) != 3 or set(tasks) != EXPECTED_TASKS:
        raise ValueError(f"必须恰好覆盖三任务：{tasks}")
    shared_fields = (
        "config_sha256",
        "checkpoint_sha256",
        "checkpoint_git_commit",
    )
    for field in shared_fields:
        if len({str(report[field]) for report in reports}) != 1:
            raise ValueError(f"三任务 {field} 不一致")
    calibration_hashes = {
        str(report.get("calibration", {}).get("sha256")) for report in reports
    }
    if len(calibration_hashes) != 1 or "None" in calibration_hashes:
        raise ValueError("三任务 calibration report provenance 不一致")
    seeds = reports[0]["config"]["seeds"]
    if any(report["config"]["seeds"] != seeds for report in reports[1:]):
        raise ValueError("三任务 seeds 不一致")
    if tuple(reports[0]["config"]["policies"]) != CONFORMAL_POLICIES:
        raise ValueError("RSCC closed-loop policy 协议不一致")

    vectors = {
        policy: np.concatenate([_successes(report, policy) for report in reports])
        for policy in CONFORMAL_POLICIES
    }
    pooled = {
        policy: {
            "successes": int(values.sum()),
            "episodes": len(values),
            "success_rate": float(values.mean()),
        }
        for policy, values in vectors.items()
    }
    pairs = (
        ("phase_matched_copy_h6", "bcsg_h6"),
        ("phase_matched_copy_h6", "codra_h6"),
        ("phase_matched_copy_h6", "rscc_h6"),
        ("bcsg_h6", "codra_h6"),
        ("bcsg_h6", "rscc_h6"),
        ("codra_h6", "rscc_h6"),
    )
    resamples = int(reports[0]["config"]["bootstrap_resamples"])
    seed = int(reports[0]["config"]["seed_generation_seed"])
    comparisons = {
        f"{candidate}_minus_{reference}": _paired_bootstrap(
            vectors[reference],
            vectors[candidate],
            seed=seed + offset,
            resamples=resamples,
        )
        for offset, (reference, candidate) in enumerate(pairs)
    }
    task_rates = {
        report["task"]: {
            policy: float(_successes(report, policy).mean())
            for policy in CONFORMAL_POLICIES
        }
        for report in reports
    }
    latencies = [
        float(step["predictor_latency_ms"])
        for report in reports
        for rollout in report["rollouts"]["rscc_h6"]
        for step in rollout["step_records"]
        if step.get("predictor_latency_ms") is not None
    ]
    latency = _stats(latencies)
    quartiles = _pooled_acceptance_quartiles(reports, "rscc_h6")
    quartile_values = [row["acceptance"] for row in quartiles]
    monotone = all(
        current is not None
        and following is not None
        and current >= following
        for current, following in zip(quartile_values, quartile_values[1:])
    )
    exact_fallback = all(
        report["policy_diagnostics"]["rscc_h6"][
            "rejected_prediction_max_abs_error_from_copy"
        ]
        == 0.0
        for report in reports
    )
    structural = all(
        report["initial_signature_max_abs_difference"] == 0.0
        and report["bank"]["chunk_ids_match_checkpoint"]
        and all(
            value
            for key, value in report["structural_audit"].items()
            if key.endswith("exact")
        )
        for report in reports
    )
    rscc_copy = comparisons["rscc_h6_minus_phase_matched_copy_h6"]
    rscc_bcsg = comparisons["rscc_h6_minus_bcsg_h6"]
    task_deltas = [
        values["rscc_h6"] - values["phase_matched_copy_h6"]
        for values in task_rates.values()
    ]
    criteria = {
        "c1_rscc_not_below_copy": rscc_copy["ci95_low"] >= 0.0,
        "c2_rscc_better_than_bcsg": rscc_bcsg["ci95_low"] > 0.0,
        "c3_at_least_two_tasks_not_below_copy": (
            sum(delta >= 0.0 for delta in task_deltas) >= 2
        ),
        "c3_no_task_below_copy_by_over_5pp": min(task_deltas) >= -0.05,
        "c4_acceptance_monotone_decreasing": monotone,
        "c4_last_acceptance_below_first": (
            quartile_values[0] is not None
            and quartile_values[-1] is not None
            and quartile_values[-1] < quartile_values[0]
        ),
        "c5_structural_and_exact_fallback": structural and exact_fallback,
        "c5_predictor_latency_p95_below_5ms": (
            latency is not None and latency["p95"] < 5.0
        ),
    }
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(project_root),
        "protocol": {
            "tasks": sorted(EXPECTED_TASKS),
            "task_seed_pairs": len(vectors[CONFORMAL_POLICIES[0]]),
            "bootstrap_unit": "task + seed paired across policies",
            "bootstrap_resamples": resamples,
        },
        "source_reports": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in report_paths
        ],
        "shared_provenance": {
            **{field: reports[0][field] for field in shared_fields},
            "calibration_report_sha256": next(iter(calibration_hashes)),
        },
        "pooled": pooled,
        "task_success_rates": task_rates,
        "paired_bootstrap": comparisons,
        "rscc_acceptance_by_time_quartile": quartiles,
        "rscc_predictor_latency_ms": latency,
        "criteria": criteria,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(f"{output_path.suffix}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(
        project_root=arguments.project_root.resolve(),
        report_paths=[path.resolve() for path in arguments.report],
        output_path=arguments.output.resolve(),
    )

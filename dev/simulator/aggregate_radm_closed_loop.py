"""汇总三任务 RADM closed-loop 冻结结果。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from dev.pointnet.compare_retrievers import _sha256
from dev.simulator.aggregate_query_aligned_closed_loop import (
    EXPECTED_TASKS,
    _git_commit,
    _paired_bootstrap,
    _stats,
)


POLICIES = (
    "bcsg_h6",
    "retriever_prior_mixture_h6",
    "radm_h6",
)


def _successes(report: Mapping[str, Any], policy: str) -> np.ndarray:
    seeds = report["config"]["seeds"]
    by_seed = {
        int(row["seed"]): float(row["success"])
        for row in report["rollouts"][policy]
    }
    if set(by_seed) != set(seeds):
        raise ValueError(f"{report['task']} {policy} seeds 不完整")
    return np.asarray([by_seed[int(seed)] for seed in seeds])


def _mixture_structure(report: Mapping[str, Any]) -> bool:
    diagnostics = report["policy_diagnostics"]["radm_h6"]
    odds = diagnostics["maximum_odds_distortion"]
    return bool(
        odds is not None
        and odds["max"] <= 4.0 + 1e-6
        and diagnostics["translation_convex_hull_max_violation"] <= 1e-6
        and diagnostics["discrete_prior_max_abs_error"] == 0.0
    )


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
        "radm_checkpoint_sha256",
        "checkpoint_git_commit",
    )
    for field in shared_fields:
        if len({str(report[field]) for report in reports}) != 1:
            raise ValueError(f"三任务 {field} 不一致")
    seeds = reports[0]["config"]["seeds"]
    if any(report["config"]["seeds"] != seeds for report in reports[1:]):
        raise ValueError("三任务 closed-loop seeds 不一致")
    if any(tuple(report["config"]["policies"]) != POLICIES for report in reports):
        raise ValueError("RADM closed-loop policies 不一致")

    vectors = {
        policy: np.concatenate([_successes(report, policy) for report in reports])
        for policy in POLICIES
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
        ("bcsg_h6", "retriever_prior_mixture_h6"),
        ("bcsg_h6", "radm_h6"),
        ("retriever_prior_mixture_h6", "radm_h6"),
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
            for policy in POLICIES
        }
        for report in reports
    }
    radm_vs_rank1 = comparisons["radm_h6_minus_bcsg_h6"]
    radm_vs_prior = comparisons[
        "radm_h6_minus_retriever_prior_mixture_h6"
    ]
    radm_latencies = [
        float(step["predictor_latency_ms"])
        for report in reports
        for rollout in report["rollouts"]["radm_h6"]
        for step in rollout["step_records"]
        if step.get("predictor_latency_ms") is not None
    ]
    latency = _stats(radm_latencies)
    task_consistency = sum(
        rates["radm_h6"] >= rates["bcsg_h6"]
        for rates in task_rates.values()
    ) >= 2
    structural = all(
        report["initial_signature_max_abs_difference"] == 0.0
        and report["bank"]["chunk_ids_match_checkpoint"]
        and report["radm_checkpoint_sha256"] is not None
        and all(
            value
            for key, value in report["structural_audit"].items()
            if key.endswith("exact")
        )
        and report["structural_audit"]["radm_maximum_odds_distortion"] == 4.0
        and _mixture_structure(report)
        for report in reports
    )
    criteria = {
        "cl1_radm_significantly_better_than_rank1": (
            radm_vs_rank1["ci95_low"] > 0.0
        ),
        "cl2_radm_point_not_below_fixed_prior": (
            radm_vs_prior["candidate_minus_reference"] >= 0.0
        ),
        "cl2_radm_noninferior_to_fixed_prior_margin_5pp": (
            radm_vs_prior["ci95_low"] > -0.05
        ),
        "cl3_at_least_two_tasks_not_below_rank1": task_consistency,
        "cl4_structural": structural,
        "cl4_predictor_latency_p95_below_8ms": (
            latency is not None and latency["p95"] < 8.0
        ),
    }
    output = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(project_root),
        "protocol": {
            "tasks": sorted(EXPECTED_TASKS),
            "task_seed_pairs": len(vectors[POLICIES[0]]),
            "bootstrap_unit": "task + seed paired across policies",
            "bootstrap_resamples": resamples,
        },
        "source_reports": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in report_paths
        ],
        "shared_provenance": {
            field: reports[0][field] for field in shared_fields
        },
        "pooled": pooled,
        "task_success_rates": task_rates,
        "paired_bootstrap": comparisons,
        "radm_predictor_latency_ms": latency,
        "preregistered_criteria": criteria,
        "closed_loop_passed": all(criteria.values()),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(f"{output_path.suffix}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)


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

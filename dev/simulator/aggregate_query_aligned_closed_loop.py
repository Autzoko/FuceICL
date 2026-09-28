"""汇总三任务 QA-LRDT/BCSG closed-loop 预注册结果。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np

from dev.pointnet.compare_retrievers import _sha256


EXPECTED_TASKS = frozenset(("PickCube-v1", "PushCube-v1", "PullCube-v1"))
POLICIES = (
    "phase_matched_copy_h6",
    "phase_factorized_transport_h6",
    "fixed_low_rank_transport_h6",
    "query_aligned_transport_h6",
    "bcsg_h6",
)


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _stats(values: Sequence[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(array),
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def _paired_bootstrap(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    seed: int,
    resamples: int,
) -> dict[str, float | int]:
    if reference.shape != candidate.shape or reference.ndim != 1:
        raise ValueError("paired bootstrap 输入 shape 不一致")
    rng = np.random.default_rng(seed)
    deltas = candidate - reference
    sampled_indices = rng.integers(
        0,
        len(deltas),
        size=(resamples, len(deltas)),
    )
    sampled = deltas[sampled_indices].mean(axis=1)
    return {
        "candidate_minus_reference": float(deltas.mean()),
        "ci95_low": float(np.quantile(sampled, 0.025)),
        "ci95_high": float(np.quantile(sampled, 0.975)),
        "candidate_wins": int(np.sum(deltas > 0)),
        "reference_wins": int(np.sum(deltas < 0)),
        "ties": int(np.sum(deltas == 0)),
        "num_seed_pairs": len(deltas),
        "num_resamples": resamples,
    }


def _successes(
    report: Mapping[str, Any],
    policy: str,
) -> np.ndarray:
    seeds = report["config"]["seeds"]
    by_seed = {
        int(row["seed"]): float(row["success"])
        for row in report["rollouts"][policy]
    }
    if set(by_seed) != set(seeds):
        raise ValueError(f"{report['task']} {policy} seeds 不完整")
    return np.asarray([by_seed[int(seed)] for seed in seeds])


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
    seeds = reports[0]["config"]["seeds"]
    if any(report["config"]["seeds"] != seeds for report in reports[1:]):
        raise ValueError("三任务 seeds 不一致")
    if tuple(reports[0]["config"]["policies"]) != POLICIES:
        raise ValueError("closed-loop policy 协议不一致")

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
        ("phase_matched_copy_h6", "phase_factorized_transport_h6"),
        ("phase_matched_copy_h6", "fixed_low_rank_transport_h6"),
        ("phase_matched_copy_h6", "query_aligned_transport_h6"),
        ("phase_matched_copy_h6", "bcsg_h6"),
        ("fixed_low_rank_transport_h6", "bcsg_h6"),
        ("query_aligned_transport_h6", "bcsg_h6"),
        ("phase_factorized_transport_h6", "bcsg_h6"),
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
    bcsg_vs_copy = comparisons[
        "bcsg_h6_minus_phase_matched_copy_h6"
    ]
    bcsg_vs_qa = comparisons[
        "bcsg_h6_minus_query_aligned_transport_h6"
    ]
    predictor_latencies = [
        float(step["predictor_latency_ms"])
        for report in reports
        for rollout in report["rollouts"]["bcsg_h6"]
        for step in rollout["step_records"]
        if step.get("predictor_latency_ms") is not None
    ]
    latency = _stats(predictor_latencies)
    at_least_two_tasks = sum(
        rates["bcsg_h6"] >= rates["query_aligned_transport_h6"]
        for rates in task_rates.values()
    ) >= 2
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
    criteria = {
        "cl1_bcsg_significantly_better_than_phase_copy": (
            bcsg_vs_copy["ci95_low"] > 0.0
        ),
        "cl2_bcsg_point_not_below_qa": (
            bcsg_vs_qa["candidate_minus_reference"] >= 0.0
        ),
        "cl2_bcsg_noninferior_to_qa_margin_5pp": (
            bcsg_vs_qa["ci95_low"] > -0.05
        ),
        "cl3_at_least_two_tasks_not_below_qa": at_least_two_tasks,
        "cl4_structural": structural,
        "cl4_predictor_latency_p95_below_5ms": (
            latency is not None and latency["p95"] < 5.0
        ),
    }
    report = {
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
        "bcsg_predictor_latency_ms": latency,
        "preregistered_criteria": criteria,
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


def main() -> None:
    arguments = parse_args()
    run(
        project_root=arguments.project_root.resolve(),
        report_paths=[path.resolve() for path in arguments.report],
        output_path=arguments.output.resolve(),
    )


if __name__ == "__main__":
    main()

"""分析冻结 ManiSkill rollouts 中的 Retriever support drift。"""

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
from dev.simulator.aggregate_query_aligned_closed_loop import (
    EXPECTED_TASKS,
    POLICIES,
)


TIME_BINS = (
    (1, 24, "step_001_024"),
    (25, 48, "step_025_048"),
    (49, 72, "step_049_072"),
    (73, 100, "step_073_100"),
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


def _rate(values: Sequence[bool]) -> dict[str, float | int] | None:
    if not values:
        return None
    return {
        "count": len(values),
        "positive": int(sum(values)),
        "rate": float(np.mean(values)),
    }


def _correlations(left: Sequence[float], right: Sequence[float]) -> dict[str, Any]:
    if len(left) != len(right):
        raise ValueError("correlation 输入长度不一致")
    if len(left) < 3:
        return {"count": len(left), "pearson": None, "rank": None}
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if np.ptp(x) <= 1e-12 or np.ptp(y) <= 1e-12:
        pearson = None
    else:
        pearson = float(np.corrcoef(x, y)[0, 1])

    def average_ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="stable")
        sorted_values = values[order]
        ranks = np.empty(len(values), dtype=np.float64)
        start = 0
        while start < len(values):
            stop = start + 1
            while stop < len(values) and sorted_values[stop] == sorted_values[start]:
                stop += 1
            ranks[order[start:stop]] = 0.5 * (start + stop - 1)
            start = stop
        return ranks

    x_rank = average_ranks(x)
    y_rank = average_ranks(y)
    rank = (
        None
        if np.ptp(x_rank) <= 1e-12 or np.ptp(y_rank) <= 1e-12
        else float(np.corrcoef(x_rank, y_rank)[0, 1])
    )
    return {"count": len(x), "pearson": pearson, "rank": rank}


def _time_bin(step: int) -> str:
    for low, high, name in TIME_BINS:
        if low <= step <= high:
            return name
    raise ValueError(f"step={step} 超出冻结分析范围")


def _first_replan(rollout: Mapping[str, Any]) -> Mapping[str, Any]:
    for step in rollout["step_records"]:
        if step["replanned"] and step["selected_chunk_id"] is not None:
            return step
    raise ValueError(f"seed={rollout['seed']} 没有有效 first replan")


def _initial_retrieval_audit(report: Mapping[str, Any]) -> dict[str, Any]:
    by_policy = {
        policy: {
            int(rollout["seed"]): _first_replan(rollout)
            for rollout in report["rollouts"][policy]
        }
        for policy in POLICIES
    }
    seeds = [int(seed) for seed in report["config"]["seeds"]]
    mismatched_chunks = []
    maximum_distance_delta = 0.0
    for seed in seeds:
        chunks = {
            str(by_policy[policy][seed]["selected_chunk_id"])
            for policy in POLICIES
        }
        distances = [
            float(by_policy[policy][seed]["retrieval_distance"])
            for policy in POLICIES
        ]
        if len(chunks) != 1:
            mismatched_chunks.append(seed)
        maximum_distance_delta = max(
            maximum_distance_delta,
            max(distances) - min(distances),
        )
    return {
        "task_seed_pairs": len(seeds),
        "same_chunk_for_all_policies": not mismatched_chunks,
        "mismatched_chunk_seeds": mismatched_chunks,
        "maximum_distance_delta": maximum_distance_delta,
    }


def _policy_rows(
    report: Mapping[str, Any],
    policy: str,
    support_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    action_rows = []
    replan_rows = []
    episode_rows = []
    for rollout in report["rollouts"][policy]:
        episode_actions = []
        for step in rollout["step_records"]:
            distance = step.get("retrieval_distance")
            if distance is None:
                continue
            row = {
                "seed": int(rollout["seed"]),
                "success": bool(rollout["success"]),
                "step": int(step["step"]),
                "time_bin": _time_bin(int(step["step"])),
                "retrieval_distance": float(distance),
                "ood": float(distance) > support_threshold,
                "translation_clipped": bool(step["translation_clipped"]),
                "predicted_gate": step.get("predicted_gate"),
                "residual": step.get(
                    "normalized_translation_residual_l2_mean"
                ),
            }
            action_rows.append(row)
            episode_actions.append(row)
            if bool(step["replanned"]):
                replan_rows.append(row)
        if not episode_actions:
            raise ValueError(f"{policy} seed={rollout['seed']} 没有检索 action")
        ood_steps = [row["step"] for row in episode_actions if row["ood"]]
        episode_rows.append(
            {
                "seed": int(rollout["seed"]),
                "success": bool(rollout["success"]),
                "initial_distance": episode_actions[0]["retrieval_distance"],
                "initial_ood": episode_actions[0]["ood"],
                "ever_ood": bool(ood_steps),
                "first_ood_step": min(ood_steps) if ood_steps else None,
                "ood_action_fraction": float(
                    np.mean([row["ood"] for row in episode_actions])
                ),
                "maximum_distance": max(
                    row["retrieval_distance"] for row in episode_actions
                ),
            }
        )
    return action_rows, replan_rows, episode_rows


def _summarize_policy(
    action_rows: Sequence[Mapping[str, Any]],
    replan_rows: Sequence[Mapping[str, Any]],
    episode_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_time = {}
    for _, _, name in TIME_BINS:
        rows = [row for row in action_rows if row["time_bin"] == name]
        by_time[name] = {
            "actions": len(rows),
            "retrieval_distance": _stats(
                [float(row["retrieval_distance"]) for row in rows]
            ),
            "ood": _rate([bool(row["ood"]) for row in rows]),
            "translation_clipped": _rate(
                [bool(row["translation_clipped"]) for row in rows]
            ),
        }
    support_groups = {}
    for label, is_ood in (("in_support", False), ("ood", True)):
        actions = [row for row in action_rows if bool(row["ood"]) == is_ood]
        replans = [row for row in replan_rows if bool(row["ood"]) == is_ood]
        gates = [
            float(row["predicted_gate"])
            for row in replans
            if row["predicted_gate"] is not None
        ]
        residuals = [
            float(row["residual"])
            for row in replans
            if row["residual"] is not None
        ]
        support_groups[label] = {
            "actions": len(actions),
            "replans": len(replans),
            "translation_clipped": _rate(
                [bool(row["translation_clipped"]) for row in actions]
            ),
            "predicted_gate": _stats(gates),
            "normalized_translation_residual_l2_mean": _stats(residuals),
        }
    episodes_by_support = {}
    for label, ever_ood in (("never_ood", False), ("ever_ood", True)):
        rows = [row for row in episode_rows if bool(row["ever_ood"]) == ever_ood]
        episodes_by_support[label] = {
            "episodes": len(rows),
            "success": _rate([bool(row["success"]) for row in rows]),
            "initial_distance": _stats(
                [float(row["initial_distance"]) for row in rows]
            ),
            "maximum_distance": _stats(
                [float(row["maximum_distance"]) for row in rows]
            ),
            "ood_action_fraction": _stats(
                [float(row["ood_action_fraction"]) for row in rows]
            ),
            "first_ood_step": _stats(
                [
                    float(row["first_ood_step"])
                    for row in rows
                    if row["first_ood_step"] is not None
                ]
            ),
        }
    episodes_by_initial_support = {}
    for label, initial_ood in (
        ("initial_in_support", False),
        ("initial_ood", True),
    ):
        rows = [
            row
            for row in episode_rows
            if bool(row["initial_ood"]) == initial_ood
        ]
        episodes_by_initial_support[label] = {
            "episodes": len(rows),
            "success": _rate([bool(row["success"]) for row in rows]),
            "initial_distance": _stats(
                [float(row["initial_distance"]) for row in rows]
            ),
            "ever_ood": _rate([bool(row["ever_ood"]) for row in rows]),
            "maximum_distance": _stats(
                [float(row["maximum_distance"]) for row in rows]
            ),
        }
    gates = [
        float(row["predicted_gate"])
        for row in replan_rows
        if row["predicted_gate"] is not None
    ]
    gated_rows = [
        row for row in replan_rows if row["predicted_gate"] is not None
    ]
    return {
        "actions": len(action_rows),
        "replans": len(replan_rows),
        "time_bins": by_time,
        "support_groups": support_groups,
        "episodes_by_support": episodes_by_support,
        "episodes_by_initial_support": episodes_by_initial_support,
        "gate_correlations": {
            "gate_vs_retrieval_distance": _correlations(
                gates,
                [float(row["retrieval_distance"]) for row in gated_rows],
            ),
            "gate_vs_residual": _correlations(
                gates,
                [float(row["residual"]) for row in gated_rows],
            ),
        },
    }


def run(
    *,
    project_root: Path,
    closed_loop_paths: Sequence[Path],
    offline_path: Path,
    output_path: Path,
) -> None:
    if output_path.exists():
        raise FileExistsError(f"输出已存在，拒绝覆盖：{output_path}")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in closed_loop_paths]
    offline = json.loads(offline_path.read_text(encoding="utf-8"))
    tasks = [str(report["task"]) for report in reports]
    if len(reports) != 3 or set(tasks) != EXPECTED_TASKS:
        raise ValueError(f"closed-loop reports 必须覆盖三任务：{tasks}")
    checkpoint_hashes = {report["checkpoint_sha256"] for report in reports}
    if checkpoint_hashes != {offline["checkpoint_sha256"]}:
        raise ValueError("offline/closed-loop checkpoint hash 不一致")

    task_reports = {}
    aggregate_rows = {policy: ([], [], []) for policy in POLICIES}
    for report in reports:
        task = str(report["task"])
        threshold = float(offline["tasks"][task]["validation_distance_p95"])
        policies = {}
        for policy in POLICIES:
            action_rows, replan_rows, episode_rows = _policy_rows(
                report,
                policy,
                threshold,
            )
            policies[policy] = _summarize_policy(
                action_rows,
                replan_rows,
                episode_rows,
            )
            aggregate_rows[policy][0].extend(action_rows)
            aggregate_rows[policy][1].extend(replan_rows)
            aggregate_rows[policy][2].extend(episode_rows)
        task_reports[task] = {
            "support_threshold_validation_distance_p95": threshold,
            "initial_retrieval_audit": _initial_retrieval_audit(report),
            "policies": policies,
        }
    aggregate = {
        policy: _summarize_policy(*aggregate_rows[policy])
        for policy in POLICIES
    }
    bcsg = aggregate["bcsg_h6"]
    first_bin = bcsg["time_bins"][TIME_BINS[0][2]]["ood"]
    last_bin = bcsg["time_bins"][TIME_BINS[-1][2]]["ood"]
    supported_gate = bcsg["support_groups"]["in_support"]["predicted_gate"]
    ood_gate = bcsg["support_groups"]["ood"]["predicted_gate"]
    checks = {
        "all_initial_retrievals_identical": all(
            row["initial_retrieval_audit"]["same_chunk_for_all_policies"]
            and row["initial_retrieval_audit"]["maximum_distance_delta"] == 0.0
            for row in task_reports.values()
        ),
        "bcsg_ood_fraction_increases_first_to_last_bin": (
            first_bin is not None
            and last_bin is not None
            and last_bin["rate"] > first_bin["rate"]
        ),
        "bcsg_gate_does_not_decrease_ood": (
            supported_gate is not None
            and ood_gate is not None
            and ood_gate["mean"] >= supported_gate["mean"]
        ),
        "bcsg_gate_distance_rank_correlation_nonnegative": (
            bcsg["gate_correlations"]["gate_vs_retrieval_distance"]["rank"]
            is not None
            and bcsg["gate_correlations"]["gate_vs_retrieval_distance"]["rank"]
            >= 0.0
        ),
    }
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_type": "post-hoc exploratory; primary closed-loop result known",
        "git_commit": _git_commit(project_root),
        "protocol": {
            "support_reference": (
                "per-task offline final-validation retrieval-distance p95"
            ),
            "time_bins": [
                {"low": low, "high": high, "name": name}
                for low, high, name in TIME_BINS
            ],
            "threshold_selection": "recorded before closed-loop; no tuning",
        },
        "inputs": {
            "offline": {"path": str(offline_path), "sha256": _sha256(offline_path)},
            "closed_loop": [
                {"path": str(path), "sha256": _sha256(path)}
                for path in closed_loop_paths
            ],
            "checkpoint_sha256": offline["checkpoint_sha256"],
        },
        "tasks": task_reports,
        "aggregate": aggregate,
        "mechanistic_checks": checks,
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
    parser.add_argument(
        "--closed-loop-report",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--offline-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    run(
        project_root=arguments.project_root.resolve(),
        closed_loop_paths=[
            path.resolve() for path in arguments.closed_loop_report
        ],
        offline_path=arguments.offline_report.resolve(),
        output_path=arguments.output.resolve(),
    )


if __name__ == "__main__":
    main()

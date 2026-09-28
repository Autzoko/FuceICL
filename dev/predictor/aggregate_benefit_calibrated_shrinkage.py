"""合并 BCSG 三个 task-heldout folds 并执行预注册统计。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from dev.pointnet.compare_retrievers import _sha256
from dev.predictor.train_action_chunks import POSE_SCALES, _physical_metrics
from dev.simulator.evaluate_maniskill_demo_prior import _bootstrap_comparison


MODEL_NAMES = (
    "demo_action_copy",
    "fixed_low_rank_transport",
    "query_aligned_transport",
    "constant_shrinkage",
    "bcsg",
)
PROVENANCE_FIELDS = (
    "config_sha256",
    "context_summary_sha256",
    "action_summary_sha256",
    "retriever_checkpoint_sha256",
    "text_scores_sha256",
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


def _load_fold(root: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    report_path = root / "report.json"
    artifact_path = root / "held_out_gate_predictions.npz"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != 1:
        raise ValueError(f"BCSG report schema 错误：{root}")
    if report["prediction_artifact_sha256"] != _sha256(artifact_path):
        raise ValueError(f"BCSG prediction hash 不匹配：{root}")
    with np.load(artifact_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    if str(arrays["fold_name"][0]) != report["fold_name"]:
        raise ValueError(f"fold name 不匹配：{root}")
    if set(arrays["held_out_tasks"].astype(str)) != set(
        report["held_out_tasks"]
    ):
        raise ValueError(f"held-out tasks 不匹配：{root}")
    target_shape = arrays["target_actions"].shape
    for name in MODEL_NAMES:
        if arrays[name].shape != target_shape:
            raise ValueError(f"prediction shape 不匹配：{root}/{name}")
    if set(arrays["tasks"].astype(str)) - set(report["held_out_tasks"]):
        raise ValueError(f"artifact 混入 source task：{root}")
    return report, arrays


def _shared_provenance(reports: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    shared = {}
    for field in PROVENANCE_FIELDS:
        values = {str(report[field]) for report in reports}
        if len(values) != 1:
            raise ValueError(f"fold provenance 不一致：{field}")
        shared[field] = values.pop()
    return shared


def _comparison(
    *,
    reference: torch.Tensor,
    candidate: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    groups: Sequence[str],
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    return _bootstrap_comparison(
        reference=reference,
        candidate=candidate,
        target=target,
        mask=mask,
        group_ids=groups,
        seed=seed,
        resamples=resamples,
        pose_scales=POSE_SCALES,
        translation_threshold_m=0.05,
        rotation_threshold_rad=0.25,
    )


def _per_task_metrics(
    *,
    tasks: np.ndarray,
    predictions: Mapping[str, torch.Tensor],
    target: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, Any]:
    report = {}
    for task in sorted(set(tasks.astype(str))):
        indices = torch.from_numpy(np.flatnonzero(tasks.astype(str) == task))
        report[task] = {
            "queries": len(indices),
            "metrics": {
                name: _physical_metrics(
                    prediction[indices],
                    target[indices],
                    mask[indices],
                )
                for name, prediction in predictions.items()
            },
        }
    return report


def _structural_verdict(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    folds = []
    for report in reports:
        audit = report["structural_audit"]
        folds.append(
            {
                "fold_name": report["fold_name"],
                "gate_range": (
                    audit["gate_min"] >= 0.0 and audit["gate_max"] <= 1.0
                ),
                "gate_zero_exact": audit["gate_zero_max_abs_error"] == 0.0,
                "gate_one_exact": audit["gate_one_max_abs_error"] == 0.0,
                "no_demo_exact": audit["no_demo_max_abs_output"] == 0.0,
                "rotation_gripper_exact": (
                    audit["rotation_gripper_max_abs_error"] == 0.0
                ),
                "gate_parameter_budget": report["gate_parameters"] < 5_000,
                "latency_budget": report["latency"]["p95_ms"] < 3.0,
            }
        )
    invariant_fields = (
        "gate_range",
        "gate_zero_exact",
        "gate_one_exact",
        "no_demo_exact",
        "rotation_gripper_exact",
    )
    efficiency_fields = ("gate_parameter_budget", "latency_budget")
    return {
        "folds": folds,
        "invariants_passed": all(
            all(bool(fold[field]) for field in invariant_fields)
            for fold in folds
        ),
        "efficiency_passed": all(
            all(bool(fold[field]) for field in efficiency_fields)
            for fold in folds
        ),
    }


def run(
    *,
    project_root: Path,
    fold_roots: Sequence[Path],
    output_path: Path,
    expected_task_count: int,
    seed: int,
    resamples: int,
) -> None:
    if output_path.exists():
        raise FileExistsError(f"输出已存在，拒绝覆盖：{output_path}")
    if len(fold_roots) != 3:
        raise ValueError("预注册协议要求恰好三个 folds")
    loaded = [_load_fold(root) for root in fold_roots]
    reports = [item[0] for item in loaded]
    arrays = [item[1] for item in loaded]
    held_sets = [set(report["held_out_tasks"]) for report in reports]
    overlaps = (
        left & right
        for index, left in enumerate(held_sets)
        for right in held_sets[index + 1 :]
    )
    if any(overlaps):
        raise ValueError("held-out task folds 必须互斥")
    all_tasks = set().union(*held_sets)
    if len(all_tasks) != expected_task_count:
        raise ValueError("held-out task union 数量不符合预注册")
    provenance = _shared_provenance(reports)
    target = torch.from_numpy(
        np.concatenate([fold["target_actions"] for fold in arrays])
    )
    mask = torch.from_numpy(
        np.concatenate([fold["target_masks"] for fold in arrays])
    ).bool()
    tasks = np.concatenate([fold["tasks"] for fold in arrays]).astype(str)
    groups = np.concatenate([fold["group_ids"] for fold in arrays]).astype(str)
    predictions = {
        name: torch.from_numpy(
            np.concatenate([fold[name] for fold in arrays])
        )
        for name in MODEL_NAMES
    }
    comparison_specs = (
        ("bcsg_minus_full_qa", "query_aligned_transport", "bcsg"),
        ("bcsg_minus_copy", "demo_action_copy", "bcsg"),
        ("bcsg_minus_constant", "constant_shrinkage", "bcsg"),
        ("constant_minus_full_qa", "query_aligned_transport", "constant_shrinkage"),
    )
    comparisons = {
        name: _comparison(
            reference=predictions[reference],
            candidate=predictions[candidate],
            target=target,
            mask=mask,
            groups=groups.tolist(),
            seed=seed + offset * 100,
            resamples=resamples,
        )
        for offset, (name, reference, candidate) in enumerate(comparison_specs)
    }
    fold_deltas = []
    for report, fold in zip(reports, arrays):
        fold_target = torch.from_numpy(fold["target_actions"])
        fold_mask = torch.from_numpy(fold["target_masks"]).bool()
        qa_error = _physical_metrics(
            torch.from_numpy(fold["query_aligned_transport"]),
            fold_target,
            fold_mask,
        )["translation_l2_m"]
        bcsg_error = _physical_metrics(
            torch.from_numpy(fold["bcsg"]),
            fold_target,
            fold_mask,
        )["translation_l2_m"]
        fold_deltas.append(
            {
                "fold_name": report["fold_name"],
                "bcsg_minus_full_qa_translation_l2_m": bcsg_error - qa_error,
                "improved": bcsg_error < qa_error,
            }
        )
    structural = _structural_verdict(reports)
    bcsg_qa = comparisons["bcsg_minus_full_qa"]
    bcsg_copy = comparisons["bcsg_minus_copy"]
    bcsg_constant = comparisons["bcsg_minus_constant"]
    criteria = {
        "s1_translation_better_than_full_qa": (
            bcsg_qa["translation_l2_m"]["ci95_high"] < 0.0
        ),
        "s2_translation_better_than_copy": (
            bcsg_copy["translation_l2_m"]["ci95_high"] < 0.0
        ),
        "s2_chunk_not_significantly_worse_than_copy": (
            bcsg_copy["chunk_threshold_accuracy"]["ci95_high"] >= 0.0
        ),
        "s3_point_translation_better_than_constant": (
            bcsg_constant["translation_l2_m"]["candidate_minus_reference"]
            < 0.0
        ),
        "s4_at_least_two_folds_better_than_full_qa": (
            sum(row["improved"] for row in fold_deltas) >= 2
        ),
        "s4_structural_invariants": structural["invariants_passed"],
        "s5_efficiency": structural["efficiency_passed"],
    }
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "fold_count": 3,
            "bootstrap_unit": "task + variation + episode",
            "bootstrap_resamples": resamples,
            "delta": "candidate minus reference",
        },
        "git_commit": _git_commit(project_root),
        "fold_roots": [str(root) for root in fold_roots],
        "fold_report_sha256": [
            _sha256(root / "report.json") for root in fold_roots
        ],
        "fold_prediction_sha256": [
            _sha256(root / "held_out_gate_predictions.npz")
            for root in fold_roots
        ],
        "shared_provenance": provenance,
        "held_out_tasks": sorted(all_tasks),
        "query_count": len(target),
        "episode_groups": len(set(groups.tolist())),
        "metrics": {
            name: _physical_metrics(prediction, target, mask)
            for name, prediction in predictions.items()
        },
        "paired_bootstrap": comparisons,
        "fold_translation_deltas": fold_deltas,
        "per_task": _per_task_metrics(
            tasks=tasks,
            predictions=predictions,
            target=target,
            mask=mask,
        ),
        "fold_gate_statistics": [
            {
                "fold_name": report["fold_name"],
                "constant_gate": report["constant_gate"],
                "held_out_gate": report["held_out_gate"],
                "calibration": report["calibration"],
                "gate_parameters": report["gate_parameters"],
                "combined_parameters": report["combined_parameters"],
                "latency": report["latency"],
            }
            for report in reports
        ],
        "structural_verdict": structural,
        "preregistered_criteria": criteria,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_path)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--fold-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-task-count", type=int, default=18)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    run(
        project_root=arguments.project_root.resolve(),
        fold_roots=[root.resolve() for root in arguments.fold_root],
        output_path=arguments.output.resolve(),
        expected_task_count=arguments.expected_task_count,
        seed=arguments.seed,
        resamples=arguments.bootstrap_resamples,
    )


if __name__ == "__main__":
    main()

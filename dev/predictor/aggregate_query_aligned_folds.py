"""合并 QA-LRDT 三个 task-heldout folds 并执行预注册统计。"""

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
)
CONDITIONS = ("retrieved", "oracle")
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


def _scalar_string(value: np.ndarray, field: str) -> str:
    if value.shape != (1,):
        raise ValueError(f"{field} 应为单元素数组")
    return str(value[0])


def _load_fold(root: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    report_path = root / "report.json"
    prediction_path = root / "held_out_predictions.npz"
    if not report_path.is_file() or not prediction_path.is_file():
        raise FileNotFoundError(f"fold 缺少 report/prediction：{root}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    artifact = report.get("held_out_prediction_artifact", {})
    if artifact.get("sha256") != _sha256(prediction_path):
        raise ValueError(f"prediction artifact hash 不匹配：{root}")
    with np.load(prediction_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    if _scalar_string(arrays["fold_name"], "fold_name") != report["fold_name"]:
        raise ValueError(f"fold_name 不匹配：{root}")
    count = int(artifact["query_count"])
    if len(arrays["query_indices"]) != count:
        raise ValueError(f"query_count 不匹配：{root}")
    held = set(map(str, arrays["held_out_tasks"].tolist()))
    if held != set(report["held_out_tasks"]):
        raise ValueError(f"held-out task 不匹配：{root}")
    if set(map(str, arrays["tasks"].tolist())) - held:
        raise ValueError(f"预测中出现 source task：{root}")
    for condition in CONDITIONS:
        for model in MODEL_NAMES:
            key = f"{condition}__{model}"
            if arrays[key].shape != arrays["target_actions"].shape:
                raise ValueError(f"prediction shape 不匹配：{root}/{key}")
    return report, arrays


def _shared_provenance(reports: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    shared = {}
    for field in PROVENANCE_FIELDS:
        values = {str(report[field]) for report in reports}
        if len(values) != 1:
            raise ValueError(f"fold provenance 不一致：{field}")
        shared[field] = values.pop()
    return shared


def _fold_translation_deltas(
    arrays_by_fold: Sequence[Mapping[str, np.ndarray]],
) -> list[dict[str, Any]]:
    values = []
    for arrays in arrays_by_fold:
        target = torch.from_numpy(arrays["target_actions"])
        mask = torch.from_numpy(arrays["target_masks"]).bool()
        copy = torch.from_numpy(arrays["retrieved__demo_action_copy"])
        query_aligned = torch.from_numpy(
            arrays["retrieved__query_aligned_transport"]
        )
        copy_error = _physical_metrics(copy, target, mask)["translation_l2_m"]
        aligned_error = _physical_metrics(
            query_aligned, target, mask
        )["translation_l2_m"]
        values.append(
            {
                "fold_name": _scalar_string(arrays["fold_name"], "fold_name"),
                "query_aligned_minus_copy_translation_l2_m": (
                    aligned_error - copy_error
                ),
                "improved": aligned_error < copy_error,
            }
        )
    return values


def _aggregate_condition(
    *,
    condition: str,
    arrays_by_fold: Sequence[Mapping[str, np.ndarray]],
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    targets = torch.from_numpy(
        np.concatenate([arrays["target_actions"] for arrays in arrays_by_fold])
    )
    masks = torch.from_numpy(
        np.concatenate([arrays["target_masks"] for arrays in arrays_by_fold])
    ).bool()
    groups = np.concatenate(
        [arrays["group_ids"] for arrays in arrays_by_fold]
    ).astype(str).tolist()
    predictions = {
        model: torch.from_numpy(
            np.concatenate(
                [arrays[f"{condition}__{model}"] for arrays in arrays_by_fold]
            )
        )
        for model in MODEL_NAMES
    }
    options = {
        "target": targets,
        "mask": masks,
        "group_ids": groups,
        "resamples": resamples,
        "pose_scales": POSE_SCALES,
        "translation_threshold_m": 0.05,
        "rotation_threshold_rad": 0.25,
    }
    comparisons = {
        "fixed_minus_copy": _bootstrap_comparison(
            reference=predictions["demo_action_copy"],
            candidate=predictions["fixed_low_rank_transport"],
            seed=seed,
            **options,
        ),
        "query_aligned_minus_copy": _bootstrap_comparison(
            reference=predictions["demo_action_copy"],
            candidate=predictions["query_aligned_transport"],
            seed=seed + 100,
            **options,
        ),
        "query_aligned_minus_fixed": _bootstrap_comparison(
            reference=predictions["fixed_low_rank_transport"],
            candidate=predictions["query_aligned_transport"],
            seed=seed + 200,
            **options,
        ),
    }
    return {
        "query_count": len(targets),
        "episode_groups": len(set(groups)),
        "metrics": {
            name: _physical_metrics(prediction, targets, masks)
            for name, prediction in predictions.items()
        },
        "paired_bootstrap": comparisons,
    }


def _structural_verdict(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    checks = []
    for report in reports:
        audit = report["structural_audit"]["query_aligned_transport"]
        checks.append(
            {
                "fold_name": report["fold_name"],
                "identity_exact": audit["identity_max_abs_error"] == 0.0,
                "no_demo_exact": audit["no_demo_max_abs_output"] == 0.0,
                "rotation_gripper_exact": (
                    audit["rotation_gripper_max_abs_error"] == 0.0
                ),
                "parameter_budget": (
                    report["model_parameters"]["query_aligned_transport"]
                    < 1_000_000
                ),
                "latency_budget": (
                    report["latency"]["query_aligned_transport"]["p95_ms"]
                    < 5.0
                ),
            }
        )
    check_fields = (
        "identity_exact",
        "no_demo_exact",
        "rotation_gripper_exact",
        "parameter_budget",
        "latency_budget",
    )
    return {
        "folds": checks,
        "all_passed": all(
            all(bool(row[field]) for field in check_fields) for row in checks
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
    arrays_by_fold = [item[1] for item in loaded]
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
        raise ValueError(
            f"held-out task union={len(all_tasks)}，预期 {expected_task_count}"
        )
    provenance = _shared_provenance(reports)
    conditions = {
        condition: _aggregate_condition(
            condition=condition,
            arrays_by_fold=arrays_by_fold,
            seed=seed + offset * 1000,
            resamples=resamples,
        )
        for offset, condition in enumerate(CONDITIONS)
    }
    fold_deltas = _fold_translation_deltas(arrays_by_fold)
    retrieved = conditions["retrieved"]["paired_bootstrap"]
    qa_copy_translation = retrieved["query_aligned_minus_copy"][
        "translation_l2_m"
    ]
    qa_fixed_translation = retrieved["query_aligned_minus_fixed"][
        "translation_l2_m"
    ]
    qa_fixed_chunk = retrieved["query_aligned_minus_fixed"][
        "chunk_threshold_accuracy"
    ]
    structural = _structural_verdict(reports)
    criteria = {
        "q1_retrieved_translation_better_than_copy": (
            qa_copy_translation["ci95_high"] < 0.0
        ),
        "q2_translation_better_than_fixed": (
            qa_fixed_translation["ci95_high"] < 0.0
        ),
        "q2_chunk_not_significantly_worse_than_fixed": (
            qa_fixed_chunk["ci95_high"] >= 0.0
        ),
        "q3_at_least_two_folds_improve_translation": (
            sum(row["improved"] for row in fold_deltas) >= 2
        ),
        "q4_structural_and_latency": structural["all_passed"],
    }
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "fold_count": 3,
            "bootstrap_unit": "task + variation + episode",
            "bootstrap_resamples": resamples,
            "delta": "candidate minus reference",
            "thresholds": {
                "translation_m": 0.05,
                "rotation_rad": 0.25,
            },
        },
        "git_commit": _git_commit(project_root),
        "fold_roots": [str(root) for root in fold_roots],
        "fold_report_sha256": [
            _sha256(root / "report.json") for root in fold_roots
        ],
        "fold_prediction_sha256": [
            _sha256(root / "held_out_predictions.npz") for root in fold_roots
        ],
        "shared_provenance": provenance,
        "held_out_tasks": sorted(all_tasks),
        "fold_translation_deltas": fold_deltas,
        "conditions": conditions,
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

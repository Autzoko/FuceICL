"""评估从单个 retrieved Demo 现场辨识的轻量局部策略。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Sequence

import numpy as np
import torch

from dev.pointnet.compare_retrievers import _sha256
from dev.predictor.in_context_local_policy import (
    POSITION_FEATURE_DIM,
    TRANSLATION_ACTION_DIM,
    bounded_translation_correction as _bounded_correction,
    demo_radius_gate as _demo_radius_gate,
    fit_local_translation_operator as _fit_local_translation_operator,
)
from dev.predictor.train_action_chunks import _physical_metrics
from dev.simulator.evaluate_maniskill_demo_prior import _bootstrap_comparison
from dev.simulator.train_maniskill_low_rank_transport import (
    TaskData,
    _load_task,
    _nearest_demo_indices,
    _phase,
    _selection_hash,
)


@dataclass(frozen=True)
class LocalPolicyConfig:
    """结果揭盲前冻结的 in-context local policy 配置。"""

    schema_version: str
    seed: int
    ridge_lambda: float
    position_scale_m: float
    correction_limit_normalized: float
    minimum_demo_radius_normalized: float
    bootstrap_resamples: int
    retrieval_top_k: int = 1
    support_aware_selection: bool = False

    @classmethod
    def from_json(cls, path: Path) -> "LocalPolicyConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def __post_init__(self) -> None:
        positive = (
            self.ridge_lambda,
            self.position_scale_m,
            self.correction_limit_normalized,
            self.minimum_demo_radius_normalized,
            self.bootstrap_resamples,
            self.retrieval_top_k,
        )
        if not self.schema_version.strip() or min(positive) <= 0:
            raise ValueError("schema_version 与所有实验参数必须有效")
        if self.support_aware_selection and self.retrieval_top_k < 2:
            raise ValueError("support-aware selection 至少需要两个候选")


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _distance_summary(values: torch.Tensor) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "p50": float(torch.quantile(values, 0.50)),
        "p95": float(torch.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def _select_demo_indices(
    task: TaskData,
    config: LocalPolicyConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """以冻结的几何 rank-1 或 Demo support-aware 规则选择单个 chunk。"""
    if not config.support_aware_selection:
        indices, distances = _nearest_demo_indices(
            candidate_records=task.train_records,
            candidate_geometry=task.train_geometry,
            query_records=task.val_records,
            query_geometry=task.val_geometry,
            exclude_same_episode=False,
        )
        return indices, indices, distances, {
            "mode": "geometry_rank1",
            "top_k": 1,
            "selection_changed_rate": 0.0,
        }
    if task.train_geometry_sequence is None:
        raise ValueError(f"{task.task_id} support-aware retrieval 缺少 sequence")

    mean = task.train_geometry.mean(dim=0)
    std = task.train_geometry.std(dim=0, unbiased=False).clamp_min(1e-4)
    distances = torch.cdist(
        (task.val_geometry - mean) / std,
        (task.train_geometry - mean) / std,
    ) / math.sqrt(task.train_geometry.shape[1])
    phase_matched = (
        _phase(task.val_geometry)[:, None]
        == _phase(task.train_geometry)[None, :]
    )
    if bool((phase_matched.sum(dim=1) < config.retrieval_top_k).any()):
        raise ValueError("至少一个 query 的同 phase candidate 数不足 top-K")
    masked = distances.masked_fill(~phase_matched, torch.inf)
    candidates = torch.argsort(masked, dim=1, stable=True)[
        :, : config.retrieval_top_k
    ]
    candidate_sequence = task.train_geometry_sequence[candidates]
    sequence = (
        candidate_sequence[..., :POSITION_FEATURE_DIM]
        / config.position_scale_m
    )
    initial = sequence[:, :, :1]
    normalization = math.sqrt(POSITION_FEATURE_DIM)
    path_distance = torch.linalg.vector_norm(
        sequence - initial,
        dim=3,
    ) / normalization
    radius = path_distance.max(dim=2).values.clamp_min(
        config.minimum_demo_radius_normalized
    )
    query = (
        task.val_geometry[:, None, :POSITION_FEATURE_DIM]
        / config.position_scale_m
    )
    query_distance = torch.linalg.vector_norm(
        query - initial[:, :, 0],
        dim=2,
    ) / normalization
    support_ratio = query_distance / radius
    contained = support_ratio <= 1.0
    has_support = contained.any(dim=1)
    supported_choice = support_ratio.masked_fill(~contained, torch.inf).argmin(dim=1)
    choice = torch.where(
        has_support,
        supported_choice,
        torch.zeros_like(supported_choice),
    )
    batch = torch.arange(len(choice))
    selected = candidates[batch, choice]
    rank1 = candidates[:, 0]
    selected_distances = masked[batch, selected]
    selected_ratio = support_ratio[batch, choice]
    rank_counts = {
        str(rank + 1): int((choice == rank).sum())
        for rank in range(config.retrieval_top_k)
    }
    return selected, rank1, selected_distances, {
        "mode": "top-k_demo_radius_support",
        "top_k": config.retrieval_top_k,
        "fallback": "geometry rank-1 when no candidate contains query",
        "queries_with_supported_candidate": int(has_support.sum()),
        "candidate_support_coverage": float(has_support.float().mean()),
        "selection_changed_rate": float((selected != rank1).float().mean()),
        "selected_candidate_rank_counts": rank_counts,
        "selected_support_ratio": _distance_summary(selected_ratio),
    }


@torch.inference_mode()
def _evaluate_task(
    task: TaskData,
    config: LocalPolicyConfig,
) -> dict[str, Any]:
    if task.train_geometry_sequence is None:
        raise ValueError(f"{task.task_id} 缺少 geometry_sequence")
    indices, rank1_indices, retrieval_distances, retrieval_diagnostics = (
        _select_demo_indices(task, config)
    )
    rank1_actions = task.train_actions[rank1_indices]
    demo_actions = task.train_actions[indices]
    demo_sequence = task.train_geometry_sequence[indices]
    operator, condition_number = _fit_local_translation_operator(
        demo_sequence,
        demo_actions,
        position_scale_m=config.position_scale_m,
        ridge_lambda=config.ridge_lambda,
    )
    correction, raw_correction_norm, clipped = _bounded_correction(
        task.val_geometry,
        demo_sequence,
        operator,
        position_scale_m=config.position_scale_m,
        correction_limit=config.correction_limit_normalized,
    )
    raw_prediction = demo_actions.clone()
    unconstrained_translation = (
        raw_prediction[..., :TRANSLATION_ACTION_DIM] + correction[:, None, :]
    )
    raw_prediction[..., :TRANSLATION_ACTION_DIM] = (
        unconstrained_translation.clamp(-1.0, 1.0)
    )
    action_clipped = unconstrained_translation != raw_prediction[..., :3]
    accepted, query_distance, demo_radius = _demo_radius_gate(
        task.val_geometry,
        demo_sequence,
        position_scale_m=config.position_scale_m,
        minimum_radius=config.minimum_demo_radius_normalized,
    )
    if config.support_aware_selection:
        expected = retrieval_diagnostics["candidate_support_coverage"]
        observed = float(accepted.float().mean())
        if not math.isclose(observed, expected, abs_tol=1e-7):
            raise RuntimeError("support-aware selection 与 Demo-radius gate 不一致")
    gated_prediction = torch.where(
        accepted[:, None, None],
        raw_prediction,
        demo_actions,
    )

    # 结构审计：query 等于 Demo 起点时必须严格退化为 Demo copy。
    identity_correction, _, _ = _bounded_correction(
        demo_sequence[:, 0],
        demo_sequence,
        operator,
        position_scale_m=config.position_scale_m,
        correction_limit=config.correction_limit_normalized,
    )
    mask = torch.ones(task.val_actions.shape[:2], dtype=torch.bool)
    options = {
        "pose_scales": task.pose_scales,
        "translation_threshold_m": task.translation_threshold_m,
        "rotation_threshold_rad": task.rotation_threshold_rad,
    }
    if config.support_aware_selection:
        predictions = {
            "geometry_rank1_copy": rank1_actions,
            "support_aware_demo_copy": demo_actions,
            "raw_support_aware_local_policy": raw_prediction,
            "gated_support_aware_local_policy": gated_prediction,
        }
    else:
        predictions = {
            "retrieved_demo_copy": demo_actions,
            "raw_in_context_local_policy": raw_prediction,
            "demo_radius_gated_local_policy": gated_prediction,
        }
    metrics = {
        name: _physical_metrics(value, task.val_actions, mask, **options)
        for name, value in predictions.items()
    }
    groups = [f"{task.task_id}:{row['episode']}" for row in task.val_records]
    bootstrap = {
        "raw_minus_copy": _bootstrap_comparison(
            reference=demo_actions,
            candidate=raw_prediction,
            target=task.val_actions,
            mask=mask,
            group_ids=groups,
            seed=config.seed,
            resamples=config.bootstrap_resamples,
            **options,
        ),
        "gated_minus_copy": _bootstrap_comparison(
            reference=demo_actions,
            candidate=gated_prediction,
            target=task.val_actions,
            mask=mask,
            group_ids=groups,
            seed=config.seed + 100,
            resamples=config.bootstrap_resamples,
            **options,
        ),
    }
    if config.support_aware_selection:
        bootstrap.update(
            {
                "selected_copy_minus_rank1_copy": _bootstrap_comparison(
                    reference=rank1_actions,
                    candidate=demo_actions,
                    target=task.val_actions,
                    mask=mask,
                    group_ids=groups,
                    seed=config.seed + 200,
                    resamples=config.bootstrap_resamples,
                    **options,
                ),
                "gated_minus_rank1_copy": _bootstrap_comparison(
                    reference=rank1_actions,
                    candidate=gated_prediction,
                    target=task.val_actions,
                    mask=mask,
                    group_ids=groups,
                    seed=config.seed + 300,
                    resamples=config.bootstrap_resamples,
                    **options,
                ),
            }
        )
    bounded_norm = torch.linalg.vector_norm(correction, dim=1)
    return {
        "train_queries": len(task.train_records),
        "validation_queries": len(task.val_records),
        "validation_episodes": len({row["episode"] for row in task.val_records}),
        "validation_demo_selection_sha256": _selection_hash(indices),
        "geometry_rank1_selection_sha256": _selection_hash(rank1_indices),
        "retrieval_distance": _distance_summary(retrieval_distances),
        "retrieval_diagnostics": retrieval_diagnostics,
        "metrics": metrics,
        "paired_episode_bootstrap": bootstrap,
        "local_policy_diagnostics": {
            "operator_frobenius_norm": _distance_summary(
                torch.linalg.matrix_norm(operator, ord="fro")
            ),
            "regularized_gram_condition_number": _distance_summary(
                condition_number
            ),
            "raw_correction_l2": _distance_summary(raw_correction_norm),
            "bounded_correction_l2": _distance_summary(bounded_norm),
            "correction_clipped_rate": float(clipped.float().mean()),
            "action_component_clip_rate": float(action_clipped.float().mean()),
            "chunks_with_action_clip_rate": float(
                action_clipped.flatten(start_dim=1).any(dim=1).float().mean()
            ),
            "query_demo_distance": _distance_summary(query_distance),
            "demo_path_radius": _distance_summary(demo_radius),
            "gate_acceptance_rate": float(accepted.float().mean()),
            "accepted_queries": int(accepted.sum()),
            "identity_max_abs_correction": float(identity_correction.abs().max()),
        },
    }


def run(
    *,
    project_root: Path,
    data_roots: Sequence[Path],
    output_path: Path,
    config_path: Path,
    config: LocalPolicyConfig,
) -> None:
    if output_path.exists():
        raise FileExistsError(f"输出已存在，拒绝覆盖：{output_path}")
    tasks = [_load_task(root) for root in data_roots]
    task_ids = [task.task_id for task in tasks]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("data roots 含重复 task_id")
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "status": "preregistered zero-training analytic experiment",
            "retrieval": (
                "phase-matched geometry top-K then Demo-radius support selection"
                if config.support_aware_selection
                else "frozen per-task standardized 17D geometry rank-1 and phase match"
            ),
            "local_identification": (
                "per-Demo centered ridge from H pairs of observed 6D position "
                "relation to normalized 3D translation action"
            ),
            "prediction": (
                "constant L2-bounded translation correction across Demo chunk; "
                "controller range clamped; rotation and gripper copied"
            ),
            "gate": "query distance <= Demo H+1 path radius; Demo-only threshold",
            "validation_usage": "validation actions used only for final metrics",
            "bootstrap_unit": "validation episode",
        },
        "config": asdict(config),
        "config_sha256": _sha256(config_path),
        "git_commit": _git_commit(project_root),
        "data_summary_sha256": {
            task.task_id: _sha256(task.root / "summary.json") for task in tasks
        },
        "tasks": {task.task_id: _evaluate_task(task, config) for task in tasks},
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
    parser.add_argument("--data-root", type=Path, action="append", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    run(
        project_root=arguments.project_root.resolve(),
        data_roots=[path.resolve() for path in arguments.data_root],
        output_path=arguments.output.resolve(),
        config_path=arguments.config.resolve(),
        config=LocalPolicyConfig.from_json(arguments.config.resolve()),
    )


if __name__ == "__main__":
    main()

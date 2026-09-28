"""比较 H=6 与 phase-local 长 Demo context 的 IC-LPI。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Sequence

import torch

from dev.pointnet.compare_retrievers import _sha256
from dev.predictor.in_context_local_policy import (
    TRANSLATION_ACTION_DIM,
    bounded_translation_correction,
    demo_radius_gate,
    fit_local_translation_operator,
)
from dev.predictor.train_action_chunks import _physical_metrics
from dev.simulator.evaluate_in_context_local_policy import _distance_summary
from dev.simulator.evaluate_maniskill_demo_prior import _bootstrap_comparison
from dev.simulator.phase_local_context import build_phase_local_context_bank
from dev.simulator.train_maniskill_low_rank_transport import (
    TaskData,
    _load_task,
    _nearest_demo_indices,
    _selection_hash,
)


@dataclass(frozen=True)
class PhaseLocalConfig:
    """结果揭盲前冻结的 phase-local long-context 配置。"""

    schema_version: str
    seed: int
    max_context_transitions: int
    minimum_context_transitions: int
    ridge_lambda: float
    position_scale_m: float
    correction_limit_normalized: float
    minimum_demo_radius_normalized: float
    bootstrap_resamples: int

    @classmethod
    def from_json(cls, path: Path) -> "PhaseLocalConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def __post_init__(self) -> None:
        positive = (
            self.max_context_transitions,
            self.minimum_context_transitions,
            self.ridge_lambda,
            self.position_scale_m,
            self.correction_limit_normalized,
            self.minimum_demo_radius_normalized,
            self.bootstrap_resamples,
        )
        if not self.schema_version.strip() or min(positive) <= 0:
            raise ValueError("schema_version 与所有实验参数必须有效")
        if self.minimum_context_transitions > self.max_context_transitions:
            raise ValueError("minimum context 不能超过 maximum context")


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _apply_translation_correction(
    demo_actions: torch.Tensor,
    correction: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    prediction = demo_actions.clone()
    unconstrained = prediction[..., :TRANSLATION_ACTION_DIM] + correction[:, None]
    prediction[..., :TRANSLATION_ACTION_DIM] = unconstrained.clamp(-1.0, 1.0)
    return prediction, unconstrained != prediction[..., :TRANSLATION_ACTION_DIM]


def _masked_summary(values: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    selected = values[mask]
    if not len(selected):
        raise ValueError("诊断 mask 未选中任何值")
    return _distance_summary(selected)


@torch.inference_mode()
def _evaluate_task(
    task: TaskData,
    config: PhaseLocalConfig,
) -> dict[str, Any]:
    if task.train_geometry_sequence is None:
        raise ValueError(f"{task.task_id} 缺少 geometry_sequence")
    indices, retrieval_distances = _nearest_demo_indices(
        candidate_records=task.train_records,
        candidate_geometry=task.train_geometry,
        query_records=task.val_records,
        query_geometry=task.val_geometry,
        exclude_same_episode=False,
    )
    demo_actions = task.train_actions[indices]
    short_sequence = task.train_geometry_sequence[indices]
    short_operator, short_condition = fit_local_translation_operator(
        short_sequence,
        demo_actions,
        position_scale_m=config.position_scale_m,
        ridge_lambda=config.ridge_lambda,
    )
    short_correction, short_raw_norm, short_capped = (
        bounded_translation_correction(
            task.val_geometry,
            short_sequence,
            short_operator,
            position_scale_m=config.position_scale_m,
            correction_limit=config.correction_limit_normalized,
        )
    )
    short_raw, short_action_clipped = _apply_translation_correction(
        demo_actions,
        short_correction,
    )
    short_accepted, short_distance, short_radius = demo_radius_gate(
        task.val_geometry,
        short_sequence,
        position_scale_m=config.position_scale_m,
        minimum_radius=config.minimum_demo_radius_normalized,
    )
    short_prediction = torch.where(
        short_accepted[:, None, None],
        short_raw,
        demo_actions,
    )

    bank = build_phase_local_context_bank(
        records=task.train_records,
        geometry=task.train_geometry,
        actions=task.train_actions,
        stored_geometry_sequence=task.train_geometry_sequence,
        max_transitions=config.max_context_transitions,
    )
    long_sequence = bank.geometry_sequence[indices]
    long_actions = bank.actions[indices]
    long_mask = bank.transition_mask[indices]
    long_lengths = bank.lengths[indices]
    eligible = long_lengths >= config.minimum_context_transitions
    if not bool(eligible.any()):
        raise ValueError(f"{task.task_id} 没有 eligible phase-local contexts")
    long_operator, long_condition = fit_local_translation_operator(
        long_sequence,
        long_actions,
        position_scale_m=config.position_scale_m,
        ridge_lambda=config.ridge_lambda,
        transition_mask=long_mask,
    )
    long_correction, long_raw_norm, long_capped = (
        bounded_translation_correction(
            task.val_geometry,
            long_sequence,
            long_operator,
            position_scale_m=config.position_scale_m,
            correction_limit=config.correction_limit_normalized,
        )
    )
    long_raw, long_action_clipped = _apply_translation_correction(
        demo_actions,
        long_correction,
    )
    long_accepted, long_distance, long_radius = demo_radius_gate(
        task.val_geometry,
        long_sequence,
        position_scale_m=config.position_scale_m,
        minimum_radius=config.minimum_demo_radius_normalized,
        transition_mask=long_mask,
    )
    long_accepted &= eligible
    long_prediction = torch.where(
        long_accepted[:, None, None],
        long_raw,
        demo_actions,
    )

    mask = torch.ones(task.val_actions.shape[:2], dtype=torch.bool)
    options = {
        "pose_scales": task.pose_scales,
        "translation_threshold_m": task.translation_threshold_m,
        "rotation_threshold_rad": task.rotation_threshold_rad,
    }
    predictions = {
        "retrieved_demo_copy": demo_actions,
        "h6_gated_in_context_local_policy": short_prediction,
        "phase_local_h24_gated_in_context_local_policy": long_prediction,
    }
    metrics = {
        name: _physical_metrics(value, task.val_actions, mask, **options)
        for name, value in predictions.items()
    }
    groups = [f"{task.task_id}:{row['episode']}" for row in task.val_records]
    comparisons = {
        "h6_minus_copy": _bootstrap_comparison(
            reference=demo_actions,
            candidate=short_prediction,
            target=task.val_actions,
            mask=mask,
            group_ids=groups,
            seed=config.seed,
            resamples=config.bootstrap_resamples,
            **options,
        ),
        "phase_local_h24_minus_copy": _bootstrap_comparison(
            reference=demo_actions,
            candidate=long_prediction,
            target=task.val_actions,
            mask=mask,
            group_ids=groups,
            seed=config.seed + 100,
            resamples=config.bootstrap_resamples,
            **options,
        ),
        "phase_local_h24_minus_h6": _bootstrap_comparison(
            reference=short_prediction,
            candidate=long_prediction,
            target=task.val_actions,
            mask=mask,
            group_ids=groups,
            seed=config.seed + 200,
            resamples=config.bootstrap_resamples,
            **options,
        ),
    }
    short_identity, _, _ = bounded_translation_correction(
        short_sequence[:, 0],
        short_sequence,
        short_operator,
        position_scale_m=config.position_scale_m,
        correction_limit=config.correction_limit_normalized,
    )
    long_identity, _, _ = bounded_translation_correction(
        long_sequence[:, 0],
        long_sequence,
        long_operator,
        position_scale_m=config.position_scale_m,
        correction_limit=config.correction_limit_normalized,
    )
    return {
        "train_queries": len(task.train_records),
        "validation_queries": len(task.val_records),
        "validation_episodes": len({row["episode"] for row in task.val_records}),
        "validation_demo_selection_sha256": _selection_hash(indices),
        "retrieval_distance": _distance_summary(retrieval_distances),
        "metrics": metrics,
        "paired_episode_bootstrap": comparisons,
        "h6_diagnostics": {
            "gate_acceptance_rate": float(short_accepted.float().mean()),
            "accepted_queries": int(short_accepted.sum()),
            "query_demo_distance": _distance_summary(short_distance),
            "demo_path_radius": _distance_summary(short_radius),
            "operator_condition_number": _distance_summary(short_condition),
            "raw_correction_l2": _distance_summary(short_raw_norm),
            "correction_capped_rate": float(short_capped.float().mean()),
            "action_component_clip_rate": float(
                short_action_clipped.float().mean()
            ),
            "identity_max_abs_correction": float(short_identity.abs().max()),
        },
        "phase_local_h24_diagnostics": {
            "context_length": _distance_summary(long_lengths.float()),
            "eligible_rate": float(eligible.float().mean()),
            "eligible_queries": int(eligible.sum()),
            "gate_acceptance_rate": float(long_accepted.float().mean()),
            "accepted_queries": int(long_accepted.sum()),
            "query_demo_distance": _distance_summary(long_distance),
            "demo_path_radius": _distance_summary(long_radius),
            "operator_condition_number_eligible": _masked_summary(
                long_condition,
                eligible,
            ),
            "raw_correction_l2_eligible": _masked_summary(
                long_raw_norm,
                eligible,
            ),
            "correction_capped_rate": float(
                (long_capped & eligible).float().sum()
                / eligible.float().sum()
            ),
            "action_component_clip_rate": float(
                (
                    long_action_clipped
                    & long_accepted[:, None, None]
                ).float().mean()
            ),
            "identity_max_abs_correction": float(long_identity.abs().max()),
        },
    }


def run(
    *,
    project_root: Path,
    data_roots: Sequence[Path],
    output_path: Path,
    config_path: Path,
    config: PhaseLocalConfig,
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
            "retrieval": "frozen phase-matched standardized 17D geometry rank-1",
            "output": "same retrieved H=6 Demo action chunk",
            "h6_context": "stored H+1 observed geometry and H actions",
            "long_context": (
                "up to 24 consecutive observed transitions from the same "
                "episode and gripper phase"
            ),
            "fit": "masked centered per-Demo ridge; no cross-Demo parameters",
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
        config=PhaseLocalConfig.from_json(arguments.config.resolve()),
    )


if __name__ == "__main__":
    main()

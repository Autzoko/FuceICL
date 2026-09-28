"""评估 phase-factorized、block-sparse 的解析 Demo action transport。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Sequence

import numpy as np
import torch

from dev.pointnet.compare_retrievers import _sha256
from dev.predictor.evaluate_jacobian_significance import (
    _episode_bootstrap,
    _per_query_metrics,
)
from dev.predictor.train_action_chunks import _physical_metrics
from dev.simulator.evaluate_maniskill_demo_prior import (
    _action_protocol,
    _load_split,
)


PHASE_NAMES = ("open_pre_grasp", "closed_post_grasp")


@dataclass(frozen=True)
class PhaseTransportConfig:
    """结果揭盲前固定的稀疏 transport 配置。"""

    schema_version: str
    seed: int
    ridge_lambda: float
    bootstrap_resamples: int

    @classmethod
    def from_json(cls, path: Path) -> "PhaseTransportConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def __post_init__(self) -> None:
        if not self.schema_version.strip():
            raise ValueError("schema_version 不能为空")
        if min(self.ridge_lambda, self.bootstrap_resamples) <= 0:
            raise ValueError("ridge lambda 与 bootstrap 次数必须为正")


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _selection_hash(indices: torch.Tensor) -> str:
    values = indices.detach().cpu().numpy().astype("<i8", copy=False)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _phase_ids(geometry: torch.Tensor) -> torch.Tensor:
    """只用当前可观测 gripper state 区分 pre/post-grasp 候选。"""
    return (geometry[:, 15] < 0.5).long()


def _phase_matched_nearest(
    *,
    distances: torch.Tensor,
    query_phase: torch.Tensor,
    candidate_phase: torch.Tensor,
    query_episodes: torch.Tensor | None = None,
    candidate_episodes: torch.Tensor | None = None,
) -> torch.Tensor:
    valid = query_phase[:, None] == candidate_phase[None, :]
    if query_episodes is not None or candidate_episodes is not None:
        if query_episodes is None or candidate_episodes is None:
            raise ValueError("query/candidate episode IDs 必须同时提供")
        valid &= query_episodes[:, None] != candidate_episodes[None, :]
    if not bool(valid.any(dim=1).all()):
        raise ValueError("至少一个 query 没有跨 episode 同 phase 候选")
    return distances.masked_fill(~valid, torch.inf).argmin(dim=1)


def _fit_phase_weights(
    *,
    query_geometry: torch.Tensor,
    demo_geometry: torch.Tensor,
    target_actions: torch.Tensor,
    demo_actions: torch.Tensor,
    phase: torch.Tensor,
    position_scale_m: float,
    ridge_lambda: float,
) -> tuple[torch.Tensor, dict[str, int]]:
    """拟合两个无偏置 3D→H×3 ridge maps，保留 identity-at-zero。"""
    horizon = int(target_actions.shape[1])
    weights = torch.zeros(2, 3, horizon * 3, dtype=torch.float64)
    sample_counts: dict[str, int] = {}
    for phase_id, phase_name in enumerate(PHASE_NAMES):
        mask = phase == phase_id
        count = int(mask.sum())
        if count < 3:
            raise ValueError(f"{phase_name} train pairs 不足：{count}")
        feature_slice = slice(0, 3) if phase_id == 0 else slice(3, 6)
        x = (
            query_geometry[mask, feature_slice]
            - demo_geometry[mask, feature_slice]
        ).double() / position_scale_m
        y = (
            target_actions[mask, :, :3] - demo_actions[mask, :, :3]
        ).reshape(count, horizon * 3).double()
        gram = x.T @ x + ridge_lambda * torch.eye(3, dtype=torch.float64)
        weights[phase_id] = torch.linalg.solve(gram, x.T @ y)
        sample_counts[phase_name] = count
    return weights.float(), sample_counts


def _transport(
    *,
    query_geometry: torch.Tensor,
    demo_geometry: torch.Tensor,
    demo_actions: torch.Tensor,
    phase: torch.Tensor,
    weights: torch.Tensor,
    position_scale_m: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    horizon = int(demo_actions.shape[1])
    delta = torch.empty(len(query_geometry), 3)
    open_mask = phase == 0
    delta[open_mask] = (
        query_geometry[open_mask, 0:3] - demo_geometry[open_mask, 0:3]
    ) / position_scale_m
    delta[~open_mask] = (
        query_geometry[~open_mask, 3:6] - demo_geometry[~open_mask, 3:6]
    ) / position_scale_m
    selected_weights = weights[phase]
    residual = torch.einsum("bi,bij->bj", delta, selected_weights).reshape(
        -1, horizon, 3
    )
    prediction = demo_actions.clone()
    translation_unclipped = prediction[:, :, :3] + residual
    prediction[:, :, :3] = translation_unclipped.clamp(-1.0, 1.0)
    clipped = translation_unclipped != prediction[:, :, :3]
    return prediction, {
        "translation_component_clip_rate": float(clipped.float().mean()),
        "chunks_with_translation_clip_rate": float(
            clipped.flatten(start_dim=1).any(dim=1).float().mean()
        ),
        "normalized_residual_abs_mean": float(residual.abs().mean()),
        "normalized_residual_abs_max": float(residual.abs().max()),
    }


def _bootstrap(
    *,
    reference: torch.Tensor,
    candidate: torch.Tensor,
    target: torch.Tensor,
    group_ids: Sequence[str],
    pose_scales: torch.Tensor,
    translation_threshold_m: float,
    rotation_threshold_rad: float,
    config: PhaseTransportConfig,
    seed_offset: int,
) -> dict[str, dict[str, float | int]]:
    mask = torch.ones(target.shape[:2], dtype=torch.bool)
    options = {
        "pose_scales": pose_scales,
        "translation_threshold_m": translation_threshold_m,
        "rotation_threshold_rad": rotation_threshold_rad,
    }
    reference_metrics = _per_query_metrics(reference, target, mask, **options)
    candidate_metrics = _per_query_metrics(candidate, target, mask, **options)
    return {
        name: _episode_bootstrap(
            reference=reference_metrics[name],
            candidate=candidate_metrics[name],
            group_ids=group_ids,
            resamples=config.bootstrap_resamples,
            seed=config.seed + seed_offset + offset,
        )
        for offset, name in enumerate(reference_metrics)
    }


@torch.inference_mode()
def run(
    *,
    project_root: Path,
    data_root: Path,
    config_path: Path,
    output_path: Path,
    config: PhaseTransportConfig,
) -> None:
    if output_path.exists():
        raise FileExistsError(f"输出已存在，拒绝覆盖：{output_path}")
    (
        pose_scales,
        translation_threshold_m,
        rotation_threshold_rad,
        action_representation,
    ) = _action_protocol(data_root)
    train_records, train_geometry, train_actions = _load_split(
        data_root, "train", pose_scales
    )
    val_records, val_geometry, val_actions = _load_split(
        data_root, "val", pose_scales
    )
    mean = train_geometry.mean(dim=0)
    std = train_geometry.std(dim=0, unbiased=False).clamp_min(1e-4)
    standardized_train = (train_geometry - mean) / std
    standardized_val = (val_geometry - mean) / std
    scale = float(np.sqrt(train_geometry.shape[1]))
    train_distances = torch.cdist(standardized_train, standardized_train) / scale
    val_distances = torch.cdist(standardized_val, standardized_train) / scale
    train_phase = _phase_ids(train_geometry)
    val_phase = _phase_ids(val_geometry)
    train_episodes = torch.tensor(
        [int(record["episode"]) for record in train_records]
    )
    train_demo_indices = _phase_matched_nearest(
        distances=train_distances,
        query_phase=train_phase,
        candidate_phase=train_phase,
        query_episodes=train_episodes,
        candidate_episodes=train_episodes,
    )
    val_demo_indices = _phase_matched_nearest(
        distances=val_distances,
        query_phase=val_phase,
        candidate_phase=train_phase,
    )
    global_indices = val_distances.argmin(dim=1)
    train_demo_geometry = train_geometry[train_demo_indices]
    train_demo_actions = train_actions[train_demo_indices]
    weights, phase_sample_counts = _fit_phase_weights(
        query_geometry=train_geometry,
        demo_geometry=train_demo_geometry,
        target_actions=train_actions,
        demo_actions=train_demo_actions,
        phase=train_phase,
        position_scale_m=float(pose_scales[0]),
        ridge_lambda=config.ridge_lambda,
    )
    train_transported, train_transport_diagnostics = _transport(
        query_geometry=train_geometry,
        demo_geometry=train_demo_geometry,
        demo_actions=train_demo_actions,
        phase=train_phase,
        weights=weights,
        position_scale_m=float(pose_scales[0]),
    )
    phase_copy = train_actions[val_demo_indices]
    transported, transport_diagnostics = _transport(
        query_geometry=val_geometry,
        demo_geometry=train_geometry[val_demo_indices],
        demo_actions=phase_copy,
        phase=val_phase,
        weights=weights,
        position_scale_m=float(pose_scales[0]),
    )
    global_copy = train_actions[global_indices]
    mask = torch.ones(val_actions.shape[:2], dtype=torch.bool)
    train_mask = torch.ones(train_actions.shape[:2], dtype=torch.bool)
    metric_options = {
        "pose_scales": pose_scales,
        "translation_threshold_m": translation_threshold_m,
        "rotation_threshold_rad": rotation_threshold_rad,
    }
    predictions = {
        "global_geometry_copy": global_copy,
        "phase_matched_copy": phase_copy,
        "phase_factorized_transport": transported,
    }
    group_ids = [f"PickCube-v1:{record['episode']}" for record in val_records]
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "phase": "current observed gripper state only; no contact/simulator flag",
            "open_pre_grasp_features": "query-demo active_in_eef xyz",
            "closed_post_grasp_features": "query-demo target_from_active_eef xyz",
            "transported_action": "translation only; Demo rotation/gripper unchanged",
            "fit": "two no-bias ridge maps on cross-episode train pairs",
            "checkpoint_selection": "closed-form single fit; validation unused",
        },
        "config": asdict(config),
        "action_protocol": {
            "representation": action_representation,
            "pose_scales": pose_scales.tolist(),
            "translation_threshold_m": translation_threshold_m,
            "rotation_threshold_rad": rotation_threshold_rad,
        },
        "git_commit": _git_commit(project_root),
        "data_summary_sha256": _sha256(data_root / "summary.json"),
        "config_sha256": _sha256(config_path),
        "train_queries": len(train_records),
        "validation_queries": len(val_records),
        "validation_episodes": len(set(group_ids)),
        "model_parameters": int(weights.numel()),
        "phase_sample_counts": phase_sample_counts,
        "weights": weights.tolist(),
        "weight_frobenius_norm": {
            PHASE_NAMES[index]: float(torch.linalg.matrix_norm(value))
            for index, value in enumerate(weights)
        },
        "selection_sha256": {
            "train_phase_matched": _selection_hash(train_demo_indices),
            "validation_global": _selection_hash(global_indices),
            "validation_phase_matched": _selection_hash(val_demo_indices),
        },
        "metrics": {
            name: _physical_metrics(
                prediction, val_actions, mask, **metric_options
            )
            for name, prediction in predictions.items()
        },
        "train_metrics": {
            "phase_matched_copy": _physical_metrics(
                train_demo_actions,
                train_actions,
                train_mask,
                **metric_options,
            ),
            "phase_factorized_transport": _physical_metrics(
                train_transported,
                train_actions,
                train_mask,
                **metric_options,
            ),
        },
        "paired_episode_bootstrap": {
            "phase_copy_minus_global_copy": _bootstrap(
                reference=global_copy,
                candidate=phase_copy,
                target=val_actions,
                group_ids=group_ids,
                pose_scales=pose_scales,
                translation_threshold_m=translation_threshold_m,
                rotation_threshold_rad=rotation_threshold_rad,
                config=config,
                seed_offset=0,
            ),
            "transport_minus_phase_copy": _bootstrap(
                reference=phase_copy,
                candidate=transported,
                target=val_actions,
                group_ids=group_ids,
                pose_scales=pose_scales,
                translation_threshold_m=translation_threshold_m,
                rotation_threshold_rad=rotation_threshold_rad,
                config=config,
                seed_offset=100,
            ),
        },
        "transport_diagnostics": {
            "train": train_transport_diagnostics,
            "validation": transport_diagnostics,
        },
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
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    run(
        project_root=arguments.project_root.resolve(),
        data_root=arguments.data_root.resolve(),
        config_path=arguments.config.resolve(),
        output_path=arguments.output.resolve(),
        config=PhaseTransportConfig.from_json(arguments.config.resolve()),
    )


if __name__ == "__main__":
    main()

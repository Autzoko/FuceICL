"""诊断 held-out LR-DAT 的训练支持域，并评估 train-only locality gate。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
from dev.predictor.low_rank_demo_transport import (
    LowRankDemoActionTransport,
    LowRankTransportConfig,
)
from dev.predictor.train_action_chunks import _physical_metrics
from dev.simulator.evaluate_maniskill_demo_prior import _bootstrap_comparison
from dev.simulator.train_maniskill_low_rank_transport import (
    TaskData,
    _load_task,
    _nearest_demo_indices,
    _predict,
)


@dataclass(frozen=True)
class LocalityConfig:
    schema_version: str
    seed: int
    train_distance_quantile: float
    bootstrap_resamples: int

    @classmethod
    def from_json(cls, path: Path) -> "LocalityConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def __post_init__(self) -> None:
        if not self.schema_version.strip():
            raise ValueError("schema_version 不能为空")
        if not 0.0 < self.train_distance_quantile < 1.0:
            raise ValueError("train distance quantile 必须位于 (0,1)")
        if self.bootstrap_resamples <= 0:
            raise ValueError("bootstrap_resamples 必须为正")


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _load_model(
    checkpoint_path: Path,
) -> tuple[LowRankDemoActionTransport, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = LowRankTransportConfig(**checkpoint["model_config"])
    model = LowRankDemoActionTransport(
        config,
        geometry_mean=checkpoint["geometry_mean"],
        geometry_std=checkpoint["geometry_std"],
    )
    model.load_state_dict(checkpoint["model"])
    return model.eval(), checkpoint


def _normalized_pair_distance(
    query: torch.Tensor,
    demo: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    delta = (query - demo) / std.clamp_min(1e-4)
    return torch.linalg.vector_norm(delta, dim=1) / math.sqrt(delta.shape[1])


def _source_train_distances(
    tasks: Sequence[TaskData],
    source_ids: set[str],
    std: torch.Tensor,
) -> torch.Tensor:
    distances = []
    for task in tasks:
        if task.task_id not in source_ids:
            continue
        indices, _ = _nearest_demo_indices(
            candidate_records=task.train_records,
            candidate_geometry=task.train_geometry,
            query_records=task.train_records,
            query_geometry=task.train_geometry,
            exclude_same_episode=True,
        )
        distances.append(
            _normalized_pair_distance(
                task.train_geometry,
                task.train_geometry[indices],
                std,
            )
        )
    return torch.cat(distances)


def _distance_summary(values: torch.Tensor) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "p50": float(torch.quantile(values, 0.5)),
        "p95": float(torch.quantile(values, 0.95)),
        "max": float(values.max()),
    }


@torch.inference_mode()
def _evaluate_fold(
    *,
    fold_root: Path,
    tasks: Sequence[TaskData],
    config: LocalityConfig,
) -> dict[str, Any]:
    report = json.loads((fold_root / "report.json").read_text(encoding="utf-8"))
    held_out = str(report["held_out_task"])
    source_ids = {str(value) for value in report["source_tasks"]}
    target = next(task for task in tasks if task.task_id == held_out)
    model, checkpoint = _load_model(fold_root / "low_rank_demo_transport.pt")
    std = torch.as_tensor(checkpoint["geometry_std"]).float()
    source_distances = _source_train_distances(tasks, source_ids, std)
    threshold = float(
        torch.quantile(source_distances, config.train_distance_quantile)
    )

    indices, _ = _nearest_demo_indices(
        candidate_records=target.train_records,
        candidate_geometry=target.train_geometry,
        query_records=target.val_records,
        query_geometry=target.val_geometry,
        exclude_same_episode=False,
    )
    demo_geometry = target.train_geometry[indices]
    demo_actions = target.train_actions[indices]
    target_distances = _normalized_pair_distance(
        target.val_geometry,
        demo_geometry,
        std,
    )
    accepted = target_distances <= threshold
    raw = _predict(
        model,
        target.val_geometry,
        demo_geometry,
        demo_actions,
        torch.device("cpu"),
    )
    gated = torch.where(accepted[:, None, None], raw, demo_actions)
    mask = torch.ones(target.val_actions.shape[:2], dtype=torch.bool)
    options = {
        "pose_scales": target.pose_scales,
        "translation_threshold_m": target.translation_threshold_m,
        "rotation_threshold_rad": target.rotation_threshold_rad,
    }
    predictions = {
        "retrieved_demo_copy": demo_actions,
        "raw_low_rank_transport": raw,
        "locality_gated_transport": gated,
    }
    metrics = {
        name: _physical_metrics(value, target.val_actions, mask, **options)
        for name, value in predictions.items()
    }
    groups = [f"{held_out}:{row['episode']}" for row in target.val_records]
    residual = raw[..., :6] - demo_actions[..., :6]
    per_query_residual = torch.linalg.vector_norm(residual, dim=2).mean(dim=1)
    return {
        "held_out_task": held_out,
        "source_tasks": sorted(source_ids),
        "checkpoint_sha256": _sha256(fold_root / "low_rank_demo_transport.pt"),
        "threshold": {
            "definition": "source-train pair normalized distance quantile",
            "quantile": config.train_distance_quantile,
            "value": threshold,
        },
        "source_train_distance": _distance_summary(source_distances),
        "target_distance": {
            **_distance_summary(target_distances),
            "outside_rate": float((~accepted).float().mean()),
            "accepted_queries": int(accepted.sum()),
            "total_queries": len(accepted),
        },
        "raw_residual": {
            "normalized_l2_per_token_mean": float(per_query_residual.mean()),
            "normalized_l2_per_token_p95": float(
                torch.quantile(per_query_residual, 0.95)
            ),
            "max_abs": float(residual.abs().max()),
        },
        "metrics": metrics,
        "paired_episode_bootstrap": {
            "raw_minus_copy": _bootstrap_comparison(
                reference=demo_actions,
                candidate=raw,
                target=target.val_actions,
                mask=mask,
                group_ids=groups,
                seed=config.seed,
                resamples=config.bootstrap_resamples,
                **options,
            ),
            "gated_minus_copy": _bootstrap_comparison(
                reference=demo_actions,
                candidate=gated,
                target=target.val_actions,
                mask=mask,
                group_ids=groups,
                seed=config.seed + 100,
                resamples=config.bootstrap_resamples,
                **options,
            ),
        },
    }


def run(
    *,
    project_root: Path,
    data_roots: Sequence[Path],
    fold_roots: Sequence[Path],
    output_path: Path,
    config_path: Path,
    config: LocalityConfig,
) -> None:
    if output_path.exists():
        raise FileExistsError(f"输出已存在，拒绝覆盖：{output_path}")
    tasks = [_load_task(root) for root in data_roots]
    folds = [
        _evaluate_fold(fold_root=root, tasks=tasks, config=config)
        for root in fold_roots
    ]
    held_out = [fold["held_out_task"] for fold in folds]
    if len(set(held_out)) != len(tasks) or set(held_out) != {
        task.task_id for task in tasks
    }:
        raise ValueError("fold roots 未恰好覆盖所有 task IDs")
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "threshold": "source-train only; target actions unused",
            "outside_support": "strict fallback to retrieved Demo copy",
            "distance": "checkpoint-standardized geometry delta L2 / sqrt(dim)",
            "status": "post-failure mechanism diagnostic",
        },
        "config": {
            "schema_version": config.schema_version,
            "seed": config.seed,
            "train_distance_quantile": config.train_distance_quantile,
            "bootstrap_resamples": config.bootstrap_resamples,
        },
        "config_sha256": _sha256(config_path),
        "git_commit": _git_commit(project_root),
        "folds": folds,
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
    parser.add_argument("--fold-root", type=Path, action="append", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    run(
        project_root=arguments.project_root.resolve(),
        data_roots=[path.resolve() for path in arguments.data_root],
        fold_roots=[path.resolve() for path in arguments.fold_root],
        output_path=arguments.output.resolve(),
        config_path=arguments.config.resolve(),
        config=LocalityConfig.from_json(arguments.config.resolve()),
    )


if __name__ == "__main__":
    main()

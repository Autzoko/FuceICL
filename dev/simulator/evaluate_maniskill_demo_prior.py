"""评估 ManiSkill dense geometry retrieval 的零训练 Demo action utility。"""

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
from dev.predictor.evaluate_jacobian_significance import (
    _episode_bootstrap,
    _per_query_metrics,
)
from dev.predictor.train_action_chunks import (
    POSE_SCALES,
    _normalize_actions,
    _physical_metrics,
)


def _git_commit(project_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _action_protocol(root: Path) -> tuple[torch.Tensor, float, float, str]:
    """按数据表示固定归一化尺度和离线 command 准确阈值。"""
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    config = summary.get("config", {})
    representation = str(
        config.get("action_representation", "cumulative_observation_delta")
    )
    if representation == "canonical_controller_command":
        position = float(config["position_limit_m"])
        rotation = abs(float(config["rotation_scale_rad"]))
        scales = torch.tensor([position] * 3 + [rotation] * 3)
        return scales, 0.1 * position, 0.1 * rotation, representation
    return POSE_SCALES.clone(), 0.05, 0.25, representation


def _load_split(
    root: Path,
    split: str,
    pose_scales: torch.Tensor | Sequence[float] | None = None,
) -> tuple[list[dict[str, Any]], torch.Tensor, torch.Tensor]:
    records = _read_jsonl(root / f"manifest-{split}.jsonl")
    cache: dict[str, Mapping[str, np.ndarray]] = {}
    geometries = []
    actions = []
    for record in records:
        relative = str(record["shard"])
        if relative not in cache:
            with np.load(root / relative) as archive:
                cache[relative] = {
                    name: np.asarray(archive[name]) for name in archive.files
                }
        row = int(record["row"])
        geometries.append(cache[relative]["geometry"][row])
        actions.append(cache[relative]["actions"][row])
    return (
        records,
        torch.from_numpy(np.stack(geometries)).float(),
        _normalize_actions(
            torch.from_numpy(np.stack(actions)).float(), pose_scales
        ),
    )


def _diverse_top_k(
    distances: torch.Tensor,
    records: Sequence[Mapping[str, Any]],
    k: int,
) -> torch.Tensor:
    selected = []
    for row in distances:
        episodes: set[int] = set()
        candidates = []
        for index in torch.argsort(row, stable=True).tolist():
            episode = int(records[index]["episode"])
            if episode in episodes:
                continue
            episodes.add(episode)
            candidates.append(index)
            if len(candidates) == k:
                break
        if len(candidates) != k:
            raise ValueError("不同 episode 的候选数不足")
        selected.append(candidates)
    return torch.tensor(selected, dtype=torch.long)


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = np.argsort(np.argsort(left, kind="stable"), kind="stable")
    right_rank = np.argsort(np.argsort(right, kind="stable"), kind="stable")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _bootstrap_comparison(
    *,
    reference: torch.Tensor,
    candidate: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    group_ids: Sequence[str],
    seed: int,
    resamples: int,
    pose_scales: torch.Tensor,
    translation_threshold_m: float,
    rotation_threshold_rad: float,
) -> dict[str, Any]:
    metric_options = {
        "pose_scales": pose_scales,
        "translation_threshold_m": translation_threshold_m,
        "rotation_threshold_rad": rotation_threshold_rad,
    }
    reference_metrics = _per_query_metrics(
        reference, target, mask, **metric_options
    )
    candidate_metrics = _per_query_metrics(
        candidate, target, mask, **metric_options
    )
    return {
        metric: _episode_bootstrap(
            reference=reference_metrics[metric],
            candidate=candidate_metrics[metric],
            group_ids=group_ids,
            resamples=resamples,
            seed=seed + offset,
        )
        for offset, metric in enumerate(reference_metrics)
    }


@torch.inference_mode()
def run(
    *,
    project_root: Path,
    data_root: Path,
    output_path: Path,
    k: int,
    seed: int,
    bootstrap_resamples: int,
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
    distances = torch.cdist(
        (val_geometry - mean) / std,
        (train_geometry - mean) / std,
    ) / np.sqrt(train_geometry.shape[1])
    candidates = _diverse_top_k(distances, train_records, k)
    hypotheses = train_actions[candidates]
    batch = torch.arange(len(val_actions))
    target_mask = torch.ones(val_actions.shape[:2], dtype=torch.bool)

    current_gripper = val_geometry[:, 15].clamp(0.0, 1.0)
    zero_physical = torch.zeros_like(val_actions)
    zero_physical[..., 6] = current_gripper[:, None]
    zero_action = _normalize_actions(zero_physical, pose_scales)
    rank1 = hypotheses[:, 0]
    errors = (hypotheses - val_actions[:, None]).abs().mean(dim=(2, 3))
    oracle2_index = errors[:, :2].argmin(dim=1)
    oracle4_index = errors.argmin(dim=1)
    predictions = {
        "zero_action": zero_action,
        "geometry_rank1_copy": rank1,
        "oracle_best_in_2": hypotheses[batch, oracle2_index],
        f"oracle_best_in_{k}": hypotheses[batch, oracle4_index],
    }
    metrics = {
        name: _physical_metrics(
            prediction,
            val_actions,
            target_mask,
            pose_scales=pose_scales,
            translation_threshold_m=translation_threshold_m,
            rotation_threshold_rad=rotation_threshold_rad,
        )
        for name, prediction in predictions.items()
    }
    group_ids = [f"PickCube-v1:{record['episode']}" for record in val_records]
    selected_distances = distances[
        torch.arange(len(val_records))[:, None], candidates
    ].numpy()
    candidate_errors = errors.numpy()
    flat_distances = selected_distances.reshape(-1)
    flat_errors = candidate_errors.reshape(-1)
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "retrieval": "standardized 17D geometry L2",
            "candidate_pool": "train episodes only; one chunk per episode in top-K",
            "predictor": "none; direct canonical Demo action copy",
            "oracle": "analysis-only target-action selection",
            "bootstrap_unit": "validation episode",
        },
        "config": {
            "k": k,
            "seed": seed,
            "bootstrap_resamples": bootstrap_resamples,
        },
        "action_protocol": {
            "representation": action_representation,
            "pose_scales": pose_scales.tolist(),
            "translation_threshold_m": translation_threshold_m,
            "rotation_threshold_rad": rotation_threshold_rad,
        },
        "git_commit": _git_commit(project_root),
        "data_summary_sha256": _sha256(data_root / "summary.json"),
        "train_queries": len(train_records),
        "validation_queries": len(val_records),
        "validation_episodes": len(set(group_ids)),
        "metrics": metrics,
        "paired_episode_bootstrap": {
            "rank1_copy_minus_zero": _bootstrap_comparison(
                reference=zero_action,
                candidate=rank1,
                target=val_actions,
                mask=target_mask,
                group_ids=group_ids,
                seed=seed,
                resamples=bootstrap_resamples,
                pose_scales=pose_scales,
                translation_threshold_m=translation_threshold_m,
                rotation_threshold_rad=rotation_threshold_rad,
            ),
            f"oracle_best_in_{k}_minus_rank1_copy": _bootstrap_comparison(
                reference=rank1,
                candidate=predictions[f"oracle_best_in_{k}"],
                target=val_actions,
                mask=target_mask,
                group_ids=group_ids,
                seed=seed + 100,
                resamples=bootstrap_resamples,
                pose_scales=pose_scales,
                translation_threshold_m=translation_threshold_m,
                rotation_threshold_rad=rotation_threshold_rad,
            ),
        },
        "geometry_distance": {
            "top1_mean": float(selected_distances[:, 0].mean()),
            "top1_p95": float(np.quantile(selected_distances[:, 0], 0.95)),
            "top4_mean": float(selected_distances.mean()),
            "pearson_with_candidate_nmae": float(
                np.corrcoef(flat_distances, flat_errors)[0, 1]
            ),
            "rank_correlation_with_candidate_nmae": _rank_correlation(
                flat_distances, flat_errors
            ),
        },
        "oracle_selection_rank": {
            str(rank + 1): int((oracle4_index == rank).sum())
            for rank in range(k)
        },
        "train_geometry_mean": mean.tolist(),
        "train_geometry_std": std.tolist(),
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    if arguments.k < 4:
        raise ValueError("本实验固定要求 k>=4")
    run(
        project_root=arguments.project_root.resolve(),
        data_root=arguments.data_root.resolve(),
        output_path=arguments.output.resolve(),
        k=arguments.k,
        seed=arguments.seed,
        bootstrap_resamples=arguments.bootstrap_resamples,
    )


if __name__ == "__main__":
    main()

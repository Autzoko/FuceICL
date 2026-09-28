"""评估 geometry Retriever 的因果单调轨迹进度约束。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
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
from dev.predictor.train_action_chunks import _physical_metrics
from dev.simulator.evaluate_maniskill_demo_prior import (
    _action_protocol,
    _load_split,
)


@dataclass(frozen=True)
class ProgressConfig:
    """结果揭盲前固定的 progress Retriever 配置。"""

    schema_version: str
    seed: int
    progress_window_frames: int
    bootstrap_resamples: int

    @classmethod
    def from_json(cls, path: Path) -> "ProgressConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def __post_init__(self) -> None:
        if not self.schema_version.strip():
            raise ValueError("schema_version 不能为空")
        if min(self.progress_window_frames, self.bootstrap_resamples) <= 0:
            raise ValueError("progress window 与 bootstrap 次数必须为正")


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


def _stateless_top1(distances: torch.Tensor) -> torch.Tensor:
    return distances.argmin(dim=1)


def _causal_monotonic_selection(
    *,
    distances: torch.Tensor,
    query_records: Sequence[Mapping[str, Any]],
    candidate_records: Sequence[Mapping[str, Any]],
    progress_window_frames: int,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """每个 query episode 首次全局检索，之后只沿同一 Demo 向前。"""
    if distances.shape != (len(query_records), len(candidate_records)):
        raise ValueError("distance matrix 与 manifest 长度不一致")
    candidate_episodes = torch.tensor(
        [int(record["episode"]) for record in candidate_records]
    )
    candidate_frames = torch.tensor(
        [int(record["frame"]) for record in candidate_records]
    )
    selected = torch.empty(len(query_records), dtype=torch.long)
    diagnostics: list[dict[str, Any]] = []
    grouped: dict[int, list[int]] = {}
    for index, record in enumerate(query_records):
        grouped.setdefault(int(record["episode"]), []).append(index)

    for query_episode, raw_indices in sorted(grouped.items()):
        indices = sorted(
            raw_indices,
            key=lambda index: int(query_records[index]["frame"]),
        )
        previous: int | None = None
        reacquisitions = 0
        constrained_steps = 0
        for query_index in indices:
            if previous is None:
                choice = int(distances[query_index].argmin())
            else:
                previous_episode = int(candidate_episodes[previous])
                previous_frame = int(candidate_frames[previous])
                valid = (
                    (candidate_episodes == previous_episode)
                    & (candidate_frames > previous_frame)
                    & (
                        candidate_frames
                        <= previous_frame + progress_window_frames
                    )
                )
                if bool(valid.any()):
                    constrained = distances[query_index].masked_fill(
                        ~valid, torch.inf
                    )
                    choice = int(constrained.argmin())
                    constrained_steps += 1
                else:
                    choice = int(distances[query_index].argmin())
                    reacquisitions += 1
            selected[query_index] = choice
            previous = choice
        diagnostics.append(
            {
                "query_episode": query_episode,
                "queries": len(indices),
                "constrained_steps": constrained_steps,
                "reacquisitions": reacquisitions,
            }
        )
    return selected, diagnostics


def _temporal_metrics(
    indices: torch.Tensor,
    records: Sequence[Mapping[str, Any]],
    query_records: Sequence[Mapping[str, Any]],
    actions: torch.Tensor,
) -> dict[str, float | int | None]:
    episode_switches = 0
    transitions = 0
    sequential = 0
    forward = 0
    frame_deltas: list[int] = []
    grouped: dict[int, list[int]] = {}
    for query_index, record in enumerate(query_records):
        grouped.setdefault(int(record["episode"]), []).append(query_index)
    for query_indices in grouped.values():
        ordered = sorted(
            query_indices,
            key=lambda index: int(query_records[index]["frame"]),
        )
        choices = indices[ordered].tolist()
        for left, right in zip(choices, choices[1:]):
            transitions += 1
            left_episode = int(records[left]["episode"])
            right_episode = int(records[right]["episode"])
            if left_episode != right_episode:
                episode_switches += 1
                continue
            delta = int(records[right]["frame"]) - int(records[left]["frame"])
            frame_deltas.append(delta)
            sequential += int(delta == 1)
            forward += int(delta > 0)
    first_translation = torch.linalg.vector_norm(
        actions[indices, 0, :3], dim=-1
    )
    first_rotation = torch.linalg.vector_norm(
        actions[indices, 0, 3:6], dim=-1
    )
    near_zero = (first_translation < 1e-5) & (first_rotation < 1e-5)
    return {
        "transitions": transitions,
        "episode_switches": episode_switches,
        "episode_switch_rate": episode_switches / max(transitions, 1),
        "same_episode_forward_rate": forward / max(transitions, 1),
        "same_episode_sequential_rate": sequential / max(transitions, 1),
        "same_episode_frame_delta_mean": (
            float(np.mean(frame_deltas)) if frame_deltas else None
        ),
        "selected_first_token_normalized_near_zero_rate": float(
            near_zero.float().mean()
        ),
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
    config: ProgressConfig,
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
            seed=config.seed + offset,
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
    config: ProgressConfig,
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
    data_horizon = int(train_actions.shape[1])
    if config.progress_window_frames != data_horizon:
        raise ValueError("v1 progress window 必须等于数据 action horizon")
    mean = train_geometry.mean(dim=0)
    std = train_geometry.std(dim=0, unbiased=False).clamp_min(1e-4)
    distances = torch.cdist(
        (val_geometry - mean) / std,
        (train_geometry - mean) / std,
    ) / np.sqrt(train_geometry.shape[1])
    stateless_indices = _stateless_top1(distances)
    progress_indices, progress_diagnostics = _causal_monotonic_selection(
        distances=distances,
        query_records=val_records,
        candidate_records=train_records,
        progress_window_frames=config.progress_window_frames,
    )
    stateless = train_actions[stateless_indices]
    progress = train_actions[progress_indices]
    mask = torch.ones(val_actions.shape[:2], dtype=torch.bool)
    metric_options = {
        "pose_scales": pose_scales,
        "translation_threshold_m": translation_threshold_m,
        "rotation_threshold_rad": rotation_threshold_rad,
    }
    group_ids = [f"PickCube-v1:{record['episode']}" for record in val_records]
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "stateless": "standardized 17D geometry global top-1 per query",
            "progress": (
                "first query global top-1; then same Demo episode with "
                "frame delta in [1,H]; global reacquire only if empty"
            ),
            "causal": True,
            "validation_used_for_selection": False,
            "window_source": "stored action horizon H=6; not tuned",
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
        "selection_sha256": {
            "stateless": _selection_hash(stateless_indices),
            "causal_monotonic": _selection_hash(progress_indices),
        },
        "metrics": {
            "stateless_top1": _physical_metrics(
                stateless, val_actions, mask, **metric_options
            ),
            "causal_monotonic": _physical_metrics(
                progress, val_actions, mask, **metric_options
            ),
        },
        "paired_episode_bootstrap": {
            "causal_monotonic_minus_stateless": _bootstrap(
                reference=stateless,
                candidate=progress,
                target=val_actions,
                group_ids=group_ids,
                pose_scales=pose_scales,
                translation_threshold_m=translation_threshold_m,
                rotation_threshold_rad=rotation_threshold_rad,
                config=config,
            )
        },
        "temporal_metrics": {
            "stateless_top1": _temporal_metrics(
                stateless_indices, train_records, val_records, train_actions
            ),
            "causal_monotonic": _temporal_metrics(
                progress_indices, train_records, val_records, train_actions
            ),
        },
        "progress_diagnostics": progress_diagnostics,
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
        config=ProgressConfig.from_json(arguments.config.resolve()),
    )


if __name__ == "__main__":
    main()

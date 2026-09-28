"""在冻结 fresh episodes 上确认 RADM，不执行训练或模型选择。"""

from __future__ import annotations

import argparse
import atexit
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from dev.pointnet.compare_retrievers import _sha256
from dev.predictor.retriever_anchored_demo_mixture import (
    RetrieverAnchoredDemoMixture,
    RetrieverAnchoredMixtureConfig,
)
from dev.predictor.train_action_chunks import _physical_metrics
from dev.simulator.evaluate_maniskill_demo_prior import _bootstrap_comparison
from dev.simulator.train_causal_history_gate import (
    EXPECTED_TASKS,
    _checkpoint_bank_indices,
    _load_frozen_models,
)
from dev.simulator.train_maniskill_low_rank_transport import (
    TaskData,
    _load_task,
    _selection_hash,
)
from dev.simulator.train_retriever_anchored_demo_mixture import (
    PreparedSplit,
    _predict,
    _prepare_split,
    _selection_digest,
)


@dataclass(frozen=True)
class ExternalConfirmationConfig:
    """冻结的 RADM 外部确认协议。"""

    schema_version: str
    seed: int
    source_checkpoint_sha256: str
    radm_checkpoint_sha256: str
    candidate_count: int
    bootstrap_resamples: int
    query_summary_sha256: dict[str, str]
    query_audit_sha256: dict[str, str]

    @classmethod
    def from_json(cls, path: Path) -> "ExternalConfirmationConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def __post_init__(self) -> None:
        if self.schema_version != "maniskill-radm-external-confirmation-v1":
            raise ValueError("未知 RADM external confirmation schema")
        if self.candidate_count != 4 or self.bootstrap_resamples <= 0:
            raise ValueError("RADM external confirmation 固定 K=4 且 bootstrap>0")
        for value in (
            self.source_checkpoint_sha256,
            self.radm_checkpoint_sha256,
            *self.query_summary_sha256.values(),
            *self.query_audit_sha256.values(),
        ):
            if len(value) != 64:
                raise ValueError("RADM external confirmation SHA256 非法")
        if (
            set(self.query_summary_sha256) != EXPECTED_TASKS
            or set(self.query_audit_sha256) != EXPECTED_TASKS
        ):
            raise ValueError("RADM external confirmation 必须冻结三个 tasks")


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _task_map(tasks: Sequence[TaskData], label: str) -> dict[str, TaskData]:
    mapping = {task.task_id: task for task in tasks}
    if len(tasks) != 3 or set(mapping) != EXPECTED_TASKS:
        raise ValueError(f"{label} 必须恰好包含冻结的三个 ManiSkill tasks")
    return mapping


def _combined_queries(
    task: TaskData,
) -> tuple[list[dict[str, Any]], torch.Tensor, torch.Tensor]:
    records = []
    for split, values in (
        ("train", task.train_records),
        ("val", task.val_records),
    ):
        records.extend({**record, "query_split": split} for record in values)
    return (
        records,
        torch.cat((task.train_geometry, task.val_geometry)),
        torch.cat((task.train_actions, task.val_actions)),
    )


def _load_radm(
    path: Path,
    source_sha256: str,
    device: torch.device,
) -> RetrieverAnchoredDemoMixture:
    payload = torch.load(path, map_location="cpu")
    if payload.get("source_checkpoint_sha256") != source_sha256:
        raise ValueError("RADM checkpoint 的 source checkpoint 不匹配")
    model = RetrieverAnchoredDemoMixture(
        RetrieverAnchoredMixtureConfig(**payload["config"])
    )
    model.load_state_dict(payload["model"])
    return model.to(device).eval()


def _protocol_options(task: TaskData) -> dict[str, Any]:
    return {
        "pose_scales": task.pose_scales,
        "translation_threshold_m": task.translation_threshold_m,
        "rotation_threshold_rad": task.rotation_threshold_rad,
    }


def _groups(prepared: PreparedSplit) -> list[str]:
    return [
        (
            f"{prepared.task_id}:{row['query_split']}:"
            f"episode-{int(row['episode'])}"
        )
        for row in prepared.records
    ]


@torch.inference_mode()
def run(
    *,
    project_root: Path,
    bank_roots: Sequence[Path],
    query_roots: Sequence[Path],
    source_checkpoint_path: Path,
    radm_checkpoint_path: Path,
    config_path: Path,
    output_root: Path,
    config: ExternalConfirmationConfig,
    device: torch.device,
) -> None:
    if output_root.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output_root}")
    source_sha256 = _sha256(source_checkpoint_path)
    radm_sha256 = _sha256(radm_checkpoint_path)
    if source_sha256 != config.source_checkpoint_sha256:
        raise ValueError("source checkpoint SHA256 不匹配")
    if radm_sha256 != config.radm_checkpoint_sha256:
        raise ValueError("RADM checkpoint SHA256 不匹配")

    temporary = output_root.with_name(f".{output_root.name}.incomplete-{os.getpid()}")
    temporary.mkdir(parents=True)

    def cleanup() -> None:
        shutil.rmtree(temporary, ignore_errors=True)

    atexit.register(cleanup)
    source_checkpoint = torch.load(source_checkpoint_path, map_location="cpu")
    transport, gate = _load_frozen_models(source_checkpoint, device)
    radm = _load_radm(radm_checkpoint_path, source_sha256, device)
    if radm.config.maximum_odds_distortion != 4.0:
        raise ValueError("RADM checkpoint odds bound 不是冻结值 4")

    bank_tasks = _task_map(
        [_load_task(root) for root in bank_roots],
        "bank roots",
    )
    query_tasks = _task_map(
        [_load_task(root) for root in query_roots],
        "query roots",
    )
    reference_query = query_tasks[sorted(EXPECTED_TASKS)[0]]
    prepared_by_task: list[PreparedSplit] = []
    data_audit: dict[str, Any] = {}
    for task_id in sorted(EXPECTED_TASKS):
        bank = bank_tasks[task_id]
        query = query_tasks[task_id]
        bank_summary = _sha256(bank.root / "summary.json")
        query_summary = _sha256(query.root / "summary.json")
        query_audit = _sha256(query.root / "audit.json")
        if source_checkpoint["data_summary_sha256"][task_id] != bank_summary:
            raise ValueError(f"{task_id} bank summary 与 source checkpoint 不匹配")
        if query_summary != config.query_summary_sha256[task_id]:
            raise ValueError(f"{task_id} query summary SHA256 不匹配")
        if query_audit != config.query_audit_sha256[task_id]:
            raise ValueError(f"{task_id} query audit SHA256 不匹配")
        for task in (bank, query):
            if task.action_representation != source_checkpoint["action_representation"]:
                raise ValueError(f"{task_id} action representation 不匹配")
            if not torch.equal(
                task.pose_scales.float(),
                torch.as_tensor(source_checkpoint["action_pose_scales"]).float(),
            ):
                raise ValueError(f"{task_id} action scales 不匹配")
        if bank.train_actions.shape[1:] != query.train_actions.shape[1:]:
            raise ValueError(f"{task_id} bank/query action shape 不匹配")
        if (
            bank.translation_threshold_m != query.translation_threshold_m
            or bank.rotation_threshold_rad != query.rotation_threshold_rad
        ):
            raise ValueError(f"{task_id} bank/query evaluation thresholds 不匹配")
        if (
            query.translation_threshold_m
            != reference_query.translation_threshold_m
            or query.rotation_threshold_rad
            != reference_query.rotation_threshold_rad
        ):
            raise ValueError("外部确认 tasks 的 evaluation thresholds 不一致")

        bank_indices = _checkpoint_bank_indices(
            bank.train_records,
            source_checkpoint["bank_chunk_ids"][task_id],
        )
        query_records, query_geometry, query_actions = _combined_queries(query)
        prepared = _prepare_split(
            task_id=task_id,
            records=query_records,
            query_geometry=query_geometry,
            target=query_actions,
            bank_records=[
                bank.train_records[index] for index in bank_indices.tolist()
            ],
            bank_geometry=bank.train_geometry[bank_indices],
            bank_actions=bank.train_actions[bank_indices],
            transport=transport,
            gate=gate,
            candidate_count=config.candidate_count,
            device=device,
        )
        prepared_by_task.append(prepared)
        data_audit[task_id] = {
            "bank_summary_sha256": bank_summary,
            "bank_chunks": len(bank_indices),
            "bank_selection_sha256": _selection_hash(bank_indices),
            "query_summary_sha256": query_summary,
            "query_audit_sha256": query_audit,
            "query_chunks": len(query_records),
            "query_episode_groups": len(set(_groups(prepared))),
            "candidate_selection_sha256": _selection_digest(
                prepared.candidate_indices
            ),
        }

    per_task: dict[str, Any] = {}
    aggregate_target = []
    aggregate_groups: list[str] = []
    aggregate_predictions: dict[str, list[torch.Tensor]] = {}
    aggregate_posterior = []
    aggregate_prior = []
    aggregate_candidates = []
    artifacts: dict[str, list[np.ndarray]] = {}
    for offset, prepared in enumerate(prepared_by_task):
        task = query_tasks[prepared.task_id]
        predictions, posterior = _predict(radm, prepared, device)
        mask = torch.ones(prepared.target.shape[:2], dtype=torch.bool)
        options = _protocol_options(task)
        metrics = {
            name: _physical_metrics(value, prepared.target, mask, **options)
            for name, value in predictions.items()
        }
        groups = _groups(prepared)
        comparisons = {
            f"radm_minus_{reference}": _bootstrap_comparison(
                reference=predictions[reference],
                candidate=predictions["radm"],
                target=prepared.target,
                mask=mask,
                group_ids=groups,
                seed=config.seed + offset * 100 + index,
                resamples=config.bootstrap_resamples,
                **options,
            )
            for index, reference in enumerate(
                ("rank1_bcsg", "retriever_prior_mixture", "uniform_mixture")
            )
        }
        denominator = (
            metrics["rank1_bcsg"]["translation_l2_m"]
            - metrics["oracle_best_in_4"]["translation_l2_m"]
        )
        recovery = (
            metrics["rank1_bcsg"]["translation_l2_m"]
            - metrics["radm"]["translation_l2_m"]
        ) / max(denominator, 1e-12)
        per_task[prepared.task_id] = {
            "queries": len(prepared.target),
            "episode_groups": len(set(groups)),
            "metrics": metrics,
            "paired_episode_bootstrap": comparisons,
            "oracle_gap_recovery": recovery,
            "posterior_rank_mean": posterior.mean(dim=0).tolist(),
        }
        aggregate_target.append(prepared.target)
        aggregate_groups.extend(groups)
        aggregate_posterior.append(posterior)
        aggregate_prior.append(prepared.prior)
        aggregate_candidates.append(prepared.hypotheses)
        for name, value in predictions.items():
            aggregate_predictions.setdefault(name, []).append(value)
        rows: dict[str, np.ndarray] = {
            "task": np.asarray([prepared.task_id] * len(prepared.target)),
            "group_id": np.asarray(groups),
            "target": prepared.target.numpy(),
            "posterior": posterior.numpy(),
            "prior": prepared.prior.numpy(),
            "candidate_indices": prepared.candidate_indices.numpy(),
            **{name: value.numpy() for name, value in predictions.items()},
        }
        for name, value in rows.items():
            artifacts.setdefault(name, []).append(value)

    target = torch.cat(aggregate_target)
    predictions = {
        name: torch.cat(values) for name, values in aggregate_predictions.items()
    }
    posterior = torch.cat(aggregate_posterior)
    prior = torch.cat(aggregate_prior)
    candidates = torch.cat(aggregate_candidates)
    mask = torch.ones(target.shape[:2], dtype=torch.bool)
    first_task = query_tasks[sorted(EXPECTED_TASKS)[0]]
    options = _protocol_options(first_task)
    metrics = {
        name: _physical_metrics(value, target, mask, **options)
        for name, value in predictions.items()
    }
    comparisons = {
        f"radm_minus_{reference}": _bootstrap_comparison(
            reference=predictions[reference],
            candidate=predictions["radm"],
            target=target,
            mask=mask,
            group_ids=aggregate_groups,
            seed=config.seed + 1000 + index,
            resamples=config.bootstrap_resamples,
            **options,
        )
        for index, reference in enumerate(
            ("rank1_bcsg", "retriever_prior_mixture", "uniform_mixture")
        )
    }
    rank1_translation = metrics["rank1_bcsg"]["translation_l2_m"]
    oracle_translation = metrics["oracle_best_in_4"]["translation_l2_m"]
    radm_translation = metrics["radm"]["translation_l2_m"]
    recovery = (rank1_translation - radm_translation) / max(
        rank1_translation - oracle_translation,
        1e-12,
    )
    odds = (posterior[:, :, None] / posterior[:, None, :]) / (
        prior[:, :, None] / prior[:, None, :]
    )
    candidate_translation = candidates[..., :3]
    radm_translation_values = predictions["radm"][..., :3]
    convex_violation = max(
        float(
            (candidate_translation.amin(dim=1) - radm_translation_values)
            .clamp_min(0)
            .max()
        ),
        float(
            (radm_translation_values - candidate_translation.amax(dim=1))
            .clamp_min(0)
            .max()
        ),
    )
    discrete_error = float(
        (
            predictions["radm"][..., 3:]
            - predictions["rank1_bcsg"][..., 3:]
        )
        .abs()
        .max()
    )
    improved_tasks = sum(
        values["metrics"]["radm"]["translation_l2_m"]
        < values["metrics"]["rank1_bcsg"]["translation_l2_m"]
        for values in per_task.values()
    )
    rank1_comparison = comparisons["radm_minus_rank1_bcsg"]["translation_l2_m"]
    prior_comparison = comparisons[
        "radm_minus_retriever_prior_mixture"
    ]["translation_l2_m"]
    criteria = {
        "c1_radm_better_than_rank1": rank1_comparison["ci95_high"] < 0.0,
        "c2_radm_better_than_fixed_prior": prior_comparison["ci95_high"] < 0.0,
        "c2_at_least_two_tasks_better_than_rank1": improved_tasks >= 2,
        "c3_recovers_at_least_20_percent_oracle_gap": recovery >= 0.20,
        "c4_odds_bound": float(odds.max()) <= 4.0 + 1e-6,
        "c4_translation_convex_hull": convex_violation <= 1e-6,
        "c4_discrete_prior_exact": discrete_error == 0.0,
        "c4_query_hashes": all(
            values["query_summary_sha256"]
            == config.query_summary_sha256[task_id]
            and values["query_audit_sha256"]
            == config.query_audit_sha256[task_id]
            for task_id, values in data_audit.items()
        ),
    }
    artifact_path = temporary / "predictions.npz"
    with artifact_path.open("wb") as stream:
        np.savez_compressed(
            stream,
            **{
                name: np.concatenate(values)
                for name, values in artifacts.items()
            },
        )
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(project_root),
        "protocol": "frozen RADM on 3x24 fresh source episodes",
        "config": asdict(config),
        "config_sha256": _sha256(config_path),
        "source_checkpoint_sha256": source_sha256,
        "radm_checkpoint_sha256": radm_sha256,
        "data": data_audit,
        "device": str(device),
        "per_task": per_task,
        "aggregate": {
            "queries": len(target),
            "episode_groups": len(set(aggregate_groups)),
            "metrics": metrics,
            "paired_episode_bootstrap": comparisons,
            "oracle_gap_recovery": recovery,
            "posterior_rank_mean": posterior.mean(dim=0).tolist(),
            "maximum_odds_distortion": float(odds.max()),
        },
        "structural": {
            "translation_convex_hull_max_violation": convex_violation,
            "discrete_prior_max_abs_error": discrete_error,
        },
        "criteria": criteria,
        "confirmation_passed": all(criteria.values()),
        "predictions_sha256": _sha256(artifact_path),
    }
    (temporary / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_root)
    atexit.unregister(cleanup)
    print(json.dumps(report["aggregate"], indent=2), flush=True)
    print(json.dumps(criteria, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--bank-root", type=Path, action="append", required=True)
    parser.add_argument("--query-root", type=Path, action="append", required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--radm-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(
        project_root=arguments.project_root.resolve(),
        bank_roots=[path.resolve() for path in arguments.bank_root],
        query_roots=[path.resolve() for path in arguments.query_root],
        source_checkpoint_path=arguments.source_checkpoint.resolve(),
        radm_checkpoint_path=arguments.radm_checkpoint.resolve(),
        config_path=arguments.config.resolve(),
        output_root=arguments.output_root.resolve(),
        config=ExternalConfirmationConfig.from_json(arguments.config.resolve()),
        device=torch.device(arguments.device),
    )

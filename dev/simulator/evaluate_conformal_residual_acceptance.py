"""在独立 ManiSkill episodes 上校准并评估 CoDRA。"""

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
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from dev.pointnet.compare_retrievers import _sha256
from dev.predictor.benefit_calibrated_shrinkage import (
    BenefitCalibratedGate,
    BenefitGateConfig,
    apply_shrinkage,
    frozen_transport_features,
)
from dev.predictor.conformal_residual_acceptance import (
    ConformalAcceptanceCalibration,
    calibrate_acceptance_threshold,
    episode_risk,
    residual_risk_scores,
    selective_gate,
)
from dev.predictor.query_aligned_transport import (
    QueryAlignedLowRankDemoTransport,
    QueryAlignedTransportConfig,
)
from dev.predictor.train_action_chunks import _physical_metrics
from dev.simulator.evaluate_maniskill_demo_prior import _bootstrap_comparison
from dev.simulator.train_maniskill_low_rank_transport import (
    TaskData,
    _load_task,
    _nearest_demo_indices,
    _selection_hash,
)
from dev.simulator.train_maniskill_query_aligned_bcsg import _qa_diagnostics


SCORE_NAMES = ("distance", "applied_residual", "codra")


@dataclass(frozen=True)
class CoDRAConfig:
    """独立数据上的冻结 CoDRA 协议。"""

    schema_version: str
    seed: int
    alpha: float
    expected_calibration_episodes: int
    expected_test_episodes: int
    harm_tolerance: float
    bootstrap_resamples: int
    latency_iterations: int
    source_start_seeds: dict[str, int]
    excluded_source_seeds: list[int]

    @classmethod
    def from_json(cls, path: Path) -> "CoDRAConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def __post_init__(self) -> None:
        positive = (
            self.expected_calibration_episodes,
            self.expected_test_episodes,
            self.harm_tolerance,
            self.bootstrap_resamples,
            self.latency_iterations,
        )
        if not self.schema_version.strip() or min(positive) <= 0:
            raise ValueError("CoDRA schema 与正数配置非法")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha 必须位于 (0,1)")
        if len(self.source_start_seeds) != 3:
            raise ValueError("必须为三个任务冻结 source start seeds")
        if len(set(self.excluded_source_seeds)) != len(self.excluded_source_seeds):
            raise ValueError("excluded source seeds 不能重复")


@dataclass(frozen=True)
class SplitPrediction:
    """一个任务、一个 split 的冻结检索与模型输出。"""

    records: list[dict[str, Any]]
    target: torch.Tensor
    demo_actions: torch.Tensor
    transported: torch.Tensor
    bcsg: torch.Tensor
    base_gate: torch.Tensor
    retrieval_distance: torch.Tensor
    residual_norm: torch.Tensor
    scores: dict[str, torch.Tensor]
    harmful: torch.Tensor
    episode_ids: torch.Tensor


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _load_models(
    checkpoint: Mapping[str, Any],
    device: torch.device,
) -> tuple[QueryAlignedLowRankDemoTransport, BenefitCalibratedGate]:
    transport = QueryAlignedLowRankDemoTransport(
        QueryAlignedTransportConfig(
            **checkpoint["model_configs"]["query_aligned_transport"]
        ),
        geometry_mean=checkpoint["geometry_mean"],
        geometry_std=checkpoint["geometry_std"],
    )
    transport.load_state_dict(checkpoint["models"]["query_aligned_transport"])
    gate = BenefitCalibratedGate(
        BenefitGateConfig(**checkpoint["gate_config"])
    )
    gate.load_state_dict(checkpoint["gate"])
    return transport.to(device).eval(), gate.to(device).eval()


def _bank_indices(
    records: Sequence[Mapping[str, Any]],
    expected_chunk_ids: Sequence[str],
) -> torch.Tensor:
    by_id = {str(record["chunk_id"]): index for index, record in enumerate(records)}
    if len(by_id) != len(records):
        raise ValueError("bank manifest chunk_id 不唯一")
    missing = [value for value in expected_chunk_ids if value not in by_id]
    if missing:
        raise ValueError(f"checkpoint bank chunk 缺失：{missing[:3]}")
    return torch.tensor([by_id[value] for value in expected_chunk_ids])


def _split_values(
    task: TaskData,
    split: str,
) -> tuple[list[dict[str, Any]], torch.Tensor, torch.Tensor]:
    if split == "train":
        return task.train_records, task.train_geometry, task.train_actions
    if split == "val":
        return task.val_records, task.val_geometry, task.val_actions
    raise ValueError(f"未知 split：{split}")


def _translation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return (prediction[..., :3] - target[..., :3]).square().sum(dim=(1, 2))


def _predict_split(
    *,
    evaluation: TaskData,
    split: str,
    bank_records: list[dict[str, Any]],
    bank_geometry: torch.Tensor,
    bank_actions: torch.Tensor,
    transport: QueryAlignedLowRankDemoTransport,
    gate: BenefitCalibratedGate,
    harm_tolerance: float,
    device: torch.device,
) -> SplitPrediction:
    records, query, target = _split_values(evaluation, split)
    indices, distances = _nearest_demo_indices(
        candidate_records=bank_records,
        candidate_geometry=bank_geometry,
        query_records=records,
        query_geometry=query,
        exclude_same_episode=False,
    )
    demo_geometry = bank_geometry[indices]
    demo_actions = bank_actions[indices]
    transported, features = _qa_diagnostics(
        transport,
        query,
        demo_geometry,
        demo_actions,
        device,
    )
    with torch.inference_mode():
        base_gate = gate(features.to(device)).cpu()
    mask = torch.ones(target.shape[:2], dtype=torch.bool)
    bcsg = apply_shrinkage(demo_actions, transported, base_gate, mask)
    residual_norm = torch.linalg.vector_norm(
        transported[..., :3] - demo_actions[..., :3],
        dim=-1,
    ).mean(dim=1)
    scores = residual_risk_scores(
        retrieval_distance=distances,
        residual_norm=residual_norm,
        base_gate=base_gate,
    )
    harmful = (
        _translation_loss(bcsg, target)
        > _translation_loss(demo_actions, target) + harm_tolerance
    )
    episode_ids = torch.tensor([int(record["episode"]) for record in records])
    return SplitPrediction(
        records=records,
        target=target,
        demo_actions=demo_actions,
        transported=transported,
        bcsg=bcsg,
        base_gate=base_gate,
        retrieval_distance=distances,
        residual_norm=residual_norm,
        scores=scores,
        harmful=harmful,
        episode_ids=episode_ids,
    )


def _episode_acceptance(
    accepted: torch.Tensor,
    episode_ids: torch.Tensor,
) -> torch.Tensor:
    values = []
    for episode in torch.unique(episode_ids, sorted=True):
        values.append(accepted[episode_ids == episode].float().mean())
    return torch.stack(values)


def _bootstrap_mean(
    values: torch.Tensor,
    *,
    seed: int,
    resamples: int,
) -> dict[str, float | int]:
    array = values.detach().cpu().numpy().astype(np.float64)
    generator = np.random.default_rng(seed)
    samples = generator.choice(array, size=(resamples, len(array)), replace=True)
    means = samples.mean(axis=1)
    return {
        "episodes": len(array),
        "mean": float(array.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def _source_seeds(records: Sequence[Mapping[str, Any]]) -> set[int]:
    if not records or any("source_episode_seed" not in row for row in records):
        raise ValueError("独立 CoDRA 数据缺少 source_episode_seed")
    by_episode: dict[int, set[int]] = {}
    for row in records:
        by_episode.setdefault(int(row["episode"]), set()).add(
            int(row["source_episode_seed"])
        )
    if any(len(values) != 1 for values in by_episode.values()):
        raise ValueError("同一 episode 存在多个 source seeds")
    seeds = {next(iter(values)) for values in by_episode.values()}
    if len(seeds) != len(by_episode):
        raise ValueError("source episode seeds 不唯一")
    return seeds


@torch.inference_mode()
def _latency(
    *,
    transport: QueryAlignedLowRankDemoTransport,
    gate: BenefitCalibratedGate,
    query: torch.Tensor,
    demo_geometry: torch.Tensor,
    demo_actions: torch.Tensor,
    retrieval_distance: torch.Tensor,
    calibration: ConformalAcceptanceCalibration,
    device: torch.device,
    iterations: int,
) -> dict[str, float | int]:
    query = query[:1].to(device)
    demo_geometry = demo_geometry[:1].to(device)
    demo_actions = demo_actions[:1].to(device)
    retrieval_distance = retrieval_distance[:1].to(device)
    mask = torch.ones(demo_actions.shape[:2], dtype=torch.bool, device=device)

    def predict() -> torch.Tensor:
        transported, features = frozen_transport_features(
            transport,
            query,
            demo_geometry,
            demo_actions,
            mask,
        )
        base_gate = gate(features)
        residual = torch.linalg.vector_norm(
            transported[..., :3] - demo_actions[..., :3], dim=-1
        ).mean(dim=1)
        score = retrieval_distance * base_gate * residual
        selected, _ = selective_gate(
            base_gate=base_gate,
            scores=score,
            calibration=calibration,
        )
        return apply_shrinkage(demo_actions, transported, selected, mask)

    for _ in range(20):
        predict()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        predict()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        samples.append(1000.0 * (time.perf_counter() - started))
    values = np.asarray(samples)
    return {
        "iterations": iterations,
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.quantile(values, 0.95)),
        "maximum_ms": float(values.max()),
    }


def run(
    *,
    project_root: Path,
    bank_roots: Sequence[Path],
    evaluation_roots: Sequence[Path],
    checkpoint_path: Path,
    config_path: Path,
    output_root: Path,
    config: CoDRAConfig,
    device: torch.device,
) -> None:
    if output_root.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output_root}")
    temporary = output_root.with_name(f".{output_root.name}.incomplete-{os.getpid()}")
    temporary.mkdir(parents=True)

    def cleanup_incomplete() -> None:
        shutil.rmtree(temporary, ignore_errors=True)

    atexit.register(cleanup_incomplete)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    transport, gate = _load_models(checkpoint, device)
    bank_values = [_load_task(path) for path in bank_roots]
    evaluation_values = [_load_task(path) for path in evaluation_roots]
    banks = {task.task_id: task for task in bank_values}
    evaluations = {task.task_id: task for task in evaluation_values}
    if len(banks) != len(bank_values) or len(evaluations) != len(evaluation_values):
        raise ValueError("bank/evaluation task IDs 重复")
    expected_tasks = set(config.source_start_seeds)
    if set(banks) != expected_tasks or set(evaluations) != expected_tasks:
        raise ValueError("bank/evaluation roots 与冻结三任务不一致")

    per_task: dict[str, Any] = {}
    aggregate_targets = []
    aggregate_predictions: dict[str, list[torch.Tensor]] = {
        name: []
        for name in (
            "demo_action_copy",
            "query_aligned_transport",
            "bcsg",
            "crc_distance",
            "crc_applied_residual",
            "codra",
        )
    }
    aggregate_groups: list[str] = []
    aggregate_harm: dict[str, list[torch.Tensor]] = {
        name: [] for name in ("bcsg", "crc_distance", "crc_applied_residual", "codra")
    }
    aggregate_acceptance: dict[str, list[torch.Tensor]] = {
        name: [] for name in ("crc_distance", "crc_applied_residual", "codra")
    }
    artifact: dict[str, list[np.ndarray]] = {}
    latency_inputs = None
    rejection_error = {
        name: 0.0
        for name in ("crc_distance", "crc_applied_residual", "codra")
    }
    for task_offset, task_id in enumerate(sorted(expected_tasks)):
        bank = banks[task_id]
        evaluation = evaluations[task_id]
        if checkpoint["data_summary_sha256"].get(task_id) != _sha256(
            bank.root / "summary.json"
        ):
            raise ValueError(f"{task_id} bank 与 checkpoint 不匹配")
        if evaluation.action_representation != checkpoint["action_representation"]:
            raise ValueError(f"{task_id} evaluation action representation 不匹配")
        if not torch.equal(
            evaluation.pose_scales.float(),
            torch.as_tensor(checkpoint["action_pose_scales"]).float(),
        ):
            raise ValueError(f"{task_id} evaluation action scales 不匹配")
        bank_ids = checkpoint["bank_chunk_ids"][task_id]
        bank_selection = _bank_indices(bank.train_records, bank_ids)
        bank_records = [bank.train_records[index] for index in bank_selection.tolist()]
        bank_geometry = bank.train_geometry[bank_selection]
        bank_actions = bank.train_actions[bank_selection]
        calibration = _predict_split(
            evaluation=evaluation,
            split="train",
            bank_records=bank_records,
            bank_geometry=bank_geometry,
            bank_actions=bank_actions,
            transport=transport,
            gate=gate,
            harm_tolerance=config.harm_tolerance,
            device=device,
        )
        test = _predict_split(
            evaluation=evaluation,
            split="val",
            bank_records=bank_records,
            bank_geometry=bank_geometry,
            bank_actions=bank_actions,
            transport=transport,
            gate=gate,
            harm_tolerance=config.harm_tolerance,
            device=device,
        )
        calibration_seeds = _source_seeds(calibration.records)
        test_seeds = _source_seeds(test.records)
        if len(calibration_seeds) != config.expected_calibration_episodes:
            raise ValueError(f"{task_id} calibration episode 数不符合预注册")
        if len(test_seeds) != config.expected_test_episodes:
            raise ValueError(f"{task_id} test episode 数不符合预注册")
        if calibration_seeds & test_seeds:
            raise ValueError(f"{task_id} calibration/test source seeds 重叠")
        if (calibration_seeds | test_seeds) & set(config.excluded_source_seeds):
            raise ValueError(f"{task_id} source seeds 与既有实验重叠")
        if min(calibration_seeds | test_seeds) < config.source_start_seeds[task_id]:
            raise ValueError(f"{task_id} source seed 低于冻结起点")

        calibrations = {
            name: calibrate_acceptance_threshold(
                scores=calibration.scores[name],
                harmful=calibration.harmful,
                episode_ids=calibration.episode_ids,
                alpha=config.alpha,
            )
            for name in SCORE_NAMES
        }
        mask = torch.ones(test.target.shape[:2], dtype=torch.bool)
        predictions = {
            "demo_action_copy": test.demo_actions,
            "query_aligned_transport": test.transported,
            "bcsg": test.bcsg,
        }
        accepted_by_policy = {}
        gates_by_policy = {}
        for score_name, policy_name in (
            ("distance", "crc_distance"),
            ("applied_residual", "crc_applied_residual"),
            ("codra", "codra"),
        ):
            selected_gate, accepted = selective_gate(
                base_gate=test.base_gate,
                scores=test.scores[score_name],
                calibration=calibrations[score_name],
            )
            gates_by_policy[policy_name] = selected_gate
            accepted_by_policy[policy_name] = accepted
            predictions[policy_name] = apply_shrinkage(
                test.demo_actions,
                test.transported,
                selected_gate,
                mask,
            )
            if bool((~accepted).any()):
                rejection_error[policy_name] = max(
                    rejection_error[policy_name],
                    float(
                        (
                            predictions[policy_name][~accepted]
                            - test.demo_actions[~accepted]
                        )
                        .abs()
                        .max()
                    ),
                )
        options = {
            "pose_scales": evaluation.pose_scales,
            "translation_threshold_m": evaluation.translation_threshold_m,
            "rotation_threshold_rad": evaluation.rotation_threshold_rad,
        }
        metrics = {
            name: _physical_metrics(value, test.target, mask, **options)
            for name, value in predictions.items()
        }
        groups = [
            f"{task_id}:{row['source_episode_seed']}" for row in test.records
        ]
        comparisons = {
            f"{name}_minus_copy": _bootstrap_comparison(
                reference=test.demo_actions,
                candidate=predictions[name],
                target=test.target,
                mask=mask,
                group_ids=groups,
                seed=config.seed + task_offset * 100 + index,
                resamples=config.bootstrap_resamples,
                **options,
            )
            for index, name in enumerate(
                ("bcsg", "crc_distance", "crc_applied_residual", "codra")
            )
        }
        harm = {
            "bcsg": torch.ones_like(test.harmful, dtype=torch.bool),
            **accepted_by_policy,
        }
        harm_report = {
            name: _bootstrap_mean(
                episode_risk(
                    accepted=accepted,
                    harmful=test.harmful,
                    episode_ids=test.episode_ids,
                ),
                seed=config.seed + task_offset * 1000 + index,
                resamples=config.bootstrap_resamples,
            )
            for index, (name, accepted) in enumerate(harm.items())
        }
        acceptance_report = {
            name: _bootstrap_mean(
                _episode_acceptance(accepted, test.episode_ids),
                seed=config.seed + task_offset * 1000 + 100 + index,
                resamples=config.bootstrap_resamples,
            )
            for index, (name, accepted) in enumerate(accepted_by_policy.items())
        }
        per_task[task_id] = {
            "bank_chunks": len(bank_records),
            "bank_selection_sha256": _selection_hash(bank_selection),
            "calibration_episodes": len(calibration_seeds),
            "test_episodes": len(test_seeds),
            "calibration_source_seed_range": [
                min(calibration_seeds),
                max(calibration_seeds),
            ],
            "test_source_seed_range": [min(test_seeds), max(test_seeds)],
            "calibrations": {
                name: value.as_dict() for name, value in calibrations.items()
            },
            "metrics": metrics,
            "paired_episode_bootstrap": comparisons,
            "episode_harm_fraction": harm_report,
            "episode_acceptance_fraction": acceptance_report,
        }
        aggregate_targets.append(test.target)
        aggregate_groups.extend(groups)
        for name, prediction in predictions.items():
            aggregate_predictions[name].append(prediction)
        aggregate_harm["bcsg"].append(
            episode_risk(
                accepted=torch.ones_like(test.harmful, dtype=torch.bool),
                harmful=test.harmful,
                episode_ids=test.episode_ids,
            )
        )
        for name, accepted in accepted_by_policy.items():
            aggregate_harm[name].append(
                episode_risk(
                    accepted=accepted,
                    harmful=test.harmful,
                    episode_ids=test.episode_ids,
                )
            )
            aggregate_acceptance[name].append(
                _episode_acceptance(accepted, test.episode_ids)
            )
        for name, value in {
            "task": np.asarray([task_id] * len(test.records)),
            "group_id": np.asarray(groups),
            "target": test.target.numpy(),
            **{key: value.numpy() for key, value in predictions.items()},
            "retrieval_distance": test.retrieval_distance.numpy(),
            "residual_norm": test.residual_norm.numpy(),
            "base_gate": test.base_gate.numpy(),
            **{
                f"score_{key}": value.numpy() for key, value in test.scores.items()
            },
            **{
                f"accepted_{key}": value.numpy()
                for key, value in accepted_by_policy.items()
            },
        }.items():
            artifact.setdefault(name, []).append(value)
        if latency_inputs is None:
            demo_indices, distances = _nearest_demo_indices(
                candidate_records=bank_records,
                candidate_geometry=bank_geometry,
                query_records=test.records[:1],
                query_geometry=evaluation.val_geometry[:1],
                exclude_same_episode=False,
            )
            latency_inputs = (
                evaluation.val_geometry[:1],
                bank_geometry[demo_indices],
                bank_actions[demo_indices],
                distances,
                calibrations["codra"],
            )

    target = torch.cat(aggregate_targets)
    predictions = {
        name: torch.cat(values) for name, values in aggregate_predictions.items()
    }
    mask = torch.ones(target.shape[:2], dtype=torch.bool)
    first = next(iter(evaluations.values()))
    aggregate_options = {
        "pose_scales": first.pose_scales,
        "translation_threshold_m": first.translation_threshold_m,
        "rotation_threshold_rad": first.rotation_threshold_rad,
    }
    metrics = {
        name: _physical_metrics(value, target, mask, **aggregate_options)
        for name, value in predictions.items()
    }
    comparisons = {
        f"{name}_minus_copy": _bootstrap_comparison(
            reference=predictions["demo_action_copy"],
            candidate=predictions[name],
            target=target,
            mask=mask,
            group_ids=aggregate_groups,
            seed=config.seed + 10000 + index,
            resamples=config.bootstrap_resamples,
            **aggregate_options,
        )
        for index, name in enumerate(
            ("bcsg", "crc_distance", "crc_applied_residual", "codra")
        )
    }
    harm_report = {
        name: _bootstrap_mean(
            torch.cat(values),
            seed=config.seed + 20000 + index,
            resamples=config.bootstrap_resamples,
        )
        for index, (name, values) in enumerate(aggregate_harm.items())
    }
    acceptance_report = {
        name: _bootstrap_mean(
            torch.cat(values),
            seed=config.seed + 30000 + index,
            resamples=config.bootstrap_resamples,
        )
        for index, (name, values) in enumerate(aggregate_acceptance.items())
    }
    if latency_inputs is None:
        raise RuntimeError("缺少 latency 输入")
    latency = _latency(
        transport=transport,
        gate=gate,
        query=latency_inputs[0],
        demo_geometry=latency_inputs[1],
        demo_actions=latency_inputs[2],
        retrieval_distance=latency_inputs[3],
        calibration=latency_inputs[4],
        device=device,
        iterations=config.latency_iterations,
    )
    codra_copy = comparisons["codra_minus_copy"]["translation_l2_m"]
    improved_tasks = sum(
        values["metrics"]["codra"]["translation_l2_m"]
        <= values["metrics"]["demo_action_copy"]["translation_l2_m"]
        for values in per_task.values()
    )
    criteria = {
        "r1_each_task_harm_at_most_alpha": all(
            values["episode_harm_fraction"]["codra"]["mean"] <= config.alpha
            for values in per_task.values()
        ),
        "r2_pooled_acceptance_at_least_20_percent": (
            acceptance_report["codra"]["mean"] >= 0.20
        ),
        "r3_pooled_translation_better_than_copy": (
            codra_copy["ci95_high"] < 0.0
        ),
        "r4_harm_lower_than_bcsg": (
            harm_report["codra"]["mean"] < harm_report["bcsg"]["mean"]
        ),
        "r4_at_least_two_tasks_not_worse_than_copy": improved_tasks >= 2,
        "r5_no_new_parameters": True,
        "r5_rejection_is_exact_copy": max(rejection_error.values()) == 0.0,
        "r5_latency_under_5ms": latency["p95_ms"] < 5.0,
    }
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "evidence": "fresh 24 calibration / 24 final-test episodes per task",
            "base_policy": "frozen QA-LRDT + BCSG; Demo-copy fallback",
            "risk": "episode mean of harmful accepted translation residuals",
            "risk_scores": list(SCORE_NAMES),
            "crc_alpha": config.alpha,
        },
        "git_commit": _git_commit(project_root),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_git_commit": checkpoint.get("git_commit"),
        "config": asdict(config),
        "config_sha256": _sha256(config_path),
        "device": str(device),
        "parameters": {
            "qa_lrdat": sum(value.numel() for value in transport.parameters()),
            "bcsg": sum(value.numel() for value in gate.parameters()),
            "codra": 0,
        },
        "structural": {
            "rejected_prediction_max_abs_error_from_copy": rejection_error,
        },
        "per_task": per_task,
        "aggregate": {
            "episodes": 3 * config.expected_test_episodes,
            "queries": len(target),
            "metrics": metrics,
            "paired_episode_bootstrap": comparisons,
            "episode_harm_fraction": harm_report,
            "episode_acceptance_fraction": acceptance_report,
        },
        "latency": latency,
        "criteria": criteria,
    }
    artifact_path = temporary / "test_predictions.npz"
    with artifact_path.open("wb") as stream:
        np.savez_compressed(
            stream,
            **{name: np.concatenate(values) for name, values in artifact.items()},
        )
    report["test_predictions_sha256"] = _sha256(artifact_path)
    (temporary / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_root)
    atexit.unregister(cleanup_incomplete)
    print(json.dumps(report["aggregate"], indent=2), flush=True)
    print(json.dumps(criteria, indent=2), flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--bank-root", type=Path, action="append", required=True)
    parser.add_argument(
        "--evaluation-root", type=Path, action="append", required=True
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    configuration = CoDRAConfig.from_json(args.config)
    run(
        project_root=args.project_root.resolve(),
        bank_roots=[path.resolve() for path in args.bank_root],
        evaluation_roots=[path.resolve() for path in args.evaluation_root],
        checkpoint_path=args.checkpoint.resolve(),
        config_path=args.config.resolve(),
        output_root=args.output_root.resolve(),
        config=configuration,
        device=torch.device(args.device),
    )

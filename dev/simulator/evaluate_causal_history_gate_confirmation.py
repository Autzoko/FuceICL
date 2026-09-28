"""在全新 ManiSkill expert episodes 上确认冻结 CHBG。"""

from __future__ import annotations

import argparse
import atexit
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from dev.pointnet.compare_retrievers import _sha256
from dev.predictor.benefit_calibrated_shrinkage import (
    apply_shrinkage,
    frozen_transport_features,
    optimal_shrinkage_target,
)
from dev.predictor.canonical_geometry import GEOMETRY_DIM
from dev.predictor.causal_history_gate import (
    CausalHistoryBenefitGate,
    CausalHistoryGateConfig,
    causal_history_features,
)
from dev.predictor.train_action_chunks import _physical_metrics
from dev.simulator.evaluate_maniskill_demo_prior import _bootstrap_comparison
from dev.simulator.train_causal_history_gate import (
    EXPECTED_TASKS,
    VARIANTS,
    _checkpoint_bank_indices,
    _load_frozen_models,
    _variant_history,
    _weighted_mse,
)
from dev.simulator.train_maniskill_low_rank_transport import (
    TaskData,
    _load_task,
    _nearest_demo_indices,
    _selection_hash,
)
from dev.simulator.train_maniskill_query_aligned_bcsg import _qa_diagnostics


@dataclass(frozen=True)
class ConfirmationConfig:
    """冻结的 CHBG fresh-episode 确认协议。"""

    schema_version: str
    seed: int
    expected_episodes_per_task: int
    source_start_seeds: dict[str, int]
    source_checkpoint_sha256: str
    history_checkpoint_sha256: str
    bootstrap_resamples: int

    @classmethod
    def from_json(cls, path: Path) -> "ConfirmationConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def __post_init__(self) -> None:
        if self.schema_version != "maniskill-causal-history-gate-confirm-v1":
            raise ValueError("未知 CHBG confirmation schema")
        if min(self.expected_episodes_per_task, self.bootstrap_resamples) <= 0:
            raise ValueError("CHBG confirmation 正数配置非法")
        if set(self.source_start_seeds) != EXPECTED_TASKS:
            raise ValueError("CHBG confirmation 必须冻结三任务 start seeds")
        if any(
            len(value) != 64
            for value in (
                self.source_checkpoint_sha256,
                self.history_checkpoint_sha256,
            )
        ):
            raise ValueError("冻结 checkpoint SHA256 非法")


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _load_history_gates(
    checkpoint_path: Path,
    *,
    expected_sha256: str,
    source_checkpoint_sha256: str,
    device: torch.device,
) -> tuple[dict[str, CausalHistoryBenefitGate], Mapping[str, Any]]:
    if _sha256(checkpoint_path) != expected_sha256:
        raise ValueError("history checkpoint SHA256 与冻结配置不匹配")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint["source_checkpoint_sha256"] != source_checkpoint_sha256:
        raise ValueError("history gate 与 source QA checkpoint 不匹配")
    gates = {}
    for name in VARIANTS:
        config = CausalHistoryGateConfig(
            **checkpoint["model_configs"][name]
        )
        gate = CausalHistoryBenefitGate(
            config,
            feature_mean=torch.zeros(config.input_dim),
            feature_std=torch.ones(config.input_dim),
        )
        gate.load_state_dict(checkpoint["models"][name])
        gates[name] = gate.to(device).eval()
    return gates, checkpoint


def _combine_splits(
    task: TaskData,
) -> tuple[list[dict[str, Any]], torch.Tensor, torch.Tensor]:
    records = [*task.train_records, *task.val_records]
    geometry = torch.cat((task.train_geometry, task.val_geometry))
    actions = torch.cat((task.train_actions, task.val_actions))
    if len({str(record["chunk_id"]) for record in records}) != len(records):
        raise ValueError(f"{task.task_id} combined query chunk IDs 不唯一")
    return records, geometry, actions


def _source_seed_audit(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_episodes: int,
    minimum_seed: int,
) -> dict[str, Any]:
    by_episode: dict[int, set[int]] = {}
    for record in records:
        if "source_episode_seed" not in record:
            raise ValueError("confirmation manifest 缺少 source_episode_seed")
        by_episode.setdefault(int(record["episode"]), set()).add(
            int(record["source_episode_seed"])
        )
    if len(by_episode) != expected_episodes:
        raise ValueError("confirmation episode 数与冻结协议不一致")
    if any(len(values) != 1 for values in by_episode.values()):
        raise ValueError("confirmation episode 映射到多个 source seeds")
    seeds = [next(iter(values)) for values in by_episode.values()]
    if len(set(seeds)) != len(seeds) or min(seeds) < minimum_seed:
        raise ValueError("confirmation source seeds 重复或低于冻结起点")
    return {
        "episodes": len(by_episode),
        "source_seed_min": min(seeds),
        "source_seed_max": max(seeds),
        "source_seed_sha256": hashlib.sha256(
            json.dumps(sorted(seeds)).encode()
        ).hexdigest(),
    }


def run(
    *,
    project_root: Path,
    bank_roots: Sequence[Path],
    query_roots: Sequence[Path],
    source_checkpoint_path: Path,
    history_checkpoint_path: Path,
    config_path: Path,
    output_root: Path,
    config: ConfirmationConfig,
    device: torch.device,
) -> None:
    if output_root.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output_root}")
    temporary = output_root.with_name(f".{output_root.name}.incomplete-{os.getpid()}")
    temporary.mkdir(parents=True)

    def cleanup() -> None:
        shutil.rmtree(temporary, ignore_errors=True)

    atexit.register(cleanup)
    source_sha256 = _sha256(source_checkpoint_path)
    if source_sha256 != config.source_checkpoint_sha256:
        raise ValueError("source checkpoint SHA256 与冻结配置不匹配")
    source_checkpoint = torch.load(source_checkpoint_path, map_location="cpu")
    transport, baseline_gate = _load_frozen_models(source_checkpoint, device)
    gates, history_checkpoint = _load_history_gates(
        history_checkpoint_path,
        expected_sha256=config.history_checkpoint_sha256,
        source_checkpoint_sha256=source_sha256,
        device=device,
    )
    bank_values = [_load_task(root) for root in bank_roots]
    query_values = [_load_task(root) for root in query_roots]
    banks = {task.task_id: task for task in bank_values}
    queries = {task.task_id: task for task in query_values}
    if len(banks) != 3 or len(queries) != 3:
        raise ValueError("bank/query task IDs 重复或缺失")
    if set(banks) != EXPECTED_TASKS or set(queries) != EXPECTED_TASKS:
        raise ValueError("bank/query roots 未覆盖冻结三任务")

    per_task: dict[str, Any] = {}
    aggregate_target = []
    aggregate_predictions: dict[str, list[torch.Tensor]] = {
        name: [] for name in ("copy", "bcsg", *VARIANTS)
    }
    aggregate_groups: list[str] = []
    aggregate_gate_values: dict[str, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {
        name: [] for name in ("bcsg", *VARIANTS)
    }
    artifact: dict[str, list[np.ndarray]] = {}
    source_seed_sets: dict[str, set[int]] = {}
    for offset, task_id in enumerate(sorted(EXPECTED_TASKS)):
        bank = banks[task_id]
        query = queries[task_id]
        if source_checkpoint["data_summary_sha256"][task_id] != _sha256(
            bank.root / "summary.json"
        ):
            raise ValueError(f"{task_id} bank 与 source checkpoint 不匹配")
        if query.action_representation != source_checkpoint["action_representation"]:
            raise ValueError(f"{task_id} query action representation 不匹配")
        if not torch.equal(query.pose_scales.float(), bank.pose_scales.float()):
            raise ValueError(f"{task_id} bank/query action scales 不匹配")
        records, geometry, target = _combine_splits(query)
        seed_audit = _source_seed_audit(
            records,
            expected_episodes=config.expected_episodes_per_task,
            minimum_seed=config.source_start_seeds[task_id],
        )
        source_seed_sets[task_id] = {
            int(record["source_episode_seed"]) for record in records
        }
        bank_indices = _checkpoint_bank_indices(
            bank.train_records,
            source_checkpoint["bank_chunk_ids"][task_id],
        )
        bank_records = [bank.train_records[index] for index in bank_indices.tolist()]
        bank_geometry = bank.train_geometry[bank_indices]
        bank_actions = bank.train_actions[bank_indices]
        demo_indices, distances = _nearest_demo_indices(
            candidate_records=bank_records,
            candidate_geometry=bank_geometry,
            query_records=records,
            query_geometry=geometry,
            exclude_same_episode=False,
        )
        demo_geometry = bank_geometry[demo_indices]
        demo_actions = bank_actions[demo_indices]
        transported, base = _qa_diagnostics(
            transport,
            geometry,
            demo_geometry,
            demo_actions,
            device,
        )
        history = causal_history_features(records, geometry)
        mask = torch.ones(target.shape[:2], dtype=torch.bool)
        optimal, energy = optimal_shrinkage_target(
            demo_actions,
            transported,
            target,
            mask,
        )
        with torch.inference_mode():
            baseline_values = baseline_gate(base.to(device)).cpu()
        values = {}
        predictions = {
            "copy": demo_actions,
            "bcsg": apply_shrinkage(
                demo_actions,
                transported,
                baseline_values,
                mask,
            ),
        }
        for name, gate in gates.items():
            current_history = _variant_history(
                name,
                records,
                history,
                seed=config.seed + offset,
            )
            features = torch.cat((base, current_history), dim=1)
            with torch.inference_mode():
                gate_values = gate(features.to(device)).cpu()
            values[name] = gate_values
            predictions[name] = apply_shrinkage(
                demo_actions,
                transported,
                gate_values,
                mask,
            )
        options = {
            "pose_scales": query.pose_scales,
            "translation_threshold_m": query.translation_threshold_m,
            "rotation_threshold_rad": query.rotation_threshold_rad,
        }
        metrics = {
            name: _physical_metrics(value, target, mask, **options)
            for name, value in predictions.items()
        }
        groups = [f"{task_id}:{row['source_episode_seed']}" for row in records]
        comparisons = {
            f"{name}_minus_bcsg": _bootstrap_comparison(
                reference=predictions["bcsg"],
                candidate=predictions[name],
                target=target,
                mask=mask,
                group_ids=groups,
                seed=config.seed + offset * 100 + index,
                resamples=config.bootstrap_resamples,
                **options,
            )
            for index, name in enumerate(VARIANTS)
        }
        comparisons["two_lag_minus_shuffled"] = _bootstrap_comparison(
            reference=predictions["shuffled_two_lag"],
            candidate=predictions["two_lag"],
            target=target,
            mask=mask,
            group_ids=groups,
            seed=config.seed + offset * 100 + 10,
            resamples=config.bootstrap_resamples,
            **options,
        )
        gate_mse = {
            "bcsg": _weighted_mse(baseline_values, optimal, energy),
            **{
                name: _weighted_mse(value, optimal, energy)
                for name, value in values.items()
            },
        }
        per_task[task_id] = {
            "source_seed_audit": seed_audit,
            "queries": len(records),
            "bank_chunks": len(bank_records),
            "bank_selection_sha256": _selection_hash(bank_indices),
            "retrieval_distance_mean": float(distances.mean()),
            "weighted_gate_mse": gate_mse,
            "metrics": metrics,
            "paired_episode_bootstrap": comparisons,
        }
        aggregate_target.append(target)
        aggregate_groups.extend(groups)
        for name, prediction in predictions.items():
            aggregate_predictions[name].append(prediction)
        aggregate_gate_values["bcsg"].append((baseline_values, optimal, energy))
        for name, value in values.items():
            aggregate_gate_values[name].append((value, optimal, energy))
        for name, value in {
            "task": np.asarray([task_id] * len(records)),
            "group_id": np.asarray(groups),
            "target": target.numpy(),
            **{key: value.numpy() for key, value in predictions.items()},
            "optimal_gate": optimal.numpy(),
            "bcsg_gate": baseline_values.numpy(),
            **{f"{key}_gate": value.numpy() for key, value in values.items()},
        }.items():
            artifact.setdefault(name, []).append(value)

    all_source_seeds = [value for values in source_seed_sets.values() for value in values]
    if len(all_source_seeds) != len(set(all_source_seeds)):
        raise ValueError("三任务 confirmation source seeds 重叠")
    target = torch.cat(aggregate_target)
    predictions = {
        name: torch.cat(values) for name, values in aggregate_predictions.items()
    }
    mask = torch.ones(target.shape[:2], dtype=torch.bool)
    first = next(iter(queries.values()))
    options = {
        "pose_scales": first.pose_scales,
        "translation_threshold_m": first.translation_threshold_m,
        "rotation_threshold_rad": first.rotation_threshold_rad,
    }
    metrics = {
        name: _physical_metrics(value, target, mask, **options)
        for name, value in predictions.items()
    }
    comparisons = {
        f"{name}_minus_bcsg": _bootstrap_comparison(
            reference=predictions["bcsg"],
            candidate=predictions[name],
            target=target,
            mask=mask,
            group_ids=aggregate_groups,
            seed=config.seed + 1000 + index,
            resamples=config.bootstrap_resamples,
            **options,
        )
        for index, name in enumerate(VARIANTS)
    }
    comparisons["two_lag_minus_shuffled"] = _bootstrap_comparison(
        reference=predictions["shuffled_two_lag"],
        candidate=predictions["two_lag"],
        target=target,
        mask=mask,
        group_ids=aggregate_groups,
        seed=config.seed + 2000,
        resamples=config.bootstrap_resamples,
        **options,
    )
    gate_mse = {}
    for name, rows in aggregate_gate_values.items():
        gate_mse[name] = _weighted_mse(
            torch.cat([row[0] for row in rows]),
            torch.cat([row[1] for row in rows]),
            torch.cat([row[2] for row in rows]),
        )
    two_lag_bcsg = comparisons["two_lag_minus_bcsg"]["translation_l2_m"]
    two_lag_shuffled = comparisons["two_lag_minus_shuffled"]["translation_l2_m"]
    improved_tasks = sum(
        values["metrics"]["two_lag"]["translation_l2_m"]
        <= values["metrics"]["bcsg"]["translation_l2_m"]
        for values in per_task.values()
    )
    no_demo_actions = torch.zeros((1, target.shape[1], target.shape[2]), device=device)
    no_demo_geometry = torch.zeros((1, GEOMETRY_DIM), device=device)
    no_demo_mask = torch.zeros((1, target.shape[1]), dtype=torch.bool, device=device)
    with torch.inference_mode():
        no_demo, no_demo_base = frozen_transport_features(
            transport,
            torch.zeros((1, GEOMETRY_DIM), device=device),
            no_demo_geometry,
            no_demo_actions,
            no_demo_mask,
        )
        history_dim = gates["two_lag"].config.input_dim - no_demo_base.shape[1]
        no_demo_gate = gates["two_lag"](
            torch.cat(
                (no_demo_base, torch.zeros((1, history_dim), device=device)),
                dim=1,
            )
        )
        no_demo_output = apply_shrinkage(
            no_demo_actions,
            no_demo,
            no_demo_gate,
            no_demo_mask,
        )
        gate_zero_output = apply_shrinkage(
            predictions["copy"][:1].to(device),
            predictions["two_lag"][:1].to(device),
            torch.zeros(1, device=device),
            torch.ones_like(no_demo_mask),
        )
        gate_zero_error = float(
            (
                gate_zero_output - predictions["copy"][:1].to(device)
            ).abs().max()
        )
    parameter_counts = {
        "qa_lrdat": sum(value.numel() for value in transport.parameters()),
        "frozen_bcsg": sum(value.numel() for value in baseline_gate.parameters()),
        **{
            name: sum(value.numel() for value in gate.parameters())
            for name, gate in gates.items()
        },
    }
    criteria = {
        "f1_gate_mse_improves_at_least_10_percent": (
            gate_mse["two_lag"] <= 0.9 * gate_mse["bcsg"]
        ),
        "f2_translation_ci_better_than_bcsg": two_lag_bcsg["ci95_high"] < 0.0,
        "f3_translation_ci_better_than_shuffled": (
            two_lag_shuffled["ci95_high"] < 0.0
        ),
        "f3_gate_mse_better_than_shuffled": (
            gate_mse["two_lag"] < gate_mse["shuffled_two_lag"]
        ),
        "f4_at_least_two_tasks_not_worse_than_bcsg": improved_tasks >= 2,
        "f5_source_seeds_unique": len(all_source_seeds) == len(set(all_source_seeds)),
        "f5_history_checkpoint_frozen": (
            _sha256(history_checkpoint_path) == config.history_checkpoint_sha256
        ),
        "f5_source_checkpoint_frozen": (
            source_sha256 == config.source_checkpoint_sha256
        ),
        "f5_parameter_budget": (
            parameter_counts["qa_lrdat"] + parameter_counts["two_lag"] < 250_000
        ),
        "f5_no_demo_exact": float(no_demo_output.abs().max()) == 0.0,
        "f5_gate_zero_exact_copy": gate_zero_error == 0.0,
    }
    artifact_path = temporary / "predictions.npz"
    with artifact_path.open("wb") as stream:
        np.savez_compressed(
            stream,
            **{name: np.concatenate(values) for name, values in artifact.items()},
        )
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(project_root),
        "protocol": "frozen CHBG on 24 fresh expert episodes per task",
        "config": asdict(config),
        "config_sha256": _sha256(config_path),
        "source_checkpoint_sha256": source_sha256,
        "history_checkpoint_sha256": _sha256(history_checkpoint_path),
        "history_checkpoint_git_commit": history_checkpoint.get("git_commit"),
        "bank_summary_sha256": {
            task_id: _sha256(task.root / "summary.json")
            for task_id, task in banks.items()
        },
        "query_summary_sha256": {
            task_id: _sha256(task.root / "summary.json")
            for task_id, task in queries.items()
        },
        "device": str(device),
        "parameters": parameter_counts,
        "per_task": per_task,
        "aggregate": {
            "episodes": len(set(aggregate_groups)),
            "queries": len(target),
            "weighted_gate_mse": gate_mse,
            "metrics": metrics,
            "paired_episode_bootstrap": comparisons,
        },
        "structural": {
            "no_demo_max_abs_output": float(no_demo_output.abs().max()),
            "gate_zero_max_abs_error_from_copy": gate_zero_error,
        },
        "criteria": criteria,
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
    parser.add_argument("--history-checkpoint", type=Path, required=True)
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
        history_checkpoint_path=arguments.history_checkpoint.resolve(),
        config_path=arguments.config.resolve(),
        output_root=arguments.output_root.resolve(),
        config=ConfirmationConfig.from_json(arguments.config.resolve()),
        device=torch.device(arguments.device),
    )

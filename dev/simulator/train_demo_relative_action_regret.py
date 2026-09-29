"""训练 Demo-relative Predictor，并以 counterfactual action regret 门控。"""

from __future__ import annotations

import argparse
import atexit
import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from dev.predictor.demo_relative_policy import (
    DemoRelativePolicy,
    DemoRelativePolicyConfig,
)
from dev.simulator.evaluate_rotated_layout_slide_transport import (
    SPLITS,
    _bootstrap_difference,
    _metrics,
    _scene_means,
    _sha256,
    _strip_rows,
)
from dev.simulator.train_belief_local_demo_residual import (
    _contexts,
    _git_commit,
    _load_data,
    _rank_demos,
    _relative_improvement,
    _seed_everything,
    _shuffled_demo_rows,
)
from dev.simulator.train_rotated_layout_slide_demo_policy import (
    _latency,
    _predict,
    _structural_audit,
)


@dataclass(frozen=True)
class DemoRelativeTrainConfig:
    """冻结的 relative-only 训练与 action-regret 判据。"""

    schema_version: str
    expected_data_sha256: str
    expected_preprocess_report_sha256: str
    seed: int
    train_neighbors: int
    ambiguous_progress_maximum: int
    minimum_ambiguous_queries: int
    hidden_dim: int
    maximum_translation_residual_m: float
    maximum_rotation_residual_rad: float
    translation_action_scale_m: float
    rotation_action_scale_rad: float
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    state_std_floor: float
    early_stopping_patience: int
    bootstrap_resamples: int
    minimum_raw_relative_improvement: float
    minimum_direction_accuracy: float
    minimum_wrong_relative_improvement: float
    minimum_shuffled_relative_improvement: float
    maximum_ambiguous_wrong_direction_accuracy: float
    maximum_parameters: int
    latency_warmup: int
    latency_iterations: int
    maximum_latency_p95_ms: float

    @classmethod
    def from_json(cls, path: Path) -> "DemoRelativeTrainConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def __post_init__(self) -> None:
        if self.schema_version != "demo-relative-action-regret-v1":
            raise ValueError("未知 Demo-relative schema")
        if not self.expected_data_sha256 or not self.expected_preprocess_report_sha256:
            raise ValueError("必须冻结输入数据与预处理报告 hash")
        positive = (
            self.train_neighbors,
            self.minimum_ambiguous_queries,
            self.hidden_dim,
            self.maximum_translation_residual_m,
            self.maximum_rotation_residual_rad,
            self.translation_action_scale_m,
            self.rotation_action_scale_rad,
            self.epochs,
            self.batch_size,
            self.learning_rate,
            self.gradient_clip_norm,
            self.state_std_floor,
            self.early_stopping_patience,
            self.bootstrap_resamples,
            self.minimum_raw_relative_improvement,
            self.minimum_direction_accuracy,
            self.minimum_wrong_relative_improvement,
            self.minimum_shuffled_relative_improvement,
            self.maximum_parameters,
            self.latency_warmup,
            self.latency_iterations,
            self.maximum_latency_p95_ms,
        )
        if (
            min(positive) <= 0
            or self.seed < 0
            or self.ambiguous_progress_maximum < 0
            or self.weight_decay < 0
        ):
            raise ValueError("Demo-relative 配置非法")
        probabilities = (
            self.minimum_raw_relative_improvement,
            self.minimum_direction_accuracy,
            self.minimum_wrong_relative_improvement,
            self.minimum_shuffled_relative_improvement,
            self.maximum_ambiguous_wrong_direction_accuracy,
        )
        if any(not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("概率或相对改善门槛必须位于 [0,1]")


def _fit(
    *,
    model: DemoRelativePolicy,
    train_contexts: dict[str, np.ndarray],
    validation_contexts: dict[str, np.ndarray],
    config: DemoRelativeTrainConfig,
) -> tuple[list[dict[str, float | int]], int, float]:
    dataset = TensorDataset(
        torch.from_numpy(train_contexts["query_state"]).float(),
        torch.from_numpy(train_contexts["demo_state"]).float(),
        torch.from_numpy(train_contexts["anchor"]).float(),
        torch.from_numpy(train_contexts["target"]).float(),
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    log: list[dict[str, float | int]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_validation_mse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    for epoch in range(config.epochs):
        model.train()
        losses = []
        for query, demo, anchor, target in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(query, demo, anchor, torch.ones(len(query)))
            translation_loss = torch.mean(
                ((prediction[..., :3] - target[..., :3])
                 / config.maximum_translation_residual_m) ** 2
            )
            rotation_loss = torch.mean(
                ((prediction[..., 3:6] - target[..., 3:6])
                 / config.maximum_rotation_residual_rad) ** 2
            )
            loss = translation_loss + rotation_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip_norm
            )
            optimizer.step()
            losses.append(float(loss.detach()))
        validation_prediction = _predict(
            model, validation_contexts, batch_size=config.batch_size
        )
        validation_mse = float(
            np.mean(
                (validation_prediction - validation_contexts["target"]) ** 2
            )
        )
        log.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
                "validation_action_mse": validation_mse,
            }
        )
        if validation_mse < best_validation_mse - 1e-12:
            best_state = copy.deepcopy(model.state_dict())
            best_validation_mse = validation_mse
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= config.early_stopping_patience:
            break
    model.load_state_dict(best_state)
    return log, best_epoch, best_validation_mse


def _evaluate(
    *,
    model: DemoRelativePolicy,
    correct: dict[str, np.ndarray],
    shuffled: dict[str, np.ndarray],
    wrong: dict[str, np.ndarray],
    data: dict[str, np.ndarray],
    config: DemoRelativeTrainConfig,
) -> dict[str, Any]:
    predictions = {
        "raw_nearest_copy": correct["anchor"],
        "demo_relative": _predict(
            model, correct, batch_size=config.batch_size
        ),
        "same_operation_shuffled": _predict(
            model, shuffled, batch_size=config.batch_size
        ),
        "wrong_operation": _predict(
            model, wrong, batch_size=config.batch_size
        ),
        "no_demo": _predict(
            model,
            correct,
            batch_size=config.batch_size,
            demo_mask=0.0,
        ),
    }
    raw_metrics = {
        name: _metrics(prediction, correct["target"])
        for name, prediction in predictions.items()
    }
    pair_ids = data["pair_id"][correct["query_row"]]
    scene_mse = {
        name: _scene_means(metrics["per_row_action_mse"], pair_ids)[1]
        for name, metrics in raw_metrics.items()
    }

    def comparison(other: str, offset: int) -> dict[str, float | int]:
        return _bootstrap_difference(
            scene_mse["demo_relative"],
            scene_mse[other],
            seed=config.seed + offset,
            resamples=config.bootstrap_resamples,
        )

    progress = data["progress_index"][correct["query_row"]]
    ambiguous = progress <= config.ambiguous_progress_maximum
    ambiguous_metrics = {
        name: _strip_rows(_metrics(value[ambiguous], correct["target"][ambiguous]))
        for name, value in predictions.items()
        if name in {"demo_relative", "wrong_operation"}
    }
    stripped = {name: _strip_rows(value) for name, value in raw_metrics.items()}
    return {
        "queries": len(correct["query_row"]),
        "query_scene_pairs": len(np.unique(pair_ids)),
        "metrics": stripped,
        "relative_mse_improvement": {
            "over_raw": _relative_improvement(
                stripped["demo_relative"]["action_mse"],
                stripped["raw_nearest_copy"]["action_mse"],
            ),
            "over_wrong": _relative_improvement(
                stripped["demo_relative"]["action_mse"],
                stripped["wrong_operation"]["action_mse"],
            ),
            "over_shuffled": _relative_improvement(
                stripped["demo_relative"]["action_mse"],
                stripped["same_operation_shuffled"]["action_mse"],
            ),
        },
        "scene_pair_bootstrap": {
            "correct_minus_raw": comparison("raw_nearest_copy", 101),
            "correct_minus_wrong": comparison("wrong_operation", 102),
            "correct_minus_shuffled": comparison(
                "same_operation_shuffled", 103
            ),
        },
        "ambiguous_subset": {
            "maximum_progress_index": config.ambiguous_progress_maximum,
            "queries": int(ambiguous.sum()),
            "metrics": ambiguous_metrics,
        },
    }


def run(
    *,
    project_root: Path,
    config_path: Path,
    data_root: Path,
    output_root: Path,
    config: DemoRelativeTrainConfig,
) -> None:
    if output_root.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output_root}")
    _seed_everything(config.seed)
    data, preprocess, data_path, preprocess_path = _load_data(data_root, config)
    train_indices = np.where(data["split_id"] == SPLITS["train"])[0]
    validation_indices = np.where(data["split_id"] == SPLITS["val"])[0]
    state_mean = data["state"][train_indices].mean(axis=0).astype(np.float32)
    state_std = data["state"][train_indices].std(axis=0)
    state_std = np.maximum(state_std, config.state_std_floor).astype(np.float32)

    train_query, train_demo, train_distance = _rank_demos(
        data,
        query_indices=train_indices,
        bank_indices=train_indices,
        state_std=state_std,
        same_operation=True,
        neighbors=config.train_neighbors,
        exclude_same_pair=True,
    )
    val_query, val_demo, val_distance = _rank_demos(
        data,
        query_indices=validation_indices,
        bank_indices=train_indices,
        state_std=state_std,
        same_operation=True,
        neighbors=1,
        exclude_same_pair=True,
    )
    wrong_query, wrong_demo, wrong_distance = _rank_demos(
        data,
        query_indices=validation_indices,
        bank_indices=train_indices,
        state_std=state_std,
        same_operation=False,
        neighbors=1,
        exclude_same_pair=True,
    )
    if not np.array_equal(val_query, wrong_query):
        raise ValueError("correct/wrong validation query 顺序不一致")
    shuffled_demo = _shuffled_demo_rows(
        data,
        val_query,
        val_demo,
        train_indices,
        seed=config.seed + 17,
    )
    train_contexts = _contexts(data, train_query, train_demo)
    validation_contexts = _contexts(data, val_query, val_demo)
    shuffled_contexts = _contexts(data, val_query, shuffled_demo)
    wrong_contexts = _contexts(data, wrong_query, wrong_demo)

    policy_config = DemoRelativePolicyConfig(
        state_dim=data["state"].shape[1],
        action_horizon=data["action"].shape[1],
        hidden_dim=config.hidden_dim,
        maximum_translation_residual_m=config.maximum_translation_residual_m,
        maximum_rotation_residual_rad=config.maximum_rotation_residual_rad,
        translation_action_scale_m=config.translation_action_scale_m,
        rotation_action_scale_rad=config.rotation_action_scale_rad,
    )
    model = DemoRelativePolicy(
        policy_config,
        state_mean=torch.from_numpy(state_mean),
        state_std=torch.from_numpy(state_std),
    )
    training_log, best_epoch, best_validation_mse = _fit(
        model=model,
        train_contexts=train_contexts,
        validation_contexts=validation_contexts,
        config=config,
    )
    validation = _evaluate(
        model=model,
        correct=validation_contexts,
        shuffled=shuffled_contexts,
        wrong=wrong_contexts,
        data=data,
        config=config,
    )
    structural = _structural_audit(model, validation_contexts)
    latency = _latency(
        model,
        validation_contexts,
        warmup=config.latency_warmup,
        iterations=config.latency_iterations,
    )
    ambiguous = validation["ambiguous_subset"]
    improvements = validation["relative_mse_improvement"]
    bootstrap = validation["scene_pair_bootstrap"]
    excluded_same_scene = bool(
        np.all(data["pair_id"][train_query] != data["pair_id"][train_demo])
    )
    criteria = {
        "D1_improvement_over_raw": (
            improvements["over_raw"] >= config.minimum_raw_relative_improvement
            and bootstrap["correct_minus_raw"]["ci95_high"] < 0.0
        ),
        "D2_correct_direction": (
            validation["metrics"]["demo_relative"]["direction_accuracy"]
            >= config.minimum_direction_accuracy
        ),
        "D3_counterfactual_wrong_regret": (
            improvements["over_wrong"]
            >= config.minimum_wrong_relative_improvement
            and bootstrap["correct_minus_wrong"]["ci95_high"] < 0.0
        ),
        "D4_shuffled_demo_regret": (
            improvements["over_shuffled"]
            >= config.minimum_shuffled_relative_improvement
            and bootstrap["correct_minus_shuffled"]["ci95_high"] < 0.0
        ),
        "D5_ambiguous_demo_semantics": (
            ambiguous["queries"] >= config.minimum_ambiguous_queries
            and ambiguous["metrics"]["demo_relative"]["direction_accuracy"]
            >= config.minimum_direction_accuracy
            and ambiguous["metrics"]["wrong_operation"][
                "direction_accuracy"
            ]
            <= config.maximum_ambiguous_wrong_direction_accuracy
        ),
        "D6_structural_constraints": (
            structural["identity_anchor_max_abs_error"] == 0.0
            and structural["no_demo_max_abs_value"] == 0.0
            and structural["translation_residual_max_abs_m"]
            <= config.maximum_translation_residual_m + 1e-8
            and structural["rotation_residual_max_abs_rad"]
            <= config.maximum_rotation_residual_rad + 1e-8
            and structural["gripper_residual_max_abs"] == 0.0
        ),
        "D7_deployment_budget": (
            structural["parameters"] <= config.maximum_parameters
            and latency["p95_ms"] <= config.maximum_latency_p95_ms
        ),
        "D8_provenance_and_split": excluded_same_scene,
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(project_root),
        "config": asdict(config),
        "config_sha256": _sha256(config_path),
        "data_sha256": _sha256(data_path),
        "preprocess_report_sha256": _sha256(preprocess_path),
        "protocol": {
            "model_input": "normalized query-minus-Demo state and raw Demo action",
            "absolute_query_feature_used": False,
            "train_neighbors_distinct_scene_pairs": config.train_neighbors,
            "validation_neighbors": 1,
            "training_contexts": len(train_query),
            "validation_queries": len(val_query),
            "same_scene_training_demo_excluded": excluded_same_scene,
            "progress_feature_used_for_retrieval": False,
            "operation_feature_used_by_model": False,
            "future_query_fields_used": False,
            "test_split_used": False,
            "best_epoch_selected_only_by_validation": best_epoch,
        },
        "retrieval_distance": {
            "train_median": float(np.median(train_distance)),
            "validation_median": float(np.median(val_distance)),
            "wrong_operation_median": float(np.median(wrong_distance)),
        },
        "training": {
            "epochs_run": len(training_log),
            "best_validation_action_mse": best_validation_mse,
            "final_train_loss": training_log[-1]["train_loss"],
        },
        "validation": validation,
        "structural_audit": structural,
        "latency": latency,
        "criteria": criteria,
        "development_gate_passed": all(criteria.values()),
        "upstream_preprocess_schema_version": preprocess["schema_version"],
    }

    temporary = output_root.with_name(
        f".{output_root.name}.incomplete-{os.getpid()}"
    )
    temporary.mkdir(parents=True)

    def cleanup() -> None:
        shutil.rmtree(temporary, ignore_errors=True)

    atexit.register(cleanup)
    checkpoint_path = temporary / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(policy_config),
            "train_config": asdict(config),
            "data_sha256": _sha256(data_path),
            "development_gate_passed": report["development_gate_passed"],
            "model_variant": "demo-relative-action-regret-v1",
        },
        checkpoint_path,
    )
    log_path = temporary / "training_log.jsonl"
    with log_path.open("w", encoding="utf-8") as stream:
        for row in training_log:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    retrieval_path = temporary / "retrieval_contexts.npz"
    with retrieval_path.open("wb") as stream:
        np.savez_compressed(
            stream,
            train_query_row=train_query,
            train_demo_row=train_demo,
            train_distance=train_distance,
            validation_query_row=val_query,
            validation_demo_row=val_demo,
            validation_distance=val_distance,
            shuffled_demo_row=shuffled_demo,
            wrong_demo_row=wrong_demo,
            wrong_distance=wrong_distance,
        )
    report["files"] = {
        "checkpoint.pt": _sha256(checkpoint_path),
        "training_log.jsonl": _sha256(log_path),
        "retrieval_contexts.npz": _sha256(retrieval_path),
    }
    report_path = temporary / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_root)
    atexit.unregister(cleanup)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not report["development_gate_passed"]:
        raise RuntimeError("Demo-relative action-regret 未通过开发门控")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    resolved_config = arguments.config.resolve()
    run(
        project_root=arguments.project_root.resolve(),
        config_path=resolved_config,
        data_root=arguments.data_root.resolve(),
        output_root=arguments.output_root.resolve(),
        config=DemoRelativeTrainConfig.from_json(resolved_config),
    )

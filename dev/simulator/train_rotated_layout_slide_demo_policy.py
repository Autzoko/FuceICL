"""仅用 train/val 训练并门控 Demo-anchored layout-equivariant predictor。"""

from __future__ import annotations

import argparse
import atexit
import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from dev.predictor.layout_equivariant_demo_policy import (
    LayoutEquivariantDemoPolicy,
    LayoutEquivariantDemoPolicyConfig,
)
from dev.simulator.evaluate_rotated_layout_slide_transport import (
    OPERATIONS,
    SPLITS,
    _bootstrap_difference,
    _metrics,
    _point_yaw,
    _scene_means,
    _strip_rows,
    _transport,
    _wrap_angle,
)


@dataclass(frozen=True)
class DemoPolicyTrainConfig:
    """冻结的 predictor train/validation 协议。"""

    schema_version: str
    seed: int
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
    minimum_validation_relative_improvement: float
    minimum_validation_direction_accuracy: float
    maximum_wrong_operation_direction_accuracy: float
    maximum_parameters: int
    latency_warmup: int
    latency_iterations: int
    maximum_latency_p95_ms: float

    @classmethod
    def from_json(cls, path: Path) -> "DemoPolicyTrainConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def __post_init__(self) -> None:
        if self.schema_version != "rotated-layout-slide-demo-policy-v1":
            raise ValueError("未知 rotated-layout Demo policy schema")
        positive = (
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
            self.minimum_validation_relative_improvement,
            self.minimum_validation_direction_accuracy,
            self.maximum_parameters,
            self.latency_warmup,
            self.latency_iterations,
            self.maximum_latency_p95_ms,
        )
        if min(positive) <= 0 or self.seed < 0 or self.weight_decay < 0:
            raise ValueError("Demo policy train 配置非法")
        probabilities = (
            self.minimum_validation_direction_accuracy,
            self.maximum_wrong_operation_direction_accuracy,
        )
        if any(not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("direction accuracy threshold 非法")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def _transported_action(
    data: dict[str, np.ndarray],
    query_index: int,
    demo_index: int,
) -> np.ndarray:
    query_yaw = _point_yaw(
        data["state"][query_index], data["tcp_pose"][query_index]
    )
    demo_yaw = _point_yaw(
        data["state"][demo_index], data["tcp_pose"][demo_index]
    )
    return _transport(
        data["action"][demo_index],
        data["tcp_pose"][demo_index],
        data["tcp_pose"][query_index],
        _wrap_angle(query_yaw - demo_yaw),
    )


def _contexts(
    data: dict[str, np.ndarray],
    *,
    query_indices: np.ndarray,
    demo_indices: np.ndarray,
    same_operation: bool,
    exclude_same_pair: bool,
) -> dict[str, np.ndarray]:
    query_state = []
    demo_state = []
    anchors = []
    targets = []
    query_rows = []
    demo_rows = []
    for query_index in query_indices:
        query_operation = int(data["operation_id"][query_index])
        query_pair = int(data["pair_id"][query_index])
        for demo_index in demo_indices:
            operation_matches = (
                int(data["operation_id"][demo_index]) == query_operation
            )
            if operation_matches != same_operation:
                continue
            if (
                exclude_same_pair
                and int(data["pair_id"][demo_index]) == query_pair
            ):
                continue
            query_state.append(data["state"][query_index])
            demo_state.append(data["state"][demo_index])
            anchors.append(_transported_action(data, query_index, demo_index))
            targets.append(data["action"][query_index])
            query_rows.append(query_index)
            demo_rows.append(demo_index)
    if not query_rows:
        raise ValueError("没有构造出 Demo contexts")
    return {
        "query_state": np.stack(query_state),
        "demo_state": np.stack(demo_state),
        "anchor": np.stack(anchors),
        "target": np.stack(targets),
        "query_row": np.asarray(query_rows, dtype=np.int64),
        "demo_row": np.asarray(demo_rows, dtype=np.int64),
    }


@torch.no_grad()
def _predict(
    model: LayoutEquivariantDemoPolicy,
    contexts: dict[str, np.ndarray],
    *,
    batch_size: int,
    demo_mask: float = 1.0,
) -> np.ndarray:
    model.eval()
    outputs = []
    count = len(contexts["query_state"])
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        outputs.append(
            model(
                torch.from_numpy(contexts["query_state"][start:stop]).float(),
                torch.from_numpy(contexts["demo_state"][start:stop]).float(),
                torch.from_numpy(contexts["anchor"][start:stop]).float(),
                torch.full((stop - start,), demo_mask),
            ).cpu().numpy()
        )
    return np.concatenate(outputs)


def _evaluate(
    *,
    model: LayoutEquivariantDemoPolicy,
    correct: dict[str, np.ndarray],
    wrong: dict[str, np.ndarray],
    data: dict[str, np.ndarray],
    config: DemoPolicyTrainConfig,
) -> dict[str, Any]:
    prediction = _predict(model, correct, batch_size=config.batch_size)
    wrong_prediction = _predict(model, wrong, batch_size=config.batch_size)
    raw = {
        "point_transport": _metrics(correct["anchor"], correct["target"]),
        "demo_anchored_policy": _metrics(prediction, correct["target"]),
        "wrong_operation_policy": _metrics(
            wrong_prediction, wrong["target"]
        ),
    }
    pair_ids = data["pair_id"][correct["query_row"]]
    scene_mse = {
        name: _scene_means(metrics["per_row_action_mse"], pair_ids)[1]
        for name, metrics in raw.items()
        if name != "wrong_operation_policy"
    }
    comparison = _bootstrap_difference(
        scene_mse["demo_anchored_policy"],
        scene_mse["point_transport"],
        seed=config.seed + 101,
        resamples=config.bootstrap_resamples,
    )
    return {
        "contexts": len(correct["query_row"]),
        "query_scene_pairs": len(np.unique(pair_ids)),
        "metrics": {name: _strip_rows(value) for name, value in raw.items()},
        "policy_minus_transport": comparison,
        "relative_mse_improvement": float(
            (
                raw["point_transport"]["action_mse"]
                - raw["demo_anchored_policy"]["action_mse"]
            )
            / raw["point_transport"]["action_mse"]
        ),
    }


@torch.no_grad()
def _structural_audit(
    model: LayoutEquivariantDemoPolicy,
    contexts: dict[str, np.ndarray],
) -> dict[str, float | int]:
    count = min(16, len(contexts["query_state"]))
    query = torch.from_numpy(contexts["query_state"][:count]).float()
    anchor = torch.from_numpy(contexts["anchor"][:count]).float()
    ones = torch.ones(count)
    zeros = torch.zeros(count)
    identity = model(query, query, anchor, ones)
    no_demo = model(query, query, anchor, zeros)
    regular = model(
        query,
        torch.from_numpy(contexts["demo_state"][:count]).float(),
        anchor,
        ones,
    )
    delta = regular - anchor
    return {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "identity_anchor_max_abs_error": float(
            torch.max(torch.abs(identity - anchor)).item()
        ),
        "no_demo_max_abs_value": float(torch.max(torch.abs(no_demo)).item()),
        "translation_residual_max_abs_m": float(
            torch.max(torch.abs(delta[..., :3])).item()
        ),
        "rotation_residual_max_abs_rad": float(
            torch.max(torch.abs(delta[..., 3:6])).item()
        ),
        "gripper_residual_max_abs": float(
            torch.max(torch.abs(delta[..., 6])).item()
        ),
    }


@torch.no_grad()
def _latency(
    model: LayoutEquivariantDemoPolicy,
    contexts: dict[str, np.ndarray],
    *,
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    query = torch.from_numpy(contexts["query_state"][:1]).float()
    demo = torch.from_numpy(contexts["demo_state"][:1]).float()
    anchor = torch.from_numpy(contexts["anchor"][:1]).float()
    mask = torch.ones(1)
    for _ in range(warmup):
        model(query, demo, anchor, mask)
    samples = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        model(query, demo, anchor, mask)
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return {
        "median_ms": float(np.median(samples)),
        "p95_ms": float(np.quantile(samples, 0.95)),
    }


def run(
    *,
    project_root: Path,
    config_path: Path,
    data_root: Path,
    output_root: Path,
    config: DemoPolicyTrainConfig,
) -> None:
    if output_root.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output_root}")
    _seed_everything(config.seed)
    data_path = data_root / "branch_samples.npz"
    preprocess_path = data_root / "preprocess_report.json"
    preprocess = json.loads(preprocess_path.read_text(encoding="utf-8"))
    if _sha256(data_path) != preprocess["files"]["branch_samples.npz"]:
        raise ValueError("predictor data hash 不匹配")
    with np.load(data_path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    train_indices = np.where(data["split_id"] == SPLITS["train"])[0]
    validation_indices = np.where(data["split_id"] == SPLITS["val"])[0]
    state_mean = data["state"][train_indices].mean(axis=0).astype(np.float32)
    state_std = data["state"][train_indices].std(axis=0)
    state_std = np.maximum(state_std, config.state_std_floor).astype(np.float32)
    train_contexts = _contexts(
        data,
        query_indices=train_indices,
        demo_indices=train_indices,
        same_operation=True,
        exclude_same_pair=True,
    )
    validation_contexts = _contexts(
        data,
        query_indices=validation_indices,
        demo_indices=train_indices,
        same_operation=True,
        exclude_same_pair=False,
    )
    wrong_validation_contexts = _contexts(
        data,
        query_indices=validation_indices,
        demo_indices=train_indices,
        same_operation=False,
        exclude_same_pair=False,
    )
    policy_config = LayoutEquivariantDemoPolicyConfig(
        state_dim=data["state"].shape[1],
        action_horizon=data["action"].shape[1],
        hidden_dim=config.hidden_dim,
        maximum_translation_residual_m=(
            config.maximum_translation_residual_m
        ),
        maximum_rotation_residual_rad=config.maximum_rotation_residual_rad,
        translation_action_scale_m=config.translation_action_scale_m,
        rotation_action_scale_rad=config.rotation_action_scale_rad,
    )
    model = LayoutEquivariantDemoPolicy(
        policy_config,
        state_mean=torch.from_numpy(state_mean),
        state_std=torch.from_numpy(state_std),
    )
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
    training_log = []
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
                (
                    (prediction[..., :3] - target[..., :3])
                    / config.maximum_translation_residual_m
                )
                ** 2
            )
            rotation_loss = torch.mean(
                (
                    (prediction[..., 3:6] - target[..., 3:6])
                    / config.maximum_rotation_residual_rad
                )
                ** 2
            )
            loss = translation_loss + rotation_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip_norm
            )
            optimizer.step()
            losses.append(float(loss.detach()))
        validation_prediction = _predict(
            model,
            validation_contexts,
            batch_size=config.batch_size,
        )
        validation_mse = float(
            np.mean((validation_prediction - validation_contexts["target"]) ** 2)
        )
        training_log.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
                "validation_action_mse": validation_mse,
            }
        )
        if validation_mse < best_validation_mse - 1e-12:
            best_validation_mse = validation_mse
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= config.early_stopping_patience:
            break
    model.load_state_dict(best_state)

    validation = _evaluate(
        model=model,
        correct=validation_contexts,
        wrong=wrong_validation_contexts,
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
    criteria = {
        "V1_relative_improvement": (
            validation["relative_mse_improvement"]
            >= config.minimum_validation_relative_improvement
        ),
        "V2_scene_paired_significance": (
            validation["policy_minus_transport"]["ci95_high"] < 0.0
        ),
        "V3_direction_accuracy": (
            validation["metrics"]["demo_anchored_policy"][
                "direction_accuracy"
            ]
            >= config.minimum_validation_direction_accuracy
        ),
        "V4_wrong_operation_direction_accuracy": (
            validation["metrics"]["wrong_operation_policy"][
                "direction_accuracy"
            ]
            <= config.maximum_wrong_operation_direction_accuracy
        ),
        "V5_identity_anchor_exact": (
            structural["identity_anchor_max_abs_error"] == 0.0
        ),
        "V6_no_demo_exact_zero": structural["no_demo_max_abs_value"] == 0.0,
        "V7_residual_bounds": (
            structural["translation_residual_max_abs_m"]
            <= config.maximum_translation_residual_m + 1e-8
            and structural["rotation_residual_max_abs_rad"]
            <= config.maximum_rotation_residual_rad + 1e-8
            and structural["gripper_residual_max_abs"] == 0.0
        ),
        "V8_parameter_budget": (
            structural["parameters"] <= config.maximum_parameters
        ),
        "V9_latency_budget": latency["p95_ms"] <= config.maximum_latency_p95_ms,
    }
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(project_root),
        "config": asdict(config),
        "config_sha256": _sha256(config_path),
        "data_sha256": _sha256(data_path),
        "preprocess_report_sha256": _sha256(preprocess_path),
        "protocol": {
            "test_split_used": False,
            "training_contexts": len(train_contexts["query_row"]),
            "validation_contexts": len(validation_contexts["query_row"]),
            "wrong_validation_contexts": len(
                wrong_validation_contexts["query_row"]
            ),
            "best_epoch_selected_only_by_validation": best_epoch,
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
        "validation_gate_passed": all(criteria.values()),
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
            "validation_gate_passed": report["validation_gate_passed"],
        },
        checkpoint_path,
    )
    log_path = temporary / "training_log.jsonl"
    with log_path.open("w", encoding="utf-8") as stream:
        for row in training_log:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    report["files"] = {
        "checkpoint.pt": _sha256(checkpoint_path),
        "training_log.jsonl": _sha256(log_path),
    }
    (temporary / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_root)
    atexit.unregister(cleanup)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not report["validation_gate_passed"]:
        raise RuntimeError("Demo policy 未通过 train/validation 门控")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(
        project_root=arguments.project_root.resolve(),
        config_path=arguments.config.resolve(),
        data_root=arguments.data_root.resolve(),
        output_root=arguments.output_root.resolve(),
        config=DemoPolicyTrainConfig.from_json(arguments.config.resolve()),
    )

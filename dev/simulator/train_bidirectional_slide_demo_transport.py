"""训练并评测 Demo-anchored 低秩 residual transport。"""

from __future__ import annotations

import argparse
import atexit
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

from dev.predictor.demo_anchored_residual_transport import (
    DemoAnchoredResidualTransport,
    DemoAnchoredTransportConfig,
)
from dev.simulator.evaluate_bidirectional_slide_demo_baselines import (
    OPERATIONS,
    SPLITS,
    _bootstrap_difference,
    _metrics,
    _nearest,
    _scene_values,
    _strip_rows,
)


@dataclass(frozen=True)
class DemoTransportTrainConfig:
    """冻结的低秩 Demo transport 训练与门控协议。"""

    schema_version: str
    seed: int
    rank: int
    maximum_translation_residual_m: float
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    state_std_floor: float
    bootstrap_resamples: int
    minimum_validation_relative_improvement: float
    minimum_test_direction_accuracy: float
    maximum_wrong_demo_direction_accuracy: float
    maximum_parameters: int
    latency_warmup: int
    latency_iterations: int
    maximum_latency_p95_ms: float

    @classmethod
    def from_json(cls, path: Path) -> "DemoTransportTrainConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def __post_init__(self) -> None:
        if self.schema_version != "bidirectional-slide-demo-transport-v1":
            raise ValueError("未知 Demo transport train schema")
        positive = (
            self.rank,
            self.maximum_translation_residual_m,
            self.epochs,
            self.batch_size,
            self.learning_rate,
            self.gradient_clip_norm,
            self.state_std_floor,
            self.bootstrap_resamples,
            self.minimum_validation_relative_improvement,
            self.minimum_test_direction_accuracy,
            self.maximum_parameters,
            self.latency_warmup,
            self.latency_iterations,
            self.maximum_latency_p95_ms,
        )
        if min(positive) <= 0 or self.seed < 0 or self.weight_decay < 0:
            raise ValueError("Demo transport train 配置非法")
        if not 0.0 <= self.maximum_wrong_demo_direction_accuracy <= 1.0:
            raise ValueError("wrong Demo direction threshold 非法")


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


def _training_contexts(
    data: dict[str, np.ndarray],
    train_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    queries = []
    demos = []
    targets = []
    for query_index in train_indices:
        operation = data["operation_id"][query_index]
        pair_id = data["pair_id"][query_index]
        candidates = train_indices[
            (data["operation_id"][train_indices] == operation)
            & (data["pair_id"][train_indices] != pair_id)
        ]
        if not len(candidates):
            raise ValueError("train query 没有跨 scene 的同 operation Demo")
        for demo_index in candidates:
            queries.append(data["state"][query_index])
            demos.append(
                np.concatenate(
                    (
                        data["state"][demo_index],
                        data["action"][demo_index].reshape(-1),
                    )
                )
            )
            targets.append(data["action"][query_index])
    return np.stack(queries), np.stack(demos), np.stack(targets)


def _selected_contexts(
    *,
    data: dict[str, np.ndarray],
    normalized_state: np.ndarray,
    train_indices: np.ndarray,
    split_id: int,
    correct: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, int]]]:
    query_indices = np.where(data["split_id"] == split_id)[0]
    demo_indices = []
    selections = []
    for query_index in query_indices:
        operation = int(data["operation_id"][query_index])
        pool = train_indices[
            (data["operation_id"][train_indices] == operation)
            if correct
            else (data["operation_id"][train_indices] != operation)
        ]
        demo_index, _ = _nearest(
            normalized_state[query_index],
            normalized_state,
            pool,
        )
        demo_indices.append(demo_index)
        selections.append(
            {
                "query_row": int(query_index),
                "demo_row": demo_index,
                "query_pair_id": int(data["pair_id"][query_index]),
                "demo_pair_id": int(data["pair_id"][demo_index]),
            }
        )
    demos = np.asarray(demo_indices, dtype=np.int64)
    return (
        data["state"][query_indices],
        data["state"][demos],
        data["action"][demos],
        query_indices,
        selections,
    )


@torch.no_grad()
def _predict(
    model: DemoAnchoredResidualTransport,
    query_state: np.ndarray,
    demo_state: np.ndarray,
    demo_action: np.ndarray,
) -> np.ndarray:
    model.eval()
    output = model(
        torch.from_numpy(query_state).float(),
        torch.from_numpy(demo_state).float(),
        torch.from_numpy(demo_action).float(),
        torch.ones(len(query_state)),
    )
    return output.cpu().numpy()


def _evaluate(
    *,
    model: DemoAnchoredResidualTransport,
    data: dict[str, np.ndarray],
    normalized_state: np.ndarray,
    train_indices: np.ndarray,
    split_id: int,
    config: DemoTransportTrainConfig,
) -> dict[str, Any]:
    correct_context = _selected_contexts(
        data=data,
        normalized_state=normalized_state,
        train_indices=train_indices,
        split_id=split_id,
        correct=True,
    )
    wrong_context = _selected_contexts(
        data=data,
        normalized_state=normalized_state,
        train_indices=train_indices,
        split_id=split_id,
        correct=False,
    )
    query_indices = correct_context[3]
    target = data["action"][query_indices]
    copy_prediction = correct_context[2]
    prediction = _predict(model, *correct_context[:3])
    wrong_prediction = _predict(model, *wrong_context[:3])
    raw = {
        "demo_copy": _metrics(copy_prediction, target),
        "demo_anchored_transport": _metrics(prediction, target),
        "wrong_demo_transport": _metrics(wrong_prediction, target),
    }
    pair_ids = data["pair_id"][query_indices]
    scene_mse = {
        name: _scene_values(metrics["per_row_action_mse"], pair_ids)[1]
        for name, metrics in raw.items()
    }
    comparison = _bootstrap_difference(
        scene_mse["demo_anchored_transport"],
        scene_mse["demo_copy"],
        seed=config.seed + 100 + split_id,
        resamples=config.bootstrap_resamples,
    )
    return {
        "rows": len(query_indices),
        "scene_pairs": len(np.unique(pair_ids)),
        "metrics": {name: _strip_rows(value) for name, value in raw.items()},
        "transport_minus_copy": comparison,
        "relative_mse_improvement": float(
            (
                raw["demo_copy"]["action_mse"]
                - raw["demo_anchored_transport"]["action_mse"]
            )
            / raw["demo_copy"]["action_mse"]
        ),
        "correct_selections": correct_context[4],
        "wrong_selections": wrong_context[4],
    }


@torch.no_grad()
def _structural_audit(
    model: DemoAnchoredResidualTransport,
    state: np.ndarray,
    action: np.ndarray,
) -> dict[str, float | int]:
    model.eval()
    state_tensor = torch.from_numpy(state[:8]).float()
    action_tensor = torch.from_numpy(action[:8]).float()
    anchored = model(
        state_tensor,
        state_tensor,
        action_tensor,
        torch.ones(len(state_tensor)),
    )
    no_demo = model(
        state_tensor,
        state_tensor,
        action_tensor,
        torch.zeros(len(state_tensor)),
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return {
        "parameters": parameter_count,
        "identity_anchor_max_abs_error": float(
            torch.max(torch.abs(anchored - action_tensor)).item()
        ),
        "no_demo_max_abs_value": float(torch.max(torch.abs(no_demo)).item()),
    }


@torch.no_grad()
def _latency(
    model: DemoAnchoredResidualTransport,
    state: np.ndarray,
    action: np.ndarray,
    *,
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    query = torch.from_numpy(state[:1]).float()
    demo = torch.from_numpy(state[1:2]).float()
    demo_action = torch.from_numpy(action[1:2]).float()
    mask = torch.ones(1)
    for _ in range(warmup):
        model(query, demo, demo_action, mask)
    samples = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        model(query, demo, demo_action, mask)
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
    baseline_report_path: Path,
    output_root: Path,
    config: DemoTransportTrainConfig,
) -> None:
    if output_root.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output_root}")
    _seed_everything(config.seed)
    data_path = data_root / "branch_samples.npz"
    with np.load(data_path) as archive:
        data = {key: archive[key] for key in archive.files}
    train_indices = np.where(data["split_id"] == SPLITS["train"])[0]
    state_std = data["state"][train_indices].std(axis=0)
    state_std = np.maximum(state_std, config.state_std_floor).astype(np.float32)
    normalized_state = data["state"] / state_std
    query_train, packed_demo, target_train = _training_contexts(
        data,
        train_indices,
    )
    state_dim = data["state"].shape[1]
    horizon = data["action"].shape[1]
    demo_state_train = packed_demo[:, :state_dim]
    demo_action_train = packed_demo[:, state_dim:].reshape(-1, horizon, 7)
    model = DemoAnchoredResidualTransport(
        DemoAnchoredTransportConfig(
            state_dim=state_dim,
            action_horizon=horizon,
            rank=config.rank,
            maximum_translation_residual_m=(
                config.maximum_translation_residual_m
            ),
        ),
        state_std=torch.from_numpy(state_std),
    )
    dataset = TensorDataset(
        torch.from_numpy(query_train).float(),
        torch.from_numpy(demo_state_train).float(),
        torch.from_numpy(demo_action_train).float(),
        torch.from_numpy(target_train).float(),
    )
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    training_log = []
    model.train()
    for epoch in range(config.epochs):
        losses = []
        for query, demo_state, demo_action, target in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                query,
                demo_state,
                demo_action,
                torch.ones(len(query)),
            )
            loss = torch.mean(
                (
                    (prediction[..., :3] - target[..., :3])
                    / config.maximum_translation_residual_m
                )
                ** 2
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip_norm
            )
            optimizer.step()
            losses.append(float(loss.detach()))
        training_log.append({"epoch": epoch + 1, "loss": float(np.mean(losses))})

    validation = _evaluate(
        model=model,
        data=data,
        normalized_state=normalized_state,
        train_indices=train_indices,
        split_id=SPLITS["val"],
        config=config,
    )
    validation_gate = (
        validation["relative_mse_improvement"]
        >= config.minimum_validation_relative_improvement
    )
    test = None
    if validation_gate:
        test = _evaluate(
            model=model,
            data=data,
            normalized_state=normalized_state,
            train_indices=train_indices,
            split_id=SPLITS["test"],
            config=config,
        )
    structural = _structural_audit(model, data["state"], data["action"])
    latency = _latency(
        model,
        data["state"],
        data["action"],
        warmup=config.latency_warmup,
        iterations=config.latency_iterations,
    )
    criteria = {
        "validation_gate": validation_gate,
        "test_transport_beats_copy": bool(
            test is not None and test["transport_minus_copy"]["ci95_high"] < 0.0
        ),
        "test_direction_accuracy": bool(
            test is not None
            and test["metrics"]["demo_anchored_transport"][
                "direction_accuracy"
            ]
            >= config.minimum_test_direction_accuracy
        ),
        "wrong_demo_direction_accuracy": bool(
            test is not None
            and test["metrics"]["wrong_demo_transport"]["direction_accuracy"]
            <= config.maximum_wrong_demo_direction_accuracy
        ),
        "identity_anchor_exact": (
            structural["identity_anchor_max_abs_error"] == 0.0
        ),
        "no_demo_exact_zero": structural["no_demo_max_abs_value"] == 0.0,
        "parameter_budget": structural["parameters"] <= config.maximum_parameters,
        "latency_budget": latency["p95_ms"] <= config.maximum_latency_p95_ms,
    }
    baseline_report = json.loads(
        baseline_report_path.read_text(encoding="utf-8")
    )
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(project_root),
        "config": asdict(config),
        "config_sha256": _sha256(config_path),
        "data_sha256": _sha256(data_path),
        "baseline_report_sha256": _sha256(baseline_report_path),
        "baseline_primary_passed": baseline_report.get(
            "primary_test_all_criteria_passed"
        ),
        "training_contexts": len(dataset),
        "training_final_loss": training_log[-1]["loss"],
        "validation": validation,
        "test": test
        if test is not None
        else {"status": "not_evaluated_by_validation_gate"},
        "structural_audit": structural,
        "latency": latency,
        "criteria": criteria,
        "all_criteria_passed": all(criteria.values()),
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
            "model_config": asdict(model.config),
            "train_config": asdict(config),
            "data_sha256": _sha256(data_path),
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
    report_path = temporary / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_root)
    atexit.unregister(cleanup)
    printable = {
        key: value
        for key, value in report.items()
        if key not in {"validation", "test"}
    }
    printable["validation"] = {
        key: value
        for key, value in validation.items()
        if "selections" not in key
    }
    printable["test"] = (
        {
            key: value for key, value in test.items() if "selections" not in key
        }
        if test is not None
        else report["test"]
    )
    print(json.dumps(printable, indent=2, sort_keys=True), flush=True)
    if not report["all_criteria_passed"]:
        raise RuntimeError("Demo anchored transport 未通过全部预注册判据")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(
        project_root=arguments.project_root.resolve(),
        config_path=arguments.config.resolve(),
        data_root=arguments.data_root.resolve(),
        baseline_report_path=arguments.baseline_report.resolve(),
        output_root=arguments.output_root.resolve(),
        config=DemoTransportTrainConfig.from_json(arguments.config.resolve()),
    )

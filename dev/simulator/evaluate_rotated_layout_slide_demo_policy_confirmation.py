"""在全新 confirmation scenes 上一次性评测冻结的 Demo-anchored policy。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import torch

from dev.predictor.layout_equivariant_demo_policy import (
    LayoutEquivariantDemoPolicy,
    LayoutEquivariantDemoPolicyConfig,
)
from dev.simulator.evaluate_rotated_layout_slide_transport import (
    SPLITS,
    _bootstrap_difference,
    _metrics,
    _point_yaw,
    _scene_means,
    _strip_rows,
    _transport,
    _wrap_angle,
)
from dev.simulator.train_rotated_layout_slide_demo_policy import (
    _latency,
    _predict,
    _structural_audit,
)


@dataclass(frozen=True)
class ConfirmationConfig:
    """冻结的一次性 confirmation 判据。"""

    schema_version: str
    seed: int
    bootstrap_resamples: int
    minimum_relative_mse_improvement: float
    minimum_direction_accuracy: float
    maximum_wrong_operation_direction_accuracy: float
    maximum_parameters: int
    latency_warmup: int
    latency_iterations: int
    maximum_latency_p95_ms: float

    @classmethod
    def from_json(cls, path: Path) -> "ConfirmationConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def __post_init__(self) -> None:
        if (
            self.schema_version
            != "rotated-layout-slide-demo-policy-confirmation-v1"
        ):
            raise ValueError("未知 Demo policy confirmation schema")
        positive = (
            self.bootstrap_resamples,
            self.minimum_relative_mse_improvement,
            self.minimum_direction_accuracy,
            self.maximum_parameters,
            self.latency_warmup,
            self.latency_iterations,
            self.maximum_latency_p95_ms,
        )
        if min(positive) <= 0 or self.seed < 0:
            raise ValueError("confirmation 配置非法")
        probabilities = (
            self.minimum_direction_accuracy,
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


def _load_data(root: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    data_path = root / "branch_samples.npz"
    report_path = root / "preprocess_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if _sha256(data_path) != report["files"]["branch_samples.npz"]:
        raise ValueError(f"{root} data hash 不匹配")
    with np.load(data_path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    return data, report


def _cross_transport(
    query_data: dict[str, np.ndarray],
    bank_data: dict[str, np.ndarray],
    query_index: int,
    demo_index: int,
) -> np.ndarray:
    query_yaw = _point_yaw(
        query_data["state"][query_index],
        query_data["tcp_pose"][query_index],
    )
    demo_yaw = _point_yaw(
        bank_data["state"][demo_index], bank_data["tcp_pose"][demo_index]
    )
    return _transport(
        bank_data["action"][demo_index],
        bank_data["tcp_pose"][demo_index],
        query_data["tcp_pose"][query_index],
        _wrap_angle(query_yaw - demo_yaw),
    )


def _cross_contexts(
    query_data: dict[str, np.ndarray],
    bank_data: dict[str, np.ndarray],
    *,
    bank_indices: np.ndarray,
    same_operation: bool,
) -> dict[str, np.ndarray]:
    query_state = []
    demo_state = []
    anchors = []
    targets = []
    query_rows = []
    demo_rows = []
    for query_index in range(len(query_data["pair_id"])):
        query_operation = int(query_data["operation_id"][query_index])
        for demo_index in bank_indices:
            matches = (
                int(bank_data["operation_id"][demo_index]) == query_operation
            )
            if matches != same_operation:
                continue
            query_state.append(query_data["state"][query_index])
            demo_state.append(bank_data["state"][demo_index])
            anchors.append(
                _cross_transport(
                    query_data,
                    bank_data,
                    query_index,
                    int(demo_index),
                )
            )
            targets.append(query_data["action"][query_index])
            query_rows.append(query_index)
            demo_rows.append(demo_index)
    if not query_rows:
        raise ValueError("confirmation contexts 为空")
    return {
        "query_state": np.stack(query_state),
        "demo_state": np.stack(demo_state),
        "anchor": np.stack(anchors),
        "target": np.stack(targets),
        "query_row": np.asarray(query_rows, dtype=np.int64),
        "demo_row": np.asarray(demo_rows, dtype=np.int64),
    }


def _load_policy(
    checkpoint_path: Path,
    training_report: dict[str, Any],
) -> tuple[LayoutEquivariantDemoPolicy, dict[str, Any]]:
    if not training_report.get("validation_gate_passed", False):
        raise ValueError("训练报告未通过 validation gate")
    expected_hash = training_report["files"]["checkpoint.pt"]
    if _sha256(checkpoint_path) != expected_hash:
        raise ValueError("checkpoint hash 与训练报告不一致")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    policy_config = LayoutEquivariantDemoPolicyConfig(
        **checkpoint["model_config"]
    )
    model = LayoutEquivariantDemoPolicy(policy_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def run(
    *,
    project_root: Path,
    config_path: Path,
    bank_root: Path,
    confirmation_root: Path,
    training_root: Path,
    output_path: Path,
    config: ConfirmationConfig,
) -> None:
    if output_path.exists():
        raise FileExistsError(f"输出已存在，拒绝覆盖：{output_path}")
    torch.set_num_threads(1)
    bank_data, bank_report = _load_data(bank_root)
    confirmation_data, confirmation_report = _load_data(confirmation_root)
    training_report_path = training_root / "report.json"
    checkpoint_path = training_root / "checkpoint.pt"
    training_report = json.loads(
        training_report_path.read_text(encoding="utf-8")
    )
    model, checkpoint = _load_policy(checkpoint_path, training_report)
    bank_data_path = bank_root / "branch_samples.npz"
    if _sha256(bank_data_path) != checkpoint["data_sha256"]:
        raise ValueError("checkpoint 的训练数据 hash 不匹配")
    bank_indices = np.where(bank_data["split_id"] == SPLITS["train"])[0]
    correct = _cross_contexts(
        confirmation_data,
        bank_data,
        bank_indices=bank_indices,
        same_operation=True,
    )
    wrong = _cross_contexts(
        confirmation_data,
        bank_data,
        bank_indices=bank_indices,
        same_operation=False,
    )
    batch_size = int(checkpoint["train_config"]["batch_size"])
    policy_prediction = _predict(model, correct, batch_size=batch_size)
    wrong_prediction = _predict(model, wrong, batch_size=batch_size)
    raw = {
        "point_transport": _metrics(correct["anchor"], correct["target"]),
        "demo_anchored_policy": _metrics(
            policy_prediction, correct["target"]
        ),
        "wrong_operation_policy": _metrics(
            wrong_prediction, wrong["target"]
        ),
    }
    pair_ids = confirmation_data["pair_id"][correct["query_row"]]
    scene_mse = {
        name: _scene_means(metrics["per_row_action_mse"], pair_ids)[1]
        for name, metrics in raw.items()
        if name != "wrong_operation_policy"
    }
    comparison = _bootstrap_difference(
        scene_mse["demo_anchored_policy"],
        scene_mse["point_transport"],
        seed=config.seed,
        resamples=config.bootstrap_resamples,
    )
    relative_improvement = float(
        (
            raw["point_transport"]["action_mse"]
            - raw["demo_anchored_policy"]["action_mse"]
        )
        / raw["point_transport"]["action_mse"]
    )
    structural = _structural_audit(model, correct)
    latency = _latency(
        model,
        correct,
        warmup=config.latency_warmup,
        iterations=config.latency_iterations,
    )
    criteria = {
        "C1_relative_improvement": (
            relative_improvement >= config.minimum_relative_mse_improvement
        ),
        "C2_scene_paired_significance": comparison["ci95_high"] < 0.0,
        "C3_direction_accuracy": (
            raw["demo_anchored_policy"]["direction_accuracy"]
            >= config.minimum_direction_accuracy
        ),
        "C4_wrong_operation_direction_accuracy": (
            raw["wrong_operation_policy"]["direction_accuracy"]
            <= config.maximum_wrong_operation_direction_accuracy
        ),
        "C5_identity_anchor_exact": (
            structural["identity_anchor_max_abs_error"] == 0.0
        ),
        "C6_no_demo_exact_zero": structural["no_demo_max_abs_value"] == 0.0,
        "C7_parameter_budget": (
            structural["parameters"] <= config.maximum_parameters
        ),
        "C8_latency_budget": latency["p95_ms"] <= config.maximum_latency_p95_ms,
    }
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(project_root),
        "config": asdict(config),
        "config_sha256": _sha256(config_path),
        "protocol": {
            "confirmation_scene_pairs": int(
                len(np.unique(confirmation_data["pair_id"]))
            ),
            "confirmation_rows": len(confirmation_data["pair_id"]),
            "training_demo_scenes_per_query": int(
                len(np.unique(bank_data["pair_id"][bank_indices]))
            ),
            "correct_contexts": len(correct["query_row"]),
            "wrong_contexts": len(wrong["query_row"]),
            "checkpoint_frozen_before_confirmation": True,
            "confirmation_split_labels_ignored": True,
            "bootstrap_unit": "fresh confirmation scene pair",
        },
        "artifacts": {
            "checkpoint_sha256": _sha256(checkpoint_path),
            "training_report_sha256": _sha256(training_report_path),
            "bank_preprocess_report_sha256": _sha256(
                bank_root / "preprocess_report.json"
            ),
            "confirmation_preprocess_report_sha256": _sha256(
                confirmation_root / "preprocess_report.json"
            ),
            "bank_data_sha256": bank_report["files"]["branch_samples.npz"],
            "confirmation_data_sha256": confirmation_report["files"][
                "branch_samples.npz"
            ],
        },
        "metrics": {
            name: _strip_rows(metrics) for name, metrics in raw.items()
        },
        "policy_minus_transport": comparison,
        "relative_mse_improvement": relative_improvement,
        "structural_audit": structural,
        "latency": latency,
        "criteria": criteria,
        "all_criteria_passed": all(criteria.values()),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(f"{output_path.suffix}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not report["all_criteria_passed"]:
        raise RuntimeError("Demo policy 未通过 fresh confirmation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--confirmation-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(
        project_root=arguments.project_root.resolve(),
        config_path=arguments.config.resolve(),
        bank_root=arguments.bank_root.resolve(),
        confirmation_root=arguments.confirmation_root.resolve(),
        training_root=arguments.training_root.resolve(),
        output_path=arguments.output.resolve(),
        config=ConfirmationConfig.from_json(arguments.config.resolve()),
    )

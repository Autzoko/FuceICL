"""在封存 internal test 上一次性确认 Demo-relative action-regret。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dev.predictor.demo_relative_policy import (
    DemoRelativePolicy,
    DemoRelativePolicyConfig,
)
from dev.simulator.evaluate_rotated_layout_slide_transport import (
    SPLITS,
    _sha256,
)
from dev.simulator.train_belief_local_demo_residual import (
    _contexts,
    _git_commit,
    _load_data,
    _rank_demos,
    _shuffled_demo_rows,
)
from dev.simulator.train_demo_relative_action_regret import (
    DemoRelativeTrainConfig,
    _evaluate,
)
from dev.simulator.train_rotated_layout_slide_demo_policy import (
    _latency,
    _structural_audit,
)


@dataclass(frozen=True)
class DemoRelativeConfirmationConfig:
    """冻结的 internal-test confirmation 判据与 provenance。"""

    schema_version: str
    expected_data_sha256: str
    expected_preprocess_report_sha256: str
    expected_training_report_sha256: str
    expected_checkpoint_sha256: str
    seed: int
    bootstrap_resamples: int
    ambiguous_progress_maximum: int
    minimum_ambiguous_queries: int
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
    def from_json(cls, path: Path) -> "DemoRelativeConfirmationConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def __post_init__(self) -> None:
        if (
            self.schema_version
            != "demo-relative-action-regret-confirmation-v1"
        ):
            raise ValueError("未知 Demo-relative confirmation schema")
        hashes = (
            self.expected_data_sha256,
            self.expected_preprocess_report_sha256,
            self.expected_training_report_sha256,
            self.expected_checkpoint_sha256,
        )
        if any(len(value) != 64 for value in hashes):
            raise ValueError("confirmation hash 必须为 SHA-256")
        positive = (
            self.bootstrap_resamples,
            self.minimum_ambiguous_queries,
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
        ):
            raise ValueError("confirmation 配置非法")
        probabilities = (
            self.minimum_raw_relative_improvement,
            self.minimum_direction_accuracy,
            self.minimum_wrong_relative_improvement,
            self.minimum_shuffled_relative_improvement,
            self.maximum_ambiguous_wrong_direction_accuracy,
        )
        if any(not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("概率或相对改善门槛必须位于 [0,1]")


def _load_frozen_model(
    *,
    training_root: Path,
    config: DemoRelativeConfirmationConfig,
) -> tuple[
    DemoRelativePolicy,
    DemoRelativeTrainConfig,
    dict[str, Any],
    Path,
    Path,
]:
    """先验证 development gate 与 hash；失败时不允许读取 test 数据。"""
    report_path = training_root / "report.json"
    checkpoint_path = training_root / "checkpoint.pt"
    if _sha256(report_path) != config.expected_training_report_sha256:
        raise ValueError("training report hash 与冻结配置不一致")
    if _sha256(checkpoint_path) != config.expected_checkpoint_sha256:
        raise ValueError("checkpoint hash 与冻结配置不一致")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("development_gate_passed", False):
        raise ValueError("development gate 未通过，禁止读取 test")
    if report.get("protocol", {}).get("test_split_used", True):
        raise ValueError("development 已错误访问 test")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("model_variant") != "demo-relative-action-regret-v1":
        raise ValueError("checkpoint model variant 不匹配")
    if not checkpoint.get("development_gate_passed", False):
        raise ValueError("checkpoint 未标记 development pass")
    if checkpoint.get("data_sha256") != config.expected_data_sha256:
        raise ValueError("checkpoint data hash 不匹配")
    train_config = DemoRelativeTrainConfig(**checkpoint["train_config"])
    model = DemoRelativePolicy(
        DemoRelativePolicyConfig(**checkpoint["model_config"])
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, train_config, report, report_path, checkpoint_path


def run(
    *,
    project_root: Path,
    config_path: Path,
    data_root: Path,
    training_root: Path,
    output_path: Path,
    config: DemoRelativeConfirmationConfig,
) -> None:
    if output_path.exists():
        raise FileExistsError(f"输出已存在，拒绝覆盖：{output_path}")
    torch.set_num_threads(1)
    model, train_config, training_report, training_report_path, checkpoint_path = (
        _load_frozen_model(training_root=training_root, config=config)
    )
    # 只有上面的 development/provenance gate 通过后才加载含 test rows 的文件。
    data, preprocess, data_path, preprocess_path = _load_data(
        data_root, train_config
    )
    if _sha256(data_path) != config.expected_data_sha256:
        raise ValueError("confirmation data hash 不匹配")
    if _sha256(preprocess_path) != config.expected_preprocess_report_sha256:
        raise ValueError("confirmation preprocess hash 不匹配")
    bank_indices = np.where(data["split_id"] == SPLITS["train"])[0]
    test_indices = np.where(data["split_id"] == SPLITS["test"])[0]
    state_std = model.state_std.detach().cpu().numpy()

    test_query, test_demo, test_distance = _rank_demos(
        data,
        query_indices=test_indices,
        bank_indices=bank_indices,
        state_std=state_std,
        same_operation=True,
        neighbors=1,
        exclude_same_pair=True,
    )
    wrong_query, wrong_demo, wrong_distance = _rank_demos(
        data,
        query_indices=test_indices,
        bank_indices=bank_indices,
        state_std=state_std,
        same_operation=False,
        neighbors=1,
        exclude_same_pair=True,
    )
    if not np.array_equal(test_query, wrong_query):
        raise ValueError("correct/wrong test query 顺序不一致")
    shuffled_demo = _shuffled_demo_rows(
        data,
        test_query,
        test_demo,
        bank_indices,
        seed=config.seed + 17,
    )
    correct_contexts = _contexts(data, test_query, test_demo)
    shuffled_contexts = _contexts(data, test_query, shuffled_demo)
    wrong_contexts = _contexts(data, wrong_query, wrong_demo)
    evaluation_config = replace(
        train_config,
        seed=config.seed,
        bootstrap_resamples=config.bootstrap_resamples,
        ambiguous_progress_maximum=config.ambiguous_progress_maximum,
        minimum_ambiguous_queries=config.minimum_ambiguous_queries,
    )
    confirmation = _evaluate(
        model=model,
        correct=correct_contexts,
        shuffled=shuffled_contexts,
        wrong=wrong_contexts,
        data=data,
        config=evaluation_config,
    )
    structural = _structural_audit(model, correct_contexts)
    latency = _latency(
        model,
        correct_contexts,
        warmup=config.latency_warmup,
        iterations=config.latency_iterations,
    )
    ambiguous = confirmation["ambiguous_subset"]
    improvements = confirmation["relative_mse_improvement"]
    bootstrap = confirmation["scene_pair_bootstrap"]
    split_disjoint = not bool(
        np.intersect1d(
            data["pair_id"][bank_indices],
            data["pair_id"][test_indices],
        ).size
    )
    criteria = {
        "T1_improvement_over_raw": (
            improvements["over_raw"] >= config.minimum_raw_relative_improvement
            and bootstrap["correct_minus_raw"]["ci95_high"] < 0.0
        ),
        "T2_correct_direction": (
            confirmation["metrics"]["demo_relative"]["direction_accuracy"]
            >= config.minimum_direction_accuracy
        ),
        "T3_counterfactual_wrong_regret": (
            improvements["over_wrong"]
            >= config.minimum_wrong_relative_improvement
            and bootstrap["correct_minus_wrong"]["ci95_high"] < 0.0
        ),
        "T4_shuffled_demo_regret": (
            improvements["over_shuffled"]
            >= config.minimum_shuffled_relative_improvement
            and bootstrap["correct_minus_shuffled"]["ci95_high"] < 0.0
        ),
        "T5_ambiguous_demo_semantics": (
            ambiguous["queries"] >= config.minimum_ambiguous_queries
            and ambiguous["metrics"]["demo_relative"]["direction_accuracy"]
            >= config.minimum_direction_accuracy
            and ambiguous["metrics"]["wrong_operation"][
                "direction_accuracy"
            ]
            <= config.maximum_ambiguous_wrong_direction_accuracy
        ),
        "T6_structural_constraints": (
            structural["identity_anchor_max_abs_error"] == 0.0
            and structural["no_demo_max_abs_value"] == 0.0
            and structural["translation_residual_max_abs_m"]
            <= train_config.maximum_translation_residual_m + 1e-8
            and structural["rotation_residual_max_abs_rad"]
            <= train_config.maximum_rotation_residual_rad + 1e-8
            and structural["gripper_residual_max_abs"] == 0.0
        ),
        "T7_deployment_budget": (
            structural["parameters"] <= config.maximum_parameters
            and latency["p95_ms"] <= config.maximum_latency_p95_ms
        ),
        "T8_provenance_and_split": split_disjoint,
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(project_root),
        "config": asdict(config),
        "config_sha256": _sha256(config_path),
        "data_sha256": _sha256(data_path),
        "preprocess_report_sha256": _sha256(preprocess_path),
        "training_report_sha256": _sha256(training_report_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "protocol": {
            "training_development_gate_passed": training_report[
                "development_gate_passed"
            ],
            "test_split_used": True,
            "model_selection_on_test": False,
            "test_neighbors": 1,
            "test_queries": len(test_query),
            "train_test_scene_pairs_disjoint": split_disjoint,
            "absolute_query_feature_used": False,
            "progress_feature_used_for_retrieval": False,
            "operation_feature_used_by_model": False,
            "future_query_fields_used": False,
        },
        "retrieval_distance": {
            "test_median": float(np.median(test_distance)),
            "wrong_operation_median": float(np.median(wrong_distance)),
        },
        "confirmation": confirmation,
        "structural_audit": structural,
        "latency": latency,
        "criteria": criteria,
        "confirmation_passed": all(criteria.values()),
        "upstream_preprocess_schema_version": preprocess["schema_version"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    contexts_path = output_path.with_name(f"{output_path.stem}_contexts.npz")
    if contexts_path.exists():
        raise FileExistsError(f"context 输出已存在：{contexts_path}")
    temporary_contexts = contexts_path.with_suffix(
        f"{contexts_path.suffix}.tmp-{os.getpid()}"
    )
    with temporary_contexts.open("wb") as stream:
        np.savez_compressed(
            stream,
            test_query_row=test_query,
            test_demo_row=test_demo,
            test_distance=test_distance,
            shuffled_demo_row=shuffled_demo,
            wrong_demo_row=wrong_demo,
            wrong_distance=wrong_distance,
        )
    temporary_contexts.replace(contexts_path)
    report["files"] = {
        contexts_path.name: _sha256(contexts_path),
    }
    temporary_report = output_path.with_suffix(
        f"{output_path.suffix}.tmp-{os.getpid()}"
    )
    temporary_report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_report.replace(output_path)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not report["confirmation_passed"]:
        raise RuntimeError("Demo-relative internal test confirmation 未通过")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    resolved_config = arguments.config.resolve()
    run(
        project_root=arguments.project_root.resolve(),
        config_path=resolved_config,
        data_root=arguments.data_root.resolve(),
        training_root=arguments.training_root.resolve(),
        output_path=arguments.output.resolve(),
        config=DemoRelativeConfirmationConfig.from_json(resolved_config),
    )

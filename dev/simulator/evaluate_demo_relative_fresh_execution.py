"""在 fresh multi-progress scenes 上执行冻结 Demo-relative H6 Predictor。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from dev.simulator.evaluate_demo_relative_action_regret_confirmation import (
    _load_frozen_model,
)
from dev.simulator.evaluate_rotated_layout_slide_action_execution import (
    _environment,
    _execute,
    _load_replay,
    _stats,
)
from dev.simulator.evaluate_rotated_layout_slide_transport import (
    OPERATIONS,
    SPLITS,
    _bootstrap_difference,
    _metrics,
    _point_yaw,
    _sha256,
    _strip_rows,
)
from dev.simulator.preprocess_bidirectional_slide_predictor import _trajectory
from dev.simulator.train_belief_local_demo_residual import (
    _git_commit,
    _load_data,
)


POLICIES = (
    "expert_replay",
    "raw_demo_copy",
    "demo_relative",
    "wrong_operation",
    "no_demo",
)
OPERATION_NAMES = dict(OPERATIONS)


@dataclass(frozen=True)
class FreshExecutionConfig:
    """冻结的 fresh 多进度 controller execution 与 provenance。"""

    schema_version: str
    expected_bank_data_sha256: str
    expected_bank_report_sha256: str
    expected_training_report_sha256: str
    expected_checkpoint_sha256: str
    expected_confirmation_report_sha256: str
    seed: int
    expected_pairs: int
    expected_progress_samples: int
    action_horizon: int
    bootstrap_resamples: int
    position_limit_m: float
    rotation_scale_rad: float
    maximum_branch_alignment_error_m: float
    maximum_expert_endpoint_error_m: float
    minimum_raw_endpoint_improvement: float
    minimum_wrong_endpoint_improvement: float
    maximum_no_demo_displacement_m: float
    maximum_parameters: int
    maximum_two_forward_latency_p95_ms: float

    @classmethod
    def from_json(cls, path: Path) -> "FreshExecutionConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def __post_init__(self) -> None:
        if self.schema_version != "demo-relative-fresh-execution-v1":
            raise ValueError("未知 fresh execution schema")
        hashes = (
            self.expected_bank_data_sha256,
            self.expected_bank_report_sha256,
            self.expected_training_report_sha256,
            self.expected_checkpoint_sha256,
            self.expected_confirmation_report_sha256,
        )
        if any(len(value) != 64 for value in hashes):
            raise ValueError("execution provenance 必须使用 SHA-256")
        integers = (
            self.expected_pairs,
            self.expected_progress_samples,
            self.action_horizon,
            self.bootstrap_resamples,
            self.maximum_parameters,
        )
        if min(integers) <= 0 or self.seed < 0 or self.action_horizon != 6:
            raise ValueError("execution integer 配置非法")
        positive = (
            self.position_limit_m,
            abs(self.rotation_scale_rad),
            self.maximum_branch_alignment_error_m,
            self.maximum_expert_endpoint_error_m,
            self.maximum_no_demo_displacement_m,
            self.maximum_two_forward_latency_p95_ms,
        )
        if min(positive) <= 0 or self.rotation_scale_rad == 0:
            raise ValueError("execution scale/tolerance 非法")
        probabilities = (
            self.minimum_raw_endpoint_improvement,
            self.minimum_wrong_endpoint_improvement,
        )
        if any(not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("endpoint improvement 必须位于 [0,1]")

    @property
    def expected_data_sha256(self) -> str:
        """兼容冻结模型 loader 的训练数据 provenance 契约。"""
        return self.expected_bank_data_sha256


def _load_query_data(
    root: Path,
    config: FreshExecutionConfig,
) -> tuple[dict[str, np.ndarray], dict[str, Any], Path, Path]:
    data_path = root / "progress_chunks.npz"
    report_path = root / "preprocess_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("files", {}).get("progress_chunks.npz") != _sha256(data_path):
        raise ValueError("fresh query data hash 与 preprocess report 不一致")
    if int(report.get("pairs", -1)) != config.expected_pairs:
        raise ValueError("fresh query pair 数量不匹配")
    if report.get("protocol", {}).get("future_query_fields_used", True):
        raise ValueError("fresh query state 错误使用未来字段")
    if report.get("protocol", {}).get("actor_state_used_as_input", True):
        raise ValueError("fresh query state 错误使用 actor state")
    if report.get("protocol", {}).get("progress_feature_used_as_input", True):
        raise ValueError("fresh query state 错误使用 progress")
    if report.get("camera_variants") is None:
        raise ValueError("fresh query 缺少 camera provenance")
    if not all(
        variant == "dual-fixed-v1"
        for values in report["camera_variants"].values()
        for variant in values.values()
    ):
        raise ValueError("fresh query camera variant 不是 dual-fixed-v1")
    with np.load(data_path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    expected_rows = (
        config.expected_pairs * len(OPERATIONS) * config.expected_progress_samples
    )
    if len(data["pair_id"]) != expected_rows:
        raise ValueError("fresh query row 数量不匹配")
    return data, report, data_path, report_path


def _cross_nearest(
    *,
    query_state: np.ndarray,
    operation_id: int,
    bank_data: dict[str, np.ndarray],
    bank_indices: np.ndarray,
    state_std: np.ndarray,
) -> tuple[int, float]:
    candidates = bank_indices[
        bank_data["operation_id"][bank_indices] == operation_id
    ]
    if not len(candidates):
        raise ValueError(f"operation={operation_id} Demo bank 为空")
    delta = (bank_data["state"][candidates] - query_state) / state_std
    distances = np.mean(delta.astype(np.float64) ** 2, axis=1)
    nearest = int(np.argmin(distances))
    return int(candidates[nearest]), float(distances[nearest])


@torch.inference_mode()
def _predictions(
    *,
    model: torch.nn.Module,
    query_state: np.ndarray,
    operation_id: int,
    bank_data: dict[str, np.ndarray],
    bank_indices: np.ndarray,
    state_std: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any], float]:
    correct_demo, correct_distance = _cross_nearest(
        query_state=query_state,
        operation_id=operation_id,
        bank_data=bank_data,
        bank_indices=bank_indices,
        state_std=state_std,
    )
    wrong_demo, wrong_distance = _cross_nearest(
        query_state=query_state,
        operation_id=1 - operation_id,
        bank_data=bank_data,
        bank_indices=bank_indices,
        state_std=state_std,
    )

    def infer(demo_index: int, mask: float) -> np.ndarray:
        prediction = model(
            torch.from_numpy(query_state[None]).float(),
            torch.from_numpy(bank_data["state"][demo_index][None]).float(),
            torch.from_numpy(bank_data["action"][demo_index][None]).float(),
            torch.tensor([mask], dtype=torch.float32),
        )
        return prediction[0].cpu().numpy()

    started = time.perf_counter_ns()
    correct = infer(correct_demo, 1.0)
    wrong = infer(wrong_demo, 1.0)
    latency_ms = (time.perf_counter_ns() - started) / 1e6
    no_demo = infer(correct_demo, 0.0)
    return (
        {
            "raw_demo_copy": bank_data["action"][correct_demo],
            "demo_relative": correct,
            "wrong_operation": wrong,
            "no_demo": no_demo,
        },
        {
            "correct_demo_row": correct_demo,
            "correct_distance": correct_distance,
            "wrong_demo_row": wrong_demo,
            "wrong_distance": wrong_distance,
        },
        latency_ms,
    )


def _summaries(
    rows: list[dict[str, Any]], config: FreshExecutionConfig
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for policy in POLICIES:
        selected = [row for row in rows if row["policy"] == policy]
        endpoint = [
            row["endpoint_error_to_expert_m"]
            for row in selected
            if row["endpoint_error_to_expert_m"] is not None
        ]
        output[policy] = {
            "queries": len(selected),
            "tcp_displacement_m": _stats(
                [row["tcp_displacement_norm_m"] for row in selected]
            ),
            "object_displacement_m": _stats(
                [row["object_displacement_norm_m"] for row in selected]
            ),
            "branch_alignment_error_m": _stats(
                [row["branch_alignment_error_m"] for row in selected]
            ),
            "endpoint_error_to_expert_m": _stats(endpoint) if endpoint else None,
            "translation_clip_rate": sum(
                row["translation_clips"] for row in selected
            )
            / (len(selected) * config.action_horizon),
            "rotation_clip_rate": sum(
                row["rotation_clips"] for row in selected
            )
            / (len(selected) * config.action_horizon),
            "early_termination_or_truncation_rate": float(
                np.mean(
                    [
                        row["terminated_early"] or row["truncated_early"]
                        for row in selected
                    ]
                )
            ),
            "success_rate_after_h6": float(
                np.mean([row["final_success"] for row in selected])
            ),
        }
    return output


def _pair_endpoint(rows: list[dict[str, Any]], policy: str) -> np.ndarray:
    selected = [row for row in rows if row["policy"] == policy]
    pair_ids = sorted({int(row["pair_id"]) for row in selected})
    return np.asarray(
        [
            np.mean(
                [
                    row["endpoint_error_to_expert_m"]
                    for row in selected
                    if int(row["pair_id"]) == pair_id
                ]
            )
            for pair_id in pair_ids
        ],
        dtype=np.float64,
    )


def _progress_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for progress in sorted({int(row["progress_index"]) for row in rows}):
        output[str(progress)] = {}
        for policy in POLICIES:
            selected = [
                row
                for row in rows
                if row["policy"] == policy
                and int(row["progress_index"]) == progress
            ]
            endpoint = [
                row["endpoint_error_to_expert_m"]
                for row in selected
                if row["endpoint_error_to_expert_m"] is not None
            ]
            output[str(progress)][policy] = {
                "queries": len(selected),
                "endpoint_error_to_expert_m": (
                    _stats(endpoint) if endpoint else None
                ),
                "object_displacement_m": _stats(
                    [row["object_displacement_norm_m"] for row in selected]
                ),
                "success_rate_after_h6": float(
                    np.mean([row["final_success"] for row in selected])
                ),
            }
    return output


def run(
    *,
    project_root: Path,
    config_path: Path,
    bank_root: Path,
    query_root: Path,
    training_root: Path,
    confirmation_path: Path,
    replay_root: Path,
    output_path: Path,
    config: FreshExecutionConfig,
) -> None:
    if output_path.exists():
        raise FileExistsError(f"输出已存在，拒绝覆盖：{output_path}")
    torch.set_num_threads(1)
    if _sha256(confirmation_path) != config.expected_confirmation_report_sha256:
        raise ValueError("internal-test confirmation hash 不匹配")
    confirmation_report = json.loads(
        confirmation_path.read_text(encoding="utf-8")
    )
    if not confirmation_report.get("confirmation_passed", False):
        raise ValueError("internal-test confirmation 未通过")
    model, train_config, training_report, training_report_path, checkpoint_path = (
        _load_frozen_model(training_root=training_root, config=config)
    )
    bank_data, bank_report, bank_data_path, bank_report_path = _load_data(
        bank_root, train_config
    )
    if _sha256(bank_data_path) != config.expected_bank_data_sha256:
        raise ValueError("Demo bank data hash 不匹配")
    if _sha256(bank_report_path) != config.expected_bank_report_sha256:
        raise ValueError("Demo bank report hash 不匹配")
    query_data, query_report, query_data_path, query_report_path = (
        _load_query_data(query_root, config)
    )
    if set(query_data["seed"].tolist()) & set(bank_data["seed"].tolist()):
        raise ValueError("fresh query seed 与 Demo bank 重叠")
    state_std = model.state_std.detach().cpu().numpy()
    bank_indices = np.where(bank_data["split_id"] == SPLITS["train"])[0]
    handles, episodes, replay_audit = _load_replay(replay_root)
    audit_path = replay_root / "replay_audit_report.json"
    if query_report.get("replay_audit_sha256") != _sha256(audit_path):
        raise ValueError("fresh query data 与 replay audit hash 不匹配")
    if len(replay_audit.get("pairs", [])) != config.expected_pairs:
        raise ValueError("fresh replay pair 数量不匹配")
    audit_pairs = {
        int(row["seed"]): row for row in replay_audit["pairs"]
    }

    environments = {
        name: _environment(name) for name in OPERATION_NAMES.values()
    }
    rows: list[dict[str, Any]] = []
    latencies = []
    try:
        for query_index in range(len(query_data["pair_id"])):
            operation_id = int(query_data["operation_id"][query_index])
            operation = OPERATION_NAMES[operation_id]
            seed = int(query_data["seed"][query_index])
            pair_id = int(query_data["pair_id"][query_index])
            progress_index = int(query_data["progress_index"][query_index])
            frame = int(query_data["frame"][query_index])
            point_yaw = _point_yaw(
                query_data["state"][query_index],
                query_data["tcp_pose"][query_index],
            )
            if seed not in audit_pairs:
                raise ValueError(f"seed={seed} 不在 replay audit")
            yaw = math.radians(float(audit_pairs[seed]["axis_yaw_degrees"]))
            trajectory = _trajectory(handles[operation], episodes[operation][seed])
            controller_actions = np.asarray(
                trajectory["actions"], dtype=np.float32
            )
            recorded_tcp = np.asarray(
                trajectory["obs/extra/tcp_pose"], dtype=np.float64
            )
            if len(controller_actions) < frame + config.action_horizon:
                raise ValueError(f"seed={seed} frame={frame} action 长度不足")
            prefix = controller_actions[:frame]
            expert_chunk = controller_actions[
                frame : frame + config.action_horizon
            ]
            predictions, retrieval, latency = _predictions(
                model=model,
                query_state=query_data["state"][query_index],
                operation_id=operation_id,
                bank_data=bank_data,
                bank_indices=bank_indices,
                state_std=state_std,
            )
            latencies.append(latency)
            action_by_policy = {"expert_replay": expert_chunk, **predictions}
            expert_final: np.ndarray | None = None
            pending: list[dict[str, Any]] = []
            for policy in POLICIES:
                rollout = _execute(
                    environment=environments[operation],
                    seed=seed,
                    yaw=yaw,
                    prefix=prefix,
                    actions=action_by_policy[policy],
                    expert_controller=policy == "expert_replay",
                    config=config,
                )
                branch_tcp = np.asarray(rollout["branch_tcp_pose"])
                final_tcp = np.asarray(rollout["final_tcp_pose"])
                row = {
                    "query_row": query_index,
                    "pair_id": pair_id,
                    "seed": seed,
                    "operation": operation,
                    "progress_index": progress_index,
                    "frame": frame,
                    "axis_yaw_degrees": math.degrees(yaw),
                    "point_yaw_error_degrees": abs(
                        math.degrees(
                            (point_yaw - yaw + math.pi) % (2.0 * math.pi)
                            - math.pi
                        )
                    ),
                    "policy": policy,
                    "branch_alignment_error_m": float(
                        np.linalg.norm(
                            branch_tcp[:3]
                            - query_data["tcp_pose"][query_index, :3]
                        )
                    ),
                    "tcp_displacement_norm_m": float(
                        np.linalg.norm(rollout["tcp_displacement_m"])
                    ),
                    "object_displacement_norm_m": float(
                        np.linalg.norm(rollout["object_displacement_m"])
                    ),
                    "action_mse_to_expert": (
                        None
                        if policy == "expert_replay"
                        else _strip_rows(
                            _metrics(
                                action_by_policy[policy][None],
                                query_data["action"][query_index][None],
                            )
                        )["action_mse"]
                    ),
                    "retrieval": retrieval if policy != "expert_replay" else None,
                    **rollout,
                }
                if policy == "expert_replay":
                    expert_final = final_tcp
                    row["recorded_endpoint_error_m"] = float(
                        np.linalg.norm(
                            final_tcp[:3]
                            - recorded_tcp[frame + config.action_horizon, :3]
                        )
                    )
                    row["endpoint_error_to_expert_m"] = None
                else:
                    pending.append(row)
                rows.append(row)
            if expert_final is None:
                raise RuntimeError("expert reference 未执行")
            for row in pending:
                row["recorded_endpoint_error_m"] = None
                row["endpoint_error_to_expert_m"] = float(
                    np.linalg.norm(
                        np.asarray(row["final_tcp_pose"])[:3]
                        - expert_final[:3]
                    )
                )
    finally:
        for environment in environments.values():
            environment.close()
        for handle in handles.values():
            handle.close()

    summaries = _summaries(rows, config)
    raw_pair = _pair_endpoint(rows, "raw_demo_copy")
    policy_pair = _pair_endpoint(rows, "demo_relative")
    wrong_pair = _pair_endpoint(rows, "wrong_operation")
    policy_minus_raw = _bootstrap_difference(
        policy_pair,
        raw_pair,
        seed=config.seed,
        resamples=config.bootstrap_resamples,
    )
    policy_minus_wrong = _bootstrap_difference(
        policy_pair,
        wrong_pair,
        seed=config.seed + 1,
        resamples=config.bootstrap_resamples,
    )
    raw_improvement = float(
        (raw_pair.mean() - policy_pair.mean()) / max(raw_pair.mean(), 1e-12)
    )
    wrong_improvement = float(
        (wrong_pair.mean() - policy_pair.mean())
        / max(wrong_pair.mean(), 1e-12)
    )
    maximum_branch_alignment = max(
        float(row["branch_alignment_error_m"]) for row in rows
    )
    expert_rows = [row for row in rows if row["policy"] == "expert_replay"]
    maximum_expert_endpoint = max(
        float(row["recorded_endpoint_error_m"]) for row in expert_rows
    )
    any_early_stop = any(
        row["terminated_early"] or row["truncated_early"] for row in rows
    )
    action_policies = ("raw_demo_copy", "demo_relative", "wrong_operation")
    no_clipping = all(
        summaries[name]["translation_clip_rate"] == 0.0
        and summaries[name]["rotation_clip_rate"] == 0.0
        for name in action_policies
    )
    latency = _stats(latencies)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    criteria = {
        "X1_execution_valid": (
            maximum_branch_alignment
            <= config.maximum_branch_alignment_error_m
            and maximum_expert_endpoint
            <= config.maximum_expert_endpoint_error_m
            and not any_early_stop
        ),
        "X2_predictor_beats_raw": (
            raw_improvement >= config.minimum_raw_endpoint_improvement
            and policy_minus_raw["ci95_high"] < 0.0
        ),
        "X3_counterfactual_demo_regret": (
            wrong_improvement >= config.minimum_wrong_endpoint_improvement
            and policy_minus_wrong["ci95_high"] < 0.0
        ),
        "X4_structure_and_safety": (
            no_clipping
            and summaries["no_demo"]["tcp_displacement_m"]["maximum"]
            <= config.maximum_no_demo_displacement_m
            and parameters <= config.maximum_parameters
            and latency["p95"] <= config.maximum_two_forward_latency_p95_ms
        ),
        "X5_provenance": True,
    }
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(project_root),
        "config": asdict(config),
        "config_sha256": _sha256(config_path),
        "protocol": {
            "execution": "reset plus recorded controller prefix; no state forcing",
            "queries": len(query_data["pair_id"]),
            "fresh_pairs": config.expected_pairs,
            "progress_samples": config.expected_progress_samples,
            "policies": list(POLICIES),
            "absolute_query_feature_used": False,
            "progress_feature_used_for_retrieval": False,
            "operation_feature_used_by_model": False,
            "future_query_fields_used": False,
        },
        "provenance": {
            "bank_data_sha256": _sha256(bank_data_path),
            "bank_report_sha256": _sha256(bank_report_path),
            "query_data_sha256": _sha256(query_data_path),
            "query_report_sha256": _sha256(query_report_path),
            "training_report_sha256": _sha256(training_report_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "confirmation_report_sha256": _sha256(confirmation_path),
            "replay_audit_sha256": _sha256(audit_path),
            "query_replay_hash_matches": True,
            "seed_overlap": False,
            "development_gate_passed": training_report[
                "development_gate_passed"
            ],
            "internal_test_confirmation_passed": confirmation_report[
                "confirmation_passed"
            ],
        },
        "model_parameters": parameters,
        "two_forward_latency_ms": latency,
        "summaries": summaries,
        "progress_summaries": _progress_summaries(rows),
        "policy_minus_raw_endpoint_pair_bootstrap": policy_minus_raw,
        "policy_minus_wrong_endpoint_pair_bootstrap": policy_minus_wrong,
        "raw_endpoint_relative_improvement": raw_improvement,
        "wrong_endpoint_relative_improvement": wrong_improvement,
        "maximum_branch_alignment_error_m": maximum_branch_alignment,
        "maximum_expert_endpoint_error_m": maximum_expert_endpoint,
        "criteria": criteria,
        "all_criteria_passed": all(criteria.values()),
        "rows": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(f"{output_path.suffix}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    print(
        json.dumps(
            {
                "summaries": summaries,
                "raw_endpoint_relative_improvement": raw_improvement,
                "wrong_endpoint_relative_improvement": wrong_improvement,
                "policy_minus_raw": policy_minus_raw,
                "policy_minus_wrong": policy_minus_wrong,
                "criteria": criteria,
                "all_criteria_passed": all(criteria.values()),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    if not report["all_criteria_passed"]:
        raise RuntimeError("Demo-relative fresh execution 未通过全部门控")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--query-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    resolved_config = arguments.config.resolve()
    run(
        project_root=arguments.project_root.resolve(),
        config_path=resolved_config,
        bank_root=arguments.bank_root.resolve(),
        query_root=arguments.query_root.resolve(),
        training_root=arguments.training_root.resolve(),
        confirmation_path=arguments.confirmation.resolve(),
        replay_root=arguments.replay_root.resolve(),
        output_path=arguments.output.resolve(),
        config=FreshExecutionConfig.from_json(resolved_config),
    )

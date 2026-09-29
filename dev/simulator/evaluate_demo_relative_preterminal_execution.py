"""在 fresh 终止前 queries 上确认冻结 Demo-relative H6 Predictor。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dev.simulator.evaluate_demo_relative_action_regret_confirmation import (
    _load_frozen_model,
)
from dev.simulator.evaluate_demo_relative_fresh_execution import (
    OPERATION_NAMES,
    POLICIES,
    _load_query_data,
    _predictions,
    _progress_summaries,
    _summaries,
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


@dataclass(frozen=True)
class PreterminalExecutionConfig:
    """冻结的终止前执行确认、有效性门槛与 provenance。"""

    schema_version: str
    expected_bank_data_sha256: str
    expected_bank_report_sha256: str
    expected_training_report_sha256: str
    expected_checkpoint_sha256: str
    expected_confirmation_report_sha256: str
    seed: int
    expected_pairs: int
    expected_progress_samples: int
    expected_progress_fraction_maximum: float
    action_horizon: int
    bootstrap_resamples: int
    position_limit_m: float
    rotation_scale_rad: float
    maximum_branch_alignment_error_m: float
    maximum_expert_endpoint_error_m: float
    minimum_valid_query_fraction: float
    minimum_progress_valid_fraction: float
    minimum_pair_valid_fraction: float
    minimum_raw_endpoint_improvement: float
    minimum_wrong_endpoint_improvement: float
    minimum_hold_endpoint_improvement: float
    maximum_no_demo_action_absolute: float
    maximum_parameters: int
    maximum_two_forward_latency_p95_ms: float
    expected_belief_risk_gate_maximum_tcp_net_displacement_m: (
        float | None
    ) = None
    minimum_belief_risk_coverage: float = 1.0
    minimum_belief_risk_progress_coverage: float = 1.0
    minimum_belief_risk_pair_coverage: float = 1.0
    minimum_missing_belief_queries: int = 0
    minimum_belief_risk_rejected_queries: int = 0

    @classmethod
    def from_json(cls, path: Path) -> "PreterminalExecutionConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def __post_init__(self) -> None:
        if self.schema_version != "demo-relative-preterminal-execution-v1":
            raise ValueError("未知 preterminal execution schema")
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
        optional_counts = (
            self.minimum_missing_belief_queries,
            self.minimum_belief_risk_rejected_queries,
        )
        if min(optional_counts) < 0:
            raise ValueError("belief risk 最少样本数不得为负")
        positive = (
            self.position_limit_m,
            abs(self.rotation_scale_rad),
            self.maximum_branch_alignment_error_m,
            self.maximum_expert_endpoint_error_m,
            self.maximum_no_demo_action_absolute,
            self.maximum_two_forward_latency_p95_ms,
        )
        if min(positive) <= 0 or self.rotation_scale_rad == 0:
            raise ValueError("execution scale/tolerance 非法")
        probabilities = (
            self.expected_progress_fraction_maximum,
            self.minimum_valid_query_fraction,
            self.minimum_progress_valid_fraction,
            self.minimum_pair_valid_fraction,
            self.minimum_raw_endpoint_improvement,
            self.minimum_wrong_endpoint_improvement,
            self.minimum_hold_endpoint_improvement,
            self.minimum_belief_risk_coverage,
            self.minimum_belief_risk_progress_coverage,
            self.minimum_belief_risk_pair_coverage,
        )
        if any(not 0.0 < value <= 1.0 for value in probabilities):
            raise ValueError("execution fraction 必须位于 (0,1]")
        if (
            self.expected_belief_risk_gate_maximum_tcp_net_displacement_m
            is not None
            and self.expected_belief_risk_gate_maximum_tcp_net_displacement_m
            <= 0.0
        ):
            raise ValueError("belief risk gate 位移门槛必须为正")

    @property
    def expected_data_sha256(self) -> str:
        """兼容冻结模型 loader 的训练数据 provenance 契约。"""
        return self.expected_bank_data_sha256


def _pair_endpoint(rows: list[dict[str, Any]], policy: str) -> np.ndarray:
    """先在每个 scene pair 内平均有效 queries，避免 query 伪重复。"""
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


def _coverage(
    audits: list[dict[str, Any]],
    *,
    expected_progress_samples: int,
) -> dict[str, Any]:
    """汇总预先定义的 expert replay evaluation support。"""
    valid = [row for row in audits if row["valid"]]
    progress = {}
    for index in range(expected_progress_samples):
        selected = [row for row in audits if row["progress_index"] == index]
        progress[str(index)] = {
            "valid": sum(row["valid"] for row in selected),
            "total": len(selected),
            "fraction": float(np.mean([row["valid"] for row in selected])),
        }
    pairs = {}
    for pair_id in sorted({int(row["pair_id"]) for row in audits}):
        selected = [row for row in audits if int(row["pair_id"]) == pair_id]
        pairs[str(pair_id)] = {
            "valid": sum(row["valid"] for row in selected),
            "total": len(selected),
            "fraction": float(np.mean([row["valid"] for row in selected])),
        }
    return {
        "valid": len(valid),
        "total": len(audits),
        "fraction": len(valid) / len(audits),
        "by_progress": progress,
        "by_pair": pairs,
        "minimum_progress_fraction": min(
            row["fraction"] for row in progress.values()
        ),
        "minimum_pair_fraction": min(
            row["fraction"] for row in pairs.values()
        ),
    }


def _displacement_norms(rollout: dict[str, Any]) -> dict[str, float]:
    """将执行器的三维位移向量转换为共享汇总所需的标量。"""
    return {
        "tcp_displacement_norm_m": float(
            np.linalg.norm(rollout["tcp_displacement_m"])
        ),
        "object_displacement_norm_m": float(
            np.linalg.norm(rollout["object_displacement_m"])
        ),
    }


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
    config: PreterminalExecutionConfig,
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
    bank_data, _, bank_data_path, bank_report_path = _load_data(
        bank_root, train_config
    )
    if _sha256(bank_data_path) != config.expected_bank_data_sha256:
        raise ValueError("Demo bank data hash 不匹配")
    if _sha256(bank_report_path) != config.expected_bank_report_sha256:
        raise ValueError("Demo bank report hash 不匹配")
    query_data, query_report, query_data_path, query_report_path = (
        _load_query_data(query_root, config)
    )
    scheduled_maximum = query_report.get("protocol", {}).get(
        "progress_fraction_maximum"
    )
    if scheduled_maximum != config.expected_progress_fraction_maximum:
        raise ValueError("fresh query progress 上界与预注册不一致")
    expected_risk_threshold = (
        config.expected_belief_risk_gate_maximum_tcp_net_displacement_m
    )
    risk_report = query_report.get("belief_risk_gate", {})
    if expected_risk_threshold is not None:
        if (
            not risk_report.get("enabled", False)
            or risk_report.get("maximum_tcp_net_displacement_m")
            != expected_risk_threshold
        ):
            raise ValueError("fresh query belief risk gate 与预注册不一致")
        if risk_report.get("privileged_error_used_for_gate", True):
            raise ValueError("belief risk gate 错误使用 privileged error")
        required = {"belief_risk_accepted", "belief_risk_score_m"}
        if not required.issubset(query_data):
            raise ValueError("fresh query 缺少 belief risk routing 字段")
        if required & set(query_report.get("state_features", [])):
            raise ValueError("belief risk routing 字段不得进入 Predictor state")
        risk_accepted = query_data["belief_risk_accepted"].astype(bool)
    else:
        risk_accepted = np.ones(len(query_data["pair_id"]), dtype=bool)
    if set(query_data["seed"].tolist()) & set(bank_data["seed"].tolist()):
        raise ValueError("fresh query seed 与 Demo bank 重叠")
    risk_audits = [
        {
            "valid": bool(risk_accepted[index]),
            "pair_id": int(query_data["pair_id"][index]),
            "progress_index": int(query_data["progress_index"][index]),
        }
        for index in range(len(risk_accepted))
    ]
    risk_coverage = _coverage(
        risk_audits,
        expected_progress_samples=config.expected_progress_samples,
    )
    missing_belief_queries = int(
        query_report.get("belief", {}).get("propagated_rows", 0)
    )
    rejected_belief_queries = int(risk_report.get("rejected_rows", 0))
    risk_support_passed = bool(
        risk_coverage["fraction"] >= config.minimum_belief_risk_coverage
        and risk_coverage["minimum_progress_fraction"]
        >= config.minimum_belief_risk_progress_coverage
        and risk_coverage["minimum_pair_fraction"]
        >= config.minimum_belief_risk_pair_coverage
        and missing_belief_queries >= config.minimum_missing_belief_queries
        and rejected_belief_queries
        >= config.minimum_belief_risk_rejected_queries
    )
    if not risk_support_passed:
        raise RuntimeError(
            "belief risk support 未通过，拒绝运行 learned policies"
        )

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
    query_audits: list[dict[str, Any]] = []
    latencies = []
    no_demo_action_maximum = 0.0
    try:
        for query_index in range(len(query_data["pair_id"])):
            if not risk_accepted[query_index]:
                continue
            operation_id = int(query_data["operation_id"][query_index])
            operation = OPERATION_NAMES[operation_id]
            seed = int(query_data["seed"][query_index])
            pair_id = int(query_data["pair_id"][query_index])
            progress_index = int(query_data["progress_index"][query_index])
            frame = int(query_data["frame"][query_index])
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
            expert = _execute(
                environment=environments[operation],
                seed=seed,
                yaw=yaw,
                prefix=prefix,
                actions=expert_chunk,
                expert_controller=True,
                config=config,
            )
            branch_tcp = np.asarray(expert["branch_tcp_pose"])
            expert_final = np.asarray(expert["final_tcp_pose"])
            branch_error = float(
                np.linalg.norm(
                    branch_tcp[:3] - query_data["tcp_pose"][query_index, :3]
                )
            )
            endpoint_error = float(
                np.linalg.norm(
                    expert_final[:3]
                    - recorded_tcp[frame + config.action_horizon, :3]
                )
            )
            valid = bool(
                branch_error <= config.maximum_branch_alignment_error_m
                and endpoint_error <= config.maximum_expert_endpoint_error_m
                and not expert["terminated_early"]
                and not expert["truncated_early"]
            )
            query_audits.append(
                {
                    "query_row": query_index,
                    "pair_id": pair_id,
                    "seed": seed,
                    "operation": operation,
                    "progress_index": progress_index,
                    "frame": frame,
                    "branch_alignment_error_m": branch_error,
                    "recorded_endpoint_error_m": endpoint_error,
                    "terminated_early": expert["terminated_early"],
                    "truncated_early": expert["truncated_early"],
                    "valid": valid,
                }
            )
            if not valid:
                continue

            predictions, retrieval, latency = _predictions(
                model=model,
                query_state=query_data["state"][query_index],
                operation_id=operation_id,
                bank_data=bank_data,
                bank_indices=bank_indices,
                state_std=state_std,
            )
            latencies.append(latency)
            no_demo_action_maximum = max(
                no_demo_action_maximum,
                float(np.max(np.abs(predictions["no_demo"]))),
            )
            point_yaw = _point_yaw(
                query_data["state"][query_index],
                query_data["tcp_pose"][query_index],
            )
            shared = {
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
                "branch_alignment_error_m": branch_error,
            }
            rows.append(
                {
                    **shared,
                    "policy": "expert_replay",
                    "recorded_endpoint_error_m": endpoint_error,
                    "endpoint_error_to_expert_m": None,
                    "action_mse_to_expert": None,
                    "retrieval": None,
                    **_displacement_norms(expert),
                    **expert,
                }
            )
            for policy in POLICIES[1:]:
                rollout = _execute(
                    environment=environments[operation],
                    seed=seed,
                    yaw=yaw,
                    prefix=prefix,
                    actions=predictions[policy],
                    expert_controller=False,
                    config=config,
                )
                final_tcp = np.asarray(rollout["final_tcp_pose"])
                rows.append(
                    {
                        **shared,
                        "policy": policy,
                        "recorded_endpoint_error_m": None,
                        "endpoint_error_to_expert_m": float(
                            np.linalg.norm(final_tcp[:3] - expert_final[:3])
                        ),
                        "action_mse_to_expert": _strip_rows(
                            _metrics(
                                predictions[policy][None],
                                query_data["action"][query_index][None],
                            )
                        )["action_mse"],
                        "retrieval": retrieval,
                        **_displacement_norms(rollout),
                        **rollout,
                    }
                )
    finally:
        for environment in environments.values():
            environment.close()
        for handle in handles.values():
            handle.close()

    coverage = _coverage(
        query_audits,
        expected_progress_samples=config.expected_progress_samples,
    )
    if not rows:
        raise RuntimeError("没有通过 expert replay 审计的 query")
    summaries = _summaries(rows, config)
    policy_pair = _pair_endpoint(rows, "demo_relative")
    comparisons = {}
    improvements = {}
    for offset, baseline in enumerate(
        ("raw_demo_copy", "wrong_operation", "no_demo")
    ):
        baseline_pair = _pair_endpoint(rows, baseline)
        comparisons[baseline] = _bootstrap_difference(
            policy_pair,
            baseline_pair,
            seed=config.seed + offset,
            resamples=config.bootstrap_resamples,
        )
        improvements[baseline] = float(
            (baseline_pair.mean() - policy_pair.mean())
            / max(baseline_pair.mean(), 1e-12)
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
        "C1_evaluation_support": (
            risk_coverage["fraction"]
            >= config.minimum_belief_risk_coverage
            and risk_coverage["minimum_progress_fraction"]
            >= config.minimum_belief_risk_progress_coverage
            and risk_coverage["minimum_pair_fraction"]
            >= config.minimum_belief_risk_pair_coverage
            and missing_belief_queries
            >= config.minimum_missing_belief_queries
            and rejected_belief_queries
            >= config.minimum_belief_risk_rejected_queries
            and coverage["fraction"] >= config.minimum_valid_query_fraction
            and coverage["minimum_progress_fraction"]
            >= config.minimum_progress_valid_fraction
            and coverage["minimum_pair_fraction"]
            >= config.minimum_pair_valid_fraction
        ),
        "C2_predictor_beats_raw": (
            improvements["raw_demo_copy"]
            >= config.minimum_raw_endpoint_improvement
            and comparisons["raw_demo_copy"]["ci95_high"] < 0.0
        ),
        "C3_counterfactual_demo_regret": (
            improvements["wrong_operation"]
            >= config.minimum_wrong_endpoint_improvement
            and comparisons["wrong_operation"]["ci95_high"] < 0.0
        ),
        "C4_predictor_beats_hold": (
            improvements["no_demo"]
            >= config.minimum_hold_endpoint_improvement
            and comparisons["no_demo"]["ci95_high"] < 0.0
        ),
        "C5_structure_and_safety": (
            no_demo_action_maximum
            <= config.maximum_no_demo_action_absolute
            and no_clipping
            and parameters <= config.maximum_parameters
            and latency["p95"] <= config.maximum_two_forward_latency_p95_ms
        ),
        "C6_provenance": True,
    }
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(project_root),
        "config": asdict(config),
        "config_sha256": _sha256(config_path),
        "protocol": {
            "execution": "expert-valid audit, then reset plus recorded prefix",
            "queries": len(query_data["pair_id"]),
            "belief_risk_accepted_queries": int(risk_accepted.sum()),
            "fresh_pairs": config.expected_pairs,
            "progress_samples": config.expected_progress_samples,
            "progress_fraction_maximum": scheduled_maximum,
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
        "no_demo_action_maximum_absolute": no_demo_action_maximum,
        "belief_risk_support": risk_coverage,
        "missing_belief_queries": missing_belief_queries,
        "rejected_belief_queries": rejected_belief_queries,
        "evaluation_support": coverage,
        "summaries": summaries,
        "progress_summaries": _progress_summaries(rows),
        "pair_bootstrap": comparisons,
        "endpoint_relative_improvement": improvements,
        "criteria": criteria,
        "all_criteria_passed": all(criteria.values()),
        "query_audits": query_audits,
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
                "evaluation_support": coverage,
                "summaries": summaries,
                "endpoint_relative_improvement": improvements,
                "pair_bootstrap": comparisons,
                "criteria": criteria,
                "all_criteria_passed": all(criteria.values()),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    if not report["all_criteria_passed"]:
        raise RuntimeError("Demo-relative preterminal execution 未通过全部门控")


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
        config=PreterminalExecutionConfig.from_json(resolved_config),
    )

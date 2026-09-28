"""在已揭盲 ManiSkill 数据上评估 RCCS 的开发可行性。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Sequence

import torch

from dev.pointnet.compare_retrievers import _sha256
from dev.predictor.benefit_calibrated_shrinkage import (
    BenefitCalibratedGate,
    BenefitGateConfig,
    apply_shrinkage,
    optimal_shrinkage_target,
)
from dev.predictor.query_aligned_transport import (
    QueryAlignedLowRankDemoTransport,
    QueryAlignedTransportConfig,
)
from dev.predictor.retrieval_cell_certified_shrinkage import (
    RetrievalCellCertificate,
    empirical_cell_lipschitz_constant,
    support_certified_gate,
)
from dev.predictor.train_action_chunks import _physical_metrics
from dev.simulator.train_maniskill_query_aligned_bcsg import (
    TrainConfig,
    _build_protocol,
    _git_commit,
    _load_task,
    _qa_diagnostics,
)


def _load_frozen_models(
    checkpoint: dict[str, Any],
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


def _stats(values: torch.Tensor) -> dict[str, float | int] | None:
    values = values.detach().float().cpu()
    if len(values) == 0:
        return None
    return {
        "count": len(values),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p95": float(torch.quantile(values, 0.95)),
        "maximum": float(values.max()),
    }


def _translation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return (prediction[..., :3] - target[..., :3]).square().sum(dim=(1, 2))


def _task_analysis(
    *,
    protocol: Any,
    transport: QueryAlignedLowRankDemoTransport,
    gate: BenefitCalibratedGate,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    task = protocol.task
    bank_geometry = task.train_geometry[protocol.bank_indices]
    bank_actions = task.train_actions[protocol.bank_indices]
    retrieval_mean = bank_geometry.mean(dim=0)
    retrieval_std = bank_geometry.std(dim=0, unbiased=False).clamp_min(1e-4)

    operator_demo_geometry = bank_geometry[protocol.train_demo_indices]
    operator_demo_actions = bank_actions[protocol.train_demo_indices]
    operator_transport, _ = _qa_diagnostics(
        transport,
        bank_geometry,
        operator_demo_geometry,
        operator_demo_actions,
        device,
    )
    operator_mask = torch.ones(bank_actions.shape[:2], dtype=torch.bool)
    operator_benefit, operator_energy = optimal_shrinkage_target(
        operator_demo_actions,
        operator_transport,
        bank_actions,
        operator_mask,
    )
    informative_operator = operator_energy > 1e-8

    calibration_query = task.train_geometry[protocol.calibration_indices]
    calibration_target = task.train_actions[protocol.calibration_indices]
    calibration_demo_geometry = bank_geometry[
        protocol.calibration_demo_indices
    ]
    calibration_demo_actions = bank_actions[protocol.calibration_demo_indices]
    calibration_transport, calibration_features = _qa_diagnostics(
        transport,
        calibration_query,
        calibration_demo_geometry,
        calibration_demo_actions,
        device,
    )
    calibration_mask = torch.ones(
        calibration_target.shape[:2],
        dtype=torch.bool,
    )
    calibration_benefit, calibration_energy = optimal_shrinkage_target(
        calibration_demo_actions,
        calibration_transport,
        calibration_target,
        calibration_mask,
    )
    informative_calibration = calibration_energy > 1e-8
    support_geometry = torch.cat(
        (
            bank_geometry[informative_operator],
            calibration_query[informative_calibration],
        )
    )
    support_benefit = torch.cat(
        (
            operator_benefit[informative_operator],
            calibration_benefit[informative_calibration],
        )
    )
    support_cell = torch.cat(
        (
            protocol.train_demo_indices[informative_operator],
            protocol.calibration_demo_indices[informative_calibration],
        )
    )
    lipschitz_constant, slope_audit = empirical_cell_lipschitz_constant(
        support_geometry,
        support_benefit,
        support_cell,
        geometry_mean=retrieval_mean,
        geometry_std=retrieval_std,
    )
    certificate = RetrievalCellCertificate(
        support_geometry=support_geometry,
        support_benefit=support_benefit,
        support_cell=support_cell,
        geometry_mean=retrieval_mean,
        geometry_std=retrieval_std,
        lipschitz_constant=lipschitz_constant,
    )
    interpolation, _, _ = certificate.lower_bound(
        support_geometry,
        support_cell,
    )
    interpolation_error = float(
        (interpolation - support_benefit).abs().max()
    )
    empirically_certifiable = (
        slope_audit["near_duplicate_conflicts"] == 0
        and interpolation_error <= 1e-5
    )

    demo_geometry = bank_geometry[protocol.validation_demo_indices]
    demo_actions = bank_actions[protocol.validation_demo_indices]
    transported, validation_features = _qa_diagnostics(
        transport,
        task.val_geometry,
        demo_geometry,
        demo_actions,
        device,
    )
    with torch.inference_mode():
        base_gate = gate(validation_features.to(device)).cpu()
    raw_lower_bound, support_distance, has_support = certificate.lower_bound(
        task.val_geometry,
        protocol.validation_demo_indices,
    )
    lower_bound = (
        raw_lower_bound
        if empirically_certifiable
        else torch.zeros_like(raw_lower_bound)
    )
    certified_gate = support_certified_gate(base_gate, lower_bound)
    mask = torch.ones(task.val_actions.shape[:2], dtype=torch.bool)
    bcsg = apply_shrinkage(demo_actions, transported, base_gate, mask)
    rccs = apply_shrinkage(demo_actions, transported, certified_gate, mask)
    validation_benefit, validation_energy = optimal_shrinkage_target(
        demo_actions,
        transported,
        task.val_actions,
        mask,
    )
    validation_informative = validation_energy > 1e-8
    copy_loss = _translation_loss(demo_actions, task.val_actions)
    bcsg_loss = _translation_loss(bcsg, task.val_actions)
    rccs_loss = _translation_loss(rccs, task.val_actions)
    options = {
        "pose_scales": task.pose_scales,
        "translation_threshold_m": task.translation_threshold_m,
        "rotation_threshold_rad": task.rotation_threshold_rad,
    }
    predictions = {
        "demo_action_copy": demo_actions,
        "query_aligned_transport": transported,
        "bcsg": bcsg,
        "rccs": rccs,
    }
    metrics = {
        name: _physical_metrics(value, task.val_actions, mask, **options)
        for name, value in predictions.items()
    }
    cell_supported = has_support.float().mean()
    positive_certificate = lower_bound > 0.0
    contracted = certified_gate < base_gate - 1e-8
    fallback = certified_gate == 0.0
    safe_bound_violation = (
        (lower_bound > validation_benefit + 1e-6) & validation_informative
    )
    rccs_loss_violation = rccs_loss > copy_loss + 1e-8
    bcsg_loss_violation = bcsg_loss > copy_loss + 1e-8
    report = {
        "support_fit": {
            "operator_queries": len(bank_geometry),
            "informative_operator_queries": int(informative_operator.sum()),
            "calibration_queries": len(calibration_query),
            "informative_calibration_queries": int(
                informative_calibration.sum()
            ),
            "support_queries": len(support_geometry),
            "selected_cells": int(
                torch.unique(support_cell).numel()
            ),
            "bank_cells": len(bank_geometry),
            "lipschitz_constant": lipschitz_constant,
            "slope_audit": slope_audit,
            "empirically_certifiable": empirically_certifiable,
            "operator_optimal_benefit": _stats(
                operator_benefit[informative_operator]
            ),
            "calibration_optimal_benefit": _stats(
                calibration_benefit[informative_calibration]
            ),
            "calibration_base_gate": _stats(
                gate(calibration_features.to(device)).detach().cpu()
            ),
            "interpolation_max_abs_error": interpolation_error,
        },
        "validation": {
            "queries": len(task.val_geometry),
            "cell_support_fraction": float(cell_supported),
            "positive_certificate_fraction": float(
                positive_certificate.float().mean()
            ),
            "fallback_fraction": float(fallback.float().mean()),
            "gate_contraction_fraction": float(contracted.float().mean()),
            "base_gate": _stats(base_gate),
            "certified_gate": _stats(certified_gate),
            "benefit_lower_bound": _stats(lower_bound),
            "same_cell_support_distance": _stats(
                support_distance[has_support]
            ),
            "empirical_lower_bound_violation_fraction": float(
                safe_bound_violation.float().mean()
            ),
            "bcsg_worse_than_copy_fraction": float(
                bcsg_loss_violation.float().mean()
            ),
            "rccs_worse_than_copy_fraction": float(
                rccs_loss_violation.float().mean()
            ),
            "metrics": metrics,
        },
    }
    artifacts = {
        "target": task.val_actions,
        **predictions,
        "base_gate": base_gate,
        "certified_gate": certified_gate,
        "lower_bound": lower_bound,
        "has_support": has_support,
        "copy_loss": copy_loss,
        "bcsg_loss": bcsg_loss,
        "rccs_loss": rccs_loss,
    }
    return report, artifacts


def run(
    *,
    project_root: Path,
    data_roots: Sequence[Path],
    checkpoint_path: Path,
    output_path: Path,
    device: torch.device,
) -> None:
    if output_path.exists():
        raise FileExistsError(f"输出已存在，拒绝覆盖：{output_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = TrainConfig(**checkpoint["train_config"])
    tasks = [_load_task(path) for path in data_roots]
    if len(tasks) != 3 or len({task.task_id for task in tasks}) != 3:
        raise ValueError("RCCS development audit 固定要求三个不同任务")
    for task in tasks:
        expected = checkpoint["data_summary_sha256"].get(task.task_id)
        if expected != _sha256(task.root / "summary.json"):
            raise ValueError(f"{task.task_id} 数据与 checkpoint 不匹配")
    protocols = [_build_protocol(task, config) for task in tasks]
    transport, gate = _load_frozen_models(checkpoint, device)
    per_task = {}
    artifacts = []
    for protocol in protocols:
        task_report, task_artifacts = _task_analysis(
            protocol=protocol,
            transport=transport,
            gate=gate,
            device=device,
        )
        per_task[protocol.task.task_id] = task_report
        artifacts.append(task_artifacts)

    target = torch.cat([item["target"] for item in artifacts])
    mask = torch.ones(target.shape[:2], dtype=torch.bool)
    first = tasks[0]
    aggregate_predictions = {
        name: torch.cat([item[name] for item in artifacts])
        for name in (
            "demo_action_copy",
            "query_aligned_transport",
            "bcsg",
            "rccs",
        )
    }
    options = {
        "pose_scales": first.pose_scales,
        "translation_threshold_m": first.translation_threshold_m,
        "rotation_threshold_rad": first.rotation_threshold_rad,
    }
    aggregate_metrics = {
        name: _physical_metrics(value, target, mask, **options)
        for name, value in aggregate_predictions.items()
    }
    base_gate = torch.cat([item["base_gate"] for item in artifacts])
    certified_gate = torch.cat(
        [item["certified_gate"] for item in artifacts]
    )
    has_support = torch.cat([item["has_support"] for item in artifacts])
    lower_bound = torch.cat([item["lower_bound"] for item in artifacts])
    copy_loss = torch.cat([item["copy_loss"] for item in artifacts])
    bcsg_loss = torch.cat([item["bcsg_loss"] for item in artifacts])
    rccs_loss = torch.cat([item["rccs_loss"] for item in artifacts])
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_status": (
            "post-hoc development only; validation was previously unblinded"
        ),
        "method": (
            "exact retrieval-cell empirical Lipschitz lower bound; "
            "gate=min(BCSG, 2*lower_bound)"
        ),
        "git_commit": _git_commit(project_root),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_git_commit": checkpoint.get("git_commit"),
        "device": str(device),
        "per_task": per_task,
        "aggregate": {
            "queries": len(target),
            "cell_support_fraction": float(has_support.float().mean()),
            "positive_certificate_fraction": float(
                (lower_bound > 0.0).float().mean()
            ),
            "fallback_fraction": float(
                (certified_gate == 0.0).float().mean()
            ),
            "gate_contraction_fraction": float(
                (certified_gate < base_gate - 1e-8).float().mean()
            ),
            "base_gate": _stats(base_gate),
            "certified_gate": _stats(certified_gate),
            "bcsg_worse_than_copy_fraction": float(
                (bcsg_loss > copy_loss + 1e-8).float().mean()
            ),
            "rccs_worse_than_copy_fraction": float(
                (rccs_loss > copy_loss + 1e-8).float().mean()
            ),
            "metrics": aggregate_metrics,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_path)
    print(json.dumps(report["aggregate"], indent=2), flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    run(
        project_root=arguments.project_root.resolve(),
        data_roots=[path.resolve() for path in arguments.data_root],
        checkpoint_path=arguments.checkpoint.resolve(),
        output_path=arguments.output.resolve(),
        device=torch.device(arguments.device),
    )

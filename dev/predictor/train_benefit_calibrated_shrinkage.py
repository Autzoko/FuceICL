"""在冻结 task-heldout QA-LRDT 上训练并评估 BCSG。"""

from __future__ import annotations

import argparse
import atexit
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dev.pointnet.compare_retrievers import (
    ComparisonConfig,
    _read_jsonl,
    _sha256,
    encode_contexts,
    load_model as load_retriever,
)
from dev.pointnet.dataset import PointNetContextStore
from dev.pointnet.evaluate_end_to_end_retriever import (
    _load_text_scores,
    _text_candidate_masks,
)
from dev.predictor.action_chunk_data import ActionChunkStore
from dev.predictor.benefit_calibrated_shrinkage import (
    BenefitCalibratedGate,
    BenefitGateConfig,
    apply_shrinkage,
    constant_optimal_shrinkage,
    frozen_transport_features,
    optimal_shrinkage_target,
)
from dev.predictor.canonical_geometry import load_canonical_geometries
from dev.predictor.query_aligned_transport import (
    QueryAlignedLowRankDemoTransport,
    QueryAlignedTransportConfig,
)
from dev.predictor.train_action_chunks import (
    POSE_SCALES,
    _condition_tensors,
    _load_action_tensors,
    _physical_metrics,
)
from dev.predictor.train_query_aligned_transport import (
    _primary_selection_indices,
    _subset_selections,
)
from dev.simulator.evaluate_maniskill_demo_prior import _bootstrap_comparison


@dataclass(frozen=True)
class GateTrainConfig:
    """结果揭盲前冻结的 BCSG calibration 配置。"""

    schema_version: str
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    hidden_dim: int
    retriever_batch_size: int
    retriever_num_workers: int
    text_group_budget: int
    text_score_weight: float
    bootstrap_resamples: int
    latency_iterations: int

    @classmethod
    def from_json(cls, path: Path) -> "GateTrainConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def __post_init__(self) -> None:
        positive = (
            self.seed,
            self.epochs,
            self.batch_size,
            self.learning_rate,
            self.gradient_clip_norm,
            self.hidden_dim,
            self.retriever_batch_size,
            self.text_group_budget,
            self.bootstrap_resamples,
            self.latency_iterations,
        )
        if not self.schema_version.strip() or min(positive) <= 0:
            raise ValueError("schema_version 与训练正数参数必须有效")
        if self.weight_decay < 0 or self.retriever_num_workers < 0:
            raise ValueError("weight decay/worker 不能为负")
        if not 0.0 <= self.text_score_weight <= 1.0:
            raise ValueError("text_score_weight 必须位于 [0,1]")


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
    torch.cuda.manual_seed_all(seed)


def _load_parent(
    fold_root: Path,
    *,
    context_hash: str,
    action_hash: str,
    device: torch.device,
) -> tuple[
    QueryAlignedLowRankDemoTransport,
    dict[str, Any],
    dict[str, np.ndarray],
]:
    report_path = fold_root / "report.json"
    artifact_path = fold_root / "held_out_predictions.npz"
    checkpoint_path = fold_root / "query_aligned_transport.pt"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != 2:
        raise ValueError("BCSG 只接受 QA-LRDT v2 parent report")
    if report["context_summary_sha256"] != context_hash:
        raise ValueError("parent context hash 不匹配")
    if report["action_summary_sha256"] != action_hash:
        raise ValueError("parent action hash 不匹配")
    if report["held_out_prediction_artifact"]["sha256"] != _sha256(
        artifact_path
    ):
        raise ValueError("parent prediction artifact hash 不匹配")
    if report["checkpoint_sha256"]["query_aligned_transport"] != _sha256(
        checkpoint_path
    ):
        raise ValueError("parent QA checkpoint hash 不匹配")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint["context_summary_sha256"] != context_hash:
        raise ValueError("parent checkpoint context hash 不匹配")
    if checkpoint["action_summary_sha256"] != action_hash:
        raise ValueError("parent checkpoint action hash 不匹配")
    model = QueryAlignedLowRankDemoTransport(
        QueryAlignedTransportConfig(**checkpoint["model_config"]),
        geometry_mean=checkpoint["geometry_mean"],
        geometry_std=checkpoint["geometry_std"],
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with np.load(artifact_path, allow_pickle=False) as archive:
        artifact = {name: np.asarray(archive[name]) for name in archive.files}
    return model, report, artifact


@torch.inference_mode()
def _diagnostics(
    *,
    transport: QueryAlignedLowRankDemoTransport,
    query_geometry: torch.Tensor,
    demo_geometry: torch.Tensor,
    demo_actions: torch.Tensor,
    demo_mask: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    predictions = []
    features = []
    for start in range(0, len(query_geometry), batch_size):
        stop = start + batch_size
        prediction, feature = frozen_transport_features(
            transport,
            query_geometry[start:stop].to(device),
            demo_geometry[start:stop].to(device),
            demo_actions[start:stop].to(device),
            demo_mask[start:stop].to(device),
        )
        predictions.append(prediction.cpu())
        features.append(feature.cpu())
    return torch.cat(predictions), torch.cat(features)


def _train_gate(
    *,
    features: torch.Tensor,
    targets: torch.Tensor,
    residual_energy: torch.Tensor,
    config: GateTrainConfig,
    device: torch.device,
    log_path: Path,
) -> BenefitCalibratedGate:
    feature_mean = features.mean(dim=0)
    feature_std = features.std(dim=0, unbiased=False).clamp_min(1e-5)
    model = BenefitCalibratedGate(
        BenefitGateConfig(hidden_dim=config.hidden_dim),
        feature_mean=feature_mean,
        feature_std=feature_std,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    weights = residual_energy / residual_energy.mean().clamp_min(1e-8)
    dataset = TensorDataset(features, targets, weights)
    for epoch in range(config.epochs):
        generator = torch.Generator().manual_seed(config.seed + epoch)
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
        )
        total = 0.0
        examples = 0
        model.train()
        for feature, target, weight in loader:
            feature = feature.to(device)
            target = target.to(device)
            weight = weight.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(feature)
            loss = (weight * (prediction - target).square()).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            total += float(loss.detach()) * len(feature)
            examples += len(feature)
        if epoch == 0 or (epoch + 1) % 10 == 0:
            record = {
                "epoch": epoch + 1,
                "weighted_gate_mse": total / examples,
            }
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
    return model.eval()


def _bootstrap(
    *,
    predictions: Mapping[str, torch.Tensor],
    target: torch.Tensor,
    mask: torch.Tensor,
    group_ids: Sequence[str],
    config: GateTrainConfig,
) -> dict[str, Any]:
    options = {
        "target": target,
        "mask": mask,
        "group_ids": group_ids,
        "resamples": config.bootstrap_resamples,
        "pose_scales": POSE_SCALES,
        "translation_threshold_m": 0.05,
        "rotation_threshold_rad": 0.25,
    }
    pairs = (
        ("bcsg_minus_full_qa", "query_aligned_transport", "bcsg"),
        ("bcsg_minus_copy", "demo_action_copy", "bcsg"),
        ("bcsg_minus_constant", "constant_shrinkage", "bcsg"),
        ("constant_minus_full_qa", "query_aligned_transport", "constant_shrinkage"),
    )
    return {
        name: _bootstrap_comparison(
            reference=predictions[reference],
            candidate=predictions[candidate],
            seed=config.seed + 100 * offset,
            **options,
        )
        for offset, (name, reference, candidate) in enumerate(pairs)
    }


@torch.inference_mode()
def _latency(
    *,
    transport: QueryAlignedLowRankDemoTransport,
    gate: BenefitCalibratedGate,
    query: torch.Tensor,
    demo: torch.Tensor,
    actions: torch.Tensor,
    mask: torch.Tensor,
    iterations: int,
    device: torch.device,
) -> dict[str, float | int]:
    query = query[:1].to(device)
    demo = demo[:1].to(device)
    actions = actions[:1].to(device)
    mask = mask[:1].to(device)

    def predict() -> torch.Tensor:
        transported, features = frozen_transport_features(
            transport,
            query,
            demo,
            actions,
            mask,
        )
        return apply_shrinkage(actions, transported, gate(features), mask)

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
        "max_ms": float(values.max()),
    }


def run(
    *,
    project_root: Path,
    context_root: Path,
    action_root: Path,
    retriever_checkpoint: Path,
    text_scores_path: Path,
    parent_fold_root: Path,
    output_root: Path,
    config_path: Path,
    config: GateTrainConfig,
    device: torch.device,
) -> None:
    if output_root.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output_root}")
    temporary = output_root.with_name(f".{output_root.name}.incomplete-{os.getpid()}")
    temporary.mkdir(parents=True)

    def cleanup_incomplete() -> None:
        shutil.rmtree(temporary, ignore_errors=True)

    atexit.register(cleanup_incomplete)
    _seed_everything(config.seed)
    context_hash = _sha256(context_root / "summary.json")
    action_hash = _sha256(action_root / "summary.json")
    transport, parent_report, parent_artifact = _load_parent(
        parent_fold_root,
        context_hash=context_hash,
        action_hash=action_hash,
        device=device,
    )
    held_out_tasks = set(parent_report["held_out_tasks"])
    source_tasks = set(parent_report["source_tasks"])
    context_store = PointNetContextStore(context_root, "val", cache_size=4)
    action_store = ActionChunkStore(action_root, "val", cache_size=4)
    geometries = load_canonical_geometries(context_store)
    actions, action_masks = _load_action_tensors(context_store, action_store)
    pair_rows = _read_jsonl(context_root / "pairs-val.jsonl")

    retriever = load_retriever(retriever_checkpoint, device)
    embeddings = encode_contexts(
        retriever,
        context_store,
        ComparisonConfig(
            batch_size=config.retriever_batch_size,
            num_workers=config.retriever_num_workers,
        ),
        device,
    )
    text_scores, chunk_groups, text_metadata = _load_text_scores(
        text_scores_path,
        context_store.records,
    )
    if text_metadata.get("manifest_sha256") != _sha256(
        context_root / "manifest-val.jsonl"
    ):
        raise ValueError("文本分数与 val manifest 不匹配")
    text_mask = _text_candidate_masks(
        text_scores,
        chunk_groups,
        (config.text_group_budget,),
    )[config.text_group_budget]
    pointnet_scores = (embeddings @ embeddings.T + 1.0) / 2.0
    normalized_text = ((text_scores + 1.0) / 2.0).clamp(0.0, 1.0)
    retrieval_scores = (
        config.text_score_weight * normalized_text
        + (1.0 - config.text_score_weight) * pointnet_scores
    )

    held_all_indices, held_all_selections = _primary_selection_indices(
        records=context_store.records,
        pair_rows=pair_rows,
        action_masks=action_masks,
        retrieval_scores=retrieval_scores,
        text_mask=text_mask,
    )
    held_indices, held_selections = _subset_selections(
        query_indices=held_all_indices,
        selections=held_all_selections,
        records=context_store.records,
        tasks=held_out_tasks,
    )
    calibration_all_indices, calibration_all_selections = (
        _primary_selection_indices(
            records=context_store.records,
            pair_rows=pair_rows,
            action_masks=action_masks,
            retrieval_scores=retrieval_scores,
            text_mask=text_mask,
            candidate_tasks=source_tasks,
        )
    )
    calibration_indices, calibration_selections = _subset_selections(
        query_indices=calibration_all_indices,
        selections=calibration_all_selections,
        records=context_store.records,
        tasks=source_tasks,
    )
    parent_indices = parent_artifact["query_indices"].astype(np.int64)
    if not np.array_equal(parent_indices, np.asarray(held_indices)):
        raise ValueError("BCSG held-out query order 与 parent artifact 不一致")

    calibration_demo, calibration_actions, calibration_mask = _condition_tensors(
        calibration_selections["retrieved"],
        geometries,
        actions,
        action_masks,
    )
    calibration_transport, calibration_features = _diagnostics(
        transport=transport,
        query_geometry=geometries[calibration_indices],
        demo_geometry=calibration_demo,
        demo_actions=calibration_actions,
        demo_mask=calibration_mask,
        batch_size=config.retriever_batch_size,
        device=device,
    )
    calibration_copy = calibration_actions * calibration_mask.float().unsqueeze(-1)
    calibration_target = actions[calibration_indices]
    calibration_target_mask = action_masks[calibration_indices]
    optimal_gate, residual_energy = optimal_shrinkage_target(
        calibration_copy,
        calibration_transport,
        calibration_target,
        calibration_target_mask,
    )
    constant_gate = constant_optimal_shrinkage(
        calibration_copy,
        calibration_transport,
        calibration_target,
        calibration_target_mask,
    )
    gate = _train_gate(
        features=calibration_features,
        targets=optimal_gate,
        residual_energy=residual_energy,
        config=config,
        device=device,
        log_path=temporary / "training_metrics.jsonl",
    )

    held_demo, held_demo_actions, held_demo_mask = _condition_tensors(
        held_selections["retrieved"],
        geometries,
        actions,
        action_masks,
    )
    held_transport, held_features = _diagnostics(
        transport=transport,
        query_geometry=geometries[held_indices],
        demo_geometry=held_demo,
        demo_actions=held_demo_actions,
        demo_mask=held_demo_mask,
        batch_size=config.retriever_batch_size,
        device=device,
    )
    held_copy = held_demo_actions * held_demo_mask.float().unsqueeze(-1)
    parent_copy = torch.from_numpy(parent_artifact["retrieved__demo_action_copy"])
    parent_transport = torch.from_numpy(
        parent_artifact["retrieved__query_aligned_transport"]
    )
    if not torch.allclose(held_copy, parent_copy, atol=0.0, rtol=0.0):
        raise ValueError("重新计算的 Demo copy 与 parent artifact 不一致")
    if not torch.allclose(held_transport, parent_transport, atol=1e-6, rtol=0.0):
        raise ValueError("重新计算的 QA prediction 与 parent artifact 不一致")
    with torch.inference_mode():
        predicted_gate = gate(held_features.to(device)).cpu()
    constant_gates = torch.full_like(predicted_gate, float(constant_gate))
    bcsg = apply_shrinkage(
        held_copy,
        held_transport,
        predicted_gate,
        held_demo_mask,
    )
    constant_prediction = apply_shrinkage(
        held_copy,
        held_transport,
        constant_gates,
        held_demo_mask,
    )
    target = actions[held_indices]
    target_mask = action_masks[held_indices]
    parent_target = torch.from_numpy(parent_artifact["target_actions"])
    if not torch.equal(target, parent_target):
        raise ValueError("held-out target 与 parent artifact 不一致")
    predictions = {
        "demo_action_copy": held_copy,
        "fixed_low_rank_transport": torch.from_numpy(
            parent_artifact["retrieved__fixed_low_rank_transport"]
        ),
        "query_aligned_transport": held_transport,
        "constant_shrinkage": constant_prediction,
        "bcsg": bcsg,
    }
    group_ids = parent_artifact["group_ids"].astype(str).tolist()
    held_optimal_gate, held_energy = optimal_shrinkage_target(
        held_copy,
        held_transport,
        target,
        target_mask,
    )
    with torch.inference_mode():
        zero_mask = torch.zeros_like(held_demo_mask[:1]).to(device)
        zero_actions = held_demo_actions[:1].to(device)
        zero_transport, zero_features = frozen_transport_features(
            transport,
            geometries[held_indices[:1]].to(device),
            held_demo[:1].to(device),
            zero_actions,
            zero_mask,
        )
        zero_copy = torch.zeros_like(zero_transport)
        zero_output = apply_shrinkage(
            zero_copy,
            zero_transport,
            gate(zero_features),
            zero_mask,
        )
    checkpoint = {
        "model": {
            name: value.detach().cpu() for name, value in gate.state_dict().items()
        },
        "model_config": asdict(gate.config),
        "train_config": asdict(config),
        "fold_name": parent_report["fold_name"],
        "held_out_tasks": sorted(held_out_tasks),
        "source_tasks": sorted(source_tasks),
        "constant_gate": float(constant_gate),
        "parent_report_sha256": _sha256(parent_fold_root / "report.json"),
        "parent_prediction_sha256": _sha256(
            parent_fold_root / "held_out_predictions.npz"
        ),
        "git_commit": _git_commit(project_root),
    }
    checkpoint_path = temporary / "bcsg.pt"
    torch.save(checkpoint, checkpoint_path.with_suffix(".pt.tmp"))
    os.replace(checkpoint_path.with_suffix(".pt.tmp"), checkpoint_path)

    artifact_path = temporary / "held_out_gate_predictions.npz"
    artifact_temporary = artifact_path.with_suffix(".npz.tmp")
    with artifact_temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            schema_version=np.asarray([1], dtype=np.int64),
            fold_name=np.asarray([parent_report["fold_name"]]),
            held_out_tasks=np.asarray(sorted(held_out_tasks)),
            query_indices=np.asarray(held_indices, dtype=np.int64),
            chunk_ids=parent_artifact["chunk_ids"],
            tasks=parent_artifact["tasks"],
            group_ids=parent_artifact["group_ids"],
            target_actions=target.numpy(),
            target_masks=target_mask.numpy(),
            predicted_gate=predicted_gate.numpy(),
            optimal_gate=held_optimal_gate.numpy(),
            residual_energy=held_energy.numpy(),
            **{name: value.numpy() for name, value in predictions.items()},
        )
    os.replace(artifact_temporary, artifact_path)

    calibration_prediction = gate(calibration_features.to(device)).detach().cpu()
    calibration_weights = residual_energy / residual_energy.sum().clamp_min(1e-8)
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "transport": "frozen task-heldout QA-LRDT v2",
            "calibration_queries": "source-task validation only",
            "calibration_candidates": "source-task Demo only",
            "gate_inputs": "40D inference-only diagnostics",
            "gate_target": "analytic translation-risk optimal shrinkage",
            "bootstrap_unit": "task + variation + episode",
        },
        "fold_name": parent_report["fold_name"],
        "held_out_tasks": sorted(held_out_tasks),
        "source_tasks": sorted(source_tasks),
        "git_commit": _git_commit(project_root),
        "device": str(device),
        "cuda_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "config": asdict(config),
        "config_sha256": _sha256(config_path),
        "context_summary_sha256": context_hash,
        "action_summary_sha256": action_hash,
        "retriever_checkpoint_sha256": _sha256(retriever_checkpoint),
        "text_scores_sha256": _sha256(text_scores_path),
        "parent_report_sha256": checkpoint["parent_report_sha256"],
        "parent_prediction_sha256": checkpoint["parent_prediction_sha256"],
        "calibration_queries": len(calibration_indices),
        "held_out_eval_queries": len(held_indices),
        "constant_gate": float(constant_gate),
        "gate_parameters": sum(parameter.numel() for parameter in gate.parameters()),
        "combined_parameters": (
            sum(parameter.numel() for parameter in transport.parameters())
            + sum(parameter.numel() for parameter in gate.parameters())
        ),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "prediction_artifact_sha256": _sha256(artifact_path),
        "calibration": {
            "optimal_gate_mean": float(optimal_gate.mean()),
            "optimal_gate_zero_fraction": float((optimal_gate == 0.0).float().mean()),
            "optimal_gate_one_fraction": float((optimal_gate == 1.0).float().mean()),
            "predicted_gate_mean": float(calibration_prediction.mean()),
            "energy_weighted_gate_mse": float(
                (
                    calibration_weights
                    * (calibration_prediction - optimal_gate).square()
                ).sum()
            ),
        },
        "held_out_gate": {
            "predicted_mean": float(predicted_gate.mean()),
            "predicted_std": float(predicted_gate.std(unbiased=False)),
            "optimal_mean_evaluation_only": float(held_optimal_gate.mean()),
            "optimal_zero_fraction_evaluation_only": float(
                (held_optimal_gate == 0.0).float().mean()
            ),
        },
        "metrics": {
            name: _physical_metrics(prediction, target, target_mask)
            for name, prediction in predictions.items()
        },
        "paired_bootstrap": _bootstrap(
            predictions=predictions,
            target=target,
            mask=target_mask,
            group_ids=group_ids,
            config=config,
        ),
        "structural_audit": {
            "gate_min": float(predicted_gate.min()),
            "gate_max": float(predicted_gate.max()),
            "gate_zero_max_abs_error": float(
                (
                    apply_shrinkage(
                        held_copy,
                        held_transport,
                        torch.zeros_like(predicted_gate),
                        held_demo_mask,
                    )
                    - held_copy
                ).abs().max()
            ),
            "gate_one_max_abs_error": float(
                (
                    apply_shrinkage(
                        held_copy,
                        held_transport,
                        torch.ones_like(predicted_gate),
                        held_demo_mask,
                    )
                    - held_transport
                ).abs().max()
            ),
            "no_demo_max_abs_output": float(zero_output.abs().max()),
            "rotation_gripper_max_abs_error": float(
                (bcsg[..., 3:] - held_copy[..., 3:]).abs().max()
            ),
            "no_query_only_action_head": True,
            "parent_transport_frozen": True,
        },
        "latency": _latency(
            transport=transport,
            gate=gate,
            query=geometries[held_indices],
            demo=held_demo,
            actions=held_demo_actions,
            mask=held_demo_mask,
            iterations=config.latency_iterations,
            device=device,
        ),
    }
    (temporary / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (temporary / "TRAINING_COMPLETE").write_text(
        "rlbench_benefit_calibrated_shrinkage_v1\n",
        encoding="utf-8",
    )
    temporary.rename(output_root)
    atexit.unregister(cleanup_incomplete)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--context-root", type=Path, required=True)
    parser.add_argument("--action-root", type=Path, required=True)
    parser.add_argument("--retriever-checkpoint", type=Path, required=True)
    parser.add_argument("--text-scores", type=Path, required=True)
    parser.add_argument("--parent-fold-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA，但当前节点没有可用 GPU")
    run(
        project_root=arguments.project_root.resolve(),
        context_root=arguments.context_root.resolve(),
        action_root=arguments.action_root.resolve(),
        retriever_checkpoint=arguments.retriever_checkpoint.resolve(),
        text_scores_path=arguments.text_scores.resolve(),
        parent_fold_root=arguments.parent_fold_root.resolve(),
        output_root=arguments.output_root.resolve(),
        config_path=arguments.config.resolve(),
        config=GateTrainConfig.from_json(arguments.config.resolve()),
        device=device,
    )


if __name__ == "__main__":
    main()

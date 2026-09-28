"""在 ManiSkill canonical chunks 上训练并审计轻量 LJAT。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dev.pointnet.compare_retrievers import _sha256
from dev.predictor.jacobian_transport_model import (
    JacobianTransportConfig,
    LocalJacobianActionTransport,
)
from dev.predictor.train_action_chunks import (
    _denormalize_actions,
    _masked_sample_loss,
    _normalize_actions,
    _physical_metrics,
)
from dev.simulator.evaluate_maniskill_demo_prior import (
    _action_protocol,
    _diverse_top_k,
    _load_split,
)


@dataclass(frozen=True)
class TrainConfig:
    """预注册的 ManiSkill LJAT 训练配置。"""

    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    hidden_dim: int
    num_layers: int
    num_heads: int
    feedforward_dim: int
    dropout: float
    residual_limit: float
    retrieval_k: int
    bootstrap_resamples: int

    @classmethod
    def from_json(cls, path: Path) -> "TrainConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def __post_init__(self) -> None:
        positive = (
            self.epochs,
            self.batch_size,
            self.learning_rate,
            self.gradient_clip_norm,
            self.hidden_dim,
            self.num_layers,
            self.num_heads,
            self.feedforward_dim,
            self.residual_limit,
            self.retrieval_k,
            self.bootstrap_resamples,
        )
        if min(positive) <= 0:
            raise ValueError("训练配置中的正数参数必须大于零")
        if self.weight_decay < 0 or not 0.0 <= self.dropout < 1.0:
            raise ValueError("weight decay 或 dropout 配置非法")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim 必须能被 num_heads 整除")


class _FixedPairDataset(Dataset[dict[str, torch.Tensor]]):
    """固定 query--跨 episode Demo pair，避免隐式更换监督协议。"""

    def __init__(
        self,
        geometry: torch.Tensor,
        actions: torch.Tensor,
        demo_indices: torch.Tensor,
    ) -> None:
        if len(geometry) != len(actions) or demo_indices.shape != (len(actions),):
            raise ValueError("训练 pair 的 tensor 长度不一致")
        self.geometry = geometry.float()
        self.actions = actions.float()
        self.demo_indices = demo_indices.long()
        self.mask = torch.ones(actions.shape[:2], dtype=torch.bool)

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        demo = int(self.demo_indices[index])
        return {
            "query_geometry": self.geometry[index],
            "target_actions": self.actions[index],
            "target_mask": self.mask[index],
            "demo_geometry": self.geometry[demo],
            "demo_actions": self.actions[demo],
            "demo_mask": self.mask[demo],
        }


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


def _episode_ids(records: Sequence[dict[str, Any]]) -> torch.Tensor:
    return torch.tensor([int(record["episode"]) for record in records])


def _cross_episode_nearest(
    distances: torch.Tensor,
    query_episodes: torch.Tensor,
    candidate_episodes: torch.Tensor,
) -> torch.Tensor:
    if distances.shape != (len(query_episodes), len(candidate_episodes)):
        raise ValueError("distance matrix 与 episode IDs 不匹配")
    valid = query_episodes[:, None] != candidate_episodes[None, :]
    if not bool(valid.any(dim=1).all()):
        raise ValueError("至少一个 query 没有跨 episode 候选")
    return distances.masked_fill(~valid, torch.inf).argmin(dim=1)


def _selection_hash(indices: torch.Tensor) -> str:
    values = indices.detach().cpu().numpy().astype("<i8", copy=False)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _build_model(
    config: TrainConfig,
    horizon: int,
    geometry_dim: int,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> LocalJacobianActionTransport:
    model_config = JacobianTransportConfig(
        horizon=horizon,
        geometry_dim=geometry_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        feedforward_dim=config.feedforward_dim,
        dropout=config.dropout,
        residual_limit=config.residual_limit,
    )
    return LocalJacobianActionTransport(
        model_config,
        geometry_mean=mean,
        geometry_std=std,
    )


def _train(
    *,
    dataset: _FixedPairDataset,
    model: LocalJacobianActionTransport,
    config: TrainConfig,
    device: torch.device,
    log_path: Path,
) -> LocalJacobianActionTransport:
    """训练固定末轮模型，不读取 validation 指标。"""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
    )
    model.to(device)
    for epoch in range(config.epochs):
        generator = torch.Generator().manual_seed(config.seed + epoch)
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
        )
        model.train()
        loss_sum = 0.0
        examples = 0
        for batch in loader:
            values = {name: value.to(device) for name, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                values["query_geometry"],
                values["demo_geometry"],
                values["demo_actions"],
                values["demo_mask"],
            )
            sample_losses = _masked_sample_loss(
                prediction,
                values["target_actions"],
                values["target_mask"],
            )
            loss = sample_losses.mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            batch_size = len(prediction)
            loss_sum += float(loss.detach()) * batch_size
            examples += batch_size
        scheduler.step()
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == config.epochs:
            row = {
                "epoch": epoch + 1,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "smooth_l1_loss": loss_sum / examples,
            }
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
    return model.eval()


@torch.inference_mode()
def _predict(
    model: LocalJacobianActionTransport,
    query_geometry: torch.Tensor,
    demo_geometry: torch.Tensor,
    demo_actions: torch.Tensor,
    device: torch.device,
    *,
    valid_demo: bool = True,
) -> torch.Tensor:
    mask = torch.full(
        demo_actions.shape[:2],
        valid_demo,
        dtype=torch.bool,
        device=device,
    )
    return model(
        query_geometry.to(device),
        demo_geometry.to(device),
        demo_actions.to(device),
        mask,
    ).cpu()


def _per_query_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    pose_scales: torch.Tensor,
    translation_threshold_m: float,
    rotation_threshold_rad: float,
) -> dict[str, np.ndarray]:
    """返回 episode bootstrap 所需的 query 级完整指标。"""
    normalized_error = (prediction - target).abs().mean(dim=(1, 2))
    prediction_physical = _denormalize_actions(prediction, pose_scales)
    target_physical = _denormalize_actions(target, pose_scales)
    translation = torch.linalg.vector_norm(
        prediction_physical[..., :3] - target_physical[..., :3], dim=-1
    )
    rotation = torch.linalg.vector_norm(
        prediction_physical[..., 3:6] - target_physical[..., 3:6], dim=-1
    )
    gripper = (
        (prediction_physical[..., 6] >= 0.5)
        == (target_physical[..., 6] >= 0.5)
    )
    step = (
        (translation < translation_threshold_m)
        & (rotation < rotation_threshold_rad)
        & gripper
    )
    return {
        "normalized_mae": normalized_error.numpy(),
        "translation_l2_m": translation.mean(dim=1).numpy(),
        "rotation_l2_rad": rotation.mean(dim=1).numpy(),
        "gripper_accuracy": gripper.float().mean(dim=1).numpy(),
        "step_threshold_accuracy": step.float().mean(dim=1).numpy(),
        "chunk_threshold_accuracy": step.all(dim=1).float().numpy(),
    }


def _episode_bootstrap(
    *,
    reference: np.ndarray,
    candidate: np.ndarray,
    group_ids: Sequence[str],
    resamples: int,
    seed: int,
) -> dict[str, float | int]:
    """按 episode 成组重采样；delta 是 candidate - reference。"""
    if reference.shape != candidate.shape or len(reference) != len(group_ids):
        raise ValueError("bootstrap 输入长度不一致")
    groups = np.asarray(group_ids)
    unique = np.unique(groups)
    members = [np.flatnonzero(groups == group) for group in unique]
    rng = np.random.default_rng(seed)
    samples = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        chosen = rng.integers(0, len(members), size=len(members))
        rows = np.concatenate([members[group] for group in chosen])
        samples[index] = float(candidate[rows].mean() - reference[rows].mean())
    return {
        "candidate_minus_reference": float(candidate.mean() - reference.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "bootstrap_probability_delta_below_zero": float(np.mean(samples < 0.0)),
        "num_episode_groups": len(unique),
        "num_resamples": resamples,
    }


def _bootstrap_comparison(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    target: torch.Tensor,
    group_ids: Sequence[str],
    config: TrainConfig,
    pose_scales: torch.Tensor,
    translation_threshold_m: float,
    rotation_threshold_rad: float,
    *,
    seed_offset: int,
) -> dict[str, dict[str, float | int]]:
    metric_options = {
        "pose_scales": pose_scales,
        "translation_threshold_m": translation_threshold_m,
        "rotation_threshold_rad": rotation_threshold_rad,
    }
    reference_metrics = _per_query_metrics(reference, target, **metric_options)
    candidate_metrics = _per_query_metrics(candidate, target, **metric_options)
    return {
        name: _episode_bootstrap(
            reference=reference_metrics[name],
            candidate=candidate_metrics[name],
            group_ids=group_ids,
            resamples=config.bootstrap_resamples,
            seed=config.seed + seed_offset + offset,
        )
        for offset, name in enumerate(reference_metrics)
    }


def _held_gripper_zero(
    geometry: torch.Tensor,
    target_actions: torch.Tensor,
    pose_scales: torch.Tensor,
) -> torch.Tensor:
    physical = torch.zeros_like(target_actions)
    physical[..., 6] = geometry[:, 15, None].clamp(0.0, 1.0)
    return _normalize_actions(physical, pose_scales)


def _decision(
    metrics: dict[str, dict[str, float]],
    bootstrap: dict[str, dict[str, dict[str, float | int]]],
    train_metrics: dict[str, dict[str, float]],
) -> dict[str, bool | str]:
    copy = metrics["geometry_rank1_copy"]
    transport = metrics["local_jacobian_transport"]
    comparison = bootstrap["transport_minus_copy"]
    pose_stable = any(
        float(comparison[name]["ci95_high"]) < 0.0
        for name in ("translation_l2_m", "rotation_l2_rad")
    )
    t1 = transport["normalized_mae"] < copy["normalized_mae"] and pose_stable
    t2 = transport["gripper_accuracy"] > copy["gripper_accuracy"]
    t3 = transport["chunk_threshold_accuracy"] > max(
        metrics["zero_pose_hold_gripper"]["chunk_threshold_accuracy"],
        copy["chunk_threshold_accuracy"],
    )
    train_improves = (
        train_metrics["local_jacobian_transport"]["normalized_mae"]
        < train_metrics["geometry_rank1_copy"]["normalized_mae"]
    )
    validation_does_not_improve = (
        transport["normalized_mae"] >= copy["normalized_mae"]
    )
    return {
        "T1_pose_transport": t1,
        "T2_gripper_transition": t2,
        "T3_enter_closed_loop": t3,
        "T4_split_gripper_hazard": bool(t1 and not t2),
        "T5_train_validation_overfit": bool(
            train_improves and validation_does_not_improve
        ),
        "next_step": (
            "closed_loop_receding_horizon"
            if t1 and t2 and t3
            else "split_continuous_pose_transport_and_discrete_gripper_hazard"
            if t1 and not t2
            else "increase_episode_or_phase_diversity_without_capacity_tuning"
        ),
    }


def run(
    *,
    project_root: Path,
    data_root: Path,
    output_root: Path,
    config_path: Path,
    config: TrainConfig,
    device: torch.device,
) -> None:
    if output_root.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output_root}")
    temporary = output_root.with_name(f".{output_root.name}.incomplete-{os.getpid()}")
    temporary.mkdir(parents=True)
    _seed_everything(config.seed)

    (
        pose_scales,
        translation_threshold_m,
        rotation_threshold_rad,
        action_representation,
    ) = _action_protocol(data_root)
    train_records, train_geometry, train_actions = _load_split(
        data_root, "train", pose_scales
    )
    val_records, val_geometry, val_actions = _load_split(
        data_root, "val", pose_scales
    )
    if train_actions.shape[1:] != val_actions.shape[1:]:
        raise ValueError("train/validation action shape 不一致")
    geometry_mean = train_geometry.mean(dim=0)
    geometry_std = train_geometry.std(dim=0, unbiased=False).clamp_min(1e-4)
    standardized_train = (train_geometry - geometry_mean) / geometry_std
    standardized_val = (val_geometry - geometry_mean) / geometry_std
    scale = float(np.sqrt(train_geometry.shape[1]))
    train_distances = torch.cdist(standardized_train, standardized_train) / scale
    val_distances = torch.cdist(standardized_val, standardized_train) / scale
    train_episodes = _episode_ids(train_records)
    train_demo_indices = _cross_episode_nearest(
        train_distances,
        train_episodes,
        train_episodes,
    )
    val_candidates = _diverse_top_k(
        val_distances,
        train_records,
        config.retrieval_k,
    )
    val_demo_indices = val_candidates[:, 0]
    far_demo_indices = val_distances.argmax(dim=1)

    dataset = _FixedPairDataset(
        train_geometry,
        train_actions,
        train_demo_indices,
    )
    model = _build_model(
        config,
        horizon=int(train_actions.shape[1]),
        geometry_dim=int(train_geometry.shape[1]),
        mean=geometry_mean,
        std=geometry_std,
    )
    model = _train(
        dataset=dataset,
        model=model,
        config=config,
        device=device,
        log_path=temporary / "training_metrics.jsonl",
    )

    ones_train = torch.ones(train_actions.shape[:2], dtype=torch.bool)
    ones_val = torch.ones(val_actions.shape[:2], dtype=torch.bool)
    train_copy = train_actions[train_demo_indices]
    train_transport = _predict(
        model,
        train_geometry,
        train_geometry[train_demo_indices],
        train_copy,
        device,
    )
    val_copy = train_actions[val_demo_indices]
    val_transport = _predict(
        model,
        val_geometry,
        train_geometry[val_demo_indices],
        val_copy,
        device,
    )
    far_actions = train_actions[far_demo_indices]
    far_transport = _predict(
        model,
        val_geometry,
        train_geometry[far_demo_indices],
        far_actions,
        device,
    )
    no_demo = _predict(
        model,
        val_geometry,
        torch.zeros_like(val_geometry),
        torch.zeros_like(val_actions),
        device,
        valid_demo=False,
    )
    hypotheses = train_actions[val_candidates]
    oracle_errors = (hypotheses - val_actions[:, None]).abs().mean(dim=(2, 3))
    oracle_indices = oracle_errors.argmin(dim=1)
    oracle = hypotheses[torch.arange(len(val_actions)), oracle_indices]
    zero = _held_gripper_zero(val_geometry, val_actions, pose_scales)

    metric_options = {
        "pose_scales": pose_scales,
        "translation_threshold_m": translation_threshold_m,
        "rotation_threshold_rad": rotation_threshold_rad,
    }

    train_metrics = {
        "geometry_rank1_copy": _physical_metrics(
            train_copy, train_actions, ones_train, **metric_options
        ),
        "local_jacobian_transport": _physical_metrics(
            train_transport, train_actions, ones_train, **metric_options
        ),
    }
    predictions = {
        "zero_pose_hold_gripper": zero,
        "geometry_rank1_copy": val_copy,
        "local_jacobian_transport": val_transport,
        "far_geometry_transport": far_transport,
        f"oracle_best_in_{config.retrieval_k}": oracle,
    }
    metrics = {
        name: _physical_metrics(
            prediction, val_actions, ones_val, **metric_options
        )
        for name, prediction in predictions.items()
    }
    group_ids = [f"PickCube-v1:{record['episode']}" for record in val_records]
    bootstrap = {
        "transport_minus_copy": _bootstrap_comparison(
            val_copy,
            val_transport,
            val_actions,
            group_ids,
            config,
            pose_scales,
            translation_threshold_m,
            rotation_threshold_rad,
            seed_offset=0,
        ),
        "transport_minus_zero": _bootstrap_comparison(
            zero,
            val_transport,
            val_actions,
            group_ids,
            config,
            pose_scales,
            translation_threshold_m,
            rotation_threshold_rad,
            seed_offset=100,
        ),
    }
    decision = _decision(metrics, bootstrap, train_metrics)

    checkpoint = {
        "model": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "model_config": asdict(model.config),
        "train_config": asdict(config),
        "geometry_mean": geometry_mean,
        "geometry_std": geometry_std,
        "data_summary_sha256": _sha256(data_root / "summary.json"),
        "train_manifest_sha256": _sha256(data_root / "manifest-train.jsonl"),
        "train_demo_selection_sha256": _selection_hash(train_demo_indices),
        "action_pose_scales": pose_scales,
        "action_representation": action_representation,
        "git_commit": _git_commit(project_root),
    }
    checkpoint_path = temporary / "local_jacobian_transport.pt"
    checkpoint_tmp = temporary / "local_jacobian_transport.pt.tmp"
    torch.save(checkpoint, checkpoint_tmp)
    os.replace(checkpoint_tmp, checkpoint_path)

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "train_demo": (
                "standardized 17D geometry top-1 from another train episode"
            ),
            "validation_demo": (
                "standardized 17D geometry top-K from train episodes only"
            ),
            "checkpoint_selection": (
                "fixed final epoch; validation never used for selection"
            ),
            "predictor": "demo-dependent Local Jacobian Action Transport",
            "zero_baseline": "zero pose delta plus current observed gripper",
            "oracle": "analysis-only target-action selection within geometry top-K",
            "bootstrap_unit": "validation episode",
        },
        "device": str(device),
        "cuda_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "config": asdict(config),
        "action_protocol": {
            "representation": action_representation,
            "pose_scales": pose_scales.tolist(),
            "translation_threshold_m": translation_threshold_m,
            "rotation_threshold_rad": rotation_threshold_rad,
        },
        "git_commit": _git_commit(project_root),
        "data_summary_sha256": _sha256(data_root / "summary.json"),
        "config_sha256": _sha256(config_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "train_manifest_sha256": _sha256(data_root / "manifest-train.jsonl"),
        "validation_manifest_sha256": _sha256(data_root / "manifest-val.jsonl"),
        "train_queries": len(train_actions),
        "validation_queries": len(val_actions),
        "validation_episodes": len(set(group_ids)),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "selection": {
            "train_demo_indices_sha256": _selection_hash(train_demo_indices),
            "validation_top1_indices_sha256": _selection_hash(val_demo_indices),
            "validation_top1_distance_mean": float(
                val_distances[torch.arange(len(val_actions)), val_demo_indices].mean()
            ),
            "validation_far_distance_mean": float(
                val_distances[torch.arange(len(val_actions)), far_demo_indices].mean()
            ),
            "oracle_rank_counts": {
                str(rank + 1): int((oracle_indices == rank).sum())
                for rank in range(config.retrieval_k)
            },
        },
        "train_metrics": train_metrics,
        "validation_metrics": metrics,
        "paired_episode_bootstrap": bootstrap,
        "demo_dependency": {
            "no_demo_max_abs_normalized_output": float(no_demo.abs().max()),
            "retrieved_vs_far_prediction_mean_abs_difference": float(
                (val_transport - far_transport).abs().mean()
            ),
            "note": (
                "no-demo normalized zero maps to an artificial 0.5 gripper, "
                "so it is not used as the physical zero baseline"
            ),
        },
        "preregistered_decision": decision,
    }
    report_path = temporary / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.joinpath("TRAINING_COMPLETE").write_text(
        "maniskill_local_jacobian_transport_v1\n",
        encoding="utf-8",
    )
    temporary.rename(output_root)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
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
        data_root=arguments.data_root.resolve(),
        output_root=arguments.output_root.resolve(),
        config_path=arguments.config.resolve(),
        config=TrainConfig.from_json(arguments.config.resolve()),
        device=device,
    )


if __name__ == "__main__":
    main()

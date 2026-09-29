"""检验推理时可见信号能否辨别 fixed Demo mixture 相对 rank-1 的收益。

只使用冻结模型前向和 source-train labels 拟合固定 ridge；source-validation
不参与模型或策略选择。
"""

from __future__ import annotations

import argparse
import atexit
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from dev.pointnet.compare_retrievers import _sha256
from dev.predictor.evaluate_jacobian_significance import _episode_bootstrap
from dev.predictor.retriever_anchored_demo_mixture import mix_demo_hypotheses
from dev.predictor.train_action_chunks import _denormalize_actions
from dev.simulator.train_causal_history_gate import (
    EXPECTED_TASKS,
    _checkpoint_bank_indices,
    _load_frozen_models,
)
from dev.simulator.train_maniskill_low_rank_transport import (
    TaskData,
    _load_task,
    _selection_hash,
)
from dev.simulator.train_retriever_anchored_demo_mixture import (
    PreparedSplit,
    _prepare_split,
    _selection_digest,
)


FEATURE_NAMES = (
    "action_span_diameter",
    "mean_pairwise_action_disagreement",
    "rank1_retrieval_distance",
    "rank2_rank1_distance_gap",
    "retrieval_distance_mean",
    "retrieval_distance_std",
    "retrieval_prior_entropy",
    "previous_candidate_set_churn",
    "previous_top1_episode_switch",
    "previous_query_geometry_step_l2",
    "has_previous_query",
)


@dataclass(frozen=True)
class IdentifiabilityConfig:
    """结果揭盲前冻结的可辨识性协议。"""

    schema_version: str
    seed: int
    source_checkpoint_sha256: str
    candidate_count: int
    expected_episodes_per_split_per_task: int
    ridge_l2: float
    bootstrap_resamples: int
    feature_names: list[str]
    query_summary_sha256: dict[str, str]
    query_audit_sha256: dict[str, str]
    pooled_auc_minimum: float
    task_heldout_mean_auc_minimum: float
    minimum_improved_tasks: int
    minimum_positive_task_heldout_auc: int

    @classmethod
    def from_json(cls, path: Path) -> "IdentifiabilityConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def __post_init__(self) -> None:
        if self.schema_version != "maniskill-demo-mixture-identifiability-v1":
            raise ValueError("未知 Demo mixture identifiability schema")
        if self.candidate_count != 4:
            raise ValueError("identifiability v1 固定 K=4")
        if (
            self.expected_episodes_per_split_per_task <= 0
            or self.ridge_l2 <= 0.0
            or self.bootstrap_resamples <= 0
        ):
            raise ValueError("episode、ridge 和 bootstrap 配置必须为正")
        if tuple(self.feature_names) != FEATURE_NAMES:
            raise ValueError("identifiability feature schema 不匹配")
        if not 0.5 < self.pooled_auc_minimum <= 1.0:
            raise ValueError("pooled AUC 门槛非法")
        if not 0.5 < self.task_heldout_mean_auc_minimum <= 1.0:
            raise ValueError("task-heldout AUC 门槛非法")
        for mapping in (self.query_summary_sha256, self.query_audit_sha256):
            if set(mapping) != EXPECTED_TASKS:
                raise ValueError("query hashes 必须覆盖冻结的三个 tasks")
            if any(len(value) != 64 for value in mapping.values()):
                raise ValueError("query SHA256 非法")
        if len(self.source_checkpoint_sha256) != 64:
            raise ValueError("source checkpoint SHA256 非法")


@dataclass(frozen=True)
class SplitTable:
    """一个 task/split 的冻结特征、标签与基线误差。"""

    task_id: str
    split: str
    group_ids: list[str]
    chunk_ids: list[str]
    features: np.ndarray
    benefit_m: np.ndarray
    rank1_error_m: np.ndarray
    prior_error_m: np.ndarray
    candidate_indices: np.ndarray
    candidate_episodes: np.ndarray
    prior: np.ndarray


@dataclass(frozen=True)
class RidgeModel:
    """train-only 标准化的固定线性 ridge。"""

    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_mean: float
    coefficients: np.ndarray

    def predict(self, features: np.ndarray) -> np.ndarray:
        normalized = (features - self.feature_mean) / self.feature_std
        return self.target_mean + normalized @ self.coefficients


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _task_map(tasks: Sequence[TaskData], label: str) -> dict[str, TaskData]:
    mapping = {task.task_id: task for task in tasks}
    if len(tasks) != 3 or set(mapping) != EXPECTED_TASKS:
        raise ValueError(f"{label} 必须恰好包含冻结的三个 ManiSkill tasks")
    return mapping


def _episode_value(record: Mapping[str, Any]) -> str:
    """优先使用源采集 seed，避免 split-local episode id 冲突。"""
    if record.get("source_episode_seed") is not None:
        return f"seed-{int(record['source_episode_seed'])}"
    return f"episode-{int(record['episode'])}"


def _split_groups(
    task_id: str,
    split: str,
    records: Sequence[Mapping[str, Any]],
) -> list[str]:
    return [f"{task_id}:{split}:{_episode_value(record)}" for record in records]


def _assert_record_order(records: Sequence[Mapping[str, Any]]) -> None:
    """历史特征只允许引用 manifest 中同 episode 的上一条 query。"""
    previous_by_episode: dict[str, int] = {}
    chunk_ids = set()
    for record in records:
        chunk_id = str(record["chunk_id"])
        if chunk_id in chunk_ids:
            raise ValueError(f"query chunk_id 重复：{chunk_id}")
        chunk_ids.add(chunk_id)
        episode = _episode_value(record)
        frame = int(record["frame"])
        if episode in previous_by_episode and frame <= previous_by_episode[episode]:
            raise ValueError("query manifest 未按 episode 内 frame 严格递增")
        previous_by_episode[episode] = frame


def _translation_error_m(
    prediction: torch.Tensor,
    target: torch.Tensor,
    pose_scales: torch.Tensor,
) -> np.ndarray:
    prediction_physical = _denormalize_actions(prediction, pose_scales)
    target_physical = _denormalize_actions(target, pose_scales)
    error = torch.linalg.vector_norm(
        prediction_physical[..., :3] - target_physical[..., :3],
        dim=-1,
    ).mean(dim=1)
    return error.cpu().numpy().astype(np.float64, copy=False)


def _pairwise_action_features(
    hypotheses: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    translation = hypotheses[..., :3]
    pairwise = torch.linalg.vector_norm(
        translation[:, :, None] - translation[:, None, :],
        dim=-1,
    )
    span = pairwise.amax(dim=(1, 2, 3))
    candidates = hypotheses.shape[1]
    off_diagonal = ~torch.eye(candidates, dtype=torch.bool)
    disagreement = pairwise[:, off_diagonal].mean(dim=(1, 2))
    return span.numpy(), disagreement.numpy()


def _history_features(
    prepared: PreparedSplit,
    bank_records: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _assert_record_order(prepared.records)
    candidate_episodes = np.asarray(
        [
            [int(bank_records[index]["episode"]) for index in row]
            for row in prepared.candidate_indices.tolist()
        ],
        dtype=np.int64,
    )
    churn = np.zeros(len(prepared.records), dtype=np.float64)
    top1_switch = np.zeros(len(prepared.records), dtype=np.float64)
    geometry_step = np.zeros(len(prepared.records), dtype=np.float64)
    has_previous = np.zeros(len(prepared.records), dtype=np.float64)
    for index in range(1, len(prepared.records)):
        current = prepared.records[index]
        previous = prepared.records[index - 1]
        if _episode_value(current) != _episode_value(previous):
            continue
        current_set = set(candidate_episodes[index].tolist())
        previous_set = set(candidate_episodes[index - 1].tolist())
        union = current_set | previous_set
        churn[index] = 1.0 - len(current_set & previous_set) / len(union)
        top1_switch[index] = float(
            candidate_episodes[index, 0] != candidate_episodes[index - 1, 0]
        )
        geometry_step[index] = float(
            torch.linalg.vector_norm(
                prepared.query_geometry[index]
                - prepared.query_geometry[index - 1]
            )
        )
        has_previous[index] = 1.0
    return candidate_episodes, churn, top1_switch, geometry_step, has_previous


def _make_table(
    *,
    prepared: PreparedSplit,
    task: TaskData,
    split: str,
    bank_records: Sequence[Mapping[str, Any]],
) -> SplitTable:
    mask = prepared.mask
    rank1 = prepared.hypotheses[:, 0]
    prior_prediction = mix_demo_hypotheses(
        prepared.hypotheses,
        prepared.prior,
        mask,
    )
    rank1_error = _translation_error_m(rank1, prepared.target, task.pose_scales)
    prior_error = _translation_error_m(
        prior_prediction,
        prepared.target,
        task.pose_scales,
    )
    span, disagreement = _pairwise_action_features(prepared.hypotheses)
    (
        candidate_episodes,
        churn,
        top1_switch,
        geometry_step,
        has_previous,
    ) = _history_features(prepared, bank_records)
    distances = prepared.distances.numpy().astype(np.float64, copy=False)
    prior = prepared.prior.numpy().astype(np.float64, copy=False)
    entropy = -(prior * np.log(np.clip(prior, 1e-12, None))).sum(axis=1)
    features = np.column_stack(
        (
            span,
            disagreement,
            distances[:, 0],
            distances[:, 1] - distances[:, 0],
            distances.mean(axis=1),
            distances.std(axis=1),
            entropy,
            churn,
            top1_switch,
            geometry_step,
            has_previous,
        )
    ).astype(np.float64, copy=False)
    if features.shape[1] != len(FEATURE_NAMES) or not np.isfinite(features).all():
        raise RuntimeError("identifiability features 非有限或维度错误")
    return SplitTable(
        task_id=prepared.task_id,
        split=split,
        group_ids=_split_groups(prepared.task_id, split, prepared.records),
        chunk_ids=[str(record["chunk_id"]) for record in prepared.records],
        features=features,
        benefit_m=rank1_error - prior_error,
        rank1_error_m=rank1_error,
        prior_error_m=prior_error,
        candidate_indices=prepared.candidate_indices.numpy(),
        candidate_episodes=candidate_episodes,
        prior=prior,
    )


def _concatenate(tables: Sequence[SplitTable], split: str) -> SplitTable:
    if not tables or any(table.split != split for table in tables):
        raise ValueError("待合并 tables 的 split 不一致")
    return SplitTable(
        task_id="pooled",
        split=split,
        group_ids=sum((table.group_ids for table in tables), []),
        chunk_ids=sum((table.chunk_ids for table in tables), []),
        features=np.concatenate([table.features for table in tables]),
        benefit_m=np.concatenate([table.benefit_m for table in tables]),
        rank1_error_m=np.concatenate([table.rank1_error_m for table in tables]),
        prior_error_m=np.concatenate([table.prior_error_m for table in tables]),
        candidate_indices=np.concatenate(
            [table.candidate_indices for table in tables]
        ),
        candidate_episodes=np.concatenate(
            [table.candidate_episodes for table in tables]
        ),
        prior=np.concatenate([table.prior for table in tables]),
    )


def _episode_equal_weights(group_ids: Sequence[str]) -> np.ndarray:
    groups = np.asarray(group_ids)
    unique, inverse, counts = np.unique(
        groups,
        return_inverse=True,
        return_counts=True,
    )
    if not len(unique):
        raise ValueError("ridge groups 不能为空")
    return 1.0 / counts[inverse].astype(np.float64)


def _fit_ridge(table: SplitTable, l2: float) -> RidgeModel:
    weights = _episode_equal_weights(table.group_ids)
    total = weights.sum()
    mean = (weights[:, None] * table.features).sum(axis=0) / total
    variance = (
        weights[:, None] * (table.features - mean) ** 2
    ).sum(axis=0) / total
    std = np.sqrt(variance)
    std = np.maximum(std, 1e-8)
    normalized = (table.features - mean) / std
    target_mean = float((weights * table.benefit_m).sum() / total)
    centered_target = table.benefit_m - target_mean
    weighted_x = normalized * np.sqrt(weights[:, None])
    weighted_y = centered_target * np.sqrt(weights)
    system = weighted_x.T @ weighted_x + l2 * np.eye(normalized.shape[1])
    coefficients = np.linalg.solve(system, weighted_x.T @ weighted_y)
    return RidgeModel(
        feature_mean=mean,
        feature_std=std,
        target_mean=target_mean,
        coefficients=coefficients,
    )


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return math.nan
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    rank_sum = ranks[labels].sum()
    return float(
        (rank_sum - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def _balanced_accuracy(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    predictions = np.asarray(scores) > 0.0
    if labels.all() or (~labels).all():
        return math.nan
    sensitivity = predictions[labels].mean()
    specificity = (~predictions[~labels]).mean()
    return float(0.5 * (sensitivity + specificity))


def _bootstrap_auc(
    *,
    labels: np.ndarray,
    scores: np.ndarray,
    group_ids: Sequence[str],
    resamples: int,
    seed: int,
) -> dict[str, float | int]:
    groups = np.asarray(group_ids)
    unique = np.unique(groups)
    members = [np.flatnonzero(groups == group) for group in unique]
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(resamples):
        chosen = rng.integers(0, len(members), size=len(members))
        indices = np.concatenate([members[index] for index in chosen])
        value = _roc_auc(labels[indices], scores[indices])
        if math.isfinite(value):
            samples.append(value)
    if len(samples) < 0.99 * resamples:
        raise RuntimeError("超过 1% bootstrap AUC samples 缺少正/负类")
    values = np.asarray(samples)
    return {
        "auc": _roc_auc(labels, scores),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
        "num_episode_groups": len(unique),
        "num_valid_resamples": len(values),
    }


def _selector_errors(table: SplitTable, scores: np.ndarray) -> np.ndarray:
    return np.where(scores > 0.0, table.prior_error_m, table.rank1_error_m)


def _table_metrics(
    table: SplitTable,
    scores: np.ndarray,
    constant_policy: str,
) -> dict[str, Any]:
    labels = table.benefit_m > 0.0
    selector = _selector_errors(table, scores)
    constant = (
        table.prior_error_m
        if constant_policy == "retriever_prior_mixture"
        else table.rank1_error_m
    )
    return {
        "queries": len(labels),
        "episodes": len(set(table.group_ids)),
        "positive_rate": float(labels.mean()),
        "auc": _roc_auc(labels, scores),
        "balanced_accuracy": _balanced_accuracy(labels, scores),
        "choose_prior_rate": float((scores > 0.0).mean()),
        "translation_l2_m": {
            "rank1_bcsg": float(table.rank1_error_m.mean()),
            "retriever_prior_mixture": float(table.prior_error_m.mean()),
            "selector": float(selector.mean()),
            "train_selected_constant": float(constant.mean()),
            "oracle_binary_selector": float(
                np.minimum(table.rank1_error_m, table.prior_error_m).mean()
            ),
        },
        "selector_minus_train_selected_constant_m": float(
            selector.mean() - constant.mean()
        ),
    }


def _model_report(model: RidgeModel) -> dict[str, Any]:
    return {
        "feature_mean": model.feature_mean.tolist(),
        "feature_std": model.feature_std.tolist(),
        "target_mean_m": model.target_mean,
        "coefficients": model.coefficients.tolist(),
        "parameter_count": len(model.coefficients) + 1,
    }


@torch.inference_mode()
def run(
    *,
    project_root: Path,
    bank_roots: Sequence[Path],
    query_roots: Sequence[Path],
    source_checkpoint_path: Path,
    config_path: Path,
    output_root: Path,
    config: IdentifiabilityConfig,
    device: torch.device,
) -> None:
    if output_root.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output_root}")
    if _sha256(source_checkpoint_path) != config.source_checkpoint_sha256:
        raise ValueError("source checkpoint SHA256 不匹配")
    temporary = output_root.with_name(f".{output_root.name}.incomplete-{os.getpid()}")
    temporary.mkdir(parents=True)

    def cleanup() -> None:
        shutil.rmtree(temporary, ignore_errors=True)

    atexit.register(cleanup)
    checkpoint = torch.load(source_checkpoint_path, map_location="cpu")
    transport, gate = _load_frozen_models(checkpoint, device)
    bank_tasks = _task_map([_load_task(root) for root in bank_roots], "bank roots")
    query_tasks = _task_map(
        [_load_task(root) for root in query_roots],
        "query roots",
    )
    tables: dict[str, dict[str, SplitTable]] = {}
    data_audit: dict[str, Any] = {}
    audit_passed = True
    for task_id in sorted(EXPECTED_TASKS):
        bank = bank_tasks[task_id]
        query = query_tasks[task_id]
        bank_summary = _sha256(bank.root / "summary.json")
        query_summary = _sha256(query.root / "summary.json")
        query_audit = _sha256(query.root / "audit.json")
        if checkpoint["data_summary_sha256"][task_id] != bank_summary:
            raise ValueError(f"{task_id} bank summary 与 source checkpoint 不匹配")
        if query_summary != config.query_summary_sha256[task_id]:
            raise ValueError(f"{task_id} query summary SHA256 不匹配")
        if query_audit != config.query_audit_sha256[task_id]:
            raise ValueError(f"{task_id} query audit SHA256 不匹配")
        if bank.action_representation != query.action_representation:
            raise ValueError(f"{task_id} bank/query action representation 不匹配")
        if not torch.equal(bank.pose_scales.float(), query.pose_scales.float()):
            raise ValueError(f"{task_id} bank/query pose scales 不匹配")
        bank_indices = _checkpoint_bank_indices(
            bank.train_records,
            checkpoint["bank_chunk_ids"][task_id],
        )
        bank_records = [bank.train_records[index] for index in bank_indices.tolist()]
        bank_geometry = bank.train_geometry[bank_indices]
        bank_actions = bank.train_actions[bank_indices]
        split_inputs = {
            "train": (query.train_records, query.train_geometry, query.train_actions),
            "val": (query.val_records, query.val_geometry, query.val_actions),
        }
        task_tables = {}
        selection_hashes = {}
        episode_sets: dict[str, set[str]] = {}
        for split, (records, geometry, actions) in split_inputs.items():
            prepared = _prepare_split(
                task_id=task_id,
                records=records,
                query_geometry=geometry,
                target=actions,
                bank_records=bank_records,
                bank_geometry=bank_geometry,
                bank_actions=bank_actions,
                transport=transport,
                gate=gate,
                candidate_count=config.candidate_count,
                device=device,
            )
            task_tables[split] = _make_table(
                prepared=prepared,
                task=query,
                split=split,
                bank_records=bank_records,
            )
            selection_hashes[split] = _selection_digest(
                prepared.candidate_indices
            )
            episode_sets[split] = {_episode_value(record) for record in records}
        split_counts_ok = all(
            len(values) == config.expected_episodes_per_split_per_task
            for values in episode_sets.values()
        )
        disjoint = episode_sets["train"].isdisjoint(episode_sets["val"])
        audit_passed &= split_counts_ok and disjoint
        tables[task_id] = task_tables
        data_audit[task_id] = {
            "bank_summary_sha256": bank_summary,
            "bank_chunks": len(bank_indices),
            "bank_selection_sha256": _selection_hash(bank_indices),
            "query_summary_sha256": query_summary,
            "query_audit_sha256": query_audit,
            "train_queries": len(task_tables["train"].benefit_m),
            "validation_queries": len(task_tables["val"].benefit_m),
            "train_episodes": len(episode_sets["train"]),
            "validation_episodes": len(episode_sets["val"]),
            "train_validation_episode_disjoint": disjoint,
            "expected_split_episode_counts": split_counts_ok,
            "candidate_selection_sha256": selection_hashes,
        }

    train_tables = [tables[task]["train"] for task in sorted(tables)]
    validation_tables = [tables[task]["val"] for task in sorted(tables)]
    pooled_train = _concatenate(train_tables, "train")
    pooled_validation = _concatenate(validation_tables, "val")
    model = _fit_ridge(pooled_train, config.ridge_l2)
    train_scores = model.predict(pooled_train.features)
    validation_scores = model.predict(pooled_validation.features)
    constant_policy = (
        "retriever_prior_mixture"
        if pooled_train.prior_error_m.mean() < pooled_train.rank1_error_m.mean()
        else "rank1_bcsg"
    )
    train_metrics = _table_metrics(
        pooled_train,
        train_scores,
        constant_policy,
    )
    validation_metrics = _table_metrics(
        pooled_validation,
        validation_scores,
        constant_policy,
    )
    auc_bootstrap = _bootstrap_auc(
        labels=pooled_validation.benefit_m > 0.0,
        scores=validation_scores,
        group_ids=pooled_validation.group_ids,
        resamples=config.bootstrap_resamples,
        seed=config.seed,
    )
    selector_error = _selector_errors(pooled_validation, validation_scores)
    constant_error = (
        pooled_validation.prior_error_m
        if constant_policy == "retriever_prior_mixture"
        else pooled_validation.rank1_error_m
    )
    selector_bootstrap = _episode_bootstrap(
        reference=constant_error,
        candidate=selector_error,
        group_ids=pooled_validation.group_ids,
        resamples=config.bootstrap_resamples,
        seed=config.seed + 1,
    )
    per_task = {}
    improved_tasks = 0
    for task_id in sorted(tables):
        table = tables[task_id]["val"]
        scores = model.predict(table.features)
        metrics = _table_metrics(table, scores, constant_policy)
        improved = metrics["selector_minus_train_selected_constant_m"] < 0.0
        improved_tasks += int(improved)
        metrics["selector_better_than_train_selected_constant"] = improved
        per_task[task_id] = metrics

    task_heldout = {}
    heldout_auc_values = []
    for heldout in sorted(tables):
        fit_table = _concatenate(
            [
                tables[task]["train"]
                for task in sorted(tables)
                if task != heldout
            ],
            "train",
        )
        heldout_model = _fit_ridge(fit_table, config.ridge_l2)
        heldout_table = tables[heldout]["val"]
        heldout_scores = heldout_model.predict(heldout_table.features)
        auc = _roc_auc(heldout_table.benefit_m > 0.0, heldout_scores)
        heldout_auc_values.append(auc)
        task_heldout[heldout] = {
            "auc": auc,
            "balanced_accuracy": _balanced_accuracy(
                heldout_table.benefit_m > 0.0,
                heldout_scores,
            ),
            "choose_prior_rate": float((heldout_scores > 0.0).mean()),
            "model": _model_report(heldout_model),
        }
    heldout_mean_auc = float(np.mean(heldout_auc_values))
    heldout_positive_count = sum(value > 0.5 for value in heldout_auc_values)

    causal_feature_audit = {
        "target_or_future_action_used": False,
        "future_observation_used": False,
        "radm_posterior_used": False,
        "task_identity_feature_used": False,
        "history_restricted_to_previous_same_episode_query": True,
        "feature_schema_exact": tuple(config.feature_names) == FEATURE_NAMES,
    }
    criteria = {
        "i1_pooled_validation_auc_point": (
            auc_bootstrap["auc"] >= config.pooled_auc_minimum
        ),
        "i1_pooled_validation_auc_ci_above_chance": (
            auc_bootstrap["ci95_low"] > 0.5
        ),
        "i2_selector_significantly_better_than_train_selected_constant": (
            selector_bootstrap["ci95_high"] < 0.0
        ),
        "i3_at_least_two_tasks_improve": (
            improved_tasks >= config.minimum_improved_tasks
        ),
        "i4_task_heldout_mean_auc": (
            heldout_mean_auc >= config.task_heldout_mean_auc_minimum
        ),
        "i4_at_least_two_task_heldout_auc_above_chance": (
            heldout_positive_count >= config.minimum_positive_task_heldout_auc
        ),
        "i5_data_and_causal_feature_audit": (
            audit_passed
            and not causal_feature_audit["target_or_future_action_used"]
            and not causal_feature_audit["future_observation_used"]
            and not causal_feature_audit["radm_posterior_used"]
            and not causal_feature_audit["task_identity_feature_used"]
            and causal_feature_audit[
                "history_restricted_to_previous_same_episode_query"
            ]
            and causal_feature_audit["feature_schema_exact"]
        ),
    }

    artifact_path = temporary / "features_and_predictions.npz"
    task_values = []
    split_values = []
    group_values = []
    chunk_values = []
    feature_values = []
    benefit_values = []
    rank1_values = []
    prior_error_values = []
    score_values = []
    candidate_index_values = []
    candidate_episode_values = []
    prior_values = []
    for task_id in sorted(tables):
        for split in ("train", "val"):
            table = tables[task_id][split]
            scores = model.predict(table.features)
            count = len(scores)
            task_values.extend([task_id] * count)
            split_values.extend([split] * count)
            group_values.extend(table.group_ids)
            chunk_values.extend(table.chunk_ids)
            feature_values.append(table.features)
            benefit_values.append(table.benefit_m)
            rank1_values.append(table.rank1_error_m)
            prior_error_values.append(table.prior_error_m)
            score_values.append(scores)
            candidate_index_values.append(table.candidate_indices)
            candidate_episode_values.append(table.candidate_episodes)
            prior_values.append(table.prior)
    with artifact_path.open("wb") as stream:
        np.savez_compressed(
            stream,
            task=np.asarray(task_values),
            split=np.asarray(split_values),
            group_id=np.asarray(group_values),
            chunk_id=np.asarray(chunk_values),
            feature_names=np.asarray(FEATURE_NAMES),
            features=np.concatenate(feature_values),
            benefit_m=np.concatenate(benefit_values),
            rank1_error_m=np.concatenate(rank1_values),
            prior_error_m=np.concatenate(prior_error_values),
            predicted_benefit_m=np.concatenate(score_values),
            candidate_indices=np.concatenate(candidate_index_values),
            candidate_episodes=np.concatenate(candidate_episode_values),
            retrieval_prior=np.concatenate(prior_values),
        )
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(project_root),
        "protocol": (
            "train-only episode-equal ridge predicts signed fixed-prior "
            "mixture benefit from causal inference-time diagnostics"
        ),
        "config": asdict(config),
        "config_sha256": _sha256(config_path),
        "source_checkpoint_sha256": _sha256(source_checkpoint_path),
        "device": str(device),
        "data": data_audit,
        "feature_names": list(FEATURE_NAMES),
        "causal_feature_audit": causal_feature_audit,
        "model": _model_report(model),
        "train_selected_constant_policy": constant_policy,
        "train": train_metrics,
        "validation": {
            **validation_metrics,
            "episode_bootstrap_auc": auc_bootstrap,
            "selector_minus_train_selected_constant_episode_bootstrap": (
                selector_bootstrap
            ),
        },
        "per_task_validation": per_task,
        "task_heldout": {
            "per_task": task_heldout,
            "mean_auc": heldout_mean_auc,
            "tasks_auc_above_chance": heldout_positive_count,
        },
        "criteria": criteria,
        "identifiability_passed": all(criteria.values()),
        "features_and_predictions_sha256": _sha256(artifact_path),
    }
    (temporary / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_root)
    atexit.unregister(cleanup)
    print(json.dumps(report["validation"], indent=2), flush=True)
    print(json.dumps(report["task_heldout"], indent=2), flush=True)
    print(json.dumps(criteria, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--bank-root", type=Path, action="append", required=True)
    parser.add_argument("--query-root", type=Path, action="append", required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(
        project_root=arguments.project_root.resolve(),
        bank_roots=[path.resolve() for path in arguments.bank_root],
        query_roots=[path.resolve() for path in arguments.query_root],
        source_checkpoint_path=arguments.source_checkpoint.resolve(),
        config_path=arguments.config.resolve(),
        output_root=arguments.output_root.resolve(),
        config=IdentifiabilityConfig.from_json(arguments.config.resolve()),
        device=torch.device(arguments.device),
    )

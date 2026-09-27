"""在同一 RLBench 候选池上比较显式几何、PointNet++ 与分数融合。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from dev.pointnet.dataset import PointNetContextDataset, PointNetContextStore
from dev.pointnet.retriever_model import (
    GeometricRetrieverConfig,
    GeometricSiameseRetriever,
)


@dataclass(frozen=True)
class GeometryScales:
    """由 train positive 距离校准的几何 RBF 尺度。"""

    active_center: float
    active_extent_log: float
    target_center: float
    target_extent_log: float
    eef_position: float
    eef_rotation: float
    eef_velocity: float
    gripper_width: float


@dataclass(frozen=True)
class ComparisonConfig:
    """统一检索评测配置。"""

    recall_k: tuple[int, ...] = (1, 4, 10)
    fusion_weights: tuple[float, ...] = (0.25, 0.5, 0.75)
    batch_size: int = 64
    num_workers: int = 4
    bootstrap_samples: int = 2000
    seed: int = 20260927

    def __post_init__(self) -> None:
        if min((*self.recall_k, self.batch_size, self.bootstrap_samples)) <= 0:
            raise ValueError("Recall@K、batch size 和 bootstrap samples 必须为正")
        if self.num_workers < 0:
            raise ValueError("num_workers 不能为负")
        if any(not 0.0 <= value <= 1.0 for value in self.fusion_weights):
            raise ValueError("fusion weight 必须位于 [0, 1]")


GEOMETRY_WEIGHTS = {
    "active_center": 0.10,
    "active_extent": 0.15,
    "target_layout": 0.15,
    "eef_position": 0.25,
    "eef_rotation": 0.15,
    "eef_velocity": 0.10,
    "gripper_width": 0.10,
}

SCALE_FLOORS = GeometryScales(
    active_center=0.02,
    active_extent_log=0.10,
    target_center=0.03,
    target_extent_log=0.10,
    eef_position=0.03,
    eef_rotation=0.20,
    eef_velocity=0.03,
    gripper_width=0.005,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rotation_6d_to_matrix(values: torch.Tensor) -> torch.Tensor:
    first = torch.nn.functional.normalize(values[..., :3], dim=-1)
    second = values[..., 3:6]
    second = torch.nn.functional.normalize(
        second - (first * second).sum(dim=-1, keepdim=True) * first,
        dim=-1,
    )
    third = torch.linalg.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


def _paired_rotation_distance(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first_matrix = _rotation_6d_to_matrix(first)
    second_matrix = _rotation_6d_to_matrix(second)
    trace = (first_matrix * second_matrix).sum(dim=(-2, -1))
    cosine = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
    return torch.acos(cosine)


def _rotation_distance_matrix(values: torch.Tensor) -> torch.Tensor:
    matrices = _rotation_6d_to_matrix(values)
    trace = torch.einsum("aij,bij->ab", matrices, matrices)
    cosine = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
    return torch.acos(cosine)


def _median_rbf_scale(values: torch.Tensor, floor: float) -> float:
    finite = values[torch.isfinite(values)]
    if not len(finite):
        return floor
    # 令 train positive 的中位距离对应 0.5 相似度，并设置物理合理的下限。
    scale = float(torch.median(finite)) / math.sqrt(2.0 * math.log(2.0))
    return max(scale, floor)


def _all_states(store: PointNetContextStore) -> torch.Tensor:
    return torch.from_numpy(
        np.stack([store.get(row["chunk_id"])["state"] for row in store.records])
    ).float()


def calibrate_geometry_scales(
    store: PointNetContextStore,
    pair_rows: Sequence[Mapping[str, Any]],
) -> GeometryScales:
    """只用 train positives 校准尺度，不用验证集调参。"""
    states = _all_states(store)
    indices = {row["chunk_id"]: index for index, row in enumerate(store.records)}
    query_indices = []
    positive_indices = []
    for pair in pair_rows:
        positives = pair["positive_ids"]
        if positives and pair["query_id"] in indices and positives[0] in indices:
            query_indices.append(indices[pair["query_id"]])
            positive_indices.append(indices[positives[0]])
    if not query_indices:
        raise ValueError("train pair labels 中没有可用于尺度校准的 positive")
    first = states[query_indices]
    second = states[positive_indices]
    both_target = first[:, 28].bool() & second[:, 28].bool()

    def distance(start: int, end: int, *, logarithm: bool = False) -> torch.Tensor:
        left, right = first[:, start:end], second[:, start:end]
        if logarithm:
            left, right = left.clamp_min(1e-5).log(), right.clamp_min(1e-5).log()
        return torch.linalg.vector_norm(left - right, dim=-1)

    return GeometryScales(
        active_center=_median_rbf_scale(
            distance(0, 3), SCALE_FLOORS.active_center
        ),
        active_extent_log=_median_rbf_scale(
            distance(3, 6, logarithm=True), SCALE_FLOORS.active_extent_log
        ),
        target_center=_median_rbf_scale(
            distance(6, 9)[both_target], SCALE_FLOORS.target_center
        ),
        target_extent_log=_median_rbf_scale(
            distance(9, 12, logarithm=True)[both_target],
            SCALE_FLOORS.target_extent_log,
        ),
        eef_position=_median_rbf_scale(
            distance(12, 15), SCALE_FLOORS.eef_position
        ),
        eef_rotation=_median_rbf_scale(
            _paired_rotation_distance(first[:, 15:21], second[:, 15:21]),
            SCALE_FLOORS.eef_rotation,
        ),
        eef_velocity=_median_rbf_scale(
            distance(21, 27), SCALE_FLOORS.eef_velocity
        ),
        gripper_width=_median_rbf_scale(
            (first[:, 27] - second[:, 27]).abs(),
            SCALE_FLOORS.gripper_width,
        ),
    )


def _rbf(distance: torch.Tensor, sigma: float) -> torch.Tensor:
    return torch.exp(-0.5 * (distance / sigma).square())


def geometry_similarity(states: torch.Tensor, scales: GeometryScales) -> torch.Tensor:
    """计算可解释的纯几何相似度矩阵，不使用 future 或标签。"""
    states = states.float()
    active_center = _rbf(torch.cdist(states[:, 0:3], states[:, 0:3]), scales.active_center)
    log_extent = states[:, 3:6].clamp_min(1e-5).log()
    active_extent = _rbf(
        torch.cdist(log_extent, log_extent), scales.active_extent_log
    )
    target_valid = states[:, 28].bool()
    both_target = target_valid[:, None] & target_valid[None, :]
    both_missing = ~target_valid[:, None] & ~target_valid[None, :]
    target_center = _rbf(
        torch.cdist(states[:, 6:9], states[:, 6:9]), scales.target_center
    )
    target_extent_values = states[:, 9:12].clamp_min(1e-5).log()
    target_extent = _rbf(
        torch.cdist(target_extent_values, target_extent_values),
        scales.target_extent_log,
    )
    target_layout = torch.where(
        both_target,
        0.5 * (target_center + target_extent),
        both_missing.float(),
    )
    eef_position = _rbf(
        torch.cdist(states[:, 12:15], states[:, 12:15]), scales.eef_position
    )
    eef_rotation = _rbf(
        _rotation_distance_matrix(states[:, 15:21]), scales.eef_rotation
    )
    eef_velocity = _rbf(
        torch.cdist(states[:, 21:27], states[:, 21:27]), scales.eef_velocity
    )
    gripper = _rbf(
        (states[:, 27, None] - states[None, :, 27]).abs(),
        scales.gripper_width,
    )
    components = {
        "active_center": active_center,
        "active_extent": active_extent,
        "target_layout": target_layout,
        "eef_position": eef_position,
        "eef_rotation": eef_rotation,
        "eef_velocity": eef_velocity,
        "gripper_width": gripper,
    }
    return sum(GEOMETRY_WEIGHTS[name] * value for name, value in components.items())


def load_model(checkpoint_path: Path, device: torch.device) -> GeometricSiameseRetriever:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = GeometricSiameseRetriever(
        GeometricRetrieverConfig(**checkpoint["model_config"])
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    return model


@torch.inference_mode()
def encode_contexts(
    model: GeometricSiameseRetriever,
    store: PointNetContextStore,
    config: ComparisonConfig,
    device: torch.device,
) -> torch.Tensor:
    loader = DataLoader(
        PointNetContextDataset(store),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    embeddings = torch.empty((len(store.records), model.config.embedding_dim))
    for batch in loader:
        context = {
            name: value.to(device, non_blocking=True)
            for name, value in batch["context"].items()
        }
        embeddings[batch["index"].long()] = model(context)["embedding"].cpu()
    return embeddings


def _candidate_mask(
    records: Sequence[Mapping[str, Any]],
    query_index: int,
    pool: str,
) -> torch.Tensor:
    query = records[query_index]
    values = []
    for index, candidate in enumerate(records):
        same_episode = (
            candidate["task"] == query["task"]
            and candidate["episode"] == query["episode"]
        )
        allowed = index != query_index and not same_episode
        if pool == "same_task":
            allowed &= candidate["task"] == query["task"]
        values.append(allowed)
    return torch.tensor(values, dtype=torch.bool)


def evaluate_scores(
    scores: torch.Tensor,
    records: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    config: ComparisonConfig,
    pool: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """以固定 pair labels 评估，并记录四类 hard-negative intrusion。"""
    if scores.shape != (len(records), len(records)):
        raise ValueError("score matrix shape 与 manifest 不一致")
    id_to_index = {row["chunk_id"]: index for index, row in enumerate(records)}
    pairs = {row["query_id"]: row for row in pair_rows}
    hits = {value: 0 for value in config.recall_k}
    ranks = []
    eligible_queries = 0
    hard_categories = (
        "wrong_phase",
        "wrong_gripper",
        "wrong_layout",
        "geometry_collision",
    )
    intrusions = {
        category: {value: 0 for value in config.recall_k}
        for category in hard_categories
    }
    available = {category: 0 for category in hard_categories}
    outcomes: dict[str, dict[str, Any]] = {}
    per_task: dict[str, dict[str, list[int]]] = {}
    max_k = max(config.recall_k)

    for query_index, record in enumerate(records):
        pair = pairs[record["chunk_id"]]
        positive_indices = {
            id_to_index[value]
            for value in pair["positive_ids"]
            if value in id_to_index
        }
        mask = _candidate_mask(records, query_index, pool)
        positive_indices = {index for index in positive_indices if mask[index]}
        if not positive_indices:
            continue
        ranked = torch.argsort(
            scores[query_index].masked_fill(~mask, float("-inf")),
            descending=True,
            stable=True,
        )
        rank_lookup = torch.empty(len(records), dtype=torch.long)
        rank_lookup[ranked] = torch.arange(1, len(records) + 1)
        best_rank = min(int(rank_lookup[index]) for index in positive_indices)
        query_hits = {value: int(best_rank <= value) for value in config.recall_k}
        for value, hit in query_hits.items():
            hits[value] += hit
        eligible_queries += 1
        ranks.append(best_rank)
        task_values = per_task.setdefault(
            str(record["task"]), {str(value): [] for value in config.recall_k}
        )
        for value, hit in query_hits.items():
            task_values[str(value)].append(hit)

        top_indices = ranked[:max_k].tolist()
        for category in hard_categories:
            negative_indices = {
                id_to_index[value]
                for value in pair["hard_negatives"].get(category, [])
                if value in id_to_index and mask[id_to_index[value]]
            }
            if not negative_indices:
                continue
            available[category] += 1
            for value in config.recall_k:
                intrusions[category][value] += int(
                    bool(negative_indices.intersection(top_indices[:value]))
                )
        outcomes[record["chunk_id"]] = {
            "rank": best_rank,
            "hits": {str(key): value for key, value in query_hits.items()},
        }

    return (
        {
            "eligible_queries": eligible_queries,
            "recall": {
                f"recall@{value}": hits[value] / max(eligible_queries, 1)
                for value in config.recall_k
            },
            "mrr": sum(1.0 / rank for rank in ranks) / max(len(ranks), 1),
            "hard_negative_intrusion": {
                category: {
                    "eligible_queries": available[category],
                    **{
                        f"intrusion@{value}": (
                            intrusions[category][value] / available[category]
                            if available[category]
                            else None
                        )
                        for value in config.recall_k
                    },
                }
                for category in hard_categories
            },
            "recall_by_task": {
                task: {
                    f"recall@{value}": sum(values[str(value)]) / len(values[str(value)])
                    for value in config.recall_k
                }
                for task, values in sorted(per_task.items())
            },
        },
        outcomes,
    )


def paired_bootstrap_delta(
    first: Mapping[str, Mapping[str, Any]],
    second: Mapping[str, Mapping[str, Any]],
    *,
    recall_k: int,
    samples: int,
    seed: int,
) -> dict[str, float | int]:
    """计算 second-first 的配对 bootstrap Recall@K 差值区间。"""
    identifiers = sorted(set(first).intersection(second))
    if not identifiers:
        raise ValueError("两种方法没有共同 eligible queries")
    first_hits = np.asarray(
        [first[value]["hits"][str(recall_k)] for value in identifiers], dtype=np.float64
    )
    second_hits = np.asarray(
        [second[value]["hits"][str(recall_k)] for value in identifiers], dtype=np.float64
    )
    differences = second_hits - first_hits
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sampled = rng.integers(0, len(differences), size=len(differences))
        estimates[index] = differences[sampled].mean()
    return {
        "queries": len(identifiers),
        "delta": float(differences.mean()),
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
    }


def compare(
    *,
    data_root: Path,
    checkpoint_path: Path,
    device: torch.device,
    config: ComparisonConfig,
) -> dict[str, Any]:
    if not data_root.joinpath("PREPROCESS_COMPLETE").is_file():
        raise FileNotFoundError("数据缺少 PREPROCESS_COMPLETE")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    dataset_hash = _sha256(data_root / "summary.json")
    if checkpoint.get("dataset_summary_sha256") != dataset_hash:
        raise ValueError("checkpoint 与数据 summary hash 不匹配")

    started = time.perf_counter()
    train_store = PointNetContextStore(data_root, "train", cache_size=4)
    train_pairs = _read_jsonl(data_root / "pairs-train.jsonl")
    scales = calibrate_geometry_scales(train_store, train_pairs)
    del train_store

    val_store = PointNetContextStore(data_root, "val", cache_size=4)
    pair_rows = _read_jsonl(data_root / "pairs-val.jsonl")
    states = _all_states(val_store)
    geometry_scores = geometry_similarity(states, scales).cpu()
    model = load_model(checkpoint_path, device)
    embeddings = encode_contexts(model, val_store, config, device)
    learned_scores = ((embeddings @ embeddings.T) + 1.0) / 2.0
    score_matrices = {
        "explicit_geometry": geometry_scores,
        "pointnet": learned_scores,
        **{
            f"fusion_pointnet_{weight:.2f}": (
                weight * learned_scores + (1.0 - weight) * geometry_scores
            )
            for weight in config.fusion_weights
        },
    }

    metrics: dict[str, Any] = {}
    outcomes: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for name, scores in score_matrices.items():
        metrics[name] = {}
        outcomes[name] = {}
        for pool in ("global", "same_task"):
            report, method_outcomes = evaluate_scores(
                scores, val_store.records, pair_rows, config, pool
            )
            metrics[name][pool] = report
            outcomes[name][pool] = method_outcomes

    comparisons = {}
    for pool in ("global", "same_task"):
        comparisons[pool] = {}
        baseline = outcomes["explicit_geometry"][pool]
        for name in score_matrices:
            if name == "explicit_geometry":
                continue
            comparisons[pool][f"{name}_minus_explicit_geometry"] = (
                paired_bootstrap_delta(
                    baseline,
                    outcomes[name][pool],
                    recall_k=4,
                    samples=config.bootstrap_samples,
                    seed=config.seed,
                )
            )

    summary = json.loads((data_root / "summary.json").read_text())
    return {
        "protocol": {
            "split": "val",
            "candidate_pool": "same split, excluding same task+episode",
            "positive_definition": "precomputed cross-episode pair labels",
            "geometry_calibration": "train positives only",
            "primary_metric": "same_task Recall@4",
            "fixed_primary_fusion": "fusion_pointnet_0.50",
        },
        "config": asdict(config),
        "dataset": {
            "schema": summary["schema_version"],
            "val_chunks": len(val_store.records),
            "dataset_summary_sha256": dataset_hash,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "epoch": checkpoint["epoch"],
            "git_commit": checkpoint.get("git_commit"),
        },
        "geometry": {
            "weights": GEOMETRY_WEIGHTS,
            "scales": asdict(scales),
        },
        "metrics": metrics,
        "paired_bootstrap_recall@4": comparisons,
        "runtime_seconds": time.perf_counter() - started,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA，但当前节点没有可用 GPU")
    config = ComparisonConfig(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    report = compare(
        data_root=args.data_root,
        checkpoint_path=args.checkpoint,
        device=device,
        config=config,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

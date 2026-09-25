"""PointNet Retriever 的 sharded NPZ 数据加载与 hard-pair 采样。"""

from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
from torch.utils.data import Dataset


STATE_DIM = 29
NEGATIVE_CATEGORIES = (
    "wrong_phase",
    "wrong_gripper",
    "wrong_layout",
    "geometry_collision",
)


class _ShardCache:
    """每个 DataLoader worker 独立使用的小型 shard LRU cache。"""

    def __init__(self, root: Path, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("shard cache capacity 必须为正")
        self.root = root
        self.capacity = capacity
        self.values: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    def get(self, relative: str) -> Mapping[str, np.ndarray]:
        cached = self.values.pop(relative, None)
        if cached is None:
            with np.load(self.root / relative) as archive:
                cached = {name: archive[name] for name in archive.files}
            if len(self.values) >= self.capacity:
                self.values.popitem(last=False)
        self.values[relative] = cached
        return cached


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


class PointNetContextStore:
    """按 chunk ID 读取模型输入和辅助监督，不将 future 信息混入 state。"""

    def __init__(self, root: str | Path, split: str, cache_size: int = 4) -> None:
        self.root = Path(root)
        self.split = split
        self.records = _read_jsonl(self.root / f"manifest-{split}.jsonl")
        self.records_by_id = {row["chunk_id"]: row for row in self.records}
        if len(self.records_by_id) != len(self.records):
            raise ValueError(f"{split} manifest 存在重复 chunk_id")
        self.cache = _ShardCache(self.root, cache_size)

    @staticmethod
    def _state(arrays: Mapping[str, np.ndarray], row: int) -> np.ndarray:
        target_valid = bool(arrays["target_valid"][row])
        target_relative = (
            arrays["target_center"][row] - arrays["active_center"][row]
            if target_valid
            else np.zeros(3, dtype=np.float32)
        )
        state = np.concatenate(
            (
                arrays["active_center"][row],
                arrays["active_extent"][row],
                target_relative,
                arrays["target_extent"][row],
                arrays["eef_relative_active"][row],
                arrays["eef_velocity"][row],
                np.asarray([arrays["gripper_width"][row]], dtype=np.float32),
                np.asarray([target_valid], dtype=np.float32),
            )
        ).astype(np.float32)
        if state.shape != (STATE_DIM,):
            raise ValueError(f"state shape 错误：{state.shape}")
        return state

    def get(self, identifier: str) -> dict[str, np.ndarray]:
        metadata = self.records_by_id[identifier]
        arrays = self.cache.get(metadata["shard"])
        row = int(metadata["row"])
        future_target = np.concatenate(
            (
                arrays["future_translation"][row],
                arrays["future_rotation_axis_angle"][row],
                np.asarray([arrays["future_gripper_delta"][row]], dtype=np.float32),
                arrays["effect_translation"][row],
            )
        ).astype(np.float32)
        return {
            "active_points": arrays["active_points"][row].astype(np.float32),
            "target_points": arrays["target_points"][row].astype(np.float32),
            "state": self._state(arrays, row),
            "target_valid": np.asarray(
                arrays["target_valid"][row], dtype=np.bool_
            ),
            "phase_id": np.asarray(arrays["phase_id"][row], dtype=np.int64),
            "future_target": future_target,
            "effect_valid": np.asarray(
                arrays["effect_valid"][row], dtype=np.bool_
            ),
        }

    def state_statistics(self) -> tuple[np.ndarray, np.ndarray]:
        states = np.stack(
            [self.get(record["chunk_id"])["state"] for record in self.records]
        )
        mean = states.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = states.std(axis=0, dtype=np.float64).astype(np.float32)
        return mean, np.maximum(std, 1e-4)


class PointNetPairDataset(Dataset):
    """为每个 query 确定性采样一个跨 episode positive 和一个 hard negative。"""

    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        seed: int,
        cache_size: int = 4,
    ) -> None:
        self.store = PointNetContextStore(root, split, cache_size)
        pairs = _read_jsonl(self.store.root / f"pairs-{split}.jsonl")
        self.pairs = [
            pair
            for pair in pairs
            if pair["positive_ids"]
            and any(pair["hard_negatives"].get(name) for name in NEGATIVE_CATEGORIES)
        ]
        if not self.pairs:
            raise ValueError(f"{split} 没有可训练 positive/negative pairs")
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        pair = self.pairs[index]
        rng = random.Random(self.seed + 1_000_003 * self.epoch + 97 * index)
        positive_id = rng.choice(pair["positive_ids"])
        available = [
            name
            for name in NEGATIVE_CATEGORIES
            if pair["hard_negatives"].get(name)
        ]
        category = available[(index + self.epoch) % len(available)]
        negative_id = rng.choice(pair["hard_negatives"][category])
        return {
            "anchor": self.store.get(pair["query_id"]),
            "positive": self.store.get(positive_id),
            "negative": self.store.get(negative_id),
            "negative_category": np.asarray(
                NEGATIVE_CATEGORIES.index(category), dtype=np.int64
            ),
        }


class PointNetContextDataset(Dataset):
    """按 manifest 顺序提供全部 context，供 Recall@K 评估。"""

    def __init__(self, store: PointNetContextStore) -> None:
        self.store = store

    def __len__(self) -> int:
        return len(self.store.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.store.records[index]
        return {
            "index": np.asarray(index, dtype=np.int64),
            "context": self.store.get(record["chunk_id"]),
        }

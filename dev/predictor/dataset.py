"""聚合动作 Predictor 的跨 episode Demo pair 数据集。"""

from __future__ import annotations

import random
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import Dataset


NEGATIVE_CATEGORIES = (
    "wrong_phase",
    "wrong_gripper",
    "wrong_layout",
    "geometry_collision",
)


class AggregateActionPairDataset(Dataset):
    """为每个 query 确定性采样 D+ 和一个反事实 D-。"""

    def __init__(
        self,
        *,
        embeddings: torch.Tensor,
        actions: torch.Tensor,
        records: Sequence[Mapping[str, Any]],
        pair_rows: Sequence[Mapping[str, Any]],
        seed: int,
    ) -> None:
        if embeddings.ndim != 2 or actions.ndim != 2:
            raise ValueError("embeddings/actions 必须是二维 Tensor")
        if len(embeddings) != len(actions) or len(records) != len(actions):
            raise ValueError("embedding/action/record 数量不一致")
        self.embeddings = embeddings.float()
        self.actions = actions.float()
        self.records = list(records)
        self.id_to_index = {
            str(row["chunk_id"]): index for index, row in enumerate(records)
        }
        self.pairs = [
            dict(row)
            for row in pair_rows
            if row["positive_ids"]
            and any(row["hard_negatives"].get(name) for name in NEGATIVE_CATEGORIES)
            and row["query_id"] in self.id_to_index
        ]
        if not self.pairs:
            raise ValueError("没有可用的 positive/negative Demo pairs")
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
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
        query_index = self.id_to_index[pair["query_id"]]
        positive_index = self.id_to_index[positive_id]
        negative_index = self.id_to_index[negative_id]
        return {
            "query_context": self.embeddings[query_index],
            "target_action": self.actions[query_index],
            "demo_context": self.embeddings[positive_index],
            "demo_action": self.actions[positive_index],
            "wrong_context": self.embeddings[negative_index],
            "wrong_action": self.actions[negative_index],
            "query_index": torch.tensor(query_index, dtype=torch.long),
        }

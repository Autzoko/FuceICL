"""Action-chunk Predictor 的内存数据与确定性 pair 采样。"""

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


class ActionChunkPairDataset(Dataset):
    """为 full-horizon query 采样 full-horizon D+ 与反事实 D-。"""

    def __init__(
        self,
        *,
        embeddings: torch.Tensor,
        actions: torch.Tensor,
        action_masks: torch.Tensor,
        records: Sequence[Mapping[str, Any]],
        pair_rows: Sequence[Mapping[str, Any]],
        seed: int,
    ) -> None:
        count = len(records)
        if embeddings.ndim != 2 or actions.ndim != 3 or action_masks.ndim != 2:
            raise ValueError("embedding/actions/action_masks 维度错误")
        if not (len(embeddings) == len(actions) == len(action_masks) == count):
            raise ValueError("embedding/action/mask/record 数量不一致")
        if actions.shape[:2] != action_masks.shape:
            raise ValueError("action 与 mask shape 不一致")
        self.embeddings = embeddings.float()
        self.actions = actions.float()
        self.action_masks = action_masks.bool()
        self.records = list(records)
        self.id_to_index = {
            str(record["chunk_id"]): index
            for index, record in enumerate(records)
        }
        full = self.action_masks.all(dim=1)
        self.pairs = []
        for row in pair_rows:
            query_index = self.id_to_index.get(str(row["query_id"]))
            if query_index is None or not bool(full[query_index]):
                continue
            positives = [
                identifier
                for identifier in row["positive_ids"]
                if identifier in self.id_to_index
                and bool(full[self.id_to_index[identifier]])
            ]
            negative_ids = {
                name: [
                    identifier
                    for identifier in row["hard_negatives"].get(name, [])
                    if identifier in self.id_to_index
                    and bool(full[self.id_to_index[identifier]])
                ]
                for name in NEGATIVE_CATEGORIES
            }
            if positives and any(negative_ids.values()):
                self.pairs.append(
                    {
                        "query_id": row["query_id"],
                        "positive_ids": positives,
                        "hard_negatives": negative_ids,
                    }
                )
        if not self.pairs:
            raise ValueError("没有可用的 full-horizon action pairs")
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
            if pair["hard_negatives"][name]
        ]
        category = available[(index + self.epoch) % len(available)]
        negative_id = rng.choice(pair["hard_negatives"][category])
        query_index = self.id_to_index[str(pair["query_id"])]
        positive_index = self.id_to_index[str(positive_id)]
        negative_index = self.id_to_index[str(negative_id)]
        return {
            "query_context": self.embeddings[query_index],
            "target_actions": self.actions[query_index],
            "target_mask": self.action_masks[query_index],
            "demo_context": self.embeddings[positive_index],
            "demo_actions": self.actions[positive_index],
            "demo_mask": self.action_masks[positive_index],
            "wrong_context": self.embeddings[negative_index],
            "wrong_actions": self.actions[negative_index],
            "wrong_mask": self.action_masks[negative_index],
        }

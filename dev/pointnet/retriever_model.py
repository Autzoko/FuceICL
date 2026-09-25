"""Tiny PointNet++ 与显式几何状态融合的 Siamese Retriever encoder。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
import torch.nn.functional as functional

from dev.pointnet.dataset import STATE_DIM
from lib.tiny_pointnetpp import TinyPointNetPPConfig, TinyPointNetPPEncoder


@dataclass(frozen=True)
class GeometricRetrieverConfig:
    embedding_dim: int = 128
    state_hidden_dim: int = 64
    fusion_hidden_dim: int = 256
    dropout: float = 0.1


class GeometricSiameseRetriever(nn.Module):
    """共享点云编码器，融合 active/target shape 与当前几何状态。"""

    def __init__(
        self,
        config: GeometricRetrieverConfig | None = None,
        *,
        state_mean: torch.Tensor | None = None,
        state_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.config = config or GeometricRetrieverConfig()
        self.point_encoder = TinyPointNetPPEncoder(
            TinyPointNetPPConfig(output_dim=self.config.embedding_dim)
        )
        self.missing_target = nn.Parameter(torch.zeros(self.config.embedding_dim))
        self.state_encoder = nn.Sequential(
            nn.Linear(STATE_DIM, self.config.state_hidden_dim),
            nn.LayerNorm(self.config.state_hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.state_hidden_dim, self.config.state_hidden_dim),
            nn.SiLU(),
        )
        fusion_input = 2 * self.config.embedding_dim + self.config.state_hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, self.config.fusion_hidden_dim),
            nn.LayerNorm(self.config.fusion_hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.fusion_hidden_dim, self.config.embedding_dim),
        )
        self.phase_head = nn.Linear(self.config.embedding_dim, 4)
        self.future_head = nn.Linear(self.config.embedding_dim, 10)
        mean = torch.zeros(STATE_DIM) if state_mean is None else state_mean.float()
        std = torch.ones(STATE_DIM) if state_std is None else state_std.float()
        if mean.shape != (STATE_DIM,) or std.shape != (STATE_DIM,):
            raise ValueError(f"state mean/std 必须为 [{STATE_DIM}]")
        self.register_buffer("state_mean", mean)
        self.register_buffer("state_std", std.clamp_min(1e-4))

    def forward(
        self,
        context: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        active = context["active_points"]
        target = context["target_points"]
        if active.shape != target.shape:
            raise ValueError("active_points 与 target_points shape 必须一致")
        batch_size = active.shape[0]
        cloud_embeddings = self.point_encoder(torch.cat((active, target), dim=0))
        active_embedding = cloud_embeddings[:batch_size]
        target_embedding = cloud_embeddings[batch_size:]
        target_valid = context["target_valid"].bool().reshape(batch_size, 1)
        missing = self.missing_target.reshape(1, -1).expand_as(target_embedding)
        target_embedding = torch.where(target_valid, target_embedding, missing)
        state = (context["state"] - self.state_mean) / self.state_std
        state_embedding = self.state_encoder(state)
        fused = self.fusion(
            torch.cat((active_embedding, target_embedding, state_embedding), dim=-1)
        )
        embedding = functional.normalize(fused, p=2, dim=-1)
        return {
            "embedding": embedding,
            "phase_logits": self.phase_head(embedding),
            "future_prediction": self.future_head(embedding),
        }

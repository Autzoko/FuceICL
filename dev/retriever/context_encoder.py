"""Tiny PointNet++ + 数值状态的共享 Context Encoder。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as functional

from lib.tiny_pointnetpp import TinyPointNetPPConfig, TinyPointNetPPEncoder

from .types import GeometricContext


NUMERIC_FEATURE_DIM = 26


@dataclass(frozen=True)
class ContextEncoderConfig:
    """几何上下文编码器配置。"""

    point_embedding_dim: int = 128
    numeric_embedding_dim: int = 64
    output_dim: int = 256

    def __post_init__(self) -> None:
        values = (
            self.point_embedding_dim,
            self.numeric_embedding_dim,
            self.output_dim,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in values
        ):
            raise TypeError("ContextEncoder 维度必须是整数")
        if min(values) <= 0:
            raise ValueError("ContextEncoder 维度必须大于 0")


class GeometricContextEncoder(nn.Module):
    """编码 active/target partial cloud、layout 与 robot state。"""

    def __init__(self, config: ContextEncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or ContextEncoderConfig()
        self.point_encoder = TinyPointNetPPEncoder(
            TinyPointNetPPConfig(output_dim=self.config.point_embedding_dim)
        )
        self.numeric_encoder = nn.Sequential(
            nn.Linear(NUMERIC_FEATURE_DIM, self.config.numeric_embedding_dim),
            nn.LayerNorm(self.config.numeric_embedding_dim),
            nn.SiLU(),
            nn.Linear(
                self.config.numeric_embedding_dim,
                self.config.numeric_embedding_dim,
            ),
            nn.SiLU(),
        )
        fusion_dim = (
            2 * self.config.point_embedding_dim
            + self.config.numeric_embedding_dim
        )
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, self.config.output_dim),
            nn.LayerNorm(self.config.output_dim),
            nn.SiLU(),
            nn.Linear(self.config.output_dim, self.config.output_dim),
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @staticmethod
    def _pad_points(
        contexts: Sequence[GeometricContext],
        *,
        target: bool,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values = [
            context.target_points if target else context.active_points
            for context in contexts
        ]
        if any(value is None for value in values):
            raise ValueError("_pad_points 不能接收缺失点云")
        max_points = max(len(value) for value in values)
        points = torch.zeros(
            (len(values), max_points, 3),
            dtype=torch.float32,
            device=device,
        )
        masks = torch.zeros(
            (len(values), max_points),
            dtype=torch.bool,
            device=device,
        )
        for index, (context, value) in enumerate(zip(contexts, values, strict=True)):
            count = len(value)
            points[index, :count] = value.to(device=device, dtype=torch.float32)
            source_mask = (
                context.target_point_mask if target else context.active_point_mask
            )
            if source_mask is None:
                masks[index, :count] = True
            else:
                masks[index, :count] = source_mask.to(device=device, dtype=torch.bool)
        return points, masks

    @staticmethod
    def _numeric_features(
        contexts: Sequence[GeometricContext],
        device: torch.device,
    ) -> torch.Tensor:
        rows = []
        for context in contexts:
            zeros = torch.zeros(3, dtype=torch.float32, device=device)
            if context.has_target:
                target_extent = context.target_extent.to(
                    device=device,
                    dtype=torch.float32,
                )
                relative_center = (
                    context.target_center.to(device=device, dtype=torch.float32)
                    - context.active_center.to(device=device, dtype=torch.float32)
                )
                target_valid = torch.ones(1, dtype=torch.float32, device=device)
            else:
                target_extent = zeros
                relative_center = zeros
                target_valid = torch.zeros(1, device=device)
            rows.append(
                torch.cat(
                    (
                        context.active_extent.to(
                            device=device,
                            dtype=torch.float32,
                        ),
                        target_extent,
                        relative_center,
                        target_valid,
                        context.eef_relative_active.to(
                            device=device,
                            dtype=torch.float32,
                        ),
                        context.eef_velocity.to(
                            device=device,
                            dtype=torch.float32,
                        ),
                        torch.tensor(
                            [context.gripper_width],
                            dtype=torch.float32,
                            device=device,
                        ),
                    )
                )
            )
        return torch.stack(rows)

    def forward(self, contexts: Sequence[GeometricContext]) -> torch.Tensor:
        if isinstance(contexts, (str, bytes)) or not contexts:
            raise ValueError("contexts 必须是非空 GeometricContext 序列")
        active_points, active_masks = self._pad_points(
            contexts,
            target=False,
            device=self.device,
        )
        active_embeddings = self.point_encoder(active_points, active_masks)

        target_embeddings = torch.zeros(
            (len(contexts), self.config.point_embedding_dim),
            dtype=active_embeddings.dtype,
            device=self.device,
        )
        target_indices = [
            index for index, context in enumerate(contexts) if context.has_target
        ]
        if target_indices:
            target_contexts = [contexts[index] for index in target_indices]
            target_points, target_masks = self._pad_points(
                target_contexts,
                target=True,
                device=self.device,
            )
            encoded_targets = self.point_encoder(target_points, target_masks)
            target_embeddings[target_indices] = encoded_targets

        numeric = self.numeric_encoder(
            self._numeric_features(contexts, self.device)
        )
        fused = torch.cat((active_embeddings, target_embeddings, numeric), dim=-1)
        return functional.normalize(self.fusion(fused), p=2, dim=-1)

    @torch.inference_mode()
    def encode(
        self,
        contexts: Sequence[GeometricContext],
        *,
        batch_size: int = 32,
    ) -> torch.Tensor:
        """批量编码并返回 CPU float32 Tensor。"""
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size 必须是整数")
        if batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")
        values = list(contexts)
        if not values:
            return torch.empty((0, self.config.output_dim), dtype=torch.float32)
        was_training = self.training
        self.eval()
        try:
            outputs = [
                self(values[start : start + batch_size]).cpu().float()
                for start in range(0, len(values), batch_size)
            ]
        finally:
            self.train(was_training)
        return torch.cat(outputs, dim=0)

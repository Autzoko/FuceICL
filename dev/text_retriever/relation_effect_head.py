"""冻结句向量上的轻量 relation-effect 分类头与 conformal 预测。"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from lib.GLiNER2_Base import ParsedInstruction
from lib.all_MiniLM_L6_v2 import EMBEDDING_DIM


RELATION_LABELS = ("approach", "separate", "other")
RELATION_TO_ID = {
    label: index for index, label in enumerate(RELATION_LABELS)
}


@dataclass(frozen=True)
class RelationEffectHeadConfig:
    """小型 relation head 的容量定义。"""

    embedding_dim: int = EMBEDDING_DIM
    hidden_dim: int = 64
    classes: int = len(RELATION_LABELS)

    def __post_init__(self) -> None:
        if min(self.embedding_dim, self.hidden_dim, self.classes) <= 0:
            raise ValueError("relation head 维度必须为正")
        if self.classes != len(RELATION_LABELS):
            raise ValueError("relation head classes 与固定标签数量不一致")


class RelationEffectHead(nn.Module):
    """从原句与 object-masked 句向量预测距离变化关系。"""

    def __init__(
        self,
        config: RelationEffectHeadConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or RelationEffectHeadConfig()
        self.classifier = nn.Sequential(
            nn.Linear(2 * self.config.embedding_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, self.config.classes),
        )

    def forward(
        self,
        raw_embedding: torch.Tensor,
        masked_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """返回 `[B,3]` logits。"""
        if (
            raw_embedding.ndim != 2
            or raw_embedding.shape != masked_embedding.shape
        ):
            raise ValueError("raw/masked embedding shape 必须相同且为 [B,D]")
        if raw_embedding.shape[1] != self.config.embedding_dim:
            raise ValueError("embedding dimension 错误")
        if not bool(torch.isfinite(raw_embedding).all()) or not bool(
            torch.isfinite(masked_embedding).all()
        ):
            raise ValueError("relation head 输入含 NaN/Inf")
        return self.classifier(
            torch.cat((raw_embedding, masked_embedding), dim=1)
        )


@dataclass(frozen=True)
class TokenRelationEffectHeadConfig:
    """Token attention relation head 的容量定义。"""

    embedding_dim: int = EMBEDDING_DIM
    attention_dim: int = 64
    classes: int = len(RELATION_LABELS)

    def __post_init__(self) -> None:
        if min(self.embedding_dim, self.attention_dim, self.classes) <= 0:
            raise ValueError("token relation head 维度必须为正")
        if self.classes != len(RELATION_LABELS):
            raise ValueError(
                "token relation head classes 与固定标签数量不一致"
            )


class TokenRelationEffectHead(nn.Module):
    """在 frozen contextual tokens 上学习关系敏感的注意力池化。"""

    def __init__(
        self,
        config: TokenRelationEffectHeadConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or TokenRelationEffectHeadConfig()
        self.attention = nn.Sequential(
            nn.Linear(self.config.embedding_dim, self.config.attention_dim),
            nn.Tanh(),
            nn.Linear(self.config.attention_dim, 1),
        )
        # 同时保留 learned pooling 与稳定 mean pooling；分类层仍保持线性。
        self.classifier = nn.Linear(
            2 * self.config.embedding_dim,
            self.config.classes,
        )

    def pool(
        self,
        token_embeddings: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 attention/mean 拼接特征及可审计 token 权重。"""
        if token_embeddings.ndim != 3:
            raise ValueError("token_embeddings shape 必须为 [B,L,D]")
        if token_mask.shape != token_embeddings.shape[:2]:
            raise ValueError("token_mask shape 必须为 [B,L]")
        if token_embeddings.shape[2] != self.config.embedding_dim:
            raise ValueError("token embedding dimension 错误")
        mask = token_mask.to(torch.bool)
        if not bool(mask.any(dim=1).all()):
            raise ValueError("每个样本至少需要一个有效 token")
        if not bool(torch.isfinite(token_embeddings).all()):
            raise ValueError("token embeddings 含 NaN/Inf")
        scores = self.attention(token_embeddings).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        attended = torch.sum(weights.unsqueeze(-1) * token_embeddings, dim=1)
        mask_float = mask.unsqueeze(-1).to(token_embeddings.dtype)
        mean = torch.sum(mask_float * token_embeddings, dim=1)
        mean = mean / mask_float.sum(dim=1).clamp_min(1.0)
        return torch.cat((attended, mean), dim=1), weights

    def forward(
        self,
        token_embeddings: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        """返回 `[B,3]` logits。"""
        features, _ = self.pool(token_embeddings, token_mask)
        return self.classifier(features)


def mask_object_spans(parsed: ParsedInstruction) -> str:
    """用统一占位符替换 GLiNER object spans，保留谓词和关系结构。"""
    spans = sorted(
        {
            (span.start, span.end)
            for span in parsed.objects
            if span.start is not None and span.end is not None
        }
    )
    if not spans:
        return parsed.text
    previous_end = -1
    for start, end in spans:
        if start < 0 or end <= start or end > len(parsed.text):
            raise ValueError("object span 越界")
        if start < previous_end:
            raise ValueError("object spans 重叠")
        previous_end = end
    pieces = []
    cursor = 0
    for start, end in spans:
        pieces.extend((parsed.text[cursor:start], " [OBJECT] "))
        cursor = end
    pieces.append(parsed.text[cursor:])
    return " ".join("".join(pieces).split())


def conformal_quantile(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    *,
    alpha: float,
) -> float:
    """计算有限样本修正的 split-conformal nonconformity quantile。"""
    if (
        probabilities.ndim != 2
        or probabilities.shape[1] != len(RELATION_LABELS)
    ):
        raise ValueError("probabilities shape 必须为 [N,3]")
    if labels.shape != (len(probabilities),):
        raise ValueError("labels shape 必须为 [N]")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha 必须位于 (0,1)")
    if not len(labels):
        raise ValueError("calibration set 不能为空")
    if bool(((labels < 0) | (labels >= len(RELATION_LABELS))).any()):
        raise ValueError("label id 越界")
    if not bool(torch.isfinite(probabilities).all()):
        raise ValueError("probabilities 含 NaN/Inf")
    true_probabilities = probabilities[
        torch.arange(len(labels)), labels.to(torch.long)
    ]
    scores = torch.sort(1.0 - true_probabilities).values
    rank = math.ceil((len(scores) + 1) * (1.0 - alpha))
    rank = min(max(rank, 1), len(scores))
    return float(scores[rank - 1])


def conformal_sets(
    probabilities: torch.Tensor,
    *,
    quantile: float,
) -> torch.Tensor:
    """返回 `[N,3]` 布尔 prediction sets。"""
    if (
        probabilities.ndim != 2
        or probabilities.shape[1] != len(RELATION_LABELS)
    ):
        raise ValueError("probabilities shape 必须为 [N,3]")
    if not math.isfinite(quantile) or not 0.0 <= quantile <= 1.0:
        raise ValueError("conformal quantile 必须位于 [0,1]")
    return (1.0 - probabilities) <= quantile


def singleton_predictions(prediction_sets: torch.Tensor) -> list[str | None]:
    """singleton set 返回标签，否则返回 None 表示拒绝。"""
    if prediction_sets.ndim != 2 or prediction_sets.shape[1] != len(
        RELATION_LABELS
    ):
        raise ValueError("prediction_sets shape 必须为 [N,3]")
    outputs: list[str | None] = []
    for row in prediction_sets.to(torch.bool):
        indices = torch.where(row)[0]
        outputs.append(
            RELATION_LABELS[int(indices[0])] if len(indices) == 1 else None
        )
    return outputs

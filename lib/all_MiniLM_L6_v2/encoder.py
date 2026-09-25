"""all-MiniLM-L6-v2 的本地句向量接口。"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as functional
from transformers import AutoModel, AutoTokenizer


DEFAULT_MODEL_DIR = Path(__file__).with_name("checkpoint")
EMBEDDING_DIM = 384


class MiniLMTextEncoder:
    """生成 L2 归一化的 384 维文本向量。

    模型仅从本地目录加载。无论模型在哪个设备推理，公开接口都返回 CPU 上的
    ``float32`` Tensor，便于索引持久化与下游组件统一处理。
    """

    def __init__(
        self,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        *,
        device: str | torch.device = "cpu",
        max_length: int = 256,
    ) -> None:
        self.model_dir = Path(model_dir)
        if not self.model_dir.joinpath("model.safetensors").is_file():
            raise FileNotFoundError(
                f"MiniLM checkpoint 不完整：{self.model_dir / 'model.safetensors'}"
            )
        if max_length <= 0:
            raise ValueError("max_length 必须大于 0")

        self.device = torch.device(device)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            local_files_only=True,
        )
        self.model = AutoModel.from_pretrained(
            self.model_dir,
            local_files_only=True,
        ).to(self.device)
        self.model.eval()

    @staticmethod
    def _validate_texts(texts: Sequence[str], batch_size: int) -> list[str]:
        if isinstance(texts, str):
            raise TypeError("texts 必须是字符串序列，单条文本也需放入列表")
        if batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")
        values = list(texts)
        if any(not isinstance(text, str) for text in values):
            raise TypeError("texts 中的每个元素都必须是字符串")
        if any(not text.strip() for text in values):
            raise ValueError("编码文本不能为空字符串")
        return values

    @staticmethod
    def _mean_pool(
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """按 attention mask 做均值池化，避免 padding token 污染句向量。"""
        expanded_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        expanded_mask = expanded_mask.to(token_embeddings.dtype)
        summed = torch.sum(token_embeddings * expanded_mask, dim=1)
        counts = torch.clamp(expanded_mask.sum(dim=1), min=1e-9)
        return summed / counts

    @torch.inference_mode()
    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 64,
    ) -> torch.Tensor:
        """编码文本。

        Returns:
            CPU ``float32`` Tensor，shape 为 ``[N, 384]``，每行已 L2 归一化。
        """
        values = self._validate_texts(texts, batch_size)
        if not values:
            return torch.empty((0, EMBEDDING_DIM), dtype=torch.float32)

        embeddings: list[torch.Tensor] = []
        for start in range(0, len(values), batch_size):
            batch = values[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            output = self.model(**encoded)
            pooled = self._mean_pool(
                output.last_hidden_state,
                encoded["attention_mask"],
            )
            normalized = functional.normalize(pooled, p=2, dim=1)
            embeddings.append(normalized.cpu().float())
        return torch.cat(embeddings, dim=0)

    def similarity(
        self,
        left: Sequence[str],
        right: Sequence[str],
        *,
        batch_size: int = 64,
    ) -> torch.Tensor:
        """返回两组文本的 cosine similarity matrix，shape 为 ``[L, R]``。"""
        left_embeddings = self.encode(left, batch_size=batch_size)
        right_embeddings = self.encode(right, batch_size=batch_size)
        return left_embeddings @ right_embeddings.T

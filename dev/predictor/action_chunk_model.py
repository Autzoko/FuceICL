"""轻量 Demo-conditioned action-chunk Predictor 与受控基线。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


CONTEXT_DIM = 128
ACTION_DIM = 7


@dataclass(frozen=True)
class ChunkPredictorConfig:
    """三个序列模型共用的容量配置。"""

    horizon: int = 6
    context_dim: int = CONTEXT_DIM
    action_dim: int = ACTION_DIM
    hidden_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    feedforward_dim: int = 256
    dropout: float = 0.1
    residual_limit: float = 2.0

    def __post_init__(self) -> None:
        positive = (
            self.horizon,
            self.context_dim,
            self.action_dim,
            self.hidden_dim,
            self.num_layers,
            self.num_heads,
            self.feedforward_dim,
        )
        if min(positive) <= 0:
            raise ValueError("模型尺寸必须为正")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim 必须能被 num_heads 整除")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout 必须位于 [0, 1)")
        if self.residual_limit <= 0:
            raise ValueError("residual_limit 必须为正")


class _ChunkPredictorBase(nn.Module):
    """统一 action-chunk 模型接口。"""

    def __init__(self, config: ChunkPredictorConfig | None = None) -> None:
        super().__init__()
        self.config = config or ChunkPredictorConfig()

    def _encoder(self) -> nn.TransformerEncoder:
        layer = nn.TransformerEncoderLayer(
            d_model=self.config.hidden_dim,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.feedforward_dim,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        return nn.TransformerEncoder(
            layer,
            num_layers=self.config.num_layers,
            norm=nn.LayerNorm(self.config.hidden_dim),
            enable_nested_tensor=False,
        )

    def forward(
        self,
        query_context: torch.Tensor,
        demo_context: torch.Tensor,
        demo_actions: torch.Tensor,
        demo_mask: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


class QueryOnlyChunkPredictor(_ChunkPredictorBase):
    """只从 query observation embedding 生成动作的行为克隆对照。"""

    def __init__(self, config: ChunkPredictorConfig | None = None) -> None:
        super().__init__(config)
        self.context_projection = nn.Linear(
            self.config.context_dim, self.config.hidden_dim
        )
        self.positions = nn.Parameter(
            torch.empty(self.config.horizon, self.config.hidden_dim)
        )
        self.encoder = self._encoder()
        self.head = nn.Linear(self.config.hidden_dim, self.config.action_dim)
        nn.init.normal_(self.positions, std=0.02)

    def forward(
        self,
        query_context: torch.Tensor,
        demo_context: torch.Tensor,
        demo_actions: torch.Tensor,
        demo_mask: torch.Tensor,
    ) -> torch.Tensor:
        del demo_context, demo_actions, demo_mask
        tokens = self.context_projection(query_context)[:, None, :]
        tokens = tokens + self.positions[None, :, :]
        return self.head(self.encoder(tokens))


class DemoConcatChunkPredictor(_ChunkPredictorBase):
    """将 query、Demo context 和 Demo action 直接拼接的通用基线。"""

    def __init__(self, config: ChunkPredictorConfig | None = None) -> None:
        super().__init__(config)
        input_dim = 2 * self.config.context_dim + self.config.action_dim + 1
        self.input_projection = nn.Linear(input_dim, self.config.hidden_dim)
        self.positions = nn.Parameter(
            torch.empty(self.config.horizon, self.config.hidden_dim)
        )
        self.encoder = self._encoder()
        self.head = nn.Linear(self.config.hidden_dim, self.config.action_dim)
        nn.init.normal_(self.positions, std=0.02)

    def forward(
        self,
        query_context: torch.Tensor,
        demo_context: torch.Tensor,
        demo_actions: torch.Tensor,
        demo_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = demo_mask.float().unsqueeze(-1)
        query = query_context[:, None, :].expand(-1, self.config.horizon, -1)
        demo = demo_context[:, None, :].expand_as(query) * mask
        tokens = self.input_projection(
            torch.cat((query, demo, demo_actions * mask, mask), dim=-1)
        )
        return self.head(self.encoder(tokens + self.positions[None, :, :]))


class DemoActionPriorChunkPredictor(_ChunkPredictorBase):
    """以 Demo action sequence 为显式先验的受限残差模型。

    Query 只产生对 Demo memory 的乘性门控，不能直接进入动作头；
    没有有效 Demo 时严格输出零。
    反事实训练进一步抑制忽略 Demo 的退化解。
    """

    def __init__(self, config: ChunkPredictorConfig | None = None) -> None:
        super().__init__(config)
        self.action_projection = nn.Linear(
            self.config.action_dim, self.config.hidden_dim
        )
        self.demo_projection = nn.Linear(
            self.config.context_dim, self.config.hidden_dim
        )
        self.positions = nn.Parameter(
            torch.empty(self.config.horizon, self.config.hidden_dim)
        )
        self.demo_encoder = self._encoder()
        self.query_gate = nn.Sequential(
            nn.Linear(4 * self.config.context_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.Sigmoid(),
        )
        self.residual_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, self.config.action_dim),
            nn.Tanh(),
        )
        nn.init.normal_(self.positions, std=0.02)

    def forward(
        self,
        query_context: torch.Tensor,
        demo_context: torch.Tensor,
        demo_actions: torch.Tensor,
        demo_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid = demo_mask.bool()
        mask = valid.float().unsqueeze(-1)
        demo_context = demo_context * valid.any(dim=1, keepdim=True).float()
        tokens = (
            self.action_projection(demo_actions * mask)
            + self.demo_projection(demo_context)[:, None, :]
            + self.positions[None, :, :]
        ) * mask
        # Transformer 不能接收整行全 masked；临时开放零 token，
        # 最终输出仍由 mask 清零。
        safe_valid = valid.clone()
        no_demo = ~safe_valid.any(dim=1)
        safe_valid[no_demo, 0] = True
        memory = self.demo_encoder(
            tokens,
            src_key_padding_mask=~safe_valid,
        )
        interaction = torch.cat(
            (
                query_context,
                demo_context,
                query_context - demo_context,
                query_context * demo_context,
            ),
            dim=-1,
        )
        gate = self.query_gate(interaction)[:, None, :]
        residual = self.config.residual_limit * self.residual_head(memory * gate)
        return (demo_actions + residual) * mask


def build_chunk_predictor(
    name: str,
    config: ChunkPredictorConfig | None = None,
) -> _ChunkPredictorBase:
    """按稳定名称构造 action-chunk Predictor。"""
    classes = {
        "query_only": QueryOnlyChunkPredictor,
        "demo_concat": DemoConcatChunkPredictor,
        "demo_action_prior": DemoActionPriorChunkPredictor,
    }
    try:
        return classes[name](config)
    except KeyError as error:
        raise ValueError(f"未知 action-chunk Predictor：{name}") from error

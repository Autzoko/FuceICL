"""聚合 open-loop 动作实验使用的轻量 Predictor。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


CONTEXT_DIM = 128
ACTION_DIM = 7


@dataclass(frozen=True)
class PredictorConfig:
    """三个受控 Predictor 共用的模型尺寸。"""

    context_dim: int = CONTEXT_DIM
    action_dim: int = ACTION_DIM
    hidden_dim: int = 256
    dropout: float = 0.1
    residual_limit: float = 2.0

    def __post_init__(self) -> None:
        if min(self.context_dim, self.action_dim, self.hidden_dim) <= 0:
            raise ValueError("模型维度必须为正")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout 必须位于 [0, 1)")
        if self.residual_limit <= 0:
            raise ValueError("residual_limit 必须为正")


class _PredictorBase(nn.Module):
    """统一 Predictor 调用接口。"""

    def __init__(self, config: PredictorConfig | None = None) -> None:
        super().__init__()
        self.config = config or PredictorConfig()

    def forward(
        self,
        query_context: torch.Tensor,
        demo_context: torch.Tensor,
        demo_action: torch.Tensor,
        demo_valid: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


class QueryOnlyPredictor(_PredictorBase):
    """不使用 Demo 的行为克隆下界。"""

    def __init__(self, config: PredictorConfig | None = None) -> None:
        super().__init__(config)
        self.network = nn.Sequential(
            nn.Linear(self.config.context_dim, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, self.config.action_dim),
        )

    def forward(
        self,
        query_context: torch.Tensor,
        demo_context: torch.Tensor,
        demo_action: torch.Tensor,
        demo_valid: torch.Tensor,
    ) -> torch.Tensor:
        del demo_context, demo_action, demo_valid
        return self.network(query_context)


class DemoConcatPredictor(_PredictorBase):
    """直接拼接 query、Demo context 和 Demo action 的通用基线。"""

    def __init__(self, config: PredictorConfig | None = None) -> None:
        super().__init__(config)
        input_dim = 2 * self.config.context_dim + self.config.action_dim + 1
        self.network = nn.Sequential(
            nn.Linear(input_dim, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, self.config.action_dim),
        )

    def forward(
        self,
        query_context: torch.Tensor,
        demo_context: torch.Tensor,
        demo_action: torch.Tensor,
        demo_valid: torch.Tensor,
    ) -> torch.Tensor:
        valid = demo_valid.float().reshape(-1, 1)
        return self.network(
            torch.cat(
                (query_context, demo_context * valid, demo_action * valid, valid),
                dim=-1,
            )
        )


class DemoActionPriorPredictor(_PredictorBase):
    """以 Demo action 为基点，由 query 几何门控有限残差。

    Query 分支不能直接生成动作；没有有效 Demo 时输出严格为零。这一结构让 Demo
    依赖成为模型归纳偏置，并由反事实 Demo loss 进一步约束。
    """

    def __init__(self, config: PredictorConfig | None = None) -> None:
        super().__init__(config)
        self.demo_encoder = nn.Sequential(
            nn.Linear(
                self.config.context_dim + self.config.action_dim,
                self.config.hidden_dim,
            ),
            nn.LayerNorm(self.config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.SiLU(),
        )
        interaction_dim = 4 * self.config.context_dim
        self.query_gate = nn.Sequential(
            nn.Linear(interaction_dim, self.config.hidden_dim),
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

    def forward(
        self,
        query_context: torch.Tensor,
        demo_context: torch.Tensor,
        demo_action: torch.Tensor,
        demo_valid: torch.Tensor,
    ) -> torch.Tensor:
        valid = demo_valid.float().reshape(-1, 1)
        demo_context = demo_context * valid
        demo_action = demo_action * valid
        demo_features = self.demo_encoder(
            torch.cat((demo_context, demo_action), dim=-1)
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
        gated_prior = demo_features * self.query_gate(interaction) * valid
        residual = (
            self.config.residual_limit * self.residual_head(gated_prior) * valid
        )
        return (demo_action + residual) * valid


def build_predictor(
    name: str, config: PredictorConfig | None = None
) -> _PredictorBase:
    """按稳定名称构造 Predictor。"""
    classes = {
        "query_only": QueryOnlyPredictor,
        "demo_concat": DemoConcatPredictor,
        "demo_action_prior": DemoActionPriorPredictor,
    }
    try:
        return classes[name](config)
    except KeyError as error:
        raise ValueError(f"未知 Predictor：{name}") from error

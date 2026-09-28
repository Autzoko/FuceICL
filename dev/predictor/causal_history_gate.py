"""用最小两帧因果几何历史校准冻结 Demo residual。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import nn

from dev.predictor.benefit_calibrated_shrinkage import GATE_FEATURE_DIM
from dev.predictor.canonical_geometry import GEOMETRY_DIM


DYNAMIC_GEOMETRY_INDICES = (0, 1, 2, 3, 4, 5, 12, 13, 14, 15)
HISTORY_LAGS = 2
HISTORY_FEATURE_DIM = len(DYNAMIC_GEOMETRY_INDICES) * HISTORY_LAGS + HISTORY_LAGS


@dataclass(frozen=True)
class CausalHistoryGateConfig:
    """轻量 causal-history gate 配置。"""

    input_dim: int = GATE_FEATURE_DIM + HISTORY_FEATURE_DIM
    hidden_dim: int = 32

    def __post_init__(self) -> None:
        if min(self.input_dim, self.hidden_dim) <= 0:
            raise ValueError("history gate 尺寸必须为正")


class CausalHistoryBenefitGate(nn.Module):
    """从冻结 transport diagnostics 与两帧历史预测 residual gate。"""

    def __init__(
        self,
        config: CausalHistoryGateConfig,
        *,
        feature_mean: torch.Tensor,
        feature_std: torch.Tensor,
    ) -> None:
        super().__init__()
        if feature_mean.shape != (config.input_dim,):
            raise ValueError("feature mean shape 错误")
        if feature_std.shape != feature_mean.shape:
            raise ValueError("feature std shape 错误")
        self.config = config
        self.register_buffer("feature_mean", feature_mean.float())
        self.register_buffer("feature_std", feature_std.float().clamp_min(1e-5))
        self.network = nn.Sequential(
            nn.LayerNorm(config.input_dim),
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.config.input_dim:
            raise ValueError("history gate features shape 错误")
        normalized = (features - self.feature_mean) / self.feature_std
        return torch.sigmoid(self.network(normalized)).squeeze(-1)


def causal_history_features(
    records: Sequence[Mapping[str, object]],
    geometry: torch.Tensor,
) -> torch.Tensor:
    """按 episode/frame 构造两级因果差分；缺帧时保持零并关闭 mask。"""
    if geometry.ndim != 2 or geometry.shape != (len(records), GEOMETRY_DIM):
        raise ValueError("records 与 geometry shape 不一致")
    keys = [
        (int(record["episode"]), int(record["frame"])) for record in records
    ]
    if len(set(keys)) != len(keys):
        raise ValueError("episode/frame key 不唯一")
    by_key = {key: index for index, key in enumerate(keys)}
    selected = geometry[:, DYNAMIC_GEOMETRY_INDICES]
    output = torch.zeros(
        (len(records), HISTORY_FEATURE_DIM),
        dtype=geometry.dtype,
        device=geometry.device,
    )
    width = len(DYNAMIC_GEOMETRY_INDICES)
    for index, (episode, frame) in enumerate(keys):
        previous = by_key.get((episode, frame - 1))
        if previous is None:
            continue
        output[index, :width] = selected[index] - selected[previous]
        output[index, 2 * width] = 1.0
        second_previous = by_key.get((episode, frame - 2))
        if second_previous is None:
            continue
        output[index, width : 2 * width] = (
            selected[previous] - selected[second_previous]
        )
        output[index, 2 * width + 1] = 1.0
    if not bool(torch.isfinite(output).all()):
        raise ValueError("causal history features 含非有限值")
    return output


def first_order_history(features: torch.Tensor) -> torch.Tensor:
    """只保留最近一阶差分与 validity，作为预注册消融。"""
    if features.ndim != 2 or features.shape[1] != HISTORY_FEATURE_DIM:
        raise ValueError("history feature shape 错误")
    width = len(DYNAMIC_GEOMETRY_INDICES)
    return torch.cat((features[:, :width], features[:, 2 * width : 2 * width + 1]), dim=1)


def shuffled_history_within_episode(
    records: Sequence[Mapping[str, object]],
    features: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    """在每个 episode 内打乱 history，保留边际分布作为负对照。"""
    if features.ndim != 2 or len(features) != len(records):
        raise ValueError("records 与 history features shape 不一致")
    output = features.clone()
    episodes: dict[int, list[int]] = {}
    for index, record in enumerate(records):
        episodes.setdefault(int(record["episode"]), []).append(index)
    generator = torch.Generator().manual_seed(seed)
    for indices in episodes.values():
        if len(indices) < 2:
            continue
        source = torch.tensor(indices, dtype=torch.long)
        permutation = source[torch.randperm(len(source), generator=generator)]
        output[source] = features[permutation]
    return output

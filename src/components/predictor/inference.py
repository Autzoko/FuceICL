"""Retriever 与 Demo-anchored Predictor 之间的最小推理契约。"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .layout_equivariant_policy import LayoutEquivariantDemoPolicy


def _finite_floating_tensor(
    value: torch.Tensor,
    *,
    name: str,
    dimensions: int,
) -> None:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} 必须是浮点 Tensor")
    if value.ndim != dimensions:
        raise ValueError(f"{name} 必须是 {dimensions}D Tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} 含 NaN/Inf")


@dataclass(frozen=True)
class PreparedDemoContext:
    """检索命中经 query layout frame transport 后的最小 Demo 输入。"""

    candidate_id: str
    state: torch.Tensor
    transported_action: torch.Tensor
    retrieval_score: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ValueError("candidate_id 不能为空")
        object.__setattr__(self, "candidate_id", self.candidate_id.strip())
        _finite_floating_tensor(self.state, name="state", dimensions=1)
        _finite_floating_tensor(
            self.transported_action,
            name="transported_action",
            dimensions=2,
        )
        if self.retrieval_score is not None and (
            isinstance(self.retrieval_score, bool)
            or not isinstance(self.retrieval_score, (int, float))
            or not math.isfinite(self.retrieval_score)
            or not -1.0 <= self.retrieval_score <= 1.0
        ):
            raise ValueError("retrieval_score 必须位于 [-1,1] 或为 None")


@dataclass(frozen=True)
class ActionChunkRequest:
    """单条在线请求；query state 不包含未来信息。"""

    query_state: torch.Tensor
    demo: PreparedDemoContext | None

    def __post_init__(self) -> None:
        _finite_floating_tensor(
            self.query_state,
            name="query_state",
            dimensions=1,
        )
        if self.demo is not None and not isinstance(
            self.demo, PreparedDemoContext
        ):
            raise TypeError("demo 必须是 PreparedDemoContext 或 None")


@dataclass(frozen=True)
class ActionChunkPrediction:
    """动作输出及其 Demo provenance。"""

    action: torch.Tensor
    used_demo: bool
    candidate_id: str | None
    retrieval_score: float | None
    reason: str


class RetrievalAugmentedActionPredictor:
    """执行单 Demo、batch-1 的低延迟 action-chunk 推理。

    Point-cloud encoding、task routing、Demo retrieval 与 layout transport 均在
    上游完成。本类不接收 raw RGB/point cloud，也不在缺少 Demo 时启用
    observation-only fallback。
    """

    def __init__(self, policy: LayoutEquivariantDemoPolicy) -> None:
        if not isinstance(policy, LayoutEquivariantDemoPolicy):
            raise TypeError("policy 必须是 LayoutEquivariantDemoPolicy")
        self.policy = policy
        self.policy.eval()

    @property
    def device(self) -> torch.device:
        return self.policy.state_mean.device

    @property
    def dtype(self) -> torch.dtype:
        return self.policy.state_mean.dtype

    @torch.inference_mode()
    def predict(self, request: ActionChunkRequest) -> ActionChunkPrediction:
        """执行一次推理；无 Demo 时通过结构化 mask 返回精确零。"""
        if not isinstance(request, ActionChunkRequest):
            raise TypeError("request 必须是 ActionChunkRequest")
        config = self.policy.config
        if request.query_state.shape != (config.state_dim,):
            raise ValueError("query_state dimension 与 policy 不一致")
        query = request.query_state.to(device=self.device, dtype=self.dtype)
        if request.demo is None:
            demo_state = torch.zeros_like(query)
            transported = torch.zeros(
                (config.action_horizon, 7),
                device=self.device,
                dtype=self.dtype,
            )
            mask = torch.zeros(1, device=self.device, dtype=self.dtype)
            candidate_id = None
            retrieval_score = None
            reason = "no_demo"
        else:
            demo = request.demo
            if demo.state.shape != (config.state_dim,):
                raise ValueError("Demo state dimension 与 policy 不一致")
            expected_action = (config.action_horizon, 7)
            if demo.transported_action.shape != expected_action:
                raise ValueError("transported action shape 与 policy 不一致")
            demo_state = demo.state.to(device=self.device, dtype=self.dtype)
            transported = demo.transported_action.to(
                device=self.device,
                dtype=self.dtype,
            )
            mask = torch.ones(1, device=self.device, dtype=self.dtype)
            candidate_id = demo.candidate_id
            retrieval_score = demo.retrieval_score
            reason = "demo_conditioned"
        action = self.policy(
            query.unsqueeze(0),
            demo_state.unsqueeze(0),
            transported.unsqueeze(0),
            mask,
        )[0]
        return ActionChunkPrediction(
            action=action.detach().cpu(),
            used_demo=request.demo is not None,
            candidate_id=candidate_id,
            retrieval_score=retrieval_score,
            reason=reason,
        )

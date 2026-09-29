"""raw Demo action-prior Predictor 的低延迟 batch-1 推理契约。"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .demo_relative_policy import DemoRelativePolicy
from .inference import _finite_floating_tensor


@dataclass(frozen=True)
class PreparedRawDemoContext:
    """Retriever 命中的最小 raw Demo context 与 provenance。"""

    candidate_id: str
    state: torch.Tensor
    raw_action: torch.Tensor
    retrieval_distance: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ValueError("candidate_id 不能为空")
        object.__setattr__(self, "candidate_id", self.candidate_id.strip())
        object.__setattr__(
            self,
            "state",
            _finite_floating_tensor(self.state, name="state", dimensions=1),
        )
        object.__setattr__(
            self,
            "raw_action",
            _finite_floating_tensor(
                self.raw_action,
                name="raw_action",
                dimensions=2,
            ),
        )
        if self.retrieval_distance is not None and (
            isinstance(self.retrieval_distance, bool)
            or not isinstance(self.retrieval_distance, (int, float))
            or not math.isfinite(self.retrieval_distance)
            or self.retrieval_distance < 0.0
        ):
            raise ValueError("retrieval_distance 必须是非负有限数或 None")


@dataclass(frozen=True)
class DemoRelativeActionChunkRequest:
    """单条在线请求；query 只包含当前因果 belief/state。"""

    query_state: torch.Tensor
    demo: PreparedRawDemoContext | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "query_state",
            _finite_floating_tensor(
                self.query_state,
                name="query_state",
                dimensions=1,
            ),
        )
        if self.demo is not None and not isinstance(
            self.demo, PreparedRawDemoContext
        ):
            raise TypeError("demo 必须是 PreparedRawDemoContext 或 None")


@dataclass(frozen=True)
class DemoRelativeActionChunkPrediction:
    """action chunk 及其可审计的检索来源。"""

    action: torch.Tensor
    used_demo: bool
    candidate_id: str | None
    retrieval_distance: float | None
    reason: str


class DemoRelativeActionPredictor:
    """执行单 Demo、batch-1 的 relative-only action-chunk 推理。"""

    def __init__(self, policy: DemoRelativePolicy) -> None:
        if not isinstance(policy, DemoRelativePolicy):
            raise TypeError("policy 必须是 DemoRelativePolicy")
        self.policy = policy.eval()

    @property
    def device(self) -> torch.device:
        return self.policy.state_mean.device

    @property
    def dtype(self) -> torch.dtype:
        return self.policy.state_mean.dtype

    @torch.inference_mode()
    def predict(
        self,
        request: DemoRelativeActionChunkRequest,
    ) -> DemoRelativeActionChunkPrediction:
        """无 Demo 时结构化 mask 返回精确零，不启用 observation head。"""
        if not isinstance(request, DemoRelativeActionChunkRequest):
            raise TypeError("request 必须是 DemoRelativeActionChunkRequest")
        config = self.policy.config
        if request.query_state.shape != (config.state_dim,):
            raise ValueError("query_state dimension 与 policy 不一致")
        query = request.query_state.to(device=self.device, dtype=self.dtype)
        if request.demo is None:
            demo_state = torch.zeros_like(query)
            raw_action = torch.zeros(
                (config.action_horizon, 7),
                device=self.device,
                dtype=self.dtype,
            )
            mask = torch.zeros(1, device=self.device, dtype=self.dtype)
            candidate_id = None
            retrieval_distance = None
            reason = "no_demo"
        else:
            demo = request.demo
            if demo.state.shape != (config.state_dim,):
                raise ValueError("Demo state dimension 与 policy 不一致")
            expected_action = (config.action_horizon, 7)
            if demo.raw_action.shape != expected_action:
                raise ValueError("raw Demo action shape 与 policy 不一致")
            demo_state = demo.state.to(device=self.device, dtype=self.dtype)
            raw_action = demo.raw_action.to(
                device=self.device,
                dtype=self.dtype,
            )
            mask = torch.ones(1, device=self.device, dtype=self.dtype)
            candidate_id = demo.candidate_id
            retrieval_distance = demo.retrieval_distance
            reason = "demo_conditioned"
        action = self.policy(
            query.unsqueeze(0),
            demo_state.unsqueeze(0),
            raw_action.unsqueeze(0),
            mask,
        )[0]
        return DemoRelativeActionChunkPrediction(
            action=action.detach().cpu(),
            used_demo=request.demo is not None,
            candidate_id=candidate_id,
            retrieval_distance=retrieval_distance,
            reason=reason,
        )

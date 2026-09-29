"""将可信 belief query、单 Demo 检索与 action-prior Predictor 串联。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.components.predictor import (
    DemoRelativeActionChunkPrediction,
    DemoRelativeActionChunkRequest,
    DemoRelativeActionPredictor,
    PreparedRawDemoContext,
)
from src.components.retriever import (
    BeliefDemoQuery,
    BeliefDemoRetrievalResult,
    ExactBeliefDemoRetriever,
)


@dataclass(frozen=True)
class SelectiveBeliefActionRequest:
    """文本路由后的 typed query 及解析风险门控结论。"""

    query: BeliefDemoQuery
    belief_risk_accepted: bool

    def __post_init__(self) -> None:
        if not isinstance(self.query, BeliefDemoQuery):
            raise TypeError("query 必须是 BeliefDemoQuery")
        if not isinstance(self.belief_risk_accepted, bool):
            raise TypeError("belief_risk_accepted 必须是 bool")


@dataclass(frozen=True)
class SelectiveBeliefActionResult:
    """完整保留检索与动作预测 provenance 的单请求结果。"""

    retrieval: BeliefDemoRetrievalResult
    prediction: DemoRelativeActionChunkPrediction


class SelectiveBeliefActionPipeline:
    """风险拒绝时不检索；命中时仅向 Predictor 提供 top-1 raw Demo。"""

    def __init__(
        self,
        retriever: ExactBeliefDemoRetriever,
        predictor: DemoRelativeActionPredictor,
    ) -> None:
        if not isinstance(retriever, ExactBeliefDemoRetriever):
            raise TypeError("retriever 必须是 ExactBeliefDemoRetriever")
        if not isinstance(predictor, DemoRelativeActionPredictor):
            raise TypeError("predictor 必须是 DemoRelativeActionPredictor")
        if not retriever.is_indexed:
            raise ValueError("retriever 必须先建立索引")
        config = predictor.policy.config
        if retriever.state_std.shape != (config.state_dim,):
            raise ValueError("Retriever 与 Predictor state dimension 不一致")
        if not torch.equal(
            retriever.state_std,
            predictor.policy.state_std.detach().float().cpu(),
        ):
            raise ValueError("Retriever 与 Predictor 必须共享冻结 state_std")
        if retriever.action_shape != (config.action_horizon, 7):
            raise ValueError("Retriever Demo action shape 与 Predictor 不一致")
        self.retriever = retriever
        self.predictor = predictor

    def predict(
        self,
        request: SelectiveBeliefActionRequest,
    ) -> SelectiveBeliefActionResult:
        """执行一次 selective top-1 retrieval 与 batch-1 action 推理。"""
        if not isinstance(request, SelectiveBeliefActionRequest):
            raise TypeError("request 必须是 SelectiveBeliefActionRequest")
        if request.belief_risk_accepted:
            retrieval = self.retriever.retrieve(request.query, top_k=1)
        else:
            retrieval = BeliefDemoRetrievalResult(
                query=request.query,
                hits=(),
                accepted=False,
                reason="belief_risk_rejected",
            )
        context = None
        if retrieval.accepted:
            hit = retrieval.hits[0]
            context = PreparedRawDemoContext(
                candidate_id=hit.candidate.candidate_id,
                state=hit.candidate.state,
                raw_action=hit.candidate.raw_action,
                retrieval_distance=hit.normalized_mse_distance,
            )
        prediction = self.predictor.predict(
            DemoRelativeActionChunkRequest(request.query.state, context)
        )
        return SelectiveBeliefActionResult(retrieval, prediction)

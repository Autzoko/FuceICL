"""从 Demo action sequence 抽取并执行轻量离散 phase grammar。"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch


def _compress(symbols: Sequence[int]) -> tuple[int, ...]:
    if not symbols:
        raise ValueError("离散 action sequence 不能为空")
    compressed = [int(symbols[0])]
    for symbol in symbols[1:]:
        value = int(symbol)
        if value != compressed[-1]:
            compressed.append(value)
    return tuple(compressed)


@dataclass(frozen=True)
class DiscreteActionGrammar:
    """共享 Demo phase trace 及其 causal prefix projection。"""

    symbols: tuple[int, ...]
    episode_count: int
    trace_counts: dict[str, int]

    def __post_init__(self) -> None:
        if not self.symbols or any(symbol not in (0, 1) for symbol in self.symbols):
            raise ValueError("grammar symbols 必须是非空二值序列")
        if any(a == b for a, b in zip(self.symbols, self.symbols[1:])):
            raise ValueError("grammar 必须是 run-length compressed trace")
        if self.episode_count <= 0:
            raise ValueError("grammar episode count 必须为正")

    def project(self, desired: int, phase_index: int) -> tuple[int, int, bool]:
        """把 desired symbol 投影到当前 grammar prefix，并返回新 phase。"""
        if desired not in (0, 1):
            raise ValueError("desired symbol 必须为 0/1")
        if phase_index < 0 or phase_index >= len(self.symbols):
            raise ValueError("grammar phase index 越界")
        current = self.symbols[phase_index]
        if desired == current:
            return desired, phase_index, False
        next_index = phase_index + 1
        if next_index < len(self.symbols) and desired == self.symbols[next_index]:
            return desired, next_index, False
        return current, phase_index, True

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbols": list(self.symbols),
            "symbol_names": ["open" if value else "closed" for value in self.symbols],
            "episode_count": self.episode_count,
            "trace_counts": dict(self.trace_counts),
        }


def infer_gripper_grammar(
    *,
    records: Sequence[Mapping[str, Any]],
    normalized_actions: torch.Tensor,
) -> DiscreteActionGrammar:
    """从每个 episode 的 token-0 gripper command 抽取共享 phase trace。"""
    if normalized_actions.ndim != 3 or normalized_actions.shape[-1] != 7:
        raise ValueError("normalized actions 必须为 [N,H,7]")
    if len(records) != len(normalized_actions):
        raise ValueError("records 与 actions 数量不一致")
    by_episode: dict[int, list[tuple[int, int]]] = defaultdict(list)
    symbols = (normalized_actions[:, 0, 6] >= 0.0).long().cpu().tolist()
    for record, symbol in zip(records, symbols):
        by_episode[int(record["episode"])].append((int(record["frame"]), symbol))
    traces = []
    for values in by_episode.values():
        ordered = [symbol for _, symbol in sorted(values)]
        traces.append(_compress(ordered))
    trace_counts = Counter("".join(str(value) for value in trace) for trace in traces)
    if len(trace_counts) != 1:
        raise ValueError(f"train episodes 没有共享 gripper grammar：{dict(trace_counts)}")
    trace = traces[0]
    return DiscreteActionGrammar(
        symbols=trace,
        episode_count=len(traces),
        trace_counts=dict(trace_counts),
    )

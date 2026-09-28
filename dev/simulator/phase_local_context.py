"""从逐帧 ManiSkill chunks 构造不跨 gripper phase 的长 Demo context。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch


@dataclass(frozen=True)
class PhaseLocalContextBank:
    geometry_sequence: torch.Tensor
    actions: torch.Tensor
    transition_mask: torch.Tensor
    lengths: torch.Tensor


def _phase(geometry: torch.Tensor) -> bool:
    return bool(geometry[15] >= 0.5)


def build_phase_local_context_bank(
    *,
    records: Sequence[dict[str, Any]],
    geometry: torch.Tensor,
    actions: torch.Tensor,
    stored_geometry_sequence: torch.Tensor,
    max_transitions: int,
) -> PhaseLocalContextBank:
    """沿同 episode 连续 frame 扩展，遇到 phase 或观测缺口即停止。"""
    if max_transitions <= 0:
        raise ValueError("max_transitions 必须为正")
    if len(records) != len(geometry) or len(records) != len(actions):
        raise ValueError("records、geometry、actions 长度不一致")
    stored_horizon = actions.shape[1]
    expected = (len(records), stored_horizon + 1, geometry.shape[1])
    if stored_geometry_sequence.shape != expected:
        raise ValueError(
            "stored geometry sequence shape 错误："
            f"expected={expected}, actual={stored_geometry_sequence.shape}"
        )
    if not torch.equal(stored_geometry_sequence[:, 0], geometry):
        raise ValueError("stored sequence 首帧与 geometry 不完全一致")

    by_frame: dict[tuple[int, int], int] = {}
    for index, record in enumerate(records):
        key = (int(record["episode"]), int(record["frame"]))
        if key in by_frame:
            raise ValueError(f"重复 episode/frame：{key}")
        by_frame[key] = index

    batch = len(records)
    context_geometry = geometry.new_zeros(
        (batch, max_transitions + 1, geometry.shape[1])
    )
    context_actions = actions.new_zeros(
        (batch, max_transitions, actions.shape[2])
    )
    transition_mask = torch.zeros(
        (batch, max_transitions),
        dtype=torch.bool,
        device=geometry.device,
    )
    lengths = torch.zeros(batch, dtype=torch.long, device=geometry.device)

    for index, record in enumerate(records):
        episode = int(record["episode"])
        start_frame = int(record["frame"])
        initial_phase = _phase(stored_geometry_sequence[index, 0])
        states = [stored_geometry_sequence[index, 0]]
        transition_actions = []
        for offset in range(max_transitions):
            if offset < stored_horizon:
                current = stored_geometry_sequence[index, offset]
                following = stored_geometry_sequence[index, offset + 1]
                action = actions[index, offset]
            else:
                continuation = by_frame.get((episode, start_frame + offset))
                if continuation is None:
                    break
                current = stored_geometry_sequence[continuation, 0]
                following = stored_geometry_sequence[continuation, 1]
                action = actions[continuation, 0]
            if not torch.equal(current, states[-1]):
                raise ValueError(
                    f"episode={episode} frame={start_frame + offset} context 不连续"
                )
            if _phase(current) != initial_phase or _phase(following) != initial_phase:
                break
            transition_actions.append(action)
            states.append(following)

        length = len(transition_actions)
        lengths[index] = length
        context_geometry[index, : length + 1] = torch.stack(states)
        context_geometry[index, length + 1 :] = states[-1]
        if length:
            context_actions[index, :length] = torch.stack(transition_actions)
            transition_mask[index, :length] = True

    return PhaseLocalContextBank(
        geometry_sequence=context_geometry,
        actions=context_actions,
        transition_mask=transition_mask,
        lengths=lengths,
    )

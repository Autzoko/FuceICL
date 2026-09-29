"""对冻结 RADM closed-loop rollouts 做描述性失败机制分析。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from dev.pointnet.compare_retrievers import _sha256
from dev.simulator.aggregate_query_aligned_closed_loop import _git_commit, _stats


POLICIES = (
    "bcsg_h6",
    "retriever_prior_mixture_h6",
    "radm_h6",
)


def _entropy(weights: Sequence[float]) -> float:
    values = np.asarray(weights, dtype=np.float64)
    return float(-(values * np.log(np.clip(values, 1e-12, None))).sum())


def _trajectory_set(chunk_ids: Sequence[str]) -> frozenset[str]:
    trajectories = []
    for chunk_id in chunk_ids:
        fields = chunk_id.split(":")
        if len(fields) < 4:
            raise ValueError(f"无法解析 candidate chunk ID：{chunk_id}")
        trajectories.append(fields[-2])
    return frozenset(trajectories)


def _episode_features(rollout: Mapping[str, Any]) -> dict[str, float | int | bool]:
    steps = rollout["step_records"]
    replans = [step for step in steps if step.get("mixture_posterior") is not None]
    tv = []
    kl = []
    entropy = []
    maximum_weight = []
    spans = []
    odds = []
    trajectory_sets = []
    top_trajectories = []
    for step in replans:
        prior = np.asarray(step["mixture_prior"], dtype=np.float64)
        posterior = np.asarray(step["mixture_posterior"], dtype=np.float64)
        tv.append(float(0.5 * np.abs(posterior - prior).sum()))
        kl.append(
            float(
                (
                    posterior
                    * np.log(
                        np.clip(posterior, 1e-12, None)
                        / np.clip(prior, 1e-12, None)
                    )
                ).sum()
            )
        )
        entropy.append(_entropy(posterior))
        maximum_weight.append(float(posterior.max()))
        spans.append(float(step["normalized_action_span_diameter"]))
        odds.append(float(step["maximum_odds_distortion"]))
        candidates = step["candidate_chunk_ids"]
        trajectory_sets.append(_trajectory_set(candidates))
        top_trajectories.append(candidates[0].split(":")[-2])
    jaccard = []
    for previous, current in zip(trajectory_sets, trajectory_sets[1:]):
        union = previous | current
        jaccard.append(len(previous & current) / max(len(union), 1))
    top_switches = [
        left != right
        for left, right in zip(top_trajectories, top_trajectories[1:])
    ]
    object_goal = [
        float(step["observed_object_to_goal_m"])
        for step in steps
        if step.get("observed_object_to_goal_m") is not None
    ]
    rewards = [float(step["reward"]) for step in steps]
    return {
        "success": bool(rollout["success"]),
        "steps": len(steps),
        "replans": len(replans),
        "mean_posterior_prior_tv": float(np.mean(tv)) if tv else 0.0,
        "mean_posterior_prior_kl": float(np.mean(kl)) if kl else 0.0,
        "mean_posterior_entropy": float(np.mean(entropy)) if entropy else 0.0,
        "mean_maximum_weight": (
            float(np.mean(maximum_weight)) if maximum_weight else 0.0
        ),
        "mean_action_span": float(np.mean(spans)) if spans else 0.0,
        "maximum_action_span": max(spans, default=0.0),
        "mean_odds_distortion": float(np.mean(odds)) if odds else 0.0,
        "candidate_trajectory_churn": (
            1.0 - float(np.mean(jaccard)) if jaccard else 0.0
        ),
        "top1_trajectory_switch_rate": (
            float(np.mean(top_switches)) if top_switches else 0.0
        ),
        "refusal_rate": float(np.mean([bool(step["refusal"]) for step in steps])),
        "translation_clip_rate": float(
            np.mean([bool(step["translation_clipped"]) for step in steps])
        ),
        "minimum_object_goal_m": min(object_goal, default=math.nan),
        "maximum_reward": max(rewards, default=math.nan),
        "ever_grasped": any(bool(step["is_grasped"]) for step in steps),
        "ever_placed": any(bool(step["is_obj_placed"]) for step in steps),
    }


def _numeric_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"episodes": 0}
    keys = (
        "mean_posterior_prior_tv",
        "mean_posterior_prior_kl",
        "mean_posterior_entropy",
        "mean_maximum_weight",
        "mean_action_span",
        "maximum_action_span",
        "mean_odds_distortion",
        "candidate_trajectory_churn",
        "top1_trajectory_switch_rate",
        "refusal_rate",
        "translation_clip_rate",
        "minimum_object_goal_m",
        "maximum_reward",
        "initial_radm_prior_action_delta_m",
    )
    output: dict[str, Any] = {
        "episodes": len(rows),
        "success_rate": float(np.mean([bool(row["success"]) for row in rows])),
        "ever_grasped_rate": float(
            np.mean([bool(row["ever_grasped"]) for row in rows])
        ),
        "ever_placed_rate": float(
            np.mean([bool(row["ever_placed"]) for row in rows])
        ),
    }
    for key in keys:
        values = [
            float(row[key])
            for row in rows
            if key in row and math.isfinite(float(row[key]))
        ]
        output[key] = _stats(values)
    return output


def _first_replan(rollout: Mapping[str, Any]) -> Mapping[str, Any]:
    for step in rollout["step_records"]:
        if step.get("mixture_posterior") is not None:
            return step
    raise ValueError("mixture rollout 没有有效 replan")


def _rollout_map(
    report: Mapping[str, Any],
    policy: str,
) -> dict[int, Mapping[str, Any]]:
    rows = {int(row["seed"]): row for row in report["rollouts"][policy]}
    if len(rows) != len(report["config"]["seeds"]):
        raise ValueError(f"{report['task']} {policy} seed 不完整")
    return rows


def run(
    *,
    project_root: Path,
    report_paths: Sequence[Path],
    output_path: Path,
) -> None:
    if output_path.exists():
        raise FileExistsError(f"输出已存在，拒绝覆盖：{output_path}")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in report_paths]
    task_analysis = {}
    transition_rows: dict[str, dict[str, list[dict[str, Any]]]] = {
        "radm_vs_rank1": {"win": [], "loss": [], "tie": []},
        "radm_vs_fixed_prior": {"win": [], "loss": [], "tie": []},
    }
    discordant = []
    for report in reports:
        task = str(report["task"])
        policy_maps = {
            policy: _rollout_map(report, policy) for policy in POLICIES
        }
        features = {
            policy: {
                seed: _episode_features(rollout)
                for seed, rollout in policy_maps[policy].items()
            }
            for policy in POLICIES[1:]
        }
        task_analysis[task] = {
            policy: {
                "all": _numeric_summary(list(features[policy].values())),
                "success": _numeric_summary(
                    [row for row in features[policy].values() if row["success"]]
                ),
                "failure": _numeric_summary(
                    [row for row in features[policy].values() if not row["success"]]
                ),
            }
            for policy in POLICIES[1:]
        }
        for seed in report["config"]["seeds"]:
            radm_rollout = policy_maps["radm_h6"][int(seed)]
            radm_features = features["radm_h6"][int(seed)]
            prior_rollout = policy_maps["retriever_prior_mixture_h6"][int(seed)]
            radm_first = _first_replan(radm_rollout)
            prior_first = _first_replan(prior_rollout)
            if radm_first["candidate_chunk_ids"] != prior_first["candidate_chunk_ids"]:
                raise RuntimeError(
                    "identical seed 的初始 Top-4 candidates 不一致"
                )
            action_delta = np.linalg.norm(
                np.asarray(radm_first["canonical_first_action"][:3])
                - np.asarray(prior_first["canonical_first_action"][:3])
            )
            radm_features = {
                **radm_features,
                "task": task,
                "seed": int(seed),
                "initial_radm_prior_action_delta_m": float(action_delta),
            }
            references = {
                "radm_vs_rank1": policy_maps["bcsg_h6"][int(seed)],
                "radm_vs_fixed_prior": prior_rollout,
            }
            for comparison, reference in references.items():
                delta = int(bool(radm_rollout["success"])) - int(
                    bool(reference["success"])
                )
                label = "win" if delta > 0 else "loss" if delta < 0 else "tie"
                transition_rows[comparison][label].append(radm_features)
                if label != "tie":
                    discordant.append(
                        {
                            "comparison": comparison,
                            "direction": label,
                            "task": task,
                            "seed": int(seed),
                            "reference_success": bool(reference["success"]),
                            "radm_success": bool(radm_rollout["success"]),
                            "initial_radm_prior_action_delta_m": float(
                                action_delta
                            ),
                        }
                    )
    transition_analysis = {
        comparison: {
            label: _numeric_summary(rows)
            for label, rows in groups.items()
        }
        for comparison, groups in transition_rows.items()
    }
    output = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(project_root),
        "analysis_type": (
            "post-hoc descriptive; no new confirmatory thresholds or reruns"
        ),
        "source_reports": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in report_paths
        ],
        "task_policy_outcome_strata": task_analysis,
        "paired_outcome_transitions": transition_analysis,
        "discordant_pairs": discordant,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(f"{output_path.suffix}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    print(json.dumps(transition_analysis, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(
        project_root=arguments.project_root.resolve(),
        report_paths=[path.resolve() for path in arguments.report],
        output_path=arguments.output.resolve(),
    )

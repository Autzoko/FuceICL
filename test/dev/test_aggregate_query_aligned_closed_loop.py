"""三任务 QA-LRDT/BCSG closed-loop 汇总合约测试。"""

from __future__ import annotations

import json
from contextlib import redirect_stdout
import io
from pathlib import Path
from tempfile import TemporaryDirectory

from dev.simulator.aggregate_query_aligned_closed_loop import (
    EXPECTED_TASKS,
    POLICIES,
    run,
)


def _write_report(path: Path, task: str) -> None:
    outcomes = {
        "phase_matched_copy_h6": (False, False),
        "phase_factorized_transport_h6": (False, True),
        "fixed_low_rank_transport_h6": (False, True),
        "query_aligned_transport_h6": (False, True),
        "bcsg_h6": (True, True),
    }
    rollouts = {
        policy: [
            {
                "seed": seed,
                "success": success,
                "step_records": (
                    [{"predictor_latency_ms": 1.0}]
                    if policy == "bcsg_h6"
                    else []
                ),
            }
            for seed, success in zip((1, 2), outcomes[policy])
        ]
        for policy in POLICIES
    }
    report = {
        "task": task,
        "config": {
            "seeds": [1, 2],
            "policies": list(POLICIES),
            "bootstrap_resamples": 100,
            "seed_generation_seed": 43,
        },
        "config_sha256": "config",
        "checkpoint_sha256": "checkpoint",
        "checkpoint_git_commit": "commit",
        "initial_signature_max_abs_difference": 0.0,
        "bank": {"chunk_ids_match_checkpoint": True},
        "structural_audit": {
            "identity_exact": True,
            "no_demo_qa_exact": True,
            "no_demo_bcsg_exact": True,
            "gate_zero_exact": True,
            "gate_one_exact": True,
        },
        "rollouts": rollouts,
    }
    path.write_text(json.dumps(report), encoding="utf-8")


def test_aggregate_three_tasks() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        paths = []
        for task in sorted(EXPECTED_TASKS):
            path = root / f"{task}.json"
            _write_report(path, task)
            paths.append(path)
        output = root / "aggregate.json"
        with redirect_stdout(io.StringIO()):
            run(project_root=root, report_paths=paths, output_path=output)
        report = json.loads(output.read_text(encoding="utf-8"))
        assert report["pooled"]["bcsg_h6"]["successes"] == 6
        assert report["protocol"]["task_seed_pairs"] == 6
        assert all(report["preregistered_criteria"].values())

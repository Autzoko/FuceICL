"""三任务 RSCC closed-loop 汇总合约测试。"""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from dev.simulator.aggregate_conformal_support_closed_loop import (
    CONFORMAL_POLICIES,
    run,
)
from dev.simulator.aggregate_query_aligned_closed_loop import EXPECTED_TASKS


def _write_report(path: Path, task: str) -> None:
    outcomes = {
        "phase_matched_copy_h6": (False, True),
        "bcsg_h6": (False, False),
        "codra_h6": (False, True),
        "rscc_h6": (True, True),
    }
    rollouts = {}
    for policy in CONFORMAL_POLICIES:
        rows = []
        for seed, success in zip((1, 2), outcomes[policy]):
            steps = []
            if policy == "rscc_h6":
                for step, accepted in zip((1, 26, 51, 76), (True, True, False, False)):
                    steps.append(
                        {
                            "step": step,
                            "residual_accepted": accepted,
                            "predictor_latency_ms": 1.0,
                        }
                    )
            rows.append(
                {"seed": seed, "success": success, "step_records": steps}
            )
        rollouts[policy] = rows
    report = {
        "task": task,
        "config": {
            "seeds": [1, 2],
            "policies": list(CONFORMAL_POLICIES),
            "bootstrap_resamples": 100,
            "seed_generation_seed": 43,
            "max_episode_steps": 100,
        },
        "config_sha256": "config",
        "checkpoint_sha256": "checkpoint",
        "checkpoint_git_commit": "commit",
        "calibration": {"sha256": "calibration"},
        "initial_signature_max_abs_difference": 0.0,
        "bank": {"chunk_ids_match_checkpoint": True},
        "structural_audit": {
            "identity_exact": True,
            "no_demo_qa_exact": True,
            "no_demo_bcsg_exact": True,
            "gate_zero_exact": True,
            "gate_one_exact": True,
        },
        "policy_diagnostics": {
            "rscc_h6": {
                "rejected_prediction_max_abs_error_from_copy": 0.0
            }
        },
        "rollouts": rollouts,
    }
    path.write_text(json.dumps(report), encoding="utf-8")


def test_aggregate_conformal_support_three_tasks() -> None:
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
        assert report["pooled"]["rscc_h6"]["successes"] == 6
        assert report["protocol"]["task_seed_pairs"] == 6
        assert all(report["criteria"].values())

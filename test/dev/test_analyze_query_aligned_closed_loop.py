"""Closed-loop support drift 分析合约测试。"""

from __future__ import annotations

import json
from contextlib import redirect_stdout
import io
from pathlib import Path
from tempfile import TemporaryDirectory

from dev.simulator.aggregate_query_aligned_closed_loop import (
    EXPECTED_TASKS,
    POLICIES,
)
from dev.simulator.analyze_query_aligned_closed_loop import run


def _step(step: int, distance: float, policy: str) -> dict[str, object]:
    return {
        "step": step,
        "replanned": True,
        "selected_chunk_id": "shared",
        "retrieval_distance": distance,
        "translation_clipped": distance > 1.0,
        "predicted_gate": 0.8 if policy == "bcsg_h6" else None,
        "normalized_translation_residual_l2_mean": (
            distance if policy in {"bcsg_h6", "query_aligned_transport_h6"} else None
        ),
    }


def _write_closed_loop(path: Path, task: str) -> None:
    report = {
        "task": task,
        "checkpoint_sha256": "checkpoint",
        "config": {"seeds": [1, 2]},
        "rollouts": {
            policy: [
                {
                    "seed": seed,
                    "success": seed == 1,
                    "step_records": [
                        _step(1, 0.5, policy),
                        _step(80, 1.5, policy),
                    ],
                }
                for seed in (1, 2)
            ]
            for policy in POLICIES
        },
    }
    path.write_text(json.dumps(report), encoding="utf-8")


def test_support_drift_analysis_contract() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        offline = root / "offline.json"
        offline.write_text(
            json.dumps(
                {
                    "checkpoint_sha256": "checkpoint",
                    "tasks": {
                        task: {"validation_distance_p95": 1.0}
                        for task in EXPECTED_TASKS
                    },
                }
            ),
            encoding="utf-8",
        )
        paths = []
        for task in sorted(EXPECTED_TASKS):
            path = root / f"{task}.json"
            _write_closed_loop(path, task)
            paths.append(path)
        output = root / "analysis.json"
        with redirect_stdout(io.StringIO()):
            run(
                project_root=root,
                closed_loop_paths=paths,
                offline_path=offline,
                output_path=output,
            )
        report = json.loads(output.read_text(encoding="utf-8"))
        checks = report["mechanistic_checks"]
        assert checks["all_initial_retrievals_identical"]
        assert checks["bcsg_ood_fraction_increases_first_to_last_bin"]
        assert checks["bcsg_gate_does_not_decrease_ood"]

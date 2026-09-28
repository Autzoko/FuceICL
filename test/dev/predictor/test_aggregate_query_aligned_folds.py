"""QA-LRDT 跨 fold 汇总的最小合约测试。"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from dev.pointnet.compare_retrievers import _sha256
from dev.predictor.aggregate_query_aligned_folds import run


def _write_fold(root: Path, fold_index: int) -> None:
    root.mkdir()
    tasks = [f"task_{fold_index}_{index}" for index in range(6)]
    shape = (2, 6, 7)
    target = np.zeros(shape, dtype=np.float32)
    mask = np.ones(shape[:2], dtype=bool)
    copy = target.copy()
    fixed = target.copy()
    aligned = target.copy()
    copy[..., :3] = 1.0
    fixed[..., :3] = 0.75
    aligned[..., :3] = 0.5
    artifact = root / "held_out_predictions.npz"
    np.savez_compressed(
        artifact,
        schema_version=np.asarray([1]),
        fold_name=np.asarray([f"fold_{fold_index}"]),
        held_out_tasks=np.asarray(tasks),
        query_indices=np.asarray([0, 1]),
        chunk_ids=np.asarray([f"chunk_{fold_index}_0", f"chunk_{fold_index}_1"]),
        tasks=np.asarray(tasks[:2]),
        group_ids=np.asarray(
            [f"{tasks[0]}:0:0", f"{tasks[1]}:0:0"]
        ),
        target_actions=target,
        target_masks=mask,
        retrieved__demo_action_copy=copy,
        retrieved__fixed_low_rank_transport=fixed,
        retrieved__query_aligned_transport=aligned,
        oracle__demo_action_copy=copy,
        oracle__fixed_low_rank_transport=fixed,
        oracle__query_aligned_transport=aligned,
    )
    report = {
        "fold_name": f"fold_{fold_index}",
        "held_out_tasks": tasks,
        "held_out_prediction_artifact": {
            "sha256": _sha256(artifact),
            "query_count": 2,
        },
        "config_sha256": "config",
        "context_summary_sha256": "context",
        "action_summary_sha256": "action",
        "retriever_checkpoint_sha256": "retriever",
        "text_scores_sha256": "text",
        "structural_audit": {
            "query_aligned_transport": {
                "identity_max_abs_error": 0.0,
                "no_demo_max_abs_output": 0.0,
                "rotation_gripper_max_abs_error": 0.0,
            }
        },
        "model_parameters": {"query_aligned_transport": 100_000},
        "latency": {"query_aligned_transport": {"p95_ms": 1.0}},
    }
    (root / "report.json").write_text(json.dumps(report), encoding="utf-8")


def test_aggregate_three_disjoint_folds() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        folds = [root / f"fold_{index}" for index in range(3)]
        for index, fold in enumerate(folds):
            _write_fold(fold, index)
        output = root / "aggregate.json"
        run(
            project_root=root,
            fold_roots=folds,
            output_path=output,
            expected_task_count=18,
            seed=43,
            resamples=50,
        )
        report = json.loads(output.read_text(encoding="utf-8"))
        assert report["conditions"]["retrieved"]["query_count"] == 6
        assert report["preregistered_criteria"][
            "q1_retrieved_translation_better_than_copy"
        ]
        assert report["preregistered_criteria"][
            "q4_structural_and_latency"
        ]

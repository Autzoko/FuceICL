"""生成关系语义数据并训练冻结 MiniLM 上的轻量 conformal head。"""

from __future__ import annotations

import argparse
import atexit
import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional
from torch.utils.data import DataLoader, TensorDataset

from dev.text_retriever.relation_effect_head import (
    RELATION_LABELS,
    RELATION_TO_ID,
    RelationEffectHead,
    RelationEffectHeadConfig,
    conformal_quantile,
    conformal_sets,
    mask_object_spans,
    singleton_predictions,
)
from lib.GLiNER2_Base import (
    TEXT_SCHEMA_SHA256,
    TEXT_SCHEMA_VERSION,
    GLiNERTextParser,
)
from lib.all_MiniLM_L6_v2 import MiniLMTextEncoder


SPLITS = ("train", "validation", "calibration")


@dataclass(frozen=True)
class RelationHeadTrainConfig:
    """Relation head 数据、优化与门控配置。"""

    schema_version: str
    seed: int
    hidden_dim: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    early_stopping_patience: int
    parse_batch_size: int
    encode_batch_size: int
    conformal_alpha: float
    minimum_validation_accuracy: float
    minimum_calibration_coverage: float
    minimum_calibration_singleton_rate: float
    maximum_parameters: int
    object_pairs: tuple[tuple[str, str], ...]
    templates: dict[str, dict[str, tuple[str, ...]]]

    @classmethod
    def from_json(cls, path: Path) -> "RelationHeadTrainConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["object_pairs"] = tuple(
            tuple(values) for values in raw["object_pairs"]
        )
        raw["templates"] = {
            split: {
                label: tuple(values) for label, values in by_label.items()
            }
            for split, by_label in raw["templates"].items()
        }
        return cls(**raw)

    def __post_init__(self) -> None:
        if self.schema_version != "relation-effect-head-v1":
            raise ValueError("未知 relation head schema")
        positive = (
            self.hidden_dim,
            self.epochs,
            self.batch_size,
            self.learning_rate,
            self.early_stopping_patience,
            self.parse_batch_size,
            self.encode_batch_size,
            self.maximum_parameters,
        )
        if min(positive) <= 0 or self.seed < 0 or self.weight_decay < 0:
            raise ValueError("训练配置非法")
        probabilities = (
            self.conformal_alpha,
            self.minimum_validation_accuracy,
            self.minimum_calibration_coverage,
            self.minimum_calibration_singleton_rate,
        )
        if any(not 0.0 < value <= 1.0 for value in probabilities):
            raise ValueError("概率配置必须位于 (0,1]")
        if self.conformal_alpha >= 1.0:
            raise ValueError("conformal_alpha 必须小于 1")
        if not self.object_pairs:
            raise ValueError("object_pairs 不能为空")
        for pair in self.object_pairs:
            if len(pair) != 2 or any(not value.strip() for value in pair):
                raise ValueError("每个 object pair 必须包含两个非空字符串")
        if set(self.templates) != set(SPLITS):
            raise ValueError("templates 必须包含 train/validation/calibration")
        all_templates = []
        for split in SPLITS:
            by_label = self.templates[split]
            if set(by_label) != set(RELATION_LABELS):
                raise ValueError(f"{split} 必须覆盖全部 relation labels")
            for values in by_label.values():
                if not values or any(not value.strip() for value in values):
                    raise ValueError("template 不能为空")
                all_templates.extend(values)
        if len(all_templates) != len(set(all_templates)):
            raise ValueError("不同 split/label 之间不能复用相同 template")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def _generate_examples(config: RelationHeadTrainConfig) -> list[dict[str, Any]]:
    rows = []
    for split in SPLITS:
        for label in RELATION_LABELS:
            for template_id, template in enumerate(config.templates[split][label]):
                for object_id, (active, reference) in enumerate(config.object_pairs):
                    try:
                        text = template.format(active=active, reference=reference)
                    except (KeyError, ValueError) as error:
                        raise ValueError(
                            f"template 格式非法：{template}"
                        ) from error
                    rows.append(
                        {
                            "split": split,
                            "label": label,
                            "label_id": RELATION_TO_ID[label],
                            "template_id": template_id,
                            "object_id": object_id,
                            "active_object": active,
                            "reference_object": reference,
                            "text": text,
                        }
                    )
    texts = [row["text"].casefold().strip() for row in rows]
    if len(texts) != len(set(texts)):
        raise ValueError("生成数据存在重复文本")
    return rows


def _load_external(path: Path) -> list[dict[str, Any]]:
    config = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for pair in config["language_pairs"]:
        for operation, label in (("toward", "approach"), ("away", "separate")):
            rows.append(
                {
                    "source": path.stem,
                    "language_pair": pair["name"],
                    "category": pair["category"],
                    "operation": operation,
                    "label": label,
                    "label_id": RELATION_TO_ID[label],
                    "text": pair[operation],
                }
            )
    return rows


def _encode_rows(
    rows: list[dict[str, Any]],
    *,
    parser: GLiNERTextParser,
    encoder: MiniLMTextEncoder,
    parse_batch_size: int,
    encode_batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    texts = [row["text"] for row in rows]
    parsed = parser.parse_many(texts, batch_size=parse_batch_size)
    masked = [mask_object_spans(item) for item in parsed]
    raw_embeddings = encoder.encode(texts, batch_size=encode_batch_size)
    masked_embeddings = encoder.encode(masked, batch_size=encode_batch_size)
    diagnostics = {
        "rows": len(rows),
        "object_mask_changed_fraction": float(
            np.mean([left != right for left, right in zip(texts, masked, strict=True)])
        ),
        "mean_objects_per_text": float(
            np.mean([len(item.objects) for item in parsed])
        ),
    }
    for row, structure, masked_text in zip(rows, parsed, masked, strict=True):
        row["masked_text"] = masked_text
        row["gliner_relation_effect"] = structure.relation_effect
        row["gliner_relation_effect_confidence"] = (
            structure.relation_effect_confidence
        )
    return raw_embeddings, masked_embeddings, diagnostics


def _classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, Any]:
    probabilities = torch.softmax(logits, dim=1)
    predictions = probabilities.argmax(dim=1)
    confusion = torch.zeros(
        (len(RELATION_LABELS), len(RELATION_LABELS)), dtype=torch.int64
    )
    for truth, prediction in zip(labels, predictions, strict=True):
        confusion[int(truth), int(prediction)] += 1
    return {
        "rows": len(labels),
        "cross_entropy": float(functional.cross_entropy(logits, labels)),
        "accuracy": float((predictions == labels).float().mean()),
        "per_class_accuracy": {
            label: float(
                (predictions[labels == identifier] == identifier).float().mean()
            )
            for identifier, label in enumerate(RELATION_LABELS)
        },
        "confusion_true_rows_predicted_columns": confusion.tolist(),
    }


def _conformal_metrics(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    *,
    quantile: float,
) -> dict[str, Any]:
    prediction_sets = conformal_sets(probabilities, quantile=quantile)
    covered = prediction_sets[torch.arange(len(labels)), labels]
    sizes = prediction_sets.sum(dim=1)
    singleton = sizes == 1
    singleton_labels = prediction_sets.to(torch.int64).argmax(dim=1)
    return {
        "coverage": float(covered.float().mean()),
        "mean_set_size": float(sizes.float().mean()),
        "singleton_rate": float(singleton.float().mean()),
        "singleton_accuracy": (
            float((singleton_labels[singleton] == labels[singleton]).float().mean())
            if bool(singleton.any())
            else None
        ),
        "empty_set_rate": float((sizes == 0).float().mean()),
    }


def _external_report(
    rows: list[dict[str, Any]],
    logits: torch.Tensor,
    *,
    quantile: float,
) -> dict[str, Any]:
    labels = torch.tensor([row["label_id"] for row in rows], dtype=torch.long)
    probabilities = torch.softmax(logits, dim=1)
    predictions = probabilities.argmax(dim=1)
    prediction_sets = conformal_sets(probabilities, quantile=quantile)
    singleton = singleton_predictions(prediction_sets)
    details = []
    for row, probability, prediction, accepted in zip(
        rows, probabilities, predictions, singleton, strict=True
    ):
        details.append(
            {
                **{key: value for key, value in row.items() if key != "label_id"},
                "prediction": RELATION_LABELS[int(prediction)],
                "correct": int(prediction) == row["label_id"],
                "probabilities": {
                    label: float(probability[index])
                    for index, label in enumerate(RELATION_LABELS)
                },
                "conformal_singleton": accepted,
                "conformal_correct": accepted == row["label"] if accepted else None,
            }
        )
    by_source = {}
    for source in sorted({row["source"] for row in rows}):
        indices = torch.tensor(
            [row["source"] == source for row in rows], dtype=torch.bool
        )
        by_source[source] = {
            "classification": _classification_metrics(
                logits[indices], labels[indices]
            ),
            "conformal": _conformal_metrics(
                probabilities[indices], labels[indices], quantile=quantile
            ),
        }
    return {
        "overall": {
            "classification": _classification_metrics(logits, labels),
            "conformal": _conformal_metrics(
                probabilities, labels, quantile=quantile
            ),
        },
        "by_source": by_source,
        "details": details,
    }


@torch.no_grad()
def _head_latency(
    model: RelationEffectHead,
    raw: torch.Tensor,
    masked: torch.Tensor,
    *,
    iterations: int = 2000,
) -> dict[str, float]:
    for _ in range(100):
        model(raw[:1], masked[:1])
    samples = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        model(raw[:1], masked[:1])
        samples.append((time.perf_counter_ns() - started) / 1e6)
    return {
        "median_ms": float(np.median(samples)),
        "p95_ms": float(np.quantile(samples, 0.95)),
    }


def run(
    *,
    project_root: Path,
    config_path: Path,
    output_root: Path,
    gliner_model_dir: Path,
    minilm_model_dir: Path,
    external_config_paths: list[Path],
    config: RelationHeadTrainConfig,
) -> None:
    if output_root.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output_root}")
    _seed_everything(config.seed)
    rows = _generate_examples(config)
    external_rows = [
        row for path in external_config_paths for row in _load_external(path)
    ]
    generated_texts = {row["text"].casefold().strip() for row in rows}
    overlap = [
        row["text"]
        for row in external_rows
        if row["text"].casefold().strip() in generated_texts
    ]
    if overlap:
        raise ValueError(
            f"external development text 泄漏到生成数据：{overlap[:3]}"
        )

    parser = GLiNERTextParser(gliner_model_dir, device="cpu")
    encoder = MiniLMTextEncoder(minilm_model_dir, device="cpu")
    raw, masked, encoding_diagnostics = _encode_rows(
        rows,
        parser=parser,
        encoder=encoder,
        parse_batch_size=config.parse_batch_size,
        encode_batch_size=config.encode_batch_size,
    )
    external_raw, external_masked, external_encoding_diagnostics = _encode_rows(
        external_rows,
        parser=parser,
        encoder=encoder,
        parse_batch_size=config.parse_batch_size,
        encode_batch_size=config.encode_batch_size,
    )
    labels = torch.tensor([row["label_id"] for row in rows], dtype=torch.long)
    split_indices = {
        split: torch.tensor(
            [index for index, row in enumerate(rows) if row["split"] == split],
            dtype=torch.long,
        )
        for split in SPLITS
    }
    model_config = RelationEffectHeadConfig(hidden_dim=config.hidden_dim)
    model = RelationEffectHead(model_config)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    train_indices = split_indices["train"]
    validation_indices = split_indices["validation"]
    dataset = TensorDataset(
        raw[train_indices], masked[train_indices], labels[train_indices]
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    best_state = copy.deepcopy(model.state_dict())
    best_validation_loss = float("inf")
    best_epoch = 0
    without_improvement = 0
    training_log = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        for raw_batch, masked_batch, label_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = functional.cross_entropy(
                model(raw_batch, masked_batch), label_batch
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            validation_logits = model(
                raw[validation_indices], masked[validation_indices]
            )
            validation_loss = float(
                functional.cross_entropy(
                    validation_logits, labels[validation_indices]
                )
            )
            validation_accuracy = float(
                (
                    validation_logits.argmax(dim=1)
                    == labels[validation_indices]
                )
                .float()
                .mean()
            )
        training_log.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            }
        )
        if validation_loss < best_validation_loss - 1e-8:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            without_improvement = 0
        else:
            without_improvement += 1
        if without_improvement >= config.early_stopping_patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(raw, masked)
        external_logits = model(external_raw, external_masked)
    calibration_indices = split_indices["calibration"]
    calibration_probabilities = torch.softmax(
        logits[calibration_indices], dim=1
    )
    quantile = conformal_quantile(
        calibration_probabilities,
        labels[calibration_indices],
        alpha=config.conformal_alpha,
    )
    split_metrics = {
        split: {
            "classification": _classification_metrics(
                logits[indices], labels[indices]
            ),
            "conformal": _conformal_metrics(
                torch.softmax(logits[indices], dim=1),
                labels[indices],
                quantile=quantile,
            ),
        }
        for split, indices in split_indices.items()
    }
    external = _external_report(
        external_rows, external_logits, quantile=quantile
    )
    latency = _head_latency(model, raw, masked)
    criteria = {
        "T1_validation_accuracy": (
            split_metrics["validation"]["classification"]["accuracy"]
            >= config.minimum_validation_accuracy
        ),
        "T2_calibration_coverage": (
            split_metrics["calibration"]["conformal"]["coverage"]
            >= config.minimum_calibration_coverage
        ),
        "T3_calibration_singleton_rate": (
            split_metrics["calibration"]["conformal"]["singleton_rate"]
            >= config.minimum_calibration_singleton_rate
        ),
        "T4_parameter_budget": parameters <= config.maximum_parameters,
    }
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(project_root),
        "config": asdict(config),
        "config_sha256": _sha256(config_path),
        "model_config": asdict(model_config),
        "protocol": {
            "best_epoch_selected_only_by_validation_loss": best_epoch,
            "epochs_run": len(training_log),
            "external_development_used_for_model_selection": False,
            "external_exact_text_overlap": 0,
            "relation_labels": RELATION_LABELS,
        },
        "artifacts": {
            "gliner_schema_version": TEXT_SCHEMA_VERSION,
            "gliner_schema_sha256": TEXT_SCHEMA_SHA256,
            "minilm_model_sha256": _sha256(
                minilm_model_dir / "model.safetensors"
            ),
            "external_config_sha256": {
                path.name: _sha256(path) for path in external_config_paths
            },
        },
        "encoding": {
            "generated": encoding_diagnostics,
            "external": external_encoding_diagnostics,
        },
        "splits": split_metrics,
        "conformal": {
            "alpha": config.conformal_alpha,
            "nonconformity_quantile": quantile,
            "acceptance_rule": "prediction set must contain exactly one label",
        },
        "external_development": external,
        "parameters": parameters,
        "head_latency": latency,
        "criteria": criteria,
        "training_gate_passed": all(criteria.values()),
    }

    temporary = output_root.with_name(
        f".{output_root.name}.incomplete-{os.getpid()}"
    )
    temporary.mkdir(parents=True)

    def cleanup() -> None:
        shutil.rmtree(temporary, ignore_errors=True)

    atexit.register(cleanup)
    examples_path = temporary / "examples.jsonl"
    with examples_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    log_path = temporary / "training_log.jsonl"
    with log_path.open("w", encoding="utf-8") as stream:
        for row in training_log:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    checkpoint_path = temporary / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model_config),
            "labels": RELATION_LABELS,
            "conformal_alpha": config.conformal_alpha,
            "conformal_quantile": quantile,
            "config_sha256": _sha256(config_path),
            "minilm_model_sha256": report["artifacts"]["minilm_model_sha256"],
            "gliner_schema_sha256": TEXT_SCHEMA_SHA256,
        },
        checkpoint_path,
    )
    report["files"] = {
        "checkpoint.pt": _sha256(checkpoint_path),
        "examples.jsonl": _sha256(examples_path),
        "training_log.jsonl": _sha256(log_path),
    }
    (temporary / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_root)
    atexit.unregister(cleanup)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not report["training_gate_passed"]:
        raise RuntimeError(
            "relation head 未通过 train/validation/calibration 门控"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gliner-model-dir", type=Path, required=True)
    parser.add_argument("--minilm-model-dir", type=Path, required=True)
    parser.add_argument(
        "--external-config",
        type=Path,
        action="append",
        required=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(
        project_root=arguments.project_root.resolve(),
        config_path=arguments.config.resolve(),
        output_root=arguments.output_root.resolve(),
        gliner_model_dir=arguments.gliner_model_dir.resolve(),
        minilm_model_dir=arguments.minilm_model_dir.resolve(),
        external_config_paths=[path.resolve() for path in arguments.external_config],
        config=RelationHeadTrainConfig.from_json(arguments.config.resolve()),
    )

"""训练 frozen MiniLM contextual tokens 上的 relation attention head。"""

from __future__ import annotations

import argparse
import atexit
import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional
from torch.utils.data import DataLoader, TensorDataset

from dev.text_retriever.relation_effect_head import (
    RELATION_LABELS,
    TokenRelationEffectHead,
    TokenRelationEffectHeadConfig,
    conformal_quantile,
    mask_object_spans,
)
from dev.text_retriever.train_relation_effect_head import (
    RelationHeadTrainConfig,
    _classification_metrics,
    _conformal_metrics,
    _external_report,
    _generate_examples,
    _git_commit,
    _load_external,
    _seed_everything,
    _sha256,
)
from lib.GLiNER2_Base import (
    TEXT_SCHEMA_SHA256,
    TEXT_SCHEMA_VERSION,
    GLiNERTextParser,
)
from lib.all_MiniLM_L6_v2 import MiniLMTextEncoder


SPLITS = ("train", "validation", "calibration")


@dataclass(frozen=True)
class TokenHeadTrainConfig:
    """Token attention head 的独立训练与门控配置。"""

    schema_version: str
    seed: int
    attention_dim: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    early_stopping_patience: int
    parse_batch_size: int
    encode_batch_size: int
    max_length: int
    conformal_alpha: float
    minimum_validation_accuracy: float
    minimum_calibration_coverage: float
    minimum_calibration_singleton_rate: float
    maximum_parameters: int

    @classmethod
    def from_json(cls, path: Path) -> "TokenHeadTrainConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def __post_init__(self) -> None:
        if self.schema_version != "relation-effect-token-head-v1":
            raise ValueError("未知 token relation head schema")
        positive = (
            self.attention_dim,
            self.epochs,
            self.batch_size,
            self.learning_rate,
            self.early_stopping_patience,
            self.parse_batch_size,
            self.encode_batch_size,
            self.max_length,
            self.maximum_parameters,
        )
        if min(positive) <= 0 or self.seed < 0 or self.weight_decay < 0:
            raise ValueError("token head 训练配置非法")
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


def _parse_masked_texts(
    rows: list[dict[str, Any]],
    *,
    parser: GLiNERTextParser,
    batch_size: int,
) -> tuple[list[str], dict[str, float | int]]:
    parsed = parser.parse_many(
        [row["text"] for row in rows], batch_size=batch_size
    )
    masked = [mask_object_spans(item) for item in parsed]
    for row, structure, masked_text in zip(rows, parsed, masked, strict=True):
        row["masked_text"] = masked_text
        row["gliner_relation_effect"] = structure.relation_effect
        row["gliner_relation_effect_confidence"] = (
            structure.relation_effect_confidence
        )
    return masked, {
        "rows": len(rows),
        "object_mask_changed_fraction": float(
            np.mean(
                [
                    row["text"] != masked_text
                    for row, masked_text in zip(rows, masked, strict=True)
                ]
            )
        ),
        "mean_objects_per_text": float(
            np.mean([len(item.objects) for item in parsed])
        ),
    }


@torch.no_grad()
def _contextual_tokens(
    texts: list[str],
    *,
    encoder: MiniLMTextEncoder,
    batch_size: int,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    encoded = encoder.tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_special_tokens_mask=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"].to(torch.bool)
    special_mask = encoded["special_tokens_mask"].to(torch.bool)
    token_mask = attention_mask & ~special_mask
    hidden = []
    for start in range(0, len(texts), batch_size):
        stop = min(start + batch_size, len(texts))
        model_inputs = {
            "input_ids": input_ids[start:stop].to(encoder.device),
            "attention_mask": encoded["attention_mask"][start:stop].to(
                encoder.device
            ),
        }
        output = encoder.model(**model_inputs).last_hidden_state
        hidden.append(output.cpu().float())
    tokens = torch.cat(hidden, dim=0)
    if tokens.shape[:2] != token_mask.shape:
        raise ValueError("contextual tokens 与 mask shape 不一致")
    return tokens, token_mask, input_ids


@torch.no_grad()
def _attention_audit(
    model: TokenRelationEffectHead,
    tokens: torch.Tensor,
    token_mask: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    tokenizer: Any,
    limit: int = 5,
) -> list[list[dict[str, float | str]]]:
    _, weights = model.pool(tokens, token_mask)
    output = []
    for row_weights, row_ids, row_mask in zip(
        weights, input_ids, token_mask, strict=True
    ):
        indices = torch.where(row_mask)[0]
        ranked = indices[torch.argsort(row_weights[indices], descending=True)]
        output.append(
            [
                {
                    "token": tokenizer.convert_ids_to_tokens(
                        int(row_ids[index])
                    ),
                    "weight": float(row_weights[index]),
                }
                for index in ranked[:limit]
            ]
        )
    return output


@torch.no_grad()
def _head_latency(
    model: TokenRelationEffectHead,
    tokens: torch.Tensor,
    token_mask: torch.Tensor,
    *,
    iterations: int = 2000,
) -> dict[str, float]:
    for _ in range(100):
        model(tokens[:1], token_mask[:1])
    samples = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        model(tokens[:1], token_mask[:1])
        samples.append((time.perf_counter_ns() - started) / 1e6)
    return {
        "median_ms": float(np.median(samples)),
        "p95_ms": float(np.quantile(samples, 0.95)),
    }


def run(
    *,
    project_root: Path,
    config_path: Path,
    data_config_path: Path,
    output_root: Path,
    gliner_model_dir: Path,
    minilm_model_dir: Path,
    external_config_paths: list[Path],
    config: TokenHeadTrainConfig,
    data_config: RelationHeadTrainConfig,
) -> None:
    if output_root.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output_root}")
    _seed_everything(config.seed)
    rows = _generate_examples(data_config)
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
    masked, encoding_diagnostics = _parse_masked_texts(
        rows, parser=parser, batch_size=config.parse_batch_size
    )
    external_masked, external_encoding_diagnostics = _parse_masked_texts(
        external_rows, parser=parser, batch_size=config.parse_batch_size
    )
    tokens, token_mask, input_ids = _contextual_tokens(
        masked,
        encoder=encoder,
        batch_size=config.encode_batch_size,
        max_length=config.max_length,
    )
    external_tokens, external_token_mask, external_input_ids = (
        _contextual_tokens(
            external_masked,
            encoder=encoder,
            batch_size=config.encode_batch_size,
            max_length=config.max_length,
        )
    )
    labels = torch.tensor([row["label_id"] for row in rows], dtype=torch.long)
    split_indices = {
        split: torch.tensor(
            [index for index, row in enumerate(rows) if row["split"] == split],
            dtype=torch.long,
        )
        for split in SPLITS
    }
    model_config = TokenRelationEffectHeadConfig(
        attention_dim=config.attention_dim
    )
    model = TokenRelationEffectHead(model_config)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    train_indices = split_indices["train"]
    validation_indices = split_indices["validation"]
    dataset = TensorDataset(
        tokens[train_indices],
        token_mask[train_indices],
        labels[train_indices],
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
        for token_batch, mask_batch, label_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = functional.cross_entropy(
                model(token_batch, mask_batch), label_batch
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            validation_logits = model(
                tokens[validation_indices], token_mask[validation_indices]
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
        logits = model(tokens, token_mask)
        external_logits = model(external_tokens, external_token_mask)
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
    top_tokens = _attention_audit(
        model,
        external_tokens,
        external_token_mask,
        external_input_ids,
        tokenizer=encoder.tokenizer,
    )
    for detail, attention in zip(
        external["details"], top_tokens, strict=True
    ):
        detail["top_attention_tokens"] = attention
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
        "data_config_sha256": _sha256(data_config_path),
        "model_config": asdict(model_config),
        "protocol": {
            "best_epoch_selected_only_by_validation_loss": best_epoch,
            "epochs_run": len(training_log),
            "encoder_frozen": True,
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
            "sequence_length": int(tokens.shape[1]),
        },
        "splits": split_metrics,
        "conformal": {
            "alpha": config.conformal_alpha,
            "nonconformity_quantile": quantile,
            "acceptance_rule": "prediction set must contain exactly one label",
        },
        "external_development": external,
        "parameters": parameters,
        "head_latency": _head_latency(model, tokens, token_mask),
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
            "data_config_sha256": _sha256(data_config_path),
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
            "token relation head 未通过 train/validation/calibration 门控"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, required=True)
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
    data_config_path = arguments.data_config.resolve()
    run(
        project_root=arguments.project_root.resolve(),
        config_path=arguments.config.resolve(),
        data_config_path=data_config_path,
        output_root=arguments.output_root.resolve(),
        gliner_model_dir=arguments.gliner_model_dir.resolve(),
        minilm_model_dir=arguments.minilm_model_dir.resolve(),
        external_config_paths=[path.resolve() for path in arguments.external_config],
        config=TokenHeadTrainConfig.from_json(arguments.config.resolve()),
        data_config=RelationHeadTrainConfig.from_json(data_config_path),
    )

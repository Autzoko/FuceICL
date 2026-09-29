"""仅微调 MiniLM 最后一层与 token head 的 relation adapter 实验。"""

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
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader, TensorDataset

from dev.text_retriever.relation_effect_head import (
    RELATION_LABELS,
    TokenRelationEffectHead,
    TokenRelationEffectHeadConfig,
    conformal_quantile,
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
from dev.text_retriever.train_relation_effect_token_head import (
    _attention_audit,
    _parse_masked_texts,
)
from lib.GLiNER2_Base import (
    TEXT_SCHEMA_SHA256,
    TEXT_SCHEMA_VERSION,
    GLiNERTextParser,
)
from lib.all_MiniLM_L6_v2 import MiniLMTextEncoder


SPLITS = ("train", "validation", "calibration")


@dataclass(frozen=True)
class LastBlockTrainConfig:
    """最后一层 relation adapter 的优化与门控配置。"""

    schema_version: str
    seed: int
    attention_dim: int
    epochs: int
    batch_size: int
    head_learning_rate: float
    encoder_learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    early_stopping_patience: int
    parse_batch_size: int
    max_length: int
    conformal_alpha: float
    minimum_validation_accuracy: float
    minimum_calibration_coverage: float
    minimum_calibration_singleton_rate: float
    maximum_trainable_parameters: int
    latency_warmup: int
    latency_iterations: int

    @classmethod
    def from_json(cls, path: Path) -> "LastBlockTrainConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def __post_init__(self) -> None:
        if self.schema_version != "relation-effect-last-block-v1":
            raise ValueError("未知 last-block relation schema")
        positive = (
            self.attention_dim,
            self.epochs,
            self.batch_size,
            self.head_learning_rate,
            self.encoder_learning_rate,
            self.gradient_clip_norm,
            self.early_stopping_patience,
            self.parse_batch_size,
            self.max_length,
            self.maximum_trainable_parameters,
            self.latency_warmup,
            self.latency_iterations,
        )
        if min(positive) <= 0 or self.seed < 0 or self.weight_decay < 0:
            raise ValueError("last-block 训练配置非法")
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


class LastBlockRelationModel(nn.Module):
    """冻结 MiniLM 前五层，仅训练最后 block 与 relation head。"""

    def __init__(
        self,
        encoder: nn.Module,
        head: TokenRelationEffectHead,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head
        if not hasattr(self.encoder, "encoder") or not hasattr(
            self.encoder.encoder, "layer"
        ):
            raise TypeError("MiniLM encoder 结构不符合 BERT layer contract")
        if len(self.encoder.encoder.layer) != 6:
            raise ValueError("预期 MiniLM 包含 6 个 transformer blocks")
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in self.last_block.parameters():
            parameter.requires_grad_(True)

    @property
    def last_block(self) -> nn.Module:
        return self.encoder.encoder.layer[-1]

    def set_train_mode(self) -> None:
        """冻结层禁用 dropout，最后 block/head 保持训练模式。"""
        self.encoder.eval()
        self.last_block.train()
        self.head.train()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
        return self.head(hidden, token_mask)


def _tokenize(
    texts: list[str],
    *,
    tokenizer: Any,
    max_length: int,
) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_special_tokens_mask=True,
        return_tensors="pt",
    )
    attention_mask = encoded["attention_mask"].to(torch.bool)
    token_mask = attention_mask & ~encoded["special_tokens_mask"].to(torch.bool)
    if not bool(token_mask.any(dim=1).all()):
        raise ValueError("每条文本必须包含至少一个非 special token")
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": attention_mask,
        "token_mask": token_mask,
    }


@torch.no_grad()
def _predict(
    model: LastBlockRelationModel,
    encoded: dict[str, torch.Tensor],
    *,
    batch_size: int,
) -> torch.Tensor:
    model.eval()
    outputs = []
    for start in range(0, len(encoded["input_ids"]), batch_size):
        stop = min(start + batch_size, len(encoded["input_ids"]))
        outputs.append(
            model(
                encoded["input_ids"][start:stop],
                encoded["attention_mask"][start:stop],
                encoded["token_mask"][start:stop],
            ).cpu()
        )
    return torch.cat(outputs)


@torch.no_grad()
def _full_latency(
    model: LastBlockRelationModel,
    encoded: dict[str, torch.Tensor],
    *,
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    model.eval()
    arguments = (
        encoded["input_ids"][:1],
        encoded["attention_mask"][:1],
        encoded["token_mask"][:1],
    )
    for _ in range(warmup):
        model(*arguments)
    samples = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        model(*arguments)
        samples.append((time.perf_counter_ns() - started) / 1e6)
    return {
        "median_ms": float(np.median(samples)),
        "p95_ms": float(np.quantile(samples, 0.95)),
    }


@torch.no_grad()
def _external_attention(
    model: LastBlockRelationModel,
    encoded: dict[str, torch.Tensor],
    *,
    tokenizer: Any,
) -> list[list[dict[str, float | str]]]:
    model.eval()
    hidden = model.encoder(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
    ).last_hidden_state
    return _attention_audit(
        model.head,
        hidden,
        encoded["token_mask"],
        encoded["input_ids"],
        tokenizer=tokenizer,
    )


def run(
    *,
    project_root: Path,
    config_path: Path,
    data_config_path: Path,
    output_root: Path,
    gliner_model_dir: Path,
    minilm_model_dir: Path,
    external_config_paths: list[Path],
    config: LastBlockTrainConfig,
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
    text_encoder = MiniLMTextEncoder(minilm_model_dir, device="cpu")
    masked, encoding_diagnostics = _parse_masked_texts(
        rows, parser=parser, batch_size=config.parse_batch_size
    )
    external_masked, external_encoding_diagnostics = _parse_masked_texts(
        external_rows, parser=parser, batch_size=config.parse_batch_size
    )
    encoded = _tokenize(
        masked,
        tokenizer=text_encoder.tokenizer,
        max_length=config.max_length,
    )
    external_encoded = _tokenize(
        external_masked,
        tokenizer=text_encoder.tokenizer,
        max_length=config.max_length,
    )
    labels = torch.tensor([row["label_id"] for row in rows], dtype=torch.long)
    split_indices = {
        split: torch.tensor(
            [index for index, row in enumerate(rows) if row["split"] == split],
            dtype=torch.long,
        )
        for split in SPLITS
    }
    head_config = TokenRelationEffectHeadConfig(
        attention_dim=config.attention_dim
    )
    model = LastBlockRelationModel(
        text_encoder.model,
        TokenRelationEffectHead(head_config),
    )
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    total_parameters = sum(
        parameter.numel() for parameter in model.parameters()
    )
    train_indices = split_indices["train"]
    validation_indices = split_indices["validation"]
    dataset = TensorDataset(
        encoded["input_ids"][train_indices],
        encoded["attention_mask"][train_indices],
        encoded["token_mask"][train_indices],
        labels[train_indices],
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.head.parameters(),
                "lr": config.head_learning_rate,
            },
            {
                "params": model.last_block.parameters(),
                "lr": config.encoder_learning_rate,
            },
        ],
        weight_decay=config.weight_decay,
    )
    best_head_state = copy.deepcopy(model.head.state_dict())
    best_block_state = copy.deepcopy(model.last_block.state_dict())
    best_validation_loss = float("inf")
    best_epoch = 0
    without_improvement = 0
    training_log = []
    for epoch in range(1, config.epochs + 1):
        model.set_train_mode()
        losses = []
        for input_ids, attention_mask, token_mask, label_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = functional.cross_entropy(
                model(input_ids, attention_mask, token_mask), label_batch
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ],
                config.gradient_clip_norm,
            )
            optimizer.step()
            losses.append(float(loss.detach()))
        validation_encoded = {
            key: value[validation_indices] for key, value in encoded.items()
        }
        validation_logits = _predict(
            model,
            validation_encoded,
            batch_size=config.batch_size,
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
            best_head_state = copy.deepcopy(model.head.state_dict())
            best_block_state = copy.deepcopy(model.last_block.state_dict())
            without_improvement = 0
        else:
            without_improvement += 1
        if without_improvement >= config.early_stopping_patience:
            break
    model.head.load_state_dict(best_head_state)
    model.last_block.load_state_dict(best_block_state)
    logits = _predict(model, encoded, batch_size=config.batch_size)
    external_logits = _predict(
        model, external_encoded, batch_size=config.batch_size
    )
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
    attention = _external_attention(
        model,
        external_encoded,
        tokenizer=text_encoder.tokenizer,
    )
    for detail, top_tokens in zip(
        external["details"], attention, strict=True
    ):
        detail["top_attention_tokens"] = top_tokens
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
        "T4_trainable_parameter_budget": (
            trainable_parameters <= config.maximum_trainable_parameters
        ),
    }
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(project_root),
        "config": asdict(config),
        "config_sha256": _sha256(config_path),
        "data_config_sha256": _sha256(data_config_path),
        "head_config": asdict(head_config),
        "protocol": {
            "best_epoch_selected_only_by_validation_loss": best_epoch,
            "epochs_run": len(training_log),
            "unfrozen_transformer_blocks": 1,
            "frozen_transformer_blocks": 5,
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
            "generated_sequence_length": int(encoded["input_ids"].shape[1]),
        },
        "splits": split_metrics,
        "conformal": {
            "alpha": config.conformal_alpha,
            "nonconformity_quantile": quantile,
            "acceptance_rule": "prediction set must contain exactly one label",
        },
        "external_development": external,
        "parameters": {
            "total": total_parameters,
            "trainable": trainable_parameters,
        },
        "full_model_latency": _full_latency(
            model,
            external_encoded,
            warmup=config.latency_warmup,
            iterations=config.latency_iterations,
        ),
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
            "last_block_state_dict": model.last_block.state_dict(),
            "head_state_dict": model.head.state_dict(),
            "head_config": asdict(head_config),
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
            "last-block relation adapter 未通过 validation/calibration 门控"
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
        config=LastBlockTrainConfig.from_json(arguments.config.resolve()),
        data_config=RelationHeadTrainConfig.from_json(data_config_path),
    )

"""面向文本初筛的 GLiNER2 本地推理接口。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from gliner2 import GLiNER2


DEFAULT_MODEL_DIR = Path(__file__).with_name("checkpoint")
DEFAULT_SCHEMA_PATH = Path(__file__).with_name("schema.json")
TEXT_SCHEMA_SHA256 = hashlib.sha256(DEFAULT_SCHEMA_PATH.read_bytes()).hexdigest()
TEXT_SCHEMA_VERSION = json.loads(
    DEFAULT_SCHEMA_PATH.read_text(encoding="utf-8")
)["schema_version"]


@dataclass(frozen=True)
class TextSpan:
    """GLiNER 抽取出的原文片段。字符区间采用左闭右开约定。"""

    text: str
    confidence: float | None
    start: int | None
    end: int | None

    def to_dict(self) -> dict[str, Any]:
        """转换为可 JSON 序列化的字典。"""
        return asdict(self)


@dataclass(frozen=True)
class ParsedInstruction:
    """文本初筛所需的两类结构化信息。"""

    text: str
    goal_operation: str | None
    goal_operation_confidence: float | None
    operations: tuple[TextSpan, ...]
    objects: tuple[TextSpan, ...]

    def to_dict(self) -> dict[str, Any]:
        """转换为可 JSON 序列化的字典。"""
        return asdict(self)


class GLiNERTextParser:
    """抽取 canonical goal operation、动作片段和无角色物体集合。

    模型仅从本地目录加载，不会隐式访问网络。Operation 与 Object 使用独立
    schema 推理，避免两个实体标签在同一次抽取中相互竞争。
    """

    def __init__(
        self,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        *,
        device: str = "cpu",
        schema_path: str | Path = DEFAULT_SCHEMA_PATH,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.schema_path = Path(schema_path)
        self._validate_local_files()
        self.config = json.loads(self.schema_path.read_text(encoding="utf-8"))
        self.model = GLiNER2.from_pretrained(
            str(self.model_dir),
            map_location=device,
            local_files_only=True,
        )
        self._operation_schema = self._build_operation_schema()
        self._object_schema = self._build_object_schema()

    def _validate_local_files(self) -> None:
        if not self.model_dir.joinpath("model.safetensors").is_file():
            raise FileNotFoundError(
                f"GLiNER checkpoint 不完整：{self.model_dir / 'model.safetensors'}"
            )
        if not self.schema_path.is_file():
            raise FileNotFoundError(f"GLiNER schema 不存在：{self.schema_path}")

    def _build_operation_schema(self):
        thresholds = self.config["thresholds"]
        definition = self.config["entities"]["operation"]
        return (
            self.model.create_schema()
            .classification(
                "goal_operation",
                self.config["goal_operations"],
                cls_threshold=float(thresholds["classification"]),
            )
            .entities(
                {definition["label"]: definition["description"]},
                threshold=float(thresholds["operation_span"]),
            )
        )

    def _build_object_schema(self):
        definition = self.config["entities"]["object"]
        return self.model.create_schema().entities(
            {definition["label"]: definition["description"]},
            threshold=float(self.config["thresholds"]["object_span"]),
        )

    @staticmethod
    def _validate_texts(texts: Sequence[str], batch_size: int) -> list[str]:
        if isinstance(texts, str):
            raise TypeError("texts 必须是字符串序列，单条文本请使用 parse()")
        if batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")
        values = list(texts)
        if any(not isinstance(text, str) for text in values):
            raise TypeError("texts 中的每个元素都必须是字符串")
        if any(not text.strip() for text in values):
            raise ValueError("instruction 不能为空字符串")
        return values

    @staticmethod
    def _classification(
        result: Mapping[str, Any],
        task: str,
    ) -> tuple[str | None, float | None]:
        value = result.get(task)
        if not isinstance(value, Mapping):
            return None, None
        label = value.get("label")
        confidence = value.get("confidence")
        return (
            str(label) if label is not None else None,
            float(confidence) if confidence is not None else None,
        )

    @staticmethod
    def _spans(result: Mapping[str, Any], label: str) -> tuple[TextSpan, ...]:
        entities = result.get("entities", {})
        values = entities.get(label, []) if isinstance(entities, Mapping) else []
        spans = [
            TextSpan(
                text=str(value["text"]),
                confidence=(
                    float(value["confidence"])
                    if value.get("confidence") is not None
                    else None
                ),
                start=int(value["start"]) if value.get("start") is not None else None,
                end=int(value["end"]) if value.get("end") is not None else None,
            )
            for value in values
            if value.get("text") is not None
        ]
        return tuple(
            sorted(
                spans,
                key=lambda span: (
                    span.start is None,
                    span.start if span.start is not None else 0,
                ),
            )
        )

    def parse_many(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 8,
    ) -> list[ParsedInstruction]:
        """批量解析指令，返回结果顺序与输入严格一致。"""
        values = self._validate_texts(texts, batch_size)
        if not values:
            return []

        thresholds = self.config["thresholds"]
        operation_results = self.model.batch_extract(
            values,
            self._operation_schema,
            batch_size=batch_size,
            threshold=float(thresholds["operation_span"]),
            include_confidence=True,
            include_spans=True,
        )
        object_results = self.model.batch_extract(
            values,
            self._object_schema,
            batch_size=batch_size,
            threshold=float(thresholds["object_span"]),
            include_confidence=True,
            include_spans=True,
        )

        operation_label = self.config["entities"]["operation"]["label"]
        object_label = self.config["entities"]["object"]["label"]
        parsed = []
        for text, operation_result, object_result in zip(
            values,
            operation_results,
            object_results,
            strict=True,
        ):
            goal, goal_confidence = self._classification(
                operation_result,
                "goal_operation",
            )
            parsed.append(
                ParsedInstruction(
                    text=text,
                    goal_operation=goal,
                    goal_operation_confidence=goal_confidence,
                    operations=self._spans(operation_result, operation_label),
                    objects=self._spans(object_result, object_label),
                )
            )
        return parsed

    def parse(self, text: str) -> ParsedInstruction:
        """解析单条非空指令。"""
        return self.parse_many([text], batch_size=1)[0]

"""GLiNER2 文本初筛解析器的稳定本地接口。"""

from .parser import (
    DEFAULT_MODEL_DIR,
    DEFAULT_SCHEMA_PATH,
    TEXT_SCHEMA_SHA256,
    TEXT_SCHEMA_VERSION,
    GLiNERTextParser,
    ParsedInstruction,
    TextSpan,
)

__all__ = [
    "DEFAULT_MODEL_DIR",
    "DEFAULT_SCHEMA_PATH",
    "TEXT_SCHEMA_SHA256",
    "TEXT_SCHEMA_VERSION",
    "GLiNERTextParser",
    "ParsedInstruction",
    "TextSpan",
]

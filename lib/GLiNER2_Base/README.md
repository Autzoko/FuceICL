# GLiNER2 文本解析接口

本目录封装 `fastino/gliner2-base-v1` 的本地推理接口。稳定入口为：

```python
from lib.GLiNER2_Base import GLiNERTextParser

parser = GLiNERTextParser()
parsed = parser.parse("Put the red block into the bowl")
```

输出 `ParsedInstruction` 包含：

- `goal_operation`：规范化的最终操作类别；
- `operations`：保留原文表达的全部动作片段；
- `objects`：不区分角色的任务相关物体片段，并保留颜色、大小、位置等修饰语。

唯一 schema 为 `schema.json`。checkpoint 位于被 Git 忽略的 `checkpoint/`，接口以
`local_files_only=True` 加载，不会自动下载模型。

# FuseICL

面向单臂机器人 In-Context Imitation Learning 的研究代码库。当前文本初筛器使用：

```text
0.8 × MiniLM(原始任务描述 cosine)
+ 0.2 × MiniLM(GLiNER 无角色物体集合的双向最近邻 cosine)
```

GLiNER 同时保留 canonical goal operation 与原文 operation spans，但当前冻结分数
不使用 operation；它们作为后续独立路由或分桶的稳定数据字段。

## 最小用法

```python
from src.components.retriever import TextCandidate, TextRetriever

retriever = TextRetriever.from_local_models()
retriever.build_index(
    [
        TextCandidate("task-1", "Put the red block into the bowl"),
        TextCandidate("task-2", "Open the top drawer"),
    ]
)
result = retriever.retrieve("Place the red object in the bowl", top_k=10)
```

本地 checkpoint 分别存放于 `lib/GLiNER2_Base/checkpoint` 和
`lib/all_MiniLM_L6_v2/checkpoint`，加载过程不会隐式访问网络。

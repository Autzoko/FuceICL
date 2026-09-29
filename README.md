# FuseICL

面向单臂机器人 In-Context Imitation Learning 的研究代码库。当前文本初筛器使用：

```text
0.8 × MiniLM(原始任务描述 cosine)
+ 0.2 × MiniLM(GLiNER 无角色物体集合的双向最近邻 cosine)
```

GLiNER 同时保留 canonical goal operation 与原文 operation spans，但当前冻结分数
不使用 operation；它们作为后续独立路由或分桶的稳定数据字段。

自由文本相似度只用于高召回初筛，不能直接授权机器人执行。进入几何 chunk 检索前，数据适配器或上层任务
接口应提供规范化 `TaskKey(operation, active_object, reference_object, relation_effect, qualifiers)`；
`TaskBucketRouter` 只接受完整 key 精确匹配，未知或缺失 bucket 显式拒绝，不回退到语义最近任务。自由文本到
`TaskKey` 的可靠转换仍是研究问题，不能把 GLiNER 的高置信输出等同于安全保证。

## 最小用法

```python
from src.components.retriever import (
    TaskBucketCandidate,
    TaskBucketRouter,
    TaskKey,
    TextCandidate,
    TextRetriever,
)

retriever = TextRetriever.from_local_models()
retriever.build_index(
    [
        TextCandidate("task-1", "Put the red block into the bowl"),
        TextCandidate("task-2", "Open the top drawer"),
    ]
)
result = retriever.retrieve("Place the red object in the bowl", top_k=10)

router = TaskBucketRouter()
key = TaskKey(
    operation="push",
    active_object="red cube",
    reference_object="green marker",
    relation_effect="approach",
)
router.build_index([TaskBucketCandidate("demo-bucket-1", key)])
route = router.route(key)
assert route.accepted
```

本地 checkpoint 分别存放于 `lib/GLiNER2_Base/checkpoint` 和
`lib/all_MiniLM_L6_v2/checkpoint`，加载过程不会隐式访问网络。

## 几何 embedding 精排接口

冻结的 PointNet/几何 encoder 只负责生成 query/candidate embedding；分桶后的精确排序由
`ExactEmbeddingRetriever` 完成。它不加载文本或点云模型，默认返回 top-4，并限制同一 episode 最多一个 chunk：

```python
import torch

from src.components.retriever import (
    EmbeddingCandidate,
    EmbeddingQuery,
    ExactEmbeddingRetriever,
)

local_retriever = ExactEmbeddingRetriever()
local_retriever.build_index(
    [
        EmbeddingCandidate(
            candidate_id="episode-1:chunk-3",
            task_key=key,
            embedding=torch.tensor([0.4, 0.8]),
            episode_id="episode-1",
            payload={"action_ref": "arrays.zarr/episode-1/chunk-3"},
        )
    ]
)
local_result = local_retriever.retrieve(
    EmbeddingQuery(key, torch.tensor([0.5, 0.7]))
)
assert local_result.accepted
```

`minimum_score` 默认为空，因为拒绝阈值必须由独立 calibration 数据确定，不能在组件中硬编码。未知 task key、低于已配置
阈值以及未建立索引均有独立的可审计行为。

## Demo-anchored Action Predictor

当前稳定 Predictor 是 `LayoutEquivariantDemoPolicy`：输入 query/demo state 和已经过 layout frame transport 的
Demo H-step action，网络只能输出有界 residual；没有独立的 query-only action head。

```python
import torch

from src.components.predictor import LayoutEquivariantDemoPolicy

policy = LayoutEquivariantDemoPolicy()
action_chunk = policy(
    query_state=torch.zeros(1, 21),
    demo_state=torch.zeros(1, 21),
    transported_demo_action=torch.zeros(1, 6, 7),
    demo_mask=torch.ones(1),
)
assert policy.parameter_count == 11_940
```

默认模型无 Demo 时严格输出零，相同 query/demo state 时严格返回 transported Demo action，gripper action 不被
residual 修改。当前已确认 checkpoint 仍使用 21D 单帧几何状态；覆盖完整轨迹前必须先解决物体遮挡下的因果 belief，
不能把 simulator pose 直接填入该接口。

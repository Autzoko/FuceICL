# Retriever v1 原型

本目录实现两种“文本粗筛 + 局部重排”原型。当前推荐先以无需训练的
`ExplicitFeatureRetriever` 作为强基线；`MultistageRetriever` 保留 Tiny PointNet++
learned context，供后续训练后做对照。两者都只选择数据库中已有的 Demo chunk，不重定位、
不缩放，也不修改轨迹或 payload。

## 推荐强基线

```text
src TextRetriever Top-K
  -> active/target role-aware semantic similarity
  -> object position + confidence-gated orientation similarity
  -> active-target relative layout + metric extent similarity
  -> object-relative EEF position/orientation continuity
  -> gripper-width continuity
  -> weighted Top-N original Demo chunks
```

基础权重为 `0.15 text + 0.25 semantic + 0.15 object pose + 0.20 layout +
0.20 EEF pose + 0.05 gripper`。其中几何部分还乘以 `text_score × active_semantic_score`
compatibility gate，避免任务或物体不相关的候选仅因空间位置恰好接近而反超。权重和 RBF
尺度是可审计的初始值，需要在 RLBench validation split 上调参后冻结。对称物体或不可靠
的真实 pose 应降低 orientation confidence；confidence 为 0 时方向分数回退至 0.5，而不是
产生错误的高置信匹配。

## 数据流

```text
RetrieverCandidate[]
  ├─ text -> src/components/retriever/TextRetriever -> text Top-K
  └─ start context -> GeometricContextEncoder -> 离线 embedding

RetrieverQuery
  ├─ text -> 同一个 TextRetriever
  └─ current context -> 同一个 GeometricContextEncoder

text Top-K
  -> learned geometry cosine
  -> explicit layout similarity
  -> EEF/state continuity
  -> operation/phase/gripper/direction compatibility
  -> weighted ranking -> original Demo payload
```

`GeometricContext` 当前面向简单单臂场景：一个 active object、可选 target object、分割后
且以物体中心平移归一的 partial point cloud、base-frame 中心/尺度、object-relative EEF
pose、EEF velocity 和 gripper width。点云保留米制与 base 轴方向；EEF 旋转采用
rotation-6D。

## 使用

```python
from dev.retriever import GeometricContextEncoder, MultistageRetriever
from src.components.retriever import TextRetriever

text_retriever = TextRetriever.from_local_models(device="cpu")
context_encoder = GeometricContextEncoder()
retriever = MultistageRetriever(text_retriever, context_encoder)

retriever.build_index(candidates)
result = retriever.retrieve(query, top_k=4)
```

显式强基线不需要 Context Encoder：

```python
from dev.retriever import ExplicitFeatureRetriever

retriever = ExplicitFeatureRetriever(text_retriever)
retriever.build_index(candidates)
result = retriever.retrieve(query, top_k=4)
```

当前 `MultistageRetrieverConfig` 权重只是透明 baseline。`GeometricContextEncoder` 和其中
的 Tiny PointNet++ 目前没有训练 checkpoint；随机初始化只能验证数据流，不能用于比较
真实检索效果。正式实验必须加载训练权重，并在 index manifest 中记录 checkpoint hash。

## RLBench 真实数据基线

`rlbench_adapter.py` 可直接读取每个 task 的 ZIP，不需要整套解压。它从 front/overhead
depth、GT mask 与相机标定生成 object-centric partial cloud，并从统一机器人字段构造
EEF/gripper key；可变长且任务相关的 `task_low_dim_state` 不进入 key。

```bash
# JUBAIL 上由 Slurm 脚本抽取 train candidates + val queries
sbatch src/scripts/hpc/extract_rlbench_retrieval_eval.slurm "$PWD"

# 本机 FuseICL 环境使用真实 GLiNER/MiniLM checkpoint 评估
conda run -n FuseICL python -m dev.retriever.evaluate_rlbench \
  --input dev/retriever/results/rlbench_eval_v1/chunks.jsonl \
  --output dev/retriever/results/rlbench_eval_v1/report.json
```

首轮 6-task、192 candidates、96 held-out queries 实验中，text-only 到显式 baseline 的
strict Recall@1/4/10 从 `11.49/49.43/68.97%` 提升到 `66.67/85.06/91.95%`。这里的
strict positive 同时约束 task、phase、gripper、layout 与 EEF continuity；详细协议和边界
记录在 `_notes/research/05_rlbench_retriever_eval_v1.md`。

## 当前边界

- 只使用 candidate chunk 起点 context；candidate future action 仅作为 metadata/payload。
- 缺失 action metadata 按中性分数处理，避免不同数据集标注完整度直接决定排序。
- sim 与 real 必须走同一 depth、segmentation、坐标转换、裁剪和采样流程。
- 复杂多物体关系、对称物体 pose、短历史、质量 hard filter 和训练 loss 留待下一版。

测试步骤与数据选择见 [TEST_PLAN.md](TEST_PLAN.md)。

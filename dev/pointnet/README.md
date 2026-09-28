# Tiny PointNet++ RLBench Pilot

本目录负责生成 Tiny PointNet++ Retriever 的训练数据。PointNet++ 不作为孤立的物体分类器
训练，而是作为共享 Siamese context encoder 的点云分支；future action/effect 只用于构造
监督标签，不进入 query/candidate key。

## Pilot 数据

默认配置 `config/rlbench_pilot.json` 使用 RLBench 全部已下载任务：

- train 最多 5,000 chunks；
- val 最多 1,000 chunks；
- 每物体 512 点，front + overhead partial cloud；
- phase 优先由 gripper close/open 事件定位，无事件时显式记录 `time_fallback`；
- 机器人 segment 由“跨任务高频公共 handle + 接触前 EEF 局部系刚性轨迹”
  联合识别，不硬编码 handle ID；公共 handle 校准默认覆盖每任务两个 episode；
- active object 从排除机器人后的 segment 中，用 EEF 距离、segment motion 与
  EEF-motion coupling 联合打分；存在 gripper event 时从 contact 锁定 active handle，
  后续 phase 不允许切换到附近物体；
- target 仅在 active 确实移动、target 稳定且终点接近时保存；
- 低置信 active-object chunk 直接丢弃；
- 输出 sharded NPZ、manifest、pair labels、summary 和 SHA-256。

PointNet 输入包括 active/可选 target centered metric cloud。base-frame center、extent、
object-relative EEF pose/velocity、gripper state 作为独立数值输入。`task_low_dim_state` 不进入
模型输入。

## 输出格式

```text
pointnet_pilot_v1/
├── shards/{train,val}-*.npz
├── manifest-{train,val}.jsonl
├── pairs-{train,val}.jsonl
├── summary.json
└── SHA256SUMS
```

Manifest 保存 provenance、active/target confidence 与 shard row。Pair labels 保存最多 4 个
跨 episode positives，以及 wrong-phase、wrong-gripper、wrong-layout 和 geometry-collision
hard negatives。

## 本地验证

```bash
conda run -n FuseICL python -m unittest discover -s test -t . -v
```

## JUBAIL

```bash
module load miniconda/3-4.11.0
conda env create -f dev/pointnet/environment-jubail.yml

sbatch src/scripts/hpc/preprocess_rlbench_pointnet.slurm \
  "$PWD" \
  /scratch/ll5582/data/RLBench/processed/pointnet_pilot_v4
```

Slurm job 使用 CPU，不申请或占用 GPU，也不会操作其他队列任务。训练脚本将在数据审计通过
后单独提交。

## Retriever 训练

`dataset.py` 从 pair labels 中确定性采样跨 episode positive 和四类 hard negative。
`retriever_model.py` 共享 Tiny PointNet++ 编码 active/target cloud，再融合 layout、
object-relative EEF pose/velocity 与 gripper width。未来 action/effect 只作为辅助监督。

```bash
sbatch src/scripts/hpc/train_tiny_pointnetpp.slurm \
  "$PWD" \
  /scratch/ll5582/data/RLBench/processed/pointnet_pilot_v4 \
  /scratch/ll5582/data/RLBench/training/tiny_pointnetpp_pilot_v1
```

训练每个 epoch 报告 hard-triplet accuracy，以及 global/same-task Recall@1/4/10 和
MRR。checkpoint 保存数据 summary hash、Git commit、state normalization 和完整配置。

## 统一对比评测

`compare_retrievers.py` 在同一个 val split、相同候选池、相同 pair labels 上比较：

- 只使用当前观测的显式几何 RBF 强基线；
- Tiny PointNet++ 最佳 checkpoint；
- 固定 PointNet 权重为 0.25、0.50、0.75 的分数融合。

几何 RBF 尺度仅由 train positives 校准，禁止使用验证标签选尺度；主融合结果固定为 0.50。
评测分别报告 global 与模拟文本初筛后的 same-task Recall@1/4/10、MRR、四类 hard-negative
intrusion，以及相对显式基线的配对 bootstrap Recall@4 置信区间。

```bash
sbatch src/scripts/hpc/compare_pointnet_retrievers.slurm \
  "$PWD" \
  /scratch/ll5582/data/RLBench/processed/pointnet_pilot_v4 \
  /scratch/ll5582/data/RLBench/training/tiny_pointnetpp_pilot_v1/best.pt \
  /scratch/ll5582/data/RLBench/evaluation/pointnet_pilot_v1.json
```

## 端到端文本候选池

先在本地冻结的 GLiNER2+MiniLM 上为唯一 instruction groups 预计算可审计分数，再将小型
分数产物交给 JUBAIL 评估。正式评测将文本漏召回计为失败，并比较 Text-only、
Text+Geometry、Text+PointNet 与 Text+Fusion。

```bash
python -m dev.text_retriever.precompute_rlbench_text_scores \
  --manifest /path/to/manifest-val.jsonl \
  --output /path/to/text_scores_v1.npz

sbatch src/scripts/hpc/evaluate_end_to_end_retriever.slurm \
  "$PWD" \
  /scratch/ll5582/data/RLBench/processed/pointnet_pilot_v4 \
  /scratch/ll5582/data/RLBench/training/tiny_pointnetpp_pilot_v1/best.pt \
  /scratch/ll5582/data/RLBench/evaluation/text_scores_v1.npz \
  /scratch/ll5582/data/RLBench/evaluation/retriever_e2e_v1.json
```

## 选择性检索与拒绝

`evaluate_selective_retrieval.py` 按 episode 划分 calibration/test，只用已有文本和 PointNet
分数构造置信特征。阈值在 calibration 上选择，test 报告 precision–coverage、correct recall
和 episode-block bootstrap；该实验不增加新的视觉或文本模型前向。

```bash
sbatch src/scripts/hpc/evaluate_selective_retrieval.slurm \
  "$PWD" \
  /scratch/ll5582/data/RLBench/processed/pointnet_pilot_v4 \
  /scratch/ll5582/data/RLBench/training/tiny_pointnetpp_pilot_v1/best.pt \
  /scratch/ll5582/data/RLBench/evaluation/text_scores_v1.npz \
  /scratch/ll5582/data/RLBench/evaluation/retriever_reliability_v1.json
```

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
  联合识别，不硬编码 handle ID；
- active object 从排除机器人后的 segment 中，用 EEF 距离、segment motion 与
  EEF-motion coupling 联合打分；
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
  /scratch/ll5582/data/RLBench/processed/pointnet_pilot_v3
```

Slurm job 使用 CPU，不申请或占用 GPU，也不会操作其他队列任务。训练脚本将在数据审计通过
后单独提交。

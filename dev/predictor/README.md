# Demo-Conditioned Predictor Pilot

本目录先验证一个必要机制：在相同 query observation 下，正确 Demo 是否比随机或错误阶段
Demo 更能预测未来动作。Pilot 复用 RLBench v4 的 12-frame 聚合监督，只预测
`[delta_xyz, delta_axis_angle, gripper_delta]`，不把该结果视为最终 action-chunk 性能。

三个受控模型：

- `query_only`：只看 query context 的行为克隆下界；
- `demo_concat`：直接拼接 query、Demo context 和 Demo action；
- `demo_action_prior`：以 Demo action 为基点，query 只能门控有限残差，并加入反事实
  Demo utility margin。

评估使用 oracle、真实 Retriever、random-same-task、wrong-phase、wrong-layout、no-demo 和
shuffled-action 条件。主模型没有有效 Demo 时严格输出零，用结构保证它不能成为普通观测策略。

```bash
sbatch src/scripts/hpc/train_open_loop_predictor.slurm \
  "$PWD" \
  /scratch/ll5582/data/RLBench/processed/pointnet_pilot_v4 \
  /scratch/ll5582/data/RLBench/training/tiny_pointnetpp_pilot_v1/best.pt \
  /scratch/ll5582/data/RLBench/evaluation/text_scores_v1.npz \
  /scratch/ll5582/data/RLBench/training/open_loop_predictor_v1
```

## H-step action chunk 数据

E3 使用查询时刻末端坐标系中的累计动作：每个 token 是
`[local_translation(3), local_axis_angle(3), gripper_open_target]`。`H=6`、帧间隔为
2，因此最后一个 token 与现有 12-frame 聚合监督位于同一时刻，可直接执行一致性审计。
相对同一查询位姿的累计表示避免逐步积分误差，并对全局刚体坐标变换保持不变。

```bash
sbatch src/scripts/hpc/preprocess_rlbench_action_chunks.slurm \
  "$PWD" \
  /scratch/ll5582/data/RLBench/processed/pointnet_pilot_v4 \
  /scratch/ll5582/data/RLBench/processed/action_chunk_pilot_v1
```

数据审计通过后，使用同一冻结 Retriever embedding 训练 query-only、Demo concat、
action-prior，以及同结构但不使用反事实依赖损失的 action-prior 消融：

```bash
sbatch src/scripts/hpc/train_action_chunk_predictor.slurm \
  "$PWD" \
  /scratch/ll5582/data/RLBench/processed/pointnet_pilot_v4 \
  /scratch/ll5582/data/RLBench/processed/action_chunk_pilot_v1 \
  /scratch/ll5582/data/RLBench/training/tiny_pointnetpp_pilot_v1/best.pt \
  /scratch/ll5582/data/RLBench/evaluation/text_scores_v1.npz \
  /scratch/ll5582/data/RLBench/training/action_chunk_predictor_pilot_v1
```

若 frozen Retriever embedding 的 residual 不能超过 Demo copy，使用显式一阶搬运检验
canonical geometry 是否足以完成修正：

```bash
sbatch src/scripts/hpc/train_jacobian_transport.slurm \
  "$PWD" \
  /scratch/ll5582/data/RLBench/processed/pointnet_pilot_v4 \
  /scratch/ll5582/data/RLBench/processed/action_chunk_pilot_v1 \
  /scratch/ll5582/data/RLBench/training/tiny_pointnetpp_pilot_v1/best.pt \
  /scratch/ll5582/data/RLBench/evaluation/text_scores_v1.npz \
  /scratch/ll5582/data/RLBench/training/jacobian_transport_pilot_v1
```

固定 checkpoint 后，以 `task + variation + episode` 为 block 做 paired bootstrap，避免把同一
episode 的多个 chunks 当作独立样本：

```bash
sbatch src/scripts/hpc/evaluate_jacobian_significance.slurm \
  "$PWD" \
  /scratch/ll5582/data/RLBench/processed/pointnet_pilot_v4 \
  /scratch/ll5582/data/RLBench/processed/action_chunk_pilot_v1 \
  /scratch/ll5582/data/RLBench/training/tiny_pointnetpp_pilot_v1/best.pt \
  /scratch/ll5582/data/RLBench/evaluation/text_scores_v1.npz \
  /scratch/ll5582/data/RLBench/training/jacobian_transport_pilot_v1/local_jacobian_transport.pt \
  /scratch/ll5582/data/RLBench/evaluation/jacobian_significance_v1.json
```

# ManiSkill3 Simulation Pilot

本目录用于 E4 仿真闭环，不与 RLBench 离线训练环境混装依赖。JUBAIL 使用独立环境
`FuseICL-maniskill`，固定 ManiSkill 3.0.1、NumPy 1.26 和 PyTorch 2.1。首批任务只选
Panda 单臂的 `PickCube-v1`、`PushCube-v1`、`StackCube-v1`；双臂和移动机器人任务不进入
初始实验。

验证顺序：

1. `audit_maniskill_archive.py` 直接读取 ZIP，核对 episode、控制器、成功率和 HDF5 shape；
2. 在 V100 节点验证 `pointcloud + pd_ee_delta_pose + physx_cpu` reset/step；
3. 保存 observation structure、设备、版本和控制结果，作为后续闭环的环境门槛；
4. 将官方 Demo 重放到统一 `pd_ee_delta_pose` 后生成 canonical EEF action chunks；
5. 最后才接入 Retriever、拒绝机制与 receding-horizon Predictor。

JUBAIL 的 home cache 可能触发配额限制，Slurm 脚本会把 Matplotlib、XDG、Torch 和
Hugging Face 缓存全部定向到 `/scratch/ll5582/.cache/fuceicl-maniskill`。
ManiSkill 3.0.1 的 PickCube 构建过程即使在 `render_backend=none` 下仍创建
`RenderMaterial`，因此 JUBAIL 无 Vulkan 的普通 compute 节点不作为可运行后端。
单环境 smoke 使用 CPU physics 与 GPU rendering；不启用会向硬编码
`~/.sapien/physx` 下载运行库的 `physx_cuda`。

```bash
sbatch src/scripts/hpc/smoke_maniskill.slurm \
  "$PWD" \
  /scratch/ll5582/data/ManiSkill3/evaluation/smoke_pick_cube_pointcloud_v1.json
```

通过 smoke 后，先把 PickCube 的 RL `pd_ee_delta_pose` HDF5/JSON 解压到独立 pilot 目录，
再用官方工具按 environment states 重放 32 条，补齐 pointcloud observations。输出文件名由
ManiSkill 固定为 `trajectory.pointcloud.pd_ee_delta_pose.physx_cpu.{h5,json}`，脚本禁止覆盖：

```bash
sbatch src/scripts/hpc/replay_maniskill_pointcloud.slurm \
  "$PWD" \
  /scratch/ll5582/data/ManiSkill3/processed/replay_pick_cube_v1/source/trajectory.h5 \
  32
```

重放结束后必须运行 `audit_replayed_trajectory.py`，核对 7D action、episode 成功标签、
observation/action 时间偏移、16k pointcloud、有限值和 segmentation labels；审计通过前不进入
chunk 预处理。

审计通过后，自动用 replay 中的 cube actor position 只做离线 segmentation-label 对齐；正式
chunk 中只保存点云质心得到的 geometry、下采样点和 canonical actions：

```bash
sbatch src/scripts/hpc/preprocess_maniskill_chunks.slurm \
  "$PWD" \
  /scratch/ll5582/data/ManiSkill3/processed/replay_pick_cube_v1/source/trajectory.pointcloud.pd_ee_delta_pose.physx_cpu.h5 \
  /scratch/ll5582/data/ManiSkill3/processed/replay_pick_cube_v1/source/trajectory.pointcloud.pd_ee_delta_pose.physx_cpu.json \
  /scratch/ll5582/data/ManiSkill3/processed/pick_cube_chunks_v1
```

产物生成后运行 `audit_maniskill_chunks.py`，核对 shard allowlist、shape/finite/range、H-step
target frames、episode split 隔离与动作尺度；审计通过前不训练 Predictor。

审计通过后先运行 zero-training Demo-prior baseline，验证 17D geometry retrieval 是否足以从
train episodes 为 validation query 找到有用 action chunk：

```bash
sbatch src/scripts/hpc/evaluate_maniskill_demo_prior.slurm \
  "$PWD" \
  /scratch/ll5582/data/ManiSkill3/processed/pick_cube_chunks_v1 \
  /scratch/ll5582/data/ManiSkill3/evaluation/pick_cube_demo_prior_v1.json
```

确认 Demo prior 显著优于 zero 后，按预注册配置训练 30 万参数级 Local Jacobian Action
Transport。训练 pair 只允许跨 episode；validation 只从 train split 检索；固定使用末轮权重，
不根据 validation 选择 checkpoint：

```bash
sbatch src/scripts/hpc/train_maniskill_local_transport.slurm \
  "$PWD" \
  /scratch/ll5582/data/ManiSkill3/processed/pick_cube_chunks_v1 \
  /scratch/ll5582/data/ManiSkill3/evaluation/pick_cube_ljat_v1
```

LJAT 固定后，在与 replay 数据不重叠的新环境 seeds 上做 receding-horizon 闭环。脚本先验证
canonical EEF action 到 ManiSkill controller 的 frame/sign round-trip，再比较 zero、Demo copy、
完整 LJAT 和 factorized transport：

```bash
sbatch src/scripts/hpc/evaluate_maniskill_closed_loop.slurm \
  "$PWD" \
  /scratch/ll5582/data/ManiSkill3/processed/pick_cube_chunks_v1 \
  /scratch/ll5582/data/ManiSkill3/processed/replay_pick_cube_v1/source/trajectory.pointcloud.pd_ee_delta_pose.physx_cpu.json \
  /scratch/ll5582/data/ManiSkill3/evaluation/pick_cube_ljat_v1/local_jacobian_transport.pt \
  /scratch/ll5582/data/ManiSkill3/evaluation/pick_cube_closed_loop_v1.json
```

# ManiSkill3 Simulation Pilot

本目录用于 E4 仿真闭环，不与 RLBench 离线训练环境混装依赖。JUBAIL 使用独立环境
`FuseICL-maniskill`，固定 ManiSkill 3.0.1、NumPy 1.26 和 PyTorch 2.1。首批任务只选
Panda 单臂的 `PickCube-v1`、`PushCube-v1`、`StackCube-v1`；双臂和移动机器人任务不进入
初始实验。

验证顺序：

1. `audit_maniskill_archive.py` 直接读取 ZIP，核对 episode、控制器、成功率和 HDF5 shape；
2. `smoke_maniskill.py` 在 compute 节点执行
   `state + pd_ee_delta_pose + physx_cpu + render_backend=none` reset/step；
3. 再单独申请 GPU 验证 `pointcloud + physx_cuda`，避免把渲染依赖与控制接口问题混在一起；
4. 将官方 Demo 重放到统一 `pd_ee_delta_pose` 后生成 canonical EEF action chunks；
5. 最后才接入 Retriever、拒绝机制与 receding-horizon Predictor。

JUBAIL 的 home cache 可能触发配额限制，Slurm 脚本会把 Matplotlib、XDG、Torch 和
Hugging Face 缓存全部定向到 `/scratch/ll5582/.cache/fuceicl-maniskill`。

```bash
sbatch src/scripts/hpc/smoke_maniskill.slurm \
  "$PWD" \
  /scratch/ll5582/data/ManiSkill3/evaluation/smoke_pick_cube_state_v1.json
```

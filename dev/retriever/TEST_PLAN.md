# Retriever v1 测试计划

## 1. 测试目标

测试分开回答三个问题，不能用“代码能运行”代替“检索有效”：

1. **工程正确性**：mask、坐标、排序、索引与 payload 是否正确；
2. **离线检索质量**：是否找回任务、物体、phase、布局和状态兼容的 chunk；
3. **下游效用**：固定同一个 Predictor 时，检索结果是否提高闭环成功率。

## 2. 数据使用顺序

### A. 合成几何与单元测试（立即执行）

用程序生成立方体、圆柱体和容器 partial clouds，并改变平移、尺度、遮挡、点数、噪声和
padding。验证 Tiny PointNet++ 输出、mask 不变性、梯度，以及文本 Top-K 后的分项排序。
这一步只验证实现，不声称模型具备检索能力。

### B. RLBench（首个训练与主离线基准）

优先采用单臂、简单物体关系的 pick-place、container placement、drawer/door 和简单
stacking 任务。由 rendered depth 反投影点云，使用 GT mask/pose 生成监督和 oracle，但
线上 key 只使用经过同一感知管线得到的 partial cloud。第一轮建议：

- 训练：同任务同 phase 的跨 episode/chunk positives，加错误 phase、错误 target、反向
  operation 和不兼容 EEF/gripper hard negatives；
- 验证：隔离 episode，并额外隔离 variation、物体实例和相机扰动；
- 增强：深度量化、边缘缺失、孔洞、遮挡、外参 jitter、mask 漏分/粘连；
- 规模：先选 4–6 个任务，每任务约 100 episodes，确认标签与坐标后再扩至 18 tasks。

### C. ManiSkill 3（跨仿真域测试）

使用与 RLBench 语义重叠的 Pick/Place、Stack 和容器类单臂任务。先不联合训练，直接做
ManiSkill query → RLBench candidate，检验 encoder 是否依赖渲染器；随后加入多相机、
材质、光照、深度噪声随机化做联合训练。仿真 mesh/完整 state 只作标签或 oracle。

### D. FMB（首个真实 RGB-D 几何锚点）

FMB 的 Franka、RGB-D、CAD 物体与阶段标注适合测真实 query → sim candidate。先使用
分割和标定可信的 assembly 子集，人工审计少量 active/target role 和 phase。若任务语义
与 sim 不重叠，则只评估 shape/layout/state 通道，不把“没有对应任务”计为 encoder 错误。

### E. DROID（后续扩展，不作为第一轮 3D 主基准）

当前本地只有 language annotation JSON。DROID 先用于文本候选和任务覆盖分析；只有选出
具备可靠相机标定、depth/可重建点云、分割和 Franka state 的子集后，才进入 Stage-2。
不要把缺失深度或标定的数据填零后混入训练。

## 3. 划分与标签

- 按 episode 划分，重叠 chunk 必须位于同一 split；
- 单独构造 unseen object、unseen camera、unseen scene 与 paraphrase split；
- candidate 库与 query 禁止同 episode/self match；
- positive 需要 task/object、phase、layout/scale、EEF/gripper continuity 和 effect 兼容；
- hard negative 每类单独保留，避免总体平均值掩盖 opposite-operation 或 wrong-phase 失败；
- real→sim、sim→sim、real→real 分开报告。

## 4. 指标

Stage-1：task/object `Recall@10/50`、opposite-operation rejection、候选压缩率和 p95 延迟。

Stage-2：`Recall@1/4/10`、MRR、NDCG，以及 wrong-object、wrong-target、wrong-phase、
wrong-layout、wrong-gripper 的 rejection rate。另报告每个分项分数的校准曲线与 p50/p95
延迟。

端到端：固定 Predictor，比较 `no-demo / random / text-only / explicit-geometry /
learned-context / full / oracle`，报告成功率、action error、collision/intervention、fallback
rate 和 real-query→sim-demo 的实际占比。

## 5. 必须做的消融

- MiniLM only vs 当前 GLiNER+MiniLM；
- RGB-D raw appearance vs segmented partial cloud；
- normalized shape only vs metric cloud + extent；
- learned embedding vs explicit layout/state；
- 无 sim depth corruption vs perception-equivalent corruption；
- 只用任务文本 vs 加 phase/action semantics；
- 随机 PointNet++ vs 训练后 checkpoint，确认收益来自学习而非随机投影。

## 6. 第一轮通过门槛

进入真实机器人实验前，至少满足：单元测试全通过；RLBench held-out episode 的 full model
稳定优于 text-only；跨相机/噪声退化可解释；ManiSkill→RLBench 不出现明显 domain-only
聚类；FMB 小规模人工审计中 top-4 没有系统性错误物体、相反操作或不可接续夹爪状态。

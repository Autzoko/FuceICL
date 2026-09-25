# Tiny PointNet++

本目录提供纯 PyTorch 的轻量 PointNet++ encoder，不依赖 `pointnet2_ops` 或自定义 CUDA
扩展，可在当前 macOS CPU 与后续 Jubail GPU 上使用。

```python
import torch

from lib.tiny_pointnetpp import TinyPointNetPPEncoder

model = TinyPointNetPPEncoder()
points = torch.randn(8, 512, 3)
embeddings = model(points)  # [8, 128]，L2 normalized
```

支持 `[B, N]` point-valid mask。默认结构为两级 set abstraction：

```text
N points -> FPS 64 / kNN 24 -> 64-D local features
         -> FPS 16 / kNN 16 -> 128-D local features
         -> global max pool  -> 128-D embedding
```

默认模型共有 59,872 个可训练参数。本机单线程 CPU、batch=1、512 点的简单 warm-up 后
前向均值约 2.8 ms；该数字只用于确认轻量级规模，正式报告仍需在目标硬件上测试 p50/p95。

仓库当前没有预训练 checkpoint。随机初始化输出只能用于接口测试，不能用于正式检索；
必须在选定的 point-cloud/state retrieval 监督上训练并记录 checkpoint/hash 后使用。

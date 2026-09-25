# all-MiniLM-L6-v2 文本编码接口

稳定入口为 `MiniLMTextEncoder`：

```python
from lib.all_MiniLM_L6_v2 import MiniLMTextEncoder

encoder = MiniLMTextEncoder()
embeddings = encoder.encode(["open the drawer", "pull out the drawer"])
```

接口返回 CPU 上的 384 维 `float32` L2 归一化向量，因此向量内积即 cosine
similarity。实现使用 attention-mask-aware mean pooling；checkpoint 仅从本地
`checkpoint/` 加载。

# 文本初筛评测

本目录只保留冻结方案的 DROID 可复现实验。500 个 episode 中，第一条描述作为候选，
其余两条作为 query；同 episode 被视为相关项。该标签只是严格代理，不等价于人工标注
的任务语义相似度。

```bash
conda run -n FuseICL python dev/text_retriever/evaluate_simple_text_retrieval.py
```

首次运行会生成被 Git 忽略的解析缓存和报告。脚本对比原始 MiniLM 与冻结分数：

```text
0.8 * raw MiniLM cosine + 0.2 * symmetric object-set cosine
```

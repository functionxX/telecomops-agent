# ADR-004：BGE Reranker 精排

## Context

召回阶段（向量 + 关键词）为了覆盖率会取 Top-20 候选，其中必然混入噪声。
直接把 20 条塞给 LLM 会稀释上下文、增加幻觉风险。

## Decision

在检索与生成之间加 **Rerank** 阶段：

- 用交叉编码器 `BAAI/bge-reranker-base`（生产档位可切 `bge-reranker-v2-m3`）
  对 query-document 逐对打分；
- 召回 20 → 精排取 5（`RETRIEVAL_TOP_K` / `RERANK_TOP_K` 全配置化）；
- 模型名与设备从配置读取，与 Embedding 同为独立模块。

交叉编码器与双塔（bi-encoder）的本质区别：前者把 query 与 document
拼在一起做全注意力交互，语义判断更精细；后者各自独立编码、只能靠向量余弦
近似——所以 rerank 适合做精排，不适合做全库召回（太慢）。

## Alternatives

| 方案 | 问题 |
|---|---|
| 直接用向量分数 Top-5 | 排序质量差（见评测对比：MRR 0.90 vs 0.98） |
| 用 LLM 打分重排 | 每查询多次 LLM 调用，成本与延迟高、非确定性 |
| 只调融合权重不精排 | 权重能改善融合，但无法替代逐对语义判断 |

## Trade-offs

- CPU 上 rerank 是延迟大头（评测中单查询 ~1.3s），换 GPU 或小模型可降；
- 增加一个模型的内存占用（~1.1GB），生产可独立部署为服务。

## 数据支撑

本仓库评测（114 条真实查询，bge-small-zh-v1.5 + bge-reranker-base）：
Vector-only MRR=0.9015 → Hybrid+Rerank MRR=0.9810。详见 `scripts/evaluate_rag.py`。

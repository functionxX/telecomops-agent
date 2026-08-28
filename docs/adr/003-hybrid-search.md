# ADR-003：Hybrid Search（Dense + Keyword）与 PostgreSQL FTS

## Context

纯向量检索有已知短板：

- 精确匹配弱：套餐编号（`addon_30g`）、数字（`30GB`）、专有名词容易被语义相似
  但错误的结果淹没；
- 口语化查询（"流量咋没了"）与文档书面语存在词汇鸿沟。

关键词检索（BM25 类）恰好在精确匹配上强、语义泛化上弱——两者互补。

## Decision

第一版 Hybrid Retrieval：

- Dense：Milvus（ADR-002）；
- Keyword：**PostgreSQL Full Text Search**（`tsvector`/`ts_rank`），
  不引入 OpenSearch；
- 中文分词：PG 内置无中文 parser，采用 **bigram（二元组）分词 + 'simple' 配置**——
  零扩展依赖，任何 PG 实例可跑；查询侧用 OR 语义（命中越多 rank 越高，
  AND 语义对短文档过严）；
- 融合：**Weighted Score Fusion**——两路分数分别 min-max 归一化后加权相加
  （分数尺度不同，直接相加无意义）。权重可配置（默认 vector 0.6 / keyword 0.4），
  **默认值只是起点，最终权重由 evaluation 数据决定**（见评测章节的真实测量）。

## Alternatives

| 方案 | 问题 |
|---|---|
| OpenSearch | 引入独立集群只为关键词检索，第一版运维成本不成比例（明确否决） |
| 纯向量 | 精确匹配失败率高（评测中有真实对比数据） |
| zhparser/pg_jieba 扩展 | 需要编译安装 PG 扩展，compose 镜像与本地原生 PG 环境不一致 |
| RRF 融合 | 更好的免调参方案，但第一版选择可解释性更强的加权融合，演进可替换 |

## Trade-offs

- bigram 的词典召回不如专业分词（如 jieba），但规模（数百~数千文档）
  下差异可忽略，且零运维；
- 加权融合需要调权重 → 用评测闭环验证，不声称"最优"。

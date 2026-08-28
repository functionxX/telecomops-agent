# ADR-002：使用 Milvus 作为向量数据库

## Context

RAG 的 Dense Retrieval 需要向量存储与相似度检索。候选：Milvus、FAISS、pgvector、
OpenSearch 的 k-NN。

## Decision

使用 Milvus：

- 生产形态为 standalone 部署（docker compose：etcd + minio + milvus）；
- 开发/CI 可切 **Milvus Lite**（同一套 pymilvus 客户端 API，仅换 `MILVUS_URI`，
  代码零分叉）；
- 索引选 **IVF_FLAT**：10 万级文档规模下召回率高、构建快、内存友好；
  `nlist` 取 sqrt(N) 量级（当前配置 128），`nprobe=16` 为召回/延迟的平衡起点，
  全部可配置；
- 度量选 **COSINE**：BGE 系列模型按余弦相似度训练/评估，Embedding 侧已归一化，
  度量方向一致；
- 支持 metadata filtering（如按 category 过滤），为后续多租户/分类检索留出空间。

## Alternatives

| 方案 | 问题 |
|---|---|
| FAISS | 进程内库：无服务化、无元数据管理、生产运维能力弱 |
| pgvector | 与 PG 合并运维是优点，但大数据量（百万级）时召回性能与索引能力弱于 Milvus |
| OpenSearch k-NN | 引入整个 ES 集群只为向量检索，运维成本不匹配当前规模 |

## Trade-offs

- 多引入一个基础设施组件（etcd/minio/milvus 三容器），换取独立伸缩的向量检索层；
- Milvus Lite 与 standalone 存在实现差异（单进程、无分布式），
  但本项目规模（10^4~10^5 文档）完全在其能力范围内。

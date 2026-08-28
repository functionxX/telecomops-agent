# ADR-008：PostgreSQL 作为业务库 + 状态持久化 + 关键词检索的统一底座

## Context

系统需要三份数据：CRM 业务数据（用户/套餐/订单/服务）、Agent 元数据
（会话/消息/运行记录/审批/checkpoint）、RAG 关键词检索的倒排存储。
多引入一套存储就多一套运维。

## Decision

PostgreSQL 承担全部结构化职责：

1. **业务数据（CRM）**：users / customer_profiles / packages / user_packages /
   orders / services。业务 Tool 只访问 PG，绝不直连 Milvus 取业务信息；
2. **状态持久化**：conversations / messages / agent_runs / tool_calls /
   approvals / workflow_checkpoints（LangGraph checkpoint，见 ADR-006）；
3. **关键词检索**：knowledge_documents.content_tsv（bigram 分词，见 ADR-003）；
4. **关键工程保证**：
   - `create_order` 幂等：`idempotency_key` 唯一约束，Agent Retry /
     网络重试命中约束时返回已有订单，数据库层兜底并发重复；
   - 订单操作走事务（repository 层 commit/rollback 边界清晰）；
   - Tool 超时/重试不会产生脏订单——这正是幂等键存在的意义。

Milvus 只存知识库向量（doc_id/content/元数据），与业务数据职责分离。

## Alternatives

| 方案 | 问题 |
|---|---|
| Redis 做状态 | 引入第二套持久化语义；Redis 定位是缓存/限流，本项目第一版无此需求 |
| SQLite | 生产/并发/JSONB/FTS 能力不匹配 |
| 业务数据进 Milvus | 向量库不做事务与业务约束，职责错位 |

## Trade-offs

- 单点 PG 是风险点：生产需高可用/备份策略（第一版明确不做，文档标注）；
- checkpoint 写入频率与业务查询共用实例，量级增长后可拆库。

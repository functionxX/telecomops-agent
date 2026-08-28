# ADR-006：Human-in-the-loop 与自实现 PostgresCheckpointer

## Context

高风险操作（create_order / cancel_order / enable_roaming / disable_roaming）
必须人工确认。要求：

1. 中断必须是"真的停"——不是 sleep、不是 while 轮询；
2. 用户批准后必须从**原来的 Workflow 状态**恢复执行（不重跑前面的步骤）；
3. 中断状态必须可持久化、可查询、可审计。

## Decision

- 图中独立的 `human_approval` 节点：执行器风险检查发现高风险未批准 →
  conditional edge 进入该节点 → `interrupt(payload)` 暂停；
- LangGraph checkpoint 持久化到 **PostgreSQL**；
- 自实现 `PostgresCheckpointer`（继承 `BaseCheckpointSaver`）：
  - 自有 `workflow_checkpoints` 表（thread_id/checkpoint_id/parent/checkpoint JSONB/
    pending_writes JSONB）；
  - 序列化协议用官方 `JsonPlusSerializer`（Interrupt 对象走 msgpack 协议
    正确往返，避免把中断值写坏成字符串）；
  - 异步桥接（astream 路径）通过 `asyncio.to_thread` 委托同步实现；
- 恢复：`POST /api/v1/approvals/{id}` 用 `Command(resume={decision})` 重启图；
  审批请求同时落 `approvals` 表（可查询、重复审批返回 409）；
- **一次审批只放行一个动作**：批准执行成功后立即重置 human_decision，
  同一会话的下一个高风险动作必须重新审批。

## Alternatives

| 方案 | 问题 |
|---|---|
| 官方 langgraph-checkpoint-postgres | 表结构私有、恢复机制黑盒；本项目目标是能亲手解释 checkpoint 协议 |
| sleep/轮询模拟审批 | 假实现：服务重启即丢失，无法跨进程/跨请求恢复 |
| 审批内嵌在 executor 节点 | 图结构不显式，路由与审计难测 |

## Trade-offs

- 自实现 saver 需正确理解 pending_writes/序列化协议（本项目已用
  interrupt→落库→resume 的端到端测试覆盖）；
- 单表 JSONB 设计简单、易解释，量级增长后需考虑分区。

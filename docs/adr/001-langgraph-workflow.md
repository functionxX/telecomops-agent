# ADR-001：使用 LangGraph 构建显式 Stateful Agent Workflow

## Context

本项目需要构建一个可解释、可恢复、可控的企业级 Agent：意图路由、规划、工具执行、
校验、重试、重规划、人工审批。如果直接采用 "User → LLM → Tool → Response" 的
线性实现，或套用 LangChain 的隐式 AgentExecutor：

1. 控制流隐藏在框架内部，面试/审计时无法解释"为什么走到这一步"；
2. 无法原生支持 interrupt/resume（人工审批需要持久化暂停与恢复）；
3. Retry / Replan / 条件跳转只能靠 prompt 约定，不可测试、不可复现。

## Decision

采用 LangGraph 显式状态图：

- 每个环节（Router / Planner / Executor / Validator / Approval / Answer）都是**显式节点**；
- 分支用 **Conditional Edge** 表达（意图分派、失败分类分流、审批决策）；
- 循环真实存在于图结构中（executor 自循环推进步骤；executor→validator→retry→executor；
  executor→validator→replan→planner→executor）；
- State 是唯一的跨节点数据载体（TypedDict + reducer），checkpoint 持久化到 PostgreSQL；
- Human-in-the-loop 用官方 `interrupt()` / `Command(resume=...)`。

## Alternatives

| 方案 | 问题 |
|---|---|
| 手写 Python 状态机 | 重复造轮子：checkpoint/恢复/并发控制都要自己实现 |
| LangChain AgentExecutor（隐式循环） | 控制流不可见、不可测试；无法中断恢复 |
| Multi-Agent（多个 LLM Agent 协作） | 第一版范围外：重点是 State/Planning/循环，不是 Agent 数量（见演进章节） |

## Trade-offs

- 图结构带来一定样板代码（节点 + 边声明），换取的是**可控性、可测试性、可解释性**；
- 学习曲线：团队需要理解图语义（节点、状态、checkpoint）。

## 演进

未来若需 Multi-Agent，可将每个 Agent 编译为子图（subgraph），
通过 State 与 checkpoint 在父图中编排——不需要推翻现有架构。

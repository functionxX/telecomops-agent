# ADR-005：Planner 独立于 Tool Executor

## Context

TASK 类请求（多步 + 条件分支）不能直接丢给 LLM 边想边做：

- 执行顺序不可见、不可审计；
- 一次 LLM 输出一个工具调用的做法在长任务上延迟与成本线性放大；
- 没有计划就没有 "Replan" 的对象。

## Decision

Planner 是独立节点，只做一件事：**把任务拆解为结构化计划**（JSON mode + Pydantic 校验）：

- 每个 step 含 step_id / tool / arguments / description / status
  （PENDING/RUNNING/SUCCESS/FAILED/SKIPPED）；
- 支持 **condition 步骤**（tool=null）：`{left, op, right, then_step, else_step}`，
  left 可为状态引用 `$step_N.字段`——运行时条件由执行器**确定性求值**，
  控制流不经过 LLM；
- 步骤参数支持状态引用，执行前由执行器解析前序结果；
- Replan 时携带「已有结果 + 失败原因」，明确要求不得重复失败的工具调用。

Planner 不执行工具：计划与执行分离，执行器才能统一做校验/权限/风险/超时。

## Alternatives

| 方案 | 问题 |
|---|---|
| 每步问 LLM（无计划） | 延迟/成本线性放大、控制流不可复现、无法做 Replan 语义 |
| Planner 直接调 Tool | 绕过 ToolExecutor 的权限/风险/超时防线，安全边界塌陷 |
| 把整个 Workflow 包成一个 Tool | 失去分步可见性与失败隔离 |

## Trade-offs

- 计划质量依赖 LLM 的结构化输出能力 → 由 Pydantic 校验 + Validator 兜底；
- condition DSL 第一版只支持单条件（left op right），复杂逻辑演进为嵌套表达式。

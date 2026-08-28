# ADR-007：Tool Executor 独立，LLM 不直接执行工具

## Context

安全底线：**LLM 生成的 Tool Name + 参数是"不可信输入"**。
如果让 LLM 直接调函数（如 LangChain tool binding 直连 handler），
等于把权限校验、参数合法性、风险控制全部押在 prompt 上——prompt 不是安全边界。

## Decision

所有工具调用必须经过独立的 `ToolExecutor`，固定流程：

1. 工具存在性（Registry 查表，未知 → ToolNotFound → Replan 信号）
2. **user_id 注入**：会话用户的 user_id 由执行器注入，不信任 LLM 提供的值；
   LLM 提供了不同 user_id = 越权尝试 → PermissionDenied（Tool Guardrail）
3. Pydantic 参数校验（计划参数错误 → InvalidArguments → Replan 信号）
4. 角色权限检查（ToolPolicy.role）
5. 风险检查（ToolPolicy.require_confirmation → APPROVAL_REQUIRED，
   由人工审批节点处理）
6. 超时控制（TOOL_TIMEOUT 配置化）
7. 执行（业务 handler 只负责业务逻辑，异常归一为统一异常模型：
   ToolNotFound / InvalidArguments / PermissionDenied / Timeout /
   BusinessError / ExecutionError）
8. 结果校验 + 截断（MAX_ROWS / MAX_CELL_LENGTH，防上下文爆炸）

失败分类（FailureKind）写死在异常模型里：
transient → Retry；plan_error → Replan；fast_fail → 直接回答失败。
**控制流分类不用 LLM 临场判断。**

QUERY 路径的 bind_tools 只是"选工具"，选中后仍走同一 Executor——
工具清单本身也被策略过滤（只暴露 LOW 风险只读工具）。

## Alternatives

| 方案 | 问题 |
|---|---|
| LLM 直接调 handler | 越权/注入/参数错误全看 prompt 脸色 |
| 每工具自带校验逻辑 | 校验散落 11 处，审计与测试成本高 |
| Prompt 里声明"不要调用 XX 工具" | 已知可被注入绕过，不能作为唯一防线 |

## Trade-offs

- 每次调用多几层固定开销（毫秒级），换统一安全边界与可审计性；
- 高风险审批在图中打断了"执行"与"结果"的连续性，由 checkpoint 机制弥合。

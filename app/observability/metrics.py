"""Prometheus 指标定义。

指标名称固定（规格书要求的最小集合），各业务模块在对应位置 increment：
- HTTP 层记录 request_*
- LLM 层记录 llm_*
- Tool 执行记录 tool_*
- Workflow 层记录 workflow_* / agent_retry_*
- RAG 层记录 rag_retrieval_*
"""

from prometheus_client import Counter, Histogram

# ---------- HTTP ----------
request_count = Counter(
    "request_count",
    "HTTP 请求总数",
    ["method", "route", "status"],
)
request_latency = Histogram(
    "request_latency",
    "HTTP 请求耗时（秒）",
    ["method", "route"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

# ---------- LLM ----------
llm_latency = Histogram(
    "llm_latency",
    "LLM 调用耗时（秒）",
    ["model", "purpose"],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0),
)
llm_error_count = Counter(
    "llm_error_count",
    "LLM 调用失败次数",
    ["model", "purpose"],
)

# ---------- Tool ----------
tool_call_count = Counter(
    "tool_call_count",
    "工具调用次数",
    ["tool", "status"],  # status: success / error / timeout / denied
)
tool_error_count = Counter(
    "tool_error_count",
    "工具调用失败次数",
    ["tool", "error_type"],
)

# ---------- Workflow ----------
workflow_success_count = Counter(
    "workflow_success_count",
    "Workflow 成功完成次数",
    ["intent"],
)
workflow_failure_count = Counter(
    "workflow_failure_count",
    "Workflow 失败次数",
    ["intent", "failure_kind"],
)
agent_retry_count = Counter(
    "agent_retry_count",
    "Agent 重试/重规划次数",
    ["strategy"],  # retry / replan
)

# ---------- RAG ----------
rag_retrieval_latency = Histogram(
    "rag_retrieval_latency",
    "RAG 检索耗时（秒）",
    ["stage"],  # dense / keyword / hybrid / rerank
)
rag_retrieval_count = Counter(
    "rag_retrieval_count",
    "RAG 检索次数",
    ["stage"],
)

# TeleComOps Agent

[![CI](https://github.com/functionxX/telecomops-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/functionxX/telecomops-agent/actions/workflows/ci.yml)

**电信 CRM 场景的智能业务办理与知识助手** —— 一个用于 AI Agent / AI Application
Engineer / Agent Workflow Developer 面试展示的完整企业级项目。

`Stateful Agent Workflow（LangGraph） + RAG（自研 Pipeline） + Tool Reliability + Human-in-the-loop + Observability + Evaluation`

---

## 目录

1. [项目简介](#1-项目简介)
2. [项目背景](#2-项目背景)
3. [核心功能](#3-核心功能)
4. [整体架构](#4-整体架构)
5. [Agent Workflow 图](#5-agent-workflow-图)
6. [RAG Pipeline 图](#6-rag-pipeline-图)
7. [Tool Calling 流程](#7-tool-calling-流程)
8. [Human-in-the-loop](#8-human-in-the-loop)
9. [Memory](#9-memory)
10. [Retry / Replan](#10-retry--replan)
11. [Hybrid Search](#11-hybrid-search)
12. [Rerank](#12-rerank)
13. [Evaluation](#13-evaluation)
14. [Observability](#14-observability)
15. [技术栈](#15-技术栈)
16. [为什么选择 LangGraph](#16-为什么选择-langgraph)
17. [为什么使用 Milvus](#17-为什么使用-milvus)
18. [为什么 Hybrid Search](#18-为什么-hybrid-search)
19. [为什么 Rerank](#19-为什么-rerank)
20. [为什么 Planner](#20-为什么-planner)
21. [为什么需要 Validator](#21-为什么需要-validator)
22. [为什么 Tool Executor 独立](#22-为什么-tool-executor-独立)
23. [如何保证 Tool 安全](#23-如何保证-tool-安全)
24. [如何防止重复订单](#24-如何防止重复订单)
25. [如何处理 Agent Failure](#25-如何处理-agent-failure)
26. [如何恢复中断 Workflow](#26-如何恢复中断-workflow)
27. [如何运行](#27-如何运行)
28. [API 示例](#28-api-示例)
29. [测试](#29-测试)
30. [Evaluation](#30-evaluation)
31. [性能指标](#31-性能指标)

---

## 1. 项目简介

TeleComOps Agent 是一个面向电信 CRM 场景的智能助手，覆盖三条业务路径：

- **FAQ（知识问答）**：通过完整 RAG 链路回答套餐/漫游/账单/政策等问题，带引用；
- **QUERY（单步查询）**：LLM 从低风险只读工具中选择一个，查询真实 CRM 数据；
- **TASK（多步任务）**：Planner 生成结构化计划（支持条件分支），逐步骤执行，
  带 Retry / Replan / 人工审批。

项目不是 "ChatGPT + RAG Demo"，而是一个具备完整工程闭环的系统：
**状态机工作流、工具安全、幂等、中断恢复、Guardrails、可观测性、评测、CI/CD**。

## 2. 项目背景

电信客服场景的三大痛点：

1. **业务办理链路长**：查套餐 → 查流量 → 条件判断 → 推荐 → 下单 → 审批，
   每一步都可能失败，需要可控的重试与失败恢复；
2. **高风险操作必须留痕**：开通漫游、下单扣费不能由 LLM 自主决定，
   必须有人工确认且可审计；
3. **知识库问答的准确性与可追溯**：答案必须来自知识库并带来源，
   检索不到时明确说"没有信息"，不许编造。

## 3. 核心功能

| 能力 | 实现 |
|---|---|
| Stateful Agent Workflow | LangGraph 显式图：10 个节点、5 组条件边、3 类真实循环 |
| Planner + condition DSL | 计划内条件步骤（`left/op/right/then/else`）+ 状态引用，运行时确定性执行 |
| Tool Calling | 11 个业务工具，统一 Registry + 独立 Executor（校验/权限/风险/超时/截断） |
| Retry / Replan | 失败分类写死：transient→Retry(≤2)、计划性→Replan(≤1)、其余快速失败；全局执行 ≤10 硬兜底 |
| Human-in-the-loop | `interrupt()` 暂停 → PostgreSQL checkpoint → `Command(resume)` 恢复；一次审批只放行一个动作 |
| RAG | Query Rewrite → Hybrid(Dense+Keyword) → Fusion → BGE Rerank → Context → 引用答案 |
| 幂等 | create_order 语义幂等键（会话+套餐）+ DB 唯一约束，重试不重复下单 |
| Guardrails | 输入（注入/超长）、工具（user_id 注入+越权拦截）、输出（手机号/身份证脱敏） |
| SSE | 生命周期事件流 + 答案 token 级流式 |
| Observability | OTel（HTTP/LLM/Tool/RAG span）+ Prometheus（11 项指标）+ Grafana Dashboard |
| Evaluation | RAG（114 条查询 × 3 检索变体）+ Agent（52 任务 × 8 指标），真实运行不伪造 |

## 4. 整体架构

```
Frontend / API Client
        |  HTTP / SSE
        v
FastAPI ────────────────> PostgreSQL（业务数据 + 会话/审计/checkpoint + FTS）
        |                        ^
        v                        |
LangGraph Agent Workflow         |
        |                        |
  Router ── FAQ ──> RAG Pipeline ──> Milvus（向量） + PG（关键词）
    ├─ QUERY ──> Tool Executor ──> Validator
    └─ TASK ───> Planner ──> Tool Executor ──> Validator ──> Retry/Replan
                             （高风险）──> Human Approval ──> interrupt/resume

所有组件 → OpenTelemetry → Collector → Traces(console) / Metrics(Prometheus) → Grafana
```

模块职责（目录即架构）：

```
app/
├── main.py            # FastAPI 入口：路由挂载、中间件、统一异常
├── api/               # routes（health/chat/approvals）+ schemas
├── agent/             # LangGraph：state/router/planner/validator/executor/graph/nodes
├── tools/             # 11 个工具 + registry + policies + executor（安全边界）
├── rag/               # ingestion/embedding/query_rewrite/vector/keyword/hybrid/reranker/context/pipeline
├── llm/               # LLMClient（httpx 直连 DeepSeek）+ CustomDeepSeekChatModel + Mock
├── memory/            # PostgresCheckpointer（自实现）+ 会话历史
├── guardrails/        # input / output / tool
├── db/                # SQLAlchemy models + session + repositories
├── observability/     # tracing（OTel）+ metrics（Prometheus）
└── core/              # config（pydantic-settings）/ exceptions（统一异常模型）/ logging
```

## 5. Agent Workflow 图

```
                    START
                      |
                    Router（LLM 结构化输出，失败兜底 UNKNOWN）
              ┌───────┼───────────────┬─────────────┐
             FAQ    QUERY            TASK       UNKNOWN
              |       |               |             |
              v       v               v             |
             RAG   executor        planner         |
              |       |               |             |
              |       v               v             |
              |    validator <──── executor <───────┘
              |       |    \___________^  (Loop: 逐步推进)
              |       |          （approval_required 时）
              |  retry/replan      |
              |       |            v
              |       |       human_approval（interrupt → resume）
              |       |         /        \
              |       |    approved    rejected
              |       |         \        /
              v       v          v      v
                    Answer ──────────────> END
```

真实存在的循环（不是画出来的）：

1. `executor → executor`：计划逐步推进（success 时进入下一步，直到 END）；
2. `executor → validator → retry → executor`：瞬时失败重试（≤2 次）；
3. `executor → validator → replan → planner → executor`：计划性失败重规划（≤1 次）。

条件边由状态字段驱动（`route_after_*` 纯函数），全部可单测。

## 6. RAG Pipeline 图

```
User Query
   |
   v
Query Rewrite（口语 → 检索友好表述；失败 fallback 原 query）
   |
   v
Hybrid Retrieval
   +---- Dense：Embedding(bge-small) → Milvus IVF_FLAT / COSINE
   +---- Keyword：PG FTS（bigram 中文分词，ts_rank）
   |
   v
Weighted Fusion（两路 min-max 归一化后加权，权重可配置）
   |
   v
Top 20 候选
   |
   v
BGE Reranker（交叉编码器逐对精排）
   |
   v
Top 5
   |
   v
相关性阈值（低于阈值 → 明确回答"知识库中没有找到足够相关的信息"）
   |
   v
Context Builder（截断保护 + 引用编号）
   |
   v
DeepSeek 生成（只依据文档，句末标注 [n]）
   |
   v
Answer + Citations
```

核心链路自研（Embedding 调用 / 检索 / 融合 / 重排 / 上下文），
不包装成 LangChain Retriever/Chain——每个环节内部发生什么都可以解释。

## 7. Tool Calling 流程

```
LLM 产生 Tool Name + 参数（不可信输入）
      |
      v
Tool Exists?（Registry 查表，未知 → Replan 信号）
      |
      v
user_id 注入（Guardrail：LLM 给不同 user_id = 越权 → 拦截）
      |
      v
Pydantic 参数校验（类型错误 → Replan 信号）
      |
      v
权限检查（ToolPolicy.role）
      |
      v
风险检查（HIGH + 未批准 → APPROVAL_REQUIRED → Human Approval）
      |
      v
超时控制（TOOL_TIMEOUT，配置化）
      |
      v
执行（业务 handler；异常归一为统一异常模型）
      |
      v
结果校验 + 截断（MAX_ROWS / MAX_CELL_LENGTH）
      |
      v
Tool Result
```

QUERY 路径的 LLM bind_tools 只负责"选工具"，执行仍走同一 Executor；
且工具清单本身被策略过滤（只暴露 LOW 风险只读工具）——权限边界是程序级。

## 8. Human-in-the-loop

高风险工具（`create_order` / `cancel_order` / `enable_roaming` / `disable_roaming`）
默认要求人工确认：

1. Executor 风险检查 → `APPROVAL_REQUIRED` → conditional edge 进入独立 `human_approval` 节点；
2. 节点把审批请求落库（approvals 表，幂等 upsert）后调用 `interrupt(payload)`：
   **图真正暂停**，checkpoint 写入 PostgreSQL（workflow_checkpoints 表）；
3. `/chat` 返回 `approval` 字段（approval_id + 摘要），SSE 推送 `approval_required` 事件；
4. 用户 `POST /api/v1/approvals/{id}` → `Command(resume={decision})` 从 checkpoint
   **恢复原状态继续执行**（不重跑前面的步骤）；
5. approve → executor 执行挂起的调用；reject → 回答"已取消"，不产生任何变更；
6. **一次审批只放行一个动作**：批准执行成功后立即重置 human_decision，
   同一会话的下一个高风险操作必须重新审批。

Checkpoint 采用自实现的 `PostgresCheckpointer`（继承 `BaseCheckpointSaver`，
序列化用官方 `JsonPlusSerializer`）——表结构自有、恢复机制可亲手解释。

## 9. Memory

三层记忆职责分离（Memory ≠ messages）：

| 层 | 存储 | 用途 |
|---|---|---|
| Conversation History | `messages` 表 | 多轮对话上下文注入 |
| Workflow State | `workflow_checkpoints` + LangGraph State | 中断恢复、执行审计（tool_calls 记录每次调用的参数/结果/耗时） |
| Long-term User Info | `customer_profiles` 等业务表 | 客户画像，不是聊天记录 |

另持久化：`agent_runs`（每次运行状态）、`approvals`（审批留痕）、`evaluation_results`（评测与反馈）。

## 10. Retry / Replan

失败分类**写死在异常模型**（`FailureKind`），不用 LLM 临场判断控制流：

| 分类 | 触发 | 处理 |
|---|---|---|
| TRANSIENT | 超时、DB 连接抖动、ExecutionError | **Retry**（同一步骤重执行，≤ MAX_RETRIES=2） |
| PLAN_ERROR | 工具不存在、参数非法、计划前提错误、语义校验不通过 | **Replan**（携带失败原因重规划，≤ MAX_REPLANS=1） |
| FAST_FAIL | 权限拒绝、业务性失败 | 不进循环，直接回答失败原因 |

兜底：单次运行工具执行总次数 ≤ `MAX_TOOL_EXECUTIONS=10`——任何形态的失控循环
在架构层面被禁止（有专门测试：永久失败时重试恰好 2 次后收口）。

LLM 层面另有独立的指数退避（仅 429/5xx/网络错误，配置化重试次数）。

## 11. Hybrid Search

Dense + Keyword 互补：向量检索擅长语义相似（"流量咋没了"），
关键词检索擅长精确匹配（套餐编号、数字、专有名词）。

- **Dense**：Milvus（IVF_FLAT + COSINE，nlist/nprobe 可配置）；
- **Keyword**：PostgreSQL FTS。中文用 **bigram 二元组分词 + 'simple' 配置**
  （零扩展依赖；查询侧 OR 语义，命中越多 ts_rank 越高）；
- **Fusion**：两路分数先 min-max 归一化（尺度不同，直接相加无意义），
  再按 `VECTOR_WEIGHT` / `KEYWORD_WEIGHT` 加权。默认 0.6/0.4 只是起点，
  **最终权重由评测数据决定**，不声称最优。

不引入 OpenSearch：数百~数千条文档规模下 PG FTS 完全够用（ADR-003）。

## 12. Rerank

召回（Top 20）追求覆盖、用粗粒度分数；重排用交叉编码器
（query-document 全注意力交互）逐对精排，取 Top 5。
模型与 Top-K 全配置化（`RERANKER_MODEL` / `RETRIEVAL_TOP_K` / `RERANK_TOP_K`）。

实测数据（114 条查询，本仓库评测脚本真实运行）：

| 变体 | Recall@5 | MRR |
|---|---|---|
| Vector only | 0.9649 | 0.9015 |
| Hybrid | 0.9781 | 0.9152 |
| Hybrid + Rerank | **0.9912** | **0.9810** |

## 13. Evaluation

不硬编码、不伪造——两个评测脚本真实运行数据集后生成 JSON 报告：

- **RAG 评测**（`scripts/evaluate_rag.py`）：114 条查询（104 单文档 + 10 多文档，
  46% 口语改写），三组检索变体对比 Recall@5 / Recall@10 / MRR；
- **Agent 评测**（`scripts/evaluate_agent.py`）：52 个任务覆盖
  FAQ / QUERY / TASK / 工具选择 / 参数 / Retry / Replan / 人工审批 / Guardrails / 越权，
  统计 Intent / Tool Selection / Argument Accuracy、Task Success Rate、Retry Rate、
  Average Steps / Latency。Retry/Replan 由注入故障的执行器**真实触发循环**，
  审批任务真实走 interrupt→resume。

Mock 模式（默认，CI/离线）结果可复现但不是正式 Benchmark；
`--real-llm` 模式用真实 DeepSeek 出正式数字。用户反馈通过
`POST /api/v1/feedback` 落库，作为在线评测信号。

## 14. Observability

- **Traces（OTel）**：HTTP（自动埋点）+ LLM（model/purpose/tokens/latency）+
  Tool（name/status/error）+ RAG 各阶段 + workflow，`trace_id` 贯穿日志与 span；
  导出 OTLP → collector（第一版 debug 后端，可扩展 Tempo/Jaeger）；
- **Metrics（Prometheus）**：request_count/latency、llm_latency/error、
  tool_call/error、workflow_success/failure、agent_retry、rag_retrieval_latency/count；
  业务指标经 `/metrics` 直采，OTel 自动埋点指标经 collector 转发；
- **Grafana**：provisioning 自动加载 Prometheus 数据源与 TeleComOps Agent Dashboard。

**脱敏约定**：span/日志不记录 API Key、密码；工具结果在进入 LLM 前先脱敏
（保证流式 token 也不含手机号/身份证）。

## 15. 技术栈

Python 3.11 · FastAPI · LangGraph 1.x · LangChain Core（BaseChatModel 适配层 + 工具抽象）
· DeepSeek API（deepseek-chat）· Sentence-Transformers（BGE 系列，CPU/GPU 可配）
· Milvus（standalone + Lite 双形态）· PostgreSQL 16（业务库 + FTS + checkpoint）
· SQLAlchemy 2 · Pydantic v2 · httpx · OpenTelemetry · Prometheus · Grafana
· pytest / ruff / mypy · Docker Compose · GitHub Actions · uv

未使用：Redis（第一版无缓存/限流需求，ADR-008 说明）、OpenSearch（ADR-003 说明）、
Multi-Agent（ADR-001 演进章节说明）。

## 16. 为什么选择 LangGraph

见 [ADR-001](docs/adr/001-langgraph-workflow.md)。核心：

1. 控制流显式：Router/Planner/Validator 是图上的节点，分支是条件边——
   可解释、可单测、可审计；
2. 状态机语义：State + checkpoint，天然支持 **interrupt / resume**，
   这是人工审批与断点恢复的前提；
3. 循环是真实结构而非 prompt 约定：Retry/Replan 由边定义，不会失控。

## 17. 为什么使用 Milvus

见 [ADR-002](docs/adr/002-milvus-vector-db.md)。核心：

1. 服务化向量库：独立伸缩、索引/加载管理成熟（IVF_FLAT 起步，可升级 HNSW）；
2. 元数据过滤（category 等）支持未来多租户/分类检索；
3. 开发/CI 可用 **Milvus Lite**（同一 pymilvus API，零代码切换）；
4. pgvector 在十万~百万级文档时检索性能与索引能力不足。

## 18. 为什么 Hybrid Search

见 [ADR-003](docs/adr/003-hybrid-search.md)。纯向量检索在精确匹配
（套餐编号/数字/专有名词）上系统性弱势，关键词检索补足；评测数据证实
混合优于单路（Recall@5：0.965 → 0.978）。

## 19. 为什么 Rerank

见 [ADR-004](docs/adr/004-reranker.md)。召回是粗排（双塔向量近似），
重排是精排（交叉编码器全注意力交互）。实测 MRR 0.90 → 0.98；
同时为 Context 截断提供高质量排序，降低 LLM 幻觉。

## 20. 为什么 Planner

见 [ADR-005](docs/adr/005-planner.md)。多步任务需要**可见的计划**：

1. 执行顺序可审计（每一步的 tool/arguments/status 全留痕）；
2. 条件分支以数据（condition 步骤）表达，由执行器确定性求值——
   运行时控制流不经过 LLM，可复现、可测试；
3. Replan 有明确对象：失败时携带原因重新生成计划，而不是重跑原计划。

## 21. 为什么需要 Validator

工具"成功返回"不等于"结果可用"：

1. 失败分类分流：executor 的异常模型是分类的事实来源，Validator 把它映射为
   Retry / Replan / Fail 的路由决策——控制流分类不靠 LLM 临场判断；
2. 语义校验（可配置）：抓"工具成功了但结果不对"的场景（如计划要剩余流量、
   结果只有套餐名），失败 → Replan；
3. 空结果/形态异常归入计划性失败，驱动重规划而非盲目重试。

## 22. 为什么 Tool Executor 独立

见 [ADR-007](docs/adr/007-tool-executor.md)。LLM 生成的工具名与参数是
**不可信输入**：校验、权限、风险、超时、结果规范化必须集中在程序层，
而不是散落在 11 个工具里或押在 prompt 上。独立的 Executor 让安全边界
可审计、可单测，也让 Workflow 只关心"成功/失败/待审批"三种结果。

## 23. 如何保证 Tool 安全

四道防线：

1. **工具清单**：QUERY 路径只暴露 LOW 风险只读工具（清单即策略）；
2. **user_id 注入**：会话用户由执行器注入，LLM 提供的不同 user_id 视为越权拦截；
3. **ToolPolicy**：角色/风险等级程序化检查，高风险必须人工审批；
4. **审计**：每次调用的参数/结果/耗时/错误全量落库（tool_calls 表）。

Prompt 只是提示，Policy 才是防线。

## 24. 如何防止重复订单

双重保证（`create_order`）：

1. **语义幂等键**：`idempotency_key = ik_{conversation_id}_{package_id}`——
   与计划步骤编号无关（LLM 每次生成的 step_id 不稳定），同一会话内重复办理
   同一套餐必然命中同一键；
2. **数据库唯一约束**：`orders.idempotency_key UNIQUE`，并发重复插入时
   由约束兜底，返回已有订单（`created=false`）。

Agent Retry / Replan / 网络重试 / 用户重复点击都不会产生第二张订单
（`scripts/demo_approval.py` 中有真实断言：重试前后订单数不变）。

## 25. 如何处理 Agent Failure

```
失败 → Validator 分类
  ├── transient（超时/抖动）  → Retry 同一步骤（≤2 次，指数退避）
  ├── plan_error（计划有误）  → Replan（携带失败原因，≤1 次）
  └── fast_fail（权限/业务）  → 直接回答失败原因，不浪费重试
超限 → 收口为"任务未能完成：原因（已重试 N 次、重规划 M 次）"
全局 → 单次运行工具执行 ≤10 次，杜绝失控循环
```

所有失败路径都有对应测试（正常路径/错误路径/边界）。

## 26. 如何恢复中断 Workflow

`interrupt()` 暂停时 LangGraph 把完整状态（含 pending writes）写入
PostgreSQL `workflow_checkpoints`（自实现 Saver，序列化兼容 Interrupt 对象）。
恢复时 `Command(resume=决策)` 让框架按 thread_id 取回快照、重放 pending writes、
从暂停节点继续——**服务重启后仍可恢复**（状态不在内存）。

演示：`uv run python scripts/demo_approval.py`（中断 → 查库验证落盘 → 批准恢复 → 订单落库 → 幂等验证）。

## 27. 如何运行

### 前置条件

- Python 3.11+，[uv](https://docs.astral.sh/uv/)；
- Docker Desktop（基础设施容器）；国内网络建议设置 `HF_ENDPOINT=https://hf-mirror.com`
  加速 HuggingFace 模型下载。

### 快速开始

```bash
cp .env.example .env        # 填入 DEEPSEEK_API_KEY；无 key 则设 MOCK_LLM=true
uv sync                     # 安装依赖
docker compose up -d        # 启动全部基础设施（PG:5433 / Milvus:19530 / Prometheus:9090 / Grafana:3000）
uv run python scripts/init_db.py          # 建表 + 种子数据（幂等）
uv run python scripts/ingest_knowledge.py # 知识库导入（104 条 → PG + Milvus，首次会下载模型）
uv run uvicorn app.main:app --reload      # 启动 API（http://localhost:8000/docs）
```

Windows（无 make）等价命令见 `.env.example` 注释与 Makefile 对照；
完整 compose 模式（含 app 容器）：`docker compose up -d` 后 app 由容器运行。

### 环境变量

集中管理（`.env` / `.env.example`），关键项：

| 变量 | 说明 |
|---|---|
| `DEEPSEEK_API_KEY` / `DEEPSEEK_MODEL` / `DEEPSEEK_BASE_URL` | LLM 接入（deepseek-chat 支持 JSON mode） |
| `MOCK_LLM` | true = 确定性规则 Mock（CI/离线/测试），false = 真实 API |
| `POSTGRES_URL` | 本机默认 5433（避开原生 PG 5432）；容器内自动覆盖 |
| `MILVUS_URI` | standalone：http://localhost:19530；离线可用 ./milvus_lite.db |
| `EMBEDDING_MODEL` / `RERANKER_MODEL` | 开发档 bge-small-zh-v1.5 + bge-reranker-base；生产档 bge-m3 + v2-m3（维度从模型自动读取，勿手写） |
| `RETRIEVAL_TOP_K` / `RERANK_TOP_K` / `VECTOR_WEIGHT` / `KEYWORD_WEIGHT` | 检索与融合参数（权重由评测数据调优） |
| `MAX_RETRIES` / `MAX_REPLANS` / `MAX_TOOL_EXECUTIONS` | 循环边界 |
| `LLM_TIMEOUT` / `TOOL_TIMEOUT` | 超时控制 |
| `OTEL_EXPORTER_OTLP_ENDPOINT` / `OTEL_ENABLED` | 链路导出 |

### Mock 模式 vs 真实模式

| | Mock（`MOCK_LLM=true`） | 真实（`MOCK_LLM=false`） |
|---|---|---|
| 行为 | 确定性规则（Router/Planner/选工具按规则） | DeepSeek 真实推理 |
| 用途 | 开发、单元/集成测试、CI、Eval 复现 | 演示、正式 Benchmark |
| Eval 数字 | **可复现但不是正式 Benchmark** | 正式数字（`evaluate_agent.py --real-llm`） |

### 监控

- Prometheus：http://localhost:9090（targets: app / milvus / otel-collector）
- Grafana：http://localhost:3000（admin/admin，Dashboard 自动加载）
- Traces：otel-collector 容器日志（debug exporter；扩展 Tempo 见 collector 配置注释）

## 28. API 示例

**FAQ（RAG + 引用）**
```bash
curl -X POST http://localhost:8000/api/v1/chat -H "Content-Type: application/json" \
  -d '{"user_id":"user_001","message":"5G套餐有哪些？"}'
# → {"answer":"... [1][2]","citations":[{"index":1,"source":"package/pkg_99.md",...}],"intent":"FAQ"}
```

**QUERY（工具查询）**
```bash
curl -X POST http://localhost:8000/api/v1/chat -H "Content-Type: application/json" \
  -d '{"user_id":"user_001","message":"我的套餐还剩多少流量？"}'
# → {"answer":"...剩余流量为 8.0 GB...","intent":"QUERY","tool_calls":[...]}
```

**TASK 高风险（中断 → 审批 → 恢复）**
```bash
curl -X POST http://localhost:8000/api/v1/chat -H "Content-Type: application/json" \
  -d '{"user_id":"user_001","message":"帮我办理30GB流量包。"}'
# → {"approval":{"approval_id":"apv_xxx","tool_name":"create_order",...},"answer":""}

curl http://localhost:8000/api/v1/approvals/apv_xxx                      # 查看审批
curl -X POST http://localhost:8000/api/v1/approvals/apv_xxx \
  -H "Content-Type: application/json" -d '{"decision":"approve"}'        # 批准 → 恢复执行 → 下单
```

**SSE 流式**
```bash
curl -N -X POST http://localhost:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"user_id":"user_001","message":"帮我查一下当前套餐，如果剩余流量低于10GB就推荐一个流量包。"}'
# 事件：router_started → router_finished → planner_started → tool_started → tool_finished → answer_token* → workflow_completed
```

其他：`GET /health`、`GET /ready`（PG+Milvus 探测）、`GET /metrics`、
`POST /api/v1/feedback`、`GET /api/v1/conversations/{id}`。
统一错误 Schema：`{"error":{"code","message","trace_id"}}`（绝不返回堆栈）。

## 29. 测试

```bash
uv run pytest tests/unit -q                          # 59 项：零外部依赖
uv run pytest tests/integration tests/workflow tests/api -q   # 22 项：连 compose 真服务
```

重点覆盖（全部真实执行，非 mock 流程图）：

- **Retry 循环**：注入瞬时故障 → Validator → Retry → 第二次成功（retry_count==1）；
  永久失败 → 恰好重试 2 次后收口（不无限循环）；
- **Replan 循环**：注入计划性故障 → Replan → 新计划执行成功（replan_count==1）；
- **Human Approval**：高风险 → interrupt → 批准 resume 执行 / 拒绝终止 /
  一次审批只放行一个动作；
- **条件 DSL**：then/else 跳转与 SKIPPED 标记；
- **幂等**：相同 idempotency_key 重复下单返回同一订单（行数不变）；
- **Guardrails**：注入拦截、手机号/身份证脱敏、跨用户越权拦截；
- **API**：统一错误 Schema、SSE 事件序列、审批接口幂等（重复审批 409）。

服务不可达时集成测试自动 skip；CI（GitHub Actions）以 PG service + Milvus Lite
跑全部测试（ruff + mypy + pytest，MOCK_LLM=true，key 不进 CI）。

## 30. Evaluation

```bash
uv run python scripts/evaluate_rag.py              # 114 条查询 × 3 检索变体
uv run python scripts/evaluate_agent.py            # 52 任务（Mock，可复现）
uv run python scripts/evaluate_agent.py --real-llm # 正式数字（真实 DeepSeek）
```

结果写入 `evaluation/results/*.json`（gitignore，可随时重跑）。
本仓库最近一次真实运行（deepseek-chat + bge-small-zh-v1.5 + bge-reranker-base）：

| 模式 | Intent | Tool Selection | Argument | Task Success |
|---|---|---|---|---|
| Mock（确定性，52 任务） | 1.0000 | 1.0000 | 1.0 | 1.0000 |
| 真实 DeepSeek（52 任务） | 0.9615 | 0.9615 | — | 0.9423 |

真实模式剩余失败为 LLM 输出方差（工具选择/计划结构随采样波动），
Retry/Replan 循环与审批链路均真实触发（retry_rate=0.096、replan_rate=0.077）。
**Mock 数字用于回归对比（可复现），正式 Benchmark 以 --real-llm 为准。**

## 31. 性能指标

以下数字来自本机真实运行（RTX 3050 Laptop CPU 推理 + 本地容器，
`scripts/evaluate_rag.py` / `evaluate_agent.py` 输出，非编造；
不同硬件/网络结果会不同）：

| 指标 | 数值 | 说明 |
|---|---|---|
| RAG 检索（vector，114 查询） | 3802ms / 约 33ms/查询 | 含 embedding |
| RAG Hybrid+Rerank | 157531ms / 约 1.4s/查询 | reranker CPU 推理是主开销 |
| Router 真实 LLM | ~2.7s/次 | deepseek-chat |
| TASK 多步任务端到端 | ~15-25s（真实 LLM） | 含 planner + validator |
| QUERY 工具执行 | ~20-40ms | 含 DB 访问 |
| Agent Eval（Mock）52 任务 | 平均 880ms/任务 | 含 FAQ 的模型推理 |

定位瓶颈的方式：Grafana Dashboard 的 LLM 延迟按 purpose 分桶
（router/planner/validator/answer），RAG 延迟按 stage 分桶
（embedding/dense/keyword/hybrid/rerank）——先看指标再优化。

---

## 附：7 个业务 Demo（README 演示脚本）

`uv run python scripts/demo.py`（Mock 或真实模式，按 .env 配置）：

| 场景 | 输入 | 流程 |
|---|---|---|
| 1 | 我的套餐还剩多少流量？ | Router→QUERY→get_remaining_data→Answer |
| 2 | 5G套餐有哪些？ | Router→FAQ→Rewrite→Hybrid→Rerank→Answer+引用 |
| 3 | 帮我查一下当前套餐，如果剩余流量低于10GB就推荐一个流量包 | Router→TASK→Planner(含 condition)→执行 8.0<10→then→recommend→Answer |
| 4 | 帮我办理30GB流量包 | Router→TASK→create_order→风险检查→人工审批→执行→幂等 |
| 5 | 工具瞬时失败 | 注入故障→Validator→Retry→Success（tests/workflow） |
| 6 | 计划性失败 | 注入故障→Validator→Replan→Planner→新执行→Success |
| 7 | 中断恢复 | interrupt→checkpoint 落库→Command(resume)→继续执行 |

`scripts/demo_approval.py`：场景 7 的完整演示（含幂等断言）。
`scripts/demo_retrieval.py`：Dense/Keyword/Hybrid 三路检索对比。

## 附：架构决策记录

见 [docs/adr/](docs/adr/)（8 篇，含 Context/Decision/Alternatives/Trade-offs）。

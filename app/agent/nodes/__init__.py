"""LangGraph 节点适配：把纯逻辑模块接入图执行。

节点函数只做三件事：读 state → 调纯逻辑 → 返回状态更新片段。
控制流全部由 graph.py 的 conditional edges 决定。
"""

import json
from typing import Any

from app.agent import planner as planner_mod
from app.agent import router as router_mod
from app.agent.executor import check_global_cap, run_step
from app.agent.validator import semantic_check_enabled, validate_step
from app.core.logging import get_logger
from app.db.repositories import approval_repo
from app.db.session import session_scope
from app.guardrails.output import mask_sensitive
from app.guardrails.tool import USER_SCOPED_TOOLS
from app.observability import metrics
from app.tools.executor import ToolExecutionStatus, ToolExecutor
from app.tools.policies import get_policy
from app.tools.registry import registry

logger = get_logger(__name__)


def _emit(event: dict[str, Any]) -> None:
    """向 SSE 推送自定义事件（非流式调用时静默忽略）。"""
    try:
        from langgraph.config import get_stream_writer

        get_stream_writer()(event)
    except Exception:  # noqa: BLE001 — 未在流式上下文时不可用
        pass


def router_node(state: dict[str, Any]) -> dict[str, Any]:
    _emit({"type": "router_started", "query": state["query"]})
    decision, stats = router_mod.route(state["query"])
    logger.info("router_decision", extra={"intent": decision.intent, "stats": stats})
    _emit({"type": "router_finished", "intent": decision.intent, **stats})
    return {
        "intent": decision.intent,
        "metadata": {**(state.get("metadata") or {}), "router": stats},
    }


def planner_node(state: dict[str, Any]) -> dict[str, Any]:
    _emit({"type": "planner_started"})
    plan, stats = planner_mod.plan_task(state["query"], user_id=state["user_id"])
    _emit({"type": "planner_finished", "steps": len(plan.steps)})
    logger.info("planner_plan", extra={"steps": len(plan.steps), "stats": stats})
    updates: dict[str, Any] = {
        "plan": [s.model_dump() for s in plan.steps],
        "current_step": plan.steps[0].step_id if plan.steps else "END",
        "retry_count": state.get("retry_count", 0),
        "metadata": {**(state.get("metadata") or {}), "planner": stats},
    }
    if not plan.steps:
        updates["execution_status"] = "plan_complete"
        updates["final_answer"] = "根据已有执行结果，任务已完成，无需额外步骤。"
    return updates


def replan_node(state: dict[str, Any]) -> dict[str, Any]:
    _emit({"type": "replan_started", "failed_tool": state.get("failed_tool", "")})
    plan, stats = planner_mod.replan(
        state["query"],
        user_id=state["user_id"],
        failed_tool=state.get("failed_tool", ""),
        failure_reason=state.get("failure_reason", ""),
        previous_results=state.get("tool_results", {}),
    )
    metrics.agent_retry_count.labels("replan").inc()
    logger.info("planner_replan", extra={"steps": len(plan.steps), "stats": stats})
    metadata = state.get("metadata") or {}
    metadata["planner_replans"] = [*metadata.get("planner_replans", []), stats]
    updates: dict[str, Any] = {
        "plan": [s.model_dump() for s in plan.steps],
        "current_step": plan.steps[0].step_id if plan.steps else "END",
        "replan_count": state.get("replan_count", 0) + 1,
        "retry_count": 0,
        "validation_result": {},
        "metadata": metadata,
    }
    if not plan.steps:
        updates["execution_status"] = "plan_complete"
        updates["final_answer"] = "根据已有执行结果，任务已完成。"
    return updates


def executor_node(state: dict[str, Any], executor: ToolExecutor) -> dict[str, Any]:
    """执行节点：TASK 按计划执行一步；QUERY 走 bind_tools 单步选择。"""
    if state.get("intent") == "QUERY":
        return _run_query_selection(state, executor)
    if check_global_cap(state):
        return {
            "execution_status": "plan_complete",
            "error": f"工具执行次数达到全局上限（{state['total_tool_executions']}），任务终止以避免失控循环",
        }
    approval_granted = state.get("human_decision") == "approved"
    return run_step(state, executor, approval_granted=approval_granted)


def _run_query_selection(state: dict[str, Any], executor: ToolExecutor) -> dict[str, Any]:
    """QUERY 路径：LLM(bind_tools) 从低风险只读工具中选一个，执行并归档。"""
    from langchain_core.messages import HumanMessage
    from langchain_core.tools import StructuredTool

    from app.llm.deepseek import get_chat_model

    # 权限边界：QUERY 只暴露 LOW 风险只读工具（工具清单本身即策略）
    low_risk_tools = [
        StructuredTool(
            name=spec.name,
            description=spec.description,
            args_schema=spec.args_schema,
            func=lambda **kwargs: None,  # 只借 schema，真实执行走 ToolExecutor
        )
        for spec in registry.all().values()
        if (policy := get_policy(spec.name)) is not None and policy.risk == "LOW"
    ]

    from langchain_core.messages import SystemMessage

    selection_prompt = (
        f"你是电信 CRM 助手。当前会话用户是 {state['user_id']}。"
        "必须调用一个工具来回答用户查询（不要用文字代替工具调用）。"
        "user_id 参数由系统自动填充，**不要输出 user_id 参数**。只调用一个工具。示例：\n"
        "- '我的套餐还剩多少流量' → get_remaining_data\n"
        "- '我现在用什么套餐' → get_current_package\n"
        "- '查一下我的国际漫游状态' → query_roaming_status\n"
        "- '我的积分和等级' → get_customer_profile\n"
        "- '搜一下30GB以上的加餐包' → search_packages\n"
        "- '查订单 ord_xxx' → query_order\n"
    )
    try:
        msg = (
            get_chat_model()
            .bind_tools(low_risk_tools, tool_choice="required")
            .invoke([SystemMessage(content=selection_prompt), HumanMessage(content=state["query"])])
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "execution_status": "error",
            "failed_tool": "",
            "failure_reason": f"工具选择失败: {exc}",
            "validation_result": {"status": "pending"},
        }

    if not msg.tool_calls:
        return {
            "execution_status": "error",
            "failed_tool": "",
            "failure_reason": "LLM 未选择任何工具，无法完成该查询",
            "validation_result": {"status": "pending"},
        }

    tool_call = msg.tool_calls[0]
    # 审计记录用「注入后」的参数（user_id 由执行器写入，不信任 LLM）
    scoped_args = dict(tool_call["args"])
    if tool_call["name"] in USER_SCOPED_TOOLS:
        scoped_args["user_id"] = state["user_id"]
    result = executor.execute(
        tool_call["name"],
        tool_call["args"],
        actor_user_id=state["user_id"],
        run_context={"conversation_id": state.get("conversation_id", ""), "step_id": "step_query"},
    )
    record: dict[str, Any] = {
        "step_id": "step_query",
        "tool_name": tool_call["name"],
        "arguments": scoped_args,
        "status": result.status.value,
    }
    updates: dict[str, Any] = {"tool_calls": [record]}
    if result.status == ToolExecutionStatus.SUCCESS:
        updates["tool_results"] = {**state.get("tool_results", {}), "step_query": result.result}
        updates["total_tool_executions"] = state.get("total_tool_executions", 0) + 1
        # QUERY 是单步语义：成功即完成，直接收口（避免 success→executor 自循环）
        updates["execution_status"] = "plan_complete"
        updates["current_step"] = "END"
        record.update({"result": result.result})
    elif result.status == ToolExecutionStatus.APPROVAL_REQUIRED:
        updates["requires_human_approval"] = True
        updates["pending_approval"] = {**(result.approval.model_dump() if result.approval else {}), "step_id": "step_query"}
        updates["execution_status"] = "approval_required"
    else:
        updates["execution_status"] = "error"
        updates["failed_tool"] = tool_call["name"]
        updates["failure_reason"] = result.failure.message if result.failure else "执行失败"
        updates["validation_result"] = {"status": "pending"}
        record.update(
            {
                "error": updates["failure_reason"],
                # 失败分类随记录传递：Validator 据此分流 Retry / Replan / Fail
                "failure_kind": result.failure.kind.value if result.failure else "fast_fail",
            }
        )
    return updates


def validator_node(state: dict[str, Any]) -> dict[str, Any]:
    """Validator：读取最近一次工具调用，产出 validation_result。"""
    calls: list[dict[str, Any]] = state.get("tool_calls", [])
    last = calls[-1] if calls else {}
    if state.get("intent") == "QUERY":
        # QUERY 路径：与 TASK 同源的分流规则（transient→Retry / 计划性→Replan / 其余→Fail）
        from app.core.exceptions import FailureKind

        kind = None
        if last.get("status", "error") != "success":
            kind = FailureKind(last.get("failure_kind", "fast_fail"))
        verdict = validate_step(
            execution_status=last.get("status", "error"),
            failure_kind=kind,
            failure_message=last.get("error"),
            step_description="查询用户数据",
            tool_name=last.get("tool_name", ""),
            tool_result=None,
            use_semantic_check=False,
            purpose_hint="query",
        )
    else:
        execution_status = last.get("status", "error")
        from app.core.exceptions import FailureKind

        kind = None
        if execution_status != "success":
            kind = FailureKind(last.get("failure_kind", "fast_fail"))
        # 语义校验目标 = 当前步骤的描述（而非 plan[0]）
        from app.agent.executor import find_step

        current = find_step(state.get("plan", []), state.get("current_step", ""))
        step_description = (current or {}).get("description", "")
        verdict = validate_step(
            execution_status=execution_status,
            failure_kind=kind,
            failure_message=last.get("error"),
            step_description=step_description,
            tool_name=last.get("tool_name", ""),
            tool_result=last.get("result"),
            use_semantic_check=semantic_check_enabled(),
            purpose_hint="task",
        )
    logger.info("validator_verdict", extra={"verdict": verdict})
    return {"validation_result": verdict}


def retry_node(state: dict[str, Any]) -> dict[str, Any]:
    metrics.agent_retry_count.labels("retry").inc()
    logger.info(
        "retry_step",
        extra={"step": state.get("current_step"), "retry_count": state.get("retry_count", 0)},
    )
    return {"retry_count": state.get("retry_count", 0) + 1, "validation_result": {}}


def human_approval_node(state: dict[str, Any]) -> dict[str, Any]:
    """人工审批节点：LangGraph interrupt() 暂停 → 落库 → 等待 resume。

    - 暂停：把审批请求写入 approvals 表（幂等 upsert），然后 interrupt(payload)；
      API 层捕获 GraphInterrupt，向客户端返回 approval_required。
    - 恢复：POST /api/v1/approvals/{id} 用 Command(resume=decision) 重启图，
      本节点从头重跑（create_or_get 幂等），interrupt() 返回决策，
      据此设置 human_decision，由条件边路由到 executor（执行）或 answer（拒绝）。
    """
    from langgraph.types import interrupt

    pending = state.get("pending_approval") or {}
    approval_id = pending.get("approval_id", "")
    if approval_id:
        with session_scope() as db:
            approval_repo.create_or_get(
                db,
                approval_id,
                conversation_id=state.get("conversation_id", ""),
                agent_run_id=(state.get("metadata") or {}).get("agent_run_id", ""),
                tool_name=pending.get("tool_name", ""),
                arguments=pending.get("arguments", {}),
            )

    decision = interrupt(
        {
            "type": "approval_required",
            "approval_id": approval_id,
            "tool_name": pending.get("tool_name", ""),
            "arguments": pending.get("arguments", {}),
            "summary": pending.get("summary", ""),
        }
    )
    # resume 值：{"approval_id": ..., "decision": "approved"|"rejected"}
    final_decision = decision.get("decision", "rejected")
    if approval_id:
        with session_scope() as db:
            approval_repo.decide(
                db, approval_id, final_decision, decided_by=state.get("user_id", "system")
            )
    logger.info(
        "human_approval_decision",
        extra={"approval_id": approval_id, "decision": final_decision},
    )
    return {"human_decision": final_decision}


def rag_node(state: dict[str, Any]) -> dict[str, Any]:
    """FAQ 路径：完整 RAG Pipeline（Phase 7/8 接入）。"""
    try:
        from app.rag.pipeline import answer_with_rag

        result = answer_with_rag(state["query"], state.get("conversation_id", ""))
        return {
            "final_answer": result["answer"],
            "citations": result.get("citations", []),
            "retrieved_documents": result.get("documents", []),
            "rewritten_query": result.get("rewritten_query", state["query"]),
        }
    except (ImportError, Exception) as exc:  # noqa: BLE001 — Phase 5 占位，Phase 8 完善
        logger.warning("rag_not_ready", extra={"error": str(exc)[:200]})
        return {"final_answer": "知识库模块尚未接入，暂时无法回答该问题。"}


def answer_node(state: dict[str, Any]) -> dict[str, Any]:
    """Answer：统一收口，生成最终回答。"""
    stream_mode = bool((state.get("metadata") or {}).get("stream"))
    if stream_mode:
        answer = _compose_answer_streaming(state)
    else:
        answer = _compose_answer(state)
    # Output Guardrail：敏感信息脱敏后再交给客户端（双保险——
    # 合成前工具结果已脱敏，这里兜底）
    answer = mask_sensitive(answer)
    metrics_label = "failure" if state.get("error") else "success"
    intent = state.get("intent", "UNKNOWN")
    if state.get("error"):
        metrics.workflow_failure_count.labels(intent, state.get("failure_reason", "unknown")).inc()
    else:
        metrics.workflow_success_count.labels(intent).inc()
    logger.info("workflow_answer", extra={"intent": intent, "label": metrics_label})
    return {
        "final_answer": answer,
        "messages": [{"role": "assistant", "content": answer}],
    }


def _compose_answer(state: dict[str, Any]) -> str:
    """按状态合成最终回答（Mock 模板 / 真实 LLM 合成）。"""
    if state.get("requires_human_approval") and not state.get("human_decision"):
        pending = state.get("pending_approval", {})
        return (
            f"⚠️ 该操作需要人工审批：{pending.get('summary', '')}\n"
            f"approval_id: {pending.get('approval_id', '')}\n"
            f"请通过审批接口（POST /api/v1/approvals/{{approval_id}}）确认或拒绝。"
        )
    if state.get("human_decision") == "rejected":
        return "已按您的要求取消该操作，未产生任何变更。"

    if state.get("intent") == "FAQ":
        return state.get("final_answer") or "知识库中没有找到足够相关的信息。"

    if state.get("error"):
        reason = state.get("error") or state.get("failure_reason") or "未知错误"
        return f"抱歉，任务未能完成：{reason}"

    results = state.get("tool_results", {}) or {}
    if not results:
        return "抱歉，我无法处理这个请求。您可以尝试换一种说法。"

    from app.core.config import settings

    if settings.mock_llm:
        import json

        body = json.dumps(results, ensure_ascii=False, indent=2, default=str)
        return f"（Mock 模式）查询/办理结果如下：\n{body}"

    return _synthesize_answer(state, results, stream_tokens=False)


def _compose_answer_streaming(state: dict[str, Any]) -> str:
    """流式合成：模板路径整体推送；真实 LLM 路径逐 token 推送。"""
    text = _compose_answer_prefix(state)
    if text is not None:  # 非合成路径（审批/拒绝/错误/FAQ/Mock）
        _emit({"type": "answer", "content": text})
        return text

    results = state.get("tool_results", {}) or {}
    return _synthesize_answer(state, results, stream_tokens=True)


def _compose_answer_prefix(state: dict[str, Any]) -> str | None:
    """非 LLM 合成路径的固定文案；返回 None 表示需要 LLM 合成。"""
    if state.get("requires_human_approval") and not state.get("human_decision"):
        pending = state.get("pending_approval", {})
        return (
            f"⚠️ 该操作需要人工审批：{pending.get('summary', '')}\n"
            f"approval_id: {pending.get('approval_id', '')}\n"
            f"请通过审批接口（POST /api/v1/approvals/{{approval_id}}）确认或拒绝。"
        )
    if state.get("human_decision") == "rejected":
        return "已按您的要求取消该操作，未产生任何变更。"
    if state.get("intent") == "FAQ":
        return state.get("final_answer") or "知识库中没有找到足够相关的信息。"
    if state.get("error"):
        reason = state.get("error") or state.get("failure_reason") or "未知错误"
        return f"抱歉，任务未能完成：{reason}"
    results = state.get("tool_results", {}) or {}
    if not results:
        return "抱歉，我无法处理这个请求。您可以尝试换一种说法。"
    from app.core.config import settings

    if settings.mock_llm:
        import json

        body = json.dumps(results, ensure_ascii=False, indent=2, default=str)
        return f"（Mock 模式）查询/办理结果如下：\n{body}"
    return None


def _synthesize_answer(
    state: dict[str, Any], results: dict[str, Any], *, stream_tokens: bool
) -> str:
    """真实 LLM 合成。工具结果在进入 prompt 前先脱敏——
    保证流式 token 里也不会出现手机号/身份证等敏感数据。"""
    from app.llm.client import get_llm_client
    from app.llm.schemas import ChatMessage

    safe_results = json.loads(mask_sensitive(json.dumps(results, ensure_ascii=False, default=str)))
    messages = [
        ChatMessage(
            role="system",
            content=(
                "你是电信 CRM 智能助手。根据工具执行结果，用自然语言向用户汇报。"
                "数字准确、语言简洁；不得编造结果中没有的信息。"
            ),
        ),
        ChatMessage(
            role="user",
            content=f"用户问题：{state.get('query')}\n工具结果：{safe_results}\n请组织回答。",
        ),
    ]
    client = get_llm_client()
    try:
        if stream_tokens:
            parts: list[str] = []
            for chunk in client.stream(messages, temperature=0.3, purpose="answer"):
                parts.append(chunk)
                _emit({"type": "answer_token", "content": chunk})
            return "".join(parts)
        resp = client.chat(messages, temperature=0.3, purpose="answer")
        return resp.content
    except Exception as exc:  # noqa: BLE001 — 回答失败退回模板
        logger.exception("answer_synthesis_failed", extra={"error": str(exc)[:200]})
        body = json.dumps(safe_results, ensure_ascii=False, indent=2, default=str)
        return f"结果如下：\n{body}"

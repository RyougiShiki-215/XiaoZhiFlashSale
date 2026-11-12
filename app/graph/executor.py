"""执行器节点

步骤类型分流：TOOL / RAG / LLM / SKILL / FINISH
LLM/SKILL/FINISH 步骤内部保留"迷你 ReAct 工具调用循环"（最多3轮），
步骤执行中可自主补查工具。
"""
import json
import logging
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.config import get_stream_writer
from pydantic import BaseModel, Field

from app.config import get_settings
from app.llm import create_chat_model
from app.memory.long_term_memory import long_term_memory
from app.models import PlanStep, StepResult, StepStatus, StepType
from app.rag.retriever import RetrievalContext, main_retriever
from app.skill.registry import skill_registry
from app.tools.errors import Disposition, ErrorKind, ToolOutcome, execute_tool_with_policy
from app.tools.registry import ALL_TOOLS
from app.tools.runner import get_tool_runner

logger = logging.getLogger(__name__)


class ToolExecutionAbort(Exception):
    """ReAct 循环内不可恢复的工具失败（TRIGGER_REPLAN/FAIL_FAST）：中断循环上抛"""

    def __init__(self, tool_name: str, kind: ErrorKind, message: str):
        super().__init__(message)
        self.tool_name = tool_name
        self.kind = kind


def emit_event(event: dict) -> None:
    """向流式消费方推送自定义事件（经 graph.astream 的 custom 流出）。

    非流式调用（graph.invoke / 无 custom 模式）时静默丢弃，不影响原有行为。
    """
    try:
        writer = get_stream_writer()
        if callable(writer):
            writer(event)
    except Exception:  # noqa: BLE001 - 事件推送绝不影响主流程
        pass


class _ThinkTagStreamFilter:
    """流式输出中过滤 <think>...</think>（推理模型的思考过程不推送给用户）。

    需要跨 chunk 缓冲：标签可能被拆在多个 token 里（如 "<thi" + "nk>"）。
    """

    _OPEN, _CLOSE = "<think>", "</think>"

    def __init__(self):
        self._buf = ""
        self._in_think = False

    def feed(self, text: str) -> str:
        """送入一个 token，返回当前可安全外发的文本"""
        self._buf += text
        out = []
        while self._buf:
            if self._in_think:
                end = self._buf.find(self._CLOSE)
                if end == -1:
                    # 保留尾部可能是半个闭合标签的部分
                    keep = min(len(self._buf), len(self._CLOSE) - 1)
                    self._buf = self._buf[len(self._buf) - keep:] if keep else ""
                    break
                self._buf = self._buf[end + len(self._CLOSE):]
                self._in_think = False
            else:
                start = self._buf.find(self._OPEN)
                if start == -1:
                    # 保留尾部可能是半个开始标签的部分，其余可外发
                    safe = len(self._buf) - (len(self._OPEN) - 1)
                    if safe > 0:
                        out.append(self._buf[:safe])
                        self._buf = self._buf[safe:]
                    break
                out.append(self._buf[:start])
                self._buf = self._buf[start + len(self._OPEN):]
                self._in_think = True
        return "".join(out)

    def flush(self) -> str:
        """流结束时返回残留缓冲（处于 think 内则丢弃）"""
        rest, self._buf = self._buf, ""
        return "" if self._in_think else rest


def _stream_finish_answer(messages: list) -> str:
    """流式生成 FINISH 答案：逐 token 推送（type=token 事件），返回完整文本。

    说明：流式路径不绑定工具——FINISH 的语义是整理已执行步骤的结果，
    工具查询应由前置 TOOL/RAG 步骤完成；调用方在流式失败时回退到 run_mini_react_cycle。
    """
    llm = create_chat_model()
    tag_filter = _ThinkTagStreamFilter()
    full: list[str] = []
    for chunk in llm.stream(messages):
        token = chunk.content or ""
        if not token:
            continue
        full.append(token)
        visible = tag_filter.feed(token)
        if visible:
            emit_event({"type": "token", "content": visible})
    tail = tag_filter.flush()
    if tail:
        emit_event({"type": "token", "content": tail})
    return "".join(full)

# FINISH 步骤人设
FINISH_PERSONA = (
    "你是专业的闪购智能助手小智。你的回答必须：\n"
    "1. 中文自然友好，专业可信\n"
    "2. 直接输出答案，禁止输出任何think标签、思考过程、推理步骤\n"
    "3. 必要时用列表或分段让结构清晰。"
)


class _RepairedArgs(BaseModel):
    """LLM 参数修复的输出 schema"""
    reasoning: str = Field(description="参数错误原因与修正思路，一句话")
    arguments: dict = Field(description="修正后的完整参数键值对")


def _repair_tool_args_with_llm(tool_name: str, schema_str: str, bad_args: dict,
                               kind: ErrorKind, error_text: str) -> dict | None:
    """离线 TOOL 步骤的参数自愈：一次 LLM 调用修正错误参数（INVALID_ARGS / DB_CONSTRAINT）

    schema_str 来自 ToolRunner（本地或 MCP 皆可），故修复逻辑 transport 无关。
    """
    llm = create_chat_model().with_structured_output(_RepairedArgs, method="json_mode")
    prompt = (
        f"上一次工具调用失败，请修正参数。\n\n"
        f"工具名: {tool_name}\n"
        f"参数 Schema: {schema_str}\n"
        f"失败参数: {json.dumps(bad_args, ensure_ascii=False)}\n"
        f"失败类型: {kind.value}\n"
        f"失败信息: {error_text}\n\n"
        f"请返回修正后的完整参数（严格遵守 Schema 的字段名与类型）。"
    )
    resp = llm.invoke([HumanMessage(content=prompt)])
    return dict(resp.arguments) if resp and resp.arguments else None


def clean_think_tag(text: str | None) -> str | None:
    """清理 deepseek-r1 等推理模型的 <think> 标签"""
    if text is None:
        return None
    return re.sub(r" thinking[\s\S]*? response", "", text).strip()


def _parse_text_tool_calls(content: str) -> list[tuple[str, dict]]:
    """从模型以 XML 文本表达的工具调用中提取 (name, args)。

    模型偶发不产出结构化 tool_calls，而是输出类似
      <invoke name="read_skill_resource">
        <parameter name="skill_id">smart-marketing</parameter>
        <parameter name="file_path">prompts/coupon-priority-rules.txt</parameter>
      </invoke>
    的文本。此类文本无法被 langchain 直接执行，这里解析后交由 ReAct 手动调用。
    """
    calls: list[tuple[str, dict]] = []
    for m in re.finditer(r"<invoke\s+name=\"([^\"]+)\"[^>]*>(.*?)</invoke>", content, re.DOTALL):
        name = m.group(1).strip()
        if not name:
            continue
        args: dict = {}
        for pm in re.finditer(r"<parameter\s+name=\"([^\"]+)\"[^>]*>(.*?)</parameter>",
                              m.group(2), re.DOTALL):
            args[pm.group(1).strip()] = pm.group(2).strip()
        calls.append((name, args))
    return calls


def run_mini_react_cycle(initial_messages: list, max_turns: int | None = None) -> str:
    """迷你 ReAct 工具调用闭环

    每一轮：LLM(messages+tools) → 无 tool_calls 则返回文本；
    有则逐个执行工具，把 AiMessage + ToolMessage 写回 messages 进入下一轮。
    """
    s = get_settings()
    max_turns = max_turns or s.max_react_turns
    chat_model = create_chat_model()
    llm_with_tools = chat_model.bind_tools(ALL_TOOLS)

    messages = list(initial_messages)
    for turn in range(1, max_turns + 1):
        ai: AIMessage = llm_with_tools.invoke(messages)
        if ai.tool_calls:
            logger.info("🔧 步骤内迷你ReAct（turn=%d）：LLM请求调用 %d 个工具", turn, len(ai.tool_calls))
            calls = [(tc["name"], dict(tc.get("args") or {}), tc["id"] or tc["name"]) for tc in ai.tool_calls]
        else:
            # 模型偶发不产出结构化 tool_calls，而是用 XML 文本（如 <invoke name=...>）
            # 表达工具调用，导致原本可执行的调用被当成最终答案返回。这里尝试从文本解析并执行。
            text_calls = _parse_text_tool_calls(ai.content or "")
            if not text_calls:
                return ai.content or ""
            logger.info("🔧 步骤内迷你ReAct（turn=%d）：LLM 以文本XML表达 %d 个工具调用",
                        turn, len(text_calls))
            calls = [(name, args, f"text-{name}") for name, args in text_calls]
        messages.append(ai)
        for name, args, tool_call_id in calls:
            outcome: ToolOutcome = execute_tool_with_policy(
                name, get_tool_runner(), args,
                retry_delays=(0.1,),  # LLM 在环时不做参数修复，回传 guidance 让其下一轮自纠
            )
            if outcome.success:
                result_text = outcome.result
            elif outcome.disposition is Disposition.FEEDBACK_TO_LLM:
                # 可修复异常：回传带修正指导的 ToolMessage，LLM 下一轮自纠（不消耗重规划）
                result_text = outcome.guidance or outcome.error or "工具调用失败"
            else:
                # 不可修复：中断循环，交给上层转为重规划/失败
                raise ToolExecutionAbort(name, outcome.kind, outcome.error or "工具不可恢复失败")
            messages.append(ToolMessage(content=result_text, tool_call_id=tool_call_id))

    # 超过最大轮次：不再带工具调用，直接取文本兜底
    logger.warning("⚠️ run_mini_react_cycle 达到最大轮次 %d，取最后文本兜底", max_turns)
    return chat_model.invoke(messages).content or ""


def _build_step_context(state: dict) -> str:
    """构造步骤执行上下文：对话历史 + 已完成步骤结果"""
    parts = []
    history = state.get("chat_history", "")
    if history:
        parts.append(f"对话历史:\n{history}")
    done = []
    for step, result in state.get("past_steps", []):
        text = (result.result or result.error_message or "(无结果)")
        if len(text) > 1500:
            text = text[:1500] + "..."
        done.append(f"步骤{step.step_number}[{step.step_type.value}] {step.description} → {text}")
    if done:
        parts.append("已执行步骤结果:\n" + "\n".join(done))
    return "\n\n".join(parts)


# ================= 步骤实现 =================

def _execute_tool(step: PlanStep, state: dict) -> StepResult:
    """TOOL 步骤：走统一异常处理管道（分类→静默重试/参数修复/重规划/熔断）"""
    tool_name = (step.tool_name or "").strip()
    if not tool_name:
        return StepResult.fail(step.step_number, "TOOL步骤缺少 toolName",
                               trigger_replan=True, replan_reason="缺少工具名，需重规划补充正确工具")

    outcome = execute_tool_with_policy(
        tool_name, get_tool_runner(), step.tool_arguments or {},
        repair_fn=_repair_tool_args_with_llm,  # 离线执行：输入类异常做一次 LLM 参数自愈
    )

    if outcome.success:
        logger.info("✅ 步骤%d TOOL [%s] 执行成功(尝试%d次%s)，结果: %s",
                    step.step_number, tool_name, outcome.attempts,
                    ",含参数修复" if outcome.repaired else "",
                    (outcome.result or "")[:500])
        return StepResult.ok(step.step_number, outcome.result or "")

    detail = outcome.error or outcome.result or "未知错误"
    msg = f"工具[{tool_name}]失败({outcome.kind.value}): {detail}"
    logger.warning("❌ 步骤%d %s（处置: %s）", step.step_number, msg,
                   outcome.disposition.value if outcome.disposition else "N/A")

    if outcome.disposition is Disposition.FAIL_FAST:
        # 致命异常（认证/代码bug/资源耗尽/熔断中）：不再重规划同路径，直接失败
        # （路由层若仍有重规划额度，Planner 可选择换其他工具绕开）
        return StepResult.fail(step.step_number, msg)
    # 瞬时重试耗尽 / 修复失败 / 业务失败 / 记录不存在 → 重规划换路径
    return StepResult.fail(step.step_number, msg, trigger_replan=True, replan_reason=msg)


def _execute_rag(step: PlanStep, state: dict) -> StepResult:
    query = step.retrieval_query
    if not query or not query.strip():
        return StepResult.fail(step.step_number, "RAG步骤缺少 retrievalQuery")
    try:
        ctx = RetrievalContext(memory_id=state.get("memory_id"), user_id=state.get("user_id"))
        result = main_retriever.retrieve(query, ctx)

        # 注入长期记忆（用户画像）支撑个性化检索
        text_parts: list[str] = []
        try:
            profiles = long_term_memory.retrieve_profiles(state.get("user_id") or 123, query, top_k=3)
            if profiles:
                text_parts.append("【用户画像（历史长期记忆，仅供参考；若与当前对话矛盾，以用户当前最新表述为准）】")
                text_parts.extend(f"- {p.text}" for p in profiles)
        except Exception as e:
            logger.warning("RAG步骤-用户画像检索失败(userId=%s): %s", state.get("user_id"), e)

        text_parts.append(result.to_context_text())
        final_text = "\n".join(text_parts)
        logger.info("✅ 步骤%d RAG 命中%d块+画像，摘要: %s",
                    step.step_number, len(result.chunks), final_text[:300])
        return StepResult.ok(step.step_number, final_text)
    except Exception as e:
        logger.error("❌ 步骤%d RAG 检索异常: %s", step.step_number, e)
        return StepResult.fail(step.step_number, f"RAG检索异常: {e}")


def _execute_llm(step: PlanStep, state: dict) -> StepResult:
    prompt_parts = [
        f"=== 当前执行任务（LLM步骤{step.step_number}） ===",
        step.description or "请进行推理思考",
        "",
    ]
    if step.llm_prompt:
        prompt_parts.append(f"=== 任务附加说明（llmPrompt） ===\n{step.llm_prompt}\n")
    context_text = _build_step_context(state)
    if context_text:
        prompt_parts.append(f"=== 已有上下文（对话历史+其他步骤结果） ===\n{context_text}\n")
    prompt_parts.append("请基于以上信息完成此 LLM 推理步骤的任务，组织对应的输出内容。")

    try:
        answer = run_mini_react_cycle([HumanMessage(content="\n".join(prompt_parts))])
        return StepResult.ok(step.step_number, answer)
    except ToolExecutionAbort as abort:
        # ReAct 循环内工具不可恢复失败：转为重规划换路径
        logger.warning("❌ 步骤%d LLM 内工具[%s]不可恢复失败(%s)", step.step_number,
                       abort.tool_name, abort.kind.value)
        msg = f"步骤内工具[{abort.tool_name}]失败({abort.kind.value}): {abort}"
        return StepResult.fail(step.step_number, msg, trigger_replan=True, replan_reason=msg)
    except Exception as e:
        logger.error("❌ 步骤%d LLM 调用异常: %s", step.step_number, e)
        return StepResult.fail(step.step_number, f"LLM调用异常: {e}")


def _execute_skill(step: PlanStep, state: dict) -> StepResult:
    skill_id = step.skill_id or ""
    if not skill_id.strip():
        return StepResult.fail(step.step_number, "SKILL步骤缺少 skillId",
                               trigger_replan=True, replan_reason="缺少Skill ID，需重规划补充正确 skillId")
    # ⭐ 懒加载：走到这里才读取 skill.md 正文
    content = skill_registry.ensure_content_loaded(skill_id)
    if content is None:
        return StepResult.fail(
            step.step_number,
            f"找不到 Skill[{skill_id}]（已启用Skill: {skill_registry.list_ids()}）",
            trigger_replan=True, replan_reason="Skill不存在或已禁用，需重规划换用可用Skill或改用LLM步骤",
        )
    meta = content.metadata

    system_parts = [f"=== Skill [{meta.name}]（id={skill_id}）==="]
    if meta.description:
        system_parts.append(f"技能简介: {meta.description}\n")
    if content.markdown_body.strip():
        system_parts.append("=== 技能规则/模板（Skill Definition） ===")
        system_parts.append(content.markdown_body)
        system_parts.append("")

    # 资源清单：让 LLM 在 ReAct 循环里按需调用工具读取
    resources = skill_registry.list_resource_paths(skill_id)
    if resources:
        system_parts.append("=== Skill 自带资源文件清单（可按需调用工具读取） ===")
        system_parts.extend(f"  {i}. {p}" for i, p in enumerate(resources, 1))
        system_parts.append(
            f"\n【重要】你可以在必要时调用以下配套工具访问上述 Skill 资源：\n"
            f"  ① 【列出Skill资源文件】(skill_id=\"{skill_id}\") — 再次查看或刷新资源清单\n"
            f"  ② 【读取Skill资源文件】(skill_id=\"{skill_id}\", file_path=\"<相对路径>\") — 读取某个具体文件的文本内容（一次读一个）"
        )

    user_parts = [f"=== 当前 Skill 任务（步骤{step.step_number}） ===",
                  step.description or "使用该Skill完成用户任务", ""]
    if step.skill_prompt:
        user_parts.append(f"=== Planner 传入的 Skill 附加说明（skillPrompt） ===\n{step.skill_prompt}\n")
    context_text = _build_step_context(state)
    if context_text:
        user_parts.append(f"=== 已有上下文（对话历史+其他步骤结果） ===\n{context_text}\n")
    user_parts.append("请严格按照上面 Skill 规则/模板中规定的结构和要求组织输出，完成此 Skill 任务。")

    try:
        answer = run_mini_react_cycle([
            SystemMessage(content="\n".join(system_parts)),
            HumanMessage(content="\n".join(user_parts)),
        ])
        logger.info("✅ 步骤%d SKILL [%s] 执行成功，结果: %s", step.step_number, skill_id, answer[:500])
        return StepResult.ok(step.step_number, answer)
    except ToolExecutionAbort as abort:
        logger.warning("❌ 步骤%d SKILL[%s] 内工具[%s]不可恢复失败(%s)",
                       step.step_number, skill_id, abort.tool_name, abort.kind.value)
        msg = f"Skill[{skill_id}]步骤内工具[{abort.tool_name}]失败({abort.kind.value}): {abort}"
        return StepResult.fail(step.step_number, msg, trigger_replan=True, replan_reason=msg)
    except Exception as e:
        logger.error("❌ 步骤%d SKILL [%s] 执行异常: %s", step.step_number, skill_id, e)
        return StepResult.fail(step.step_number, f"Skill[{skill_id}]执行异常: {e}",
                               trigger_replan=True,
                               replan_reason=f"Skill调用异常，建议重规划（换可用Skill或改用LLM/RAG策略）。错误信息：{e}")


def _execute_finish(step: PlanStep, state: dict) -> StepResult:
    prompt_parts = ["=== FINALIZING：整理最终答案 ===", ""]
    context_text = _build_step_context(state)
    if context_text:
        prompt_parts.append(f"=== 已有上下文（对话历史+已执行步骤结果） ===\n{context_text}\n")
    prompt_parts.append("=== 当前任务 ===")
    prompt_parts.append(step.description or "整理所有信息，输出给用户的最终友好回答")

    messages = [
        SystemMessage(content=FINISH_PERSONA),
        HumanMessage(content="\n".join(prompt_parts)),
    ]

    # 首选流式生成：逐 token 推送给流式消费方（非流式调用时等价于普通 invoke）
    try:
        answer = _stream_finish_answer(messages)
    except Exception as e:
        logger.warning("⚠️ FINISH 流式生成失败，回退 ReAct 循环。原因: %s", e)
        answer = ""

    # 流式失败/空输出 → 回退原 ReAct 循环（可补查工具）
    if not (answer or "").strip():
        try:
            answer = run_mini_react_cycle(messages)
        except Exception as e:
            # 终极兜底：返回原始上下文
            logger.warning("⚠️ FINISH 步骤LLM调用失败，返回原始上下文兜底。原因: %s", e)
            fallback = "（系统整理答案时遇到小问题，以下是目前已得到的原始信息供您参考：）\n\n"
            fallback += context_text or "暂无信息"
            answer = fallback
    return StepResult.ok(step.step_number, answer)


def execute_step_node(state: dict) -> dict:
    """图节点：执行下一个 PENDING 步骤"""
    plan = state["plan"]
    step = plan.next_pending_step()
    if step is None:
        logger.warning("⚠️ 没有待执行步骤，直接进入 FINALIZING")
        return {"last_result": None}

    logger.info("▶️ 执行步骤%d [%s] %s", step.step_number, step.step_type.value, step.description)
    emit_event({"type": "step_start", "step_number": step.step_number,
                "step_type": step.step_type.value, "description": step.description})

    try:
        result = {
            StepType.TOOL: _execute_tool,
            StepType.RAG: _execute_rag,
            StepType.LLM: _execute_llm,
            StepType.SKILL: _execute_skill,
            StepType.FINISH: _execute_finish,
        }[step.step_type](step, state)
    except Exception as e:
        logger.error("❌ Executor 执行步骤%d[%s]异常: %s", step.step_number, step.step_type.value, e)
        result = StepResult.fail(step.step_number, f"步骤执行异常: {type(e).__name__}: {e}")

    # 更新步骤状态
    step.status = StepStatus.COMPLETED if result.success else StepStatus.FAILED
    step.execution_result = result.result if result.success else f"失败: {result.error_message}"

    # FINISH 成功 → 最终答案
    final_answer = ""
    if step.step_type == StepType.FINISH and result.success:
        final_answer = clean_think_tag(result.result) or ""

    emit_event({"type": "step_end", "step_number": step.step_number, "success": result.success})

    return {
        "plan": plan,
        "last_result": result,
        "past_steps": [(step, result)],
        "final_answer": final_answer,
    }


def finalize_node(state: dict) -> dict:
    """图节点：最终化

    若 final_answer 为空（计划中没有 FINISH 步骤或 FINISH 失败），
    合成一个 FINISH 步骤兜底生成最终答案。
    """
    final_answer = state.get("final_answer") or ""
    if not final_answer.strip():
        logger.info("🏁 FINALIZING：final_answer 为空，合成 FINISH 步骤兜底")
        synthetic = PlanStep(step_number=999, step_type=StepType.FINISH,
                             description="整理最终答案", status=StepStatus.PENDING)
        result = _execute_finish(synthetic, state)
        final_answer = clean_think_tag(result.result) or "（系统无输出）"
    return {"final_answer": final_answer}

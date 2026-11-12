"""规划器节点

采用 Pydantic 结构化输出（with_structured_output），
解析失败时同样回退到「RAG → LLM → FINISH」兜底计划。
"""
import json
import logging
from typing import Optional

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from app.config import get_settings
from app.graph.executor import emit_event
from app.llm import create_chat_model
from app.models import Plan, PlanStep, StepResult, StepStatus, StepType
from app.skill.registry import skill_registry
from app.tools.registry import render_tools_as_text

logger = logging.getLogger(__name__)


class StepSchema(BaseModel):
    """单个规划步骤的 LLM 输出 schema"""
    step_type: StepType = Field(description="步骤类型: RAG/TOOL/LLM/SKILL/FINISH")
    description: str = Field(description="步骤要做什么，一句话")
    tool_name: Optional[str] = Field(default=None, description="TOOL步骤的工具名称")
    tool_arguments: dict = Field(default_factory=dict, description="TOOL步骤的工具参数键值对")
    retrieval_query: Optional[str] = Field(default=None, description="RAG步骤的检索查询文本")
    llm_prompt: Optional[str] = Field(default=None, description="LLM步骤的提示词")
    skill_id: Optional[str] = Field(default=None, description="SKILL步骤的技能ID")
    skill_prompt: Optional[str] = Field(default=None, description="SKILL步骤的附加任务说明")


class PlanSchema(BaseModel):
    """完整计划的 LLM 输出 schema"""
    strategy: str = Field(description="一句话说明整体执行策略")
    steps: list[StepSchema] = Field(description="有序执行步骤列表，最后一步必须是FINISH")


class PlanValidityDecision(BaseModel):
    """执行后计划适配性校验的 LLM 输出 schema"""
    plan_is_valid: bool = Field(description="true=剩余计划仍然合适，继续执行；false=计划已不适用，需要重规划")
    reason: str = Field(description="判断依据，一句话说明计划为什么合适/不合适")


# 规划器系统提示词
PLANNER_SYSTEM_PROMPT = """你是一个智能任务规划器（Task Planner）。你的职责是将用户的目标拆解为若干个有序执行步骤。

【步骤类型说明】
1. RAG：调用知识库检索，参数 retrievalQuery 是要检索的查询文本。用于查询商家、菜品、业务规则等知识。
2. TOOL：调用具体业务工具，参数 toolName 是工具名称，toolArguments 是参数键值对。
3. LLM：直接调用大模型进行推理、总结、思考，参数 llmPrompt 是提示词。
4. SKILL：调用专门的技能包（Skill），参数 skillId 是可用Skill列表中给出的 id，参数 skillPrompt（可选）是给这个技能的附加任务说明。
          Skill 适合处理专业领域的复杂任务（如智能营销、菜品推荐、优惠券策略等）。
          命中 Skill 触发词或标签时，请优先使用 SKILL 步骤而不是普通 LLM 步骤，以获得更准确、结构化的能力。
5. FINISH：输出最终答案（此步骤必须是最后一步）。

【规划原则】
1. 简单问题（闲聊、常识）可以只有 LLM + FINISH 两步。
2. 需要背景知识的，先 RAG 再 LLM 再 FINISH。
3. 需要操作数据（查订单、推荐商品）的，先查询类工具再思考再 FINISH。
4. 涉及下单/修改操作，务必先查询确认（如先查商家信息）再执行变更操作。
5. 每个步骤只做一件事，不要过大。
6. 当用户问题命中某个 Skill 的触发词或标签时，优先使用 SKILL 步骤来调用专业能力。"""


def _render_skills_for_plan(objective: str) -> str:
    """按关键词匹配 TopN Skill 元数据（最多10个防膨胀）"""
    matched = skill_registry.match_by_keywords(objective)[:10]
    if not matched:
        return ""
    lines = []
    for i, m in enumerate(matched, 1):
        lines.append(f"{i}. [SKILL] id={m.skill_id}")
        lines.append(f"     名称: {m.name}")
        lines.append(f"     说明: {m.description}")
        if m.tags:
            lines.append(f"     标签: {', '.join(m.tags)}")
        if m.triggers:
            lines.append(f"     触发词: {', '.join(m.triggers)}")
    return "\n".join(lines)


def _fallback_plan(objective: str) -> Plan:
    """解析失败兜底计划"""
    return Plan(
        user_objective=objective,
        strategy="兜底策略：检索+推理+输出",
        steps=[
            PlanStep(step_number=1, step_type=StepType.RAG, description="检索知识库获取与问题相关的信息",
                     retrieval_query=objective, status=StepStatus.PENDING),
            PlanStep(step_number=2, step_type=StepType.LLM, description="结合检索到的信息，思考并生成完整答案",
                     llm_prompt=objective, status=StepStatus.PENDING),
            PlanStep(step_number=3, step_type=StepType.FINISH, description="输出最终答案给用户",
                     status=StepStatus.PENDING),
        ],
    )


def _coerce_step_type(value) -> Optional[StepType]:
    """把 LLM 返回的步骤类型字符串转成 StepType 枚举（兼容大小写及常见别名）"""
    if not value:
        return None
    v = str(value).strip().upper()
    for member in StepType:
        if member.value == v:
            return member
    return {
        "SEARCH": StepType.RAG, "QUERY": StepType.RAG, "RETRIEVE": StepType.RAG,
        "REASON": StepType.LLM, "THINK": StepType.LLM, "MCP": StepType.TOOL,
        "END": StepType.FINISH, "ANSWER": StepType.FINISH,
    }.get(v)


def _get_field(data: dict, *keys, default=None):
    """按多个候选键取字段，兼容 LLM 返回的驼峰/蛇形命名差异"""
    for k in keys:
        if k in data and data[k] is not None:
            return data[k]
    return default


def _parse_plan_json(raw: str, objective: str) -> Plan:
    """宽松解析 LLM 返回的计划 JSON

    不再依赖 Pydantic 严格 schema：DeepSeek json_mode 常输出与 PlanSchema 字段名
    不一致的键（如 type/skillId/retrievalQuery），严格校验会频繁抛错走兜底。
    这里做键名兼容 + 缺失字段宽容 + 末尾自动补 FINISH。
    """
    text = (raw or "").strip()
    if text.startswith("```"):  # 去掉 markdown 代码围栏
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        obj = json.loads(text)
    except Exception:
        obj = None
    if not isinstance(obj, (dict, list)):
        raise ValueError(f"计划输出不是有效 JSON: {text[:200]}")

    if isinstance(obj, list):  # LLM 可能直接返回步骤数组（无 strategy/steps 包裹）
        strategy = ""
        steps_raw = obj
    else:
        strategy = _get_field(obj, "strategy", "strategyText", default="") or ""
        steps_raw = obj.get("steps")
        if steps_raw is None:
            steps_raw = obj.get("plan", []) if isinstance(obj.get("plan"), list) else []
        if not isinstance(steps_raw, list) and any(
            k in obj for k in ("stepType", "step_type", "type")
        ):
            steps_raw = [obj]  # 顶层 dict 无 steps 但含步骤字段时视为单步

    steps = []
    if isinstance(steps_raw, list):
        for i, s in enumerate(steps_raw, 1):
            if not isinstance(s, dict):
                continue
            global_type = _get_field(obj, "step_type", "stepType", "type", default="")
            stype = _coerce_step_type(
                _get_field(s, "step_type", "stepType", "type", default=global_type)
            ) or StepType.LLM
            tool_args = _get_field(s, "tool_arguments", "toolArguments", "arguments", default={}) or {}
            if isinstance(tool_args, str):
                try:
                    tool_args = json.loads(tool_args)
                except Exception:
                    tool_args = {}
            steps.append(PlanStep(
                step_number=i,
                step_type=stype,
                description=_get_field(s, "description", "desc", default="") or "",
                status=StepStatus.PENDING,
                tool_name=_get_field(s, "tool_name", "toolName"),
                tool_arguments=tool_args if isinstance(tool_args, dict) else {},
                retrieval_query=_get_field(s, "retrieval_query", "retrievalQuery", "query"),
                llm_prompt=_get_field(s, "llm_prompt", "llmPrompt", "prompt"),
                skill_id=_get_field(s, "skill_id", "skillId"),
                skill_prompt=_get_field(s, "skill_prompt", "skillPrompt"),
            ))

    if not steps:
        raise ValueError("计划中没有可用的步骤")

    if steps[-1].step_type != StepType.FINISH:  # 自动保证最后一步是 FINISH
        steps.append(PlanStep(
            step_number=len(steps) + 1,
            step_type=StepType.FINISH,
            description="输出最终答案给用户",
            status=StepStatus.PENDING,
        ))
    return Plan(user_objective=objective, strategy=strategy, steps=steps)


def _call_planner(
    objective: str,
    chat_history: str,
    execution_summary: str = "",
    replan_note: str = "",
) -> Plan:
    """调用 LLM 生成计划"""
    # 用 json_object 强制 DeepSeek 返回合法的 JSON（response_format），避免它自由发挥成
    # 纯文本列表；字段名不匹配仍由 _parse_plan_json 宽松解析兜底。
    llm = create_chat_model().bind(response_format={"type": "json_object"})

    parts = [PLANNER_SYSTEM_PROMPT,
             ("【输出要求】请严格只输出一个 JSON 对象（纯 JSON，禁止用 Markdown 代码块或文字列表）："
              '{"strategy": "一句话策略", "steps": [{"step_type": "RAG", "description": "..."}]}。'
              "step_type 只能是 RAG/TOOL/LLM/SKILL/FINISH 之一，最后一步必须是 FINISH。"),
             "", f"=== 用户目标 ===\n{objective}", ""]
    tools_desc = render_tools_as_text()
    if tools_desc:
        parts.append(f"=== 可用工具 ===\n{tools_desc}\n")
    skills_desc = _render_skills_for_plan(objective)
    if skills_desc:
        parts.append(f"=== 可用Skill（专门技能包） ===\n{skills_desc}\n")
    if chat_history:
        parts.append(f"=== 对话历史摘要 ===\n{chat_history}\n")
    if execution_summary:
        parts.append(f"=== 已执行步骤结果 ===\n{execution_summary}\n")
    if replan_note:
        parts.append(replan_note)

    try:
        resp = llm.invoke("\n".join(parts))
        content = getattr(resp, "content", resp)
        if isinstance(content, list):
            content = "\n".join(m.get("text", "") if isinstance(m, dict) else str(m) for m in content)
        plan = _parse_plan_json(content or "", objective)
        if not plan.steps:
            return _fallback_plan(objective)
        return plan
    except Exception as e:
        logger.warning("⚠️ Planner 解析LLM输出失败，使用兜底策略: %s", e)
        return _fallback_plan(objective)


def plan_node(state: dict) -> dict:
    """图节点：生成初始计划"""
    objective = state["input"]
    logger.info("📋 Planner 开始生成初始计划，目标: %s", objective[:100])
    emit_event({"type": "plan_start", "objective": objective[:200]})
    plan = _call_planner(objective, state.get("chat_history", ""))
    logger.info("✅ 计划生成完成，共 %d 步: %s",
                len(plan.steps), [f"{s.step_type.value}:{s.description}" for s in plan.steps])
    emit_event({"type": "plan_ready", "strategy": plan.strategy or "",
                "steps": [f"{s.step_type.value}:{s.description}" for s in plan.steps]})
    return {"plan": plan, "past_steps": [], "replan_count": 0, "final_answer": ""}


def replan_node(state: dict) -> dict:
    """图节点：重规划"""
    objective = state["input"]
    reason = state.get("replan_reason") or "步骤执行失败"
    summary = execution_summary(state.get("past_steps", []))
    curr = state.get("replan_count", 0)
    logger.info("🔁 触发重规划（第%d次）原因: %s", curr + 1, reason)
    emit_event({"type": "replan", "count": curr + 1, "reason": reason})

    # ⭐ 最后一次机会强制收敛：直接基于已有结果输出 FINISH，不再重新规划/发起检索。
    #    避免「数据缺失 → 校验失效 → 重规划 → 再次检索」反复循环导致超时。
    if curr + 1 >= get_settings().max_replan_count:
        logger.info("🏁 已到重规划上限，强制收敛为仅 FINISH 步骤")
        plan = Plan(
            user_objective=objective,
            strategy="已到重规划上限，直接基于已有信息回答并结束",
            steps=[
                PlanStep(step_number=1, step_type=StepType.FINISH,
                         description="基于已获取的信息与执行结果，输出最终答案",
                         status=StepStatus.PENDING),
            ],
        )
    else:
        plan = _call_planner(
            objective,
            state.get("chat_history", ""),
            execution_summary=summary + f"\n\n【重规划原因】{reason}",
            replan_note=("注意：这是一次重规划。请优先根据「已执行步骤结果」直接生成最终答案——"
                         "如果之前的检索/工具已经查到了所需信息，就不要再发起 RAG 或 TOOL 检索，"
                         "直接给出答案并只包含一个 FINISH 步骤即可；"
                         "只有当你确信需要全新的信息时，才补充必要的步骤。"),
        )
    plan.needs_replan = True
    plan.replan_reason = reason
    return {"plan": plan, "replan_count": curr + 1}


def _render_remaining_steps(plan: Plan) -> str:
    """渲染剩余待执行步骤（校验时提供给 LLM 判断）"""
    pending = [s for s in plan.steps if s.status == StepStatus.PENDING]
    if not pending:
        return "（无剩余步骤）"
    lines = []
    for s in pending:
        detail = s.description
        if s.step_type == StepType.TOOL and s.tool_name:
            detail += f"（工具: {s.tool_name}）"
        elif s.step_type == StepType.RAG and s.retrieval_query:
            detail += f"（查询: {s.retrieval_query}）"
        elif s.step_type == StepType.SKILL and s.skill_id:
            detail += f"（skill: {s.skill_id}）"
        lines.append(f"步骤{s.step_number}[{s.step_type.value}] {detail}")
    return "\n".join(lines)


def check_plan_validity(state: dict) -> Optional[PlanValidityDecision]:
    """执行后校验：已执行结果是否让剩余计划失效

    由路由层在「步骤成功且仍有剩余步骤」时调用；一旦发现计划不再适用即触发重规划。
    校验本身异常/失败一律视为计划有效，绝不因校验而阻断主流程。
    """
    objective = state.get("input", "")
    plan: Plan = state["plan"]
    summary = execution_summary(state.get("past_steps", []))
    remaining = _render_remaining_steps(plan)

    prompt = (
        "你是任务执行监控器，负责在执行过程中判断当前计划是否仍然合适。\n\n"
        f"=== 用户目标 ===\n{objective}\n"
        f"=== 计划策略 ===\n{plan.strategy or ''}\n"
        f"=== 已执行步骤结果 ===\n{summary}\n"
        f"=== 剩余计划步骤 ===\n{remaining}\n\n"
        "请判断：基于已执行步骤的结果，剩余计划步骤是否仍然合理、值得继续执行？\n"
        "若执行结果与计划假设矛盾（例如数据不存在、前提已变化、剩余步骤已无意义、"
        "或已发现更优做法），返回 plan_is_valid=false 并说明原因；否则返回 true。"
        "请将判断结果以 JSON 格式输出（字段: plan_is_valid 布尔值, reason 简短理由）。"
    )

    try:
        s = get_settings()
        # 校验可用独立 provider/轻量模型（PLAN_CHECK_PROVIDER / PLAN_CHECK_MODEL），空则跟随主配置
        chat_model = create_chat_model(provider=s.plan_check_provider or None,
                                       model=s.plan_check_model or None)
        llm = chat_model.with_structured_output(PlanValidityDecision, method="json_mode")
        decision: PlanValidityDecision = llm.invoke([HumanMessage(content=prompt)])
        logger.info("🔍 计划适配性校验: valid=%s, reason=%s", decision.plan_is_valid, decision.reason)
        emit_event({"type": "plan_check", "valid": decision.plan_is_valid, "reason": decision.reason})
        return decision
    except Exception as e:
        logger.warning("⚠️ 计划适配性校验失败，按计划有效继续。原因: %s", e)
        return None


def execution_summary(past_steps: list[tuple[PlanStep, StepResult]]) -> str:
    """已执行步骤摘要"""
    if not past_steps:
        return "（尚未执行任何步骤）"
    lines = []
    for step, result in past_steps:
        if result.success:
            text = (result.result or "")[:1500]
            lines.append(f"步骤{step.step_number}[{step.step_type.value}] {step.description} → 成功: {text}")
        else:
            lines.append(f"步骤{step.step_number}[{step.step_type.value}] {step.description} → 失败: {result.error_message}")
    return "\n".join(lines)

"""图构建

执行流程：
  INITIAL → PLANNING → PLAN_READY → EXECUTING_STEP → STEP_COMPLETED
          → (REPLANNING | PLAN_READY | FINALIZING) → COMPLETED

LangGraph 图结构：
  START → plan → execute_step →(route_after_step)→ replan | execute_step | finalize
  replan → execute_step
  finalize → END
（STEP_COMPLETED 的分支判定 = 条件边 route_after_step）
"""
import logging

from langgraph.graph import END, START, StateGraph

from app.config import get_settings
from app.graph.executor import execute_step_node, finalize_node
from app.graph.planner import check_plan_validity, plan_node, replan_node
from app.graph.state import PlanExecuteState
from app.memory.working_memory import get_checkpointer
from app.models import StepType

logger = logging.getLogger(__name__)


def route_after_step(state: dict) -> str:
    """条件边：步骤完成后的路由

    返回值：
      "replan"       → 步骤失败/触发重规划/成功步骤使剩余计划失效，且重规划次数未耗尽
      "execute_step" → 成功且计划校验通过、还有待执行步骤
      "finalize"     → FINISH 完成 / 全部完成 / 失败且无重规划机会（立即整理已成功结果）
    """
    s = get_settings()
    plan = state["plan"]
    last_result = state.get("last_result")

    past_steps = state.get("past_steps", [])
    if last_result is None or not past_steps:
        return "finalize"

    last_step, _ = past_steps[-1]

    # ---- 是否需要重规划 ----
    should_replan = False
    replan_reason = None
    if not last_result.success:
        should_replan = state.get("replan_count", 0) < s.max_replan_count
        replan_reason = f"步骤执行失败: {last_result.error_message}"
    elif last_result.trigger_replan:
        should_replan = state.get("replan_count", 0) < s.max_replan_count
        replan_reason = last_result.replan_reason
    elif (s.plan_check_enabled
          and last_step.step_type != StepType.FINISH
          and not plan.all_steps_completed()
          and state.get("replan_count", 0) < s.max_replan_count):
        # ⭐ 成功步骤：校验剩余计划是否仍合适（执行中发现计划不适用→重规划换路径）
        try:
            decision = check_plan_validity(state)
        except Exception as e:
            logger.warning("⚠️ 计划适配性校验异常，按计划有效继续。原因: %s", e)
            decision = None
        if decision is not None and not decision.plan_is_valid:
            should_replan = True
            replan_reason = decision.reason or "执行中发现计划不再合适"

    if should_replan:
        logger.info("🔀 路由 → REPLANNING（原因: %s）", replan_reason)
        return "replan"

    is_finish_step = last_step.step_type == StepType.FINISH
    all_done = plan.all_steps_completed()

    # ⭐ 步骤失败且重规划已耗尽 → 立即 FINALIZING
    if not last_result.success and not is_finish_step:
        logger.warning("⚠️ 步骤%d失败且已无重规划机会，直接 FINALIZING（整理已成功的结果）",
                       last_result.step_number)
        return "finalize"
    if is_finish_step or all_done:
        return "finalize"
    return "execute_step"


def build_graph():
    """构建 Plan-and-Execute 状态图"""
    builder = StateGraph(PlanExecuteState)

    builder.add_node("plan", plan_node)
    builder.add_node("execute_step", execute_step_node)
    builder.add_node("replan", replan_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "execute_step")
    builder.add_edge("replan", "execute_step")
    builder.add_edge("finalize", END)
    builder.add_conditional_edges(
        "execute_step",
        route_after_step,
        {"replan": "replan", "execute_step": "execute_step", "finalize": "finalize"},
    )

    # 工作记忆 checkpointer：state_store=redis 且可连时挂 RedisSaver；否则不挂（内存模式）
    cp = get_checkpointer()
    return builder.compile(**({} if cp is None else {"checkpointer": cp}))


# 全局编译图
plan_execute_graph = build_graph()

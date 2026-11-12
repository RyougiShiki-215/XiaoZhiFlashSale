"""图状态定义

状态机显式维护 Phase 枚举（INITIAL/PLANNING/PLAN_READY/...），
LangGraph 版把"Phase 状态机"翻译为"图节点 + 条件边"，State 只保留数据。
"""
from typing import Annotated, Optional, TypedDict

from app.config import get_settings
from app.models import Plan, PlanStep, StepResult


def _demote(step: PlanStep, result: StepResult, skeleton_chars: int) -> tuple[PlanStep, StepResult]:
    """把一条步骤结果降级为骨架：保留成败与结果梗概，丢弃正文细节

    幂等：已压缩（compressed=True）的条目原样返回，避免重复加前缀/截断。
    """
    if result.compressed:
        return step, result
    if result.success:
        text = (result.result or "")[:skeleton_chars]
        new_result = StepResult.ok(step.step_number, f"(已压缩摘要) {text}")
    else:
        # 失败信息通常较短且对重规划有诊断价值，同样截断到骨架长度
        err = (result.error_message or "")[:skeleton_chars]
        new_result = StepResult(
            step_number=step.step_number,
            success=False,
            error_message=err,
            trigger_replan=result.trigger_replan,
            replan_reason=result.replan_reason,
        )
    new_result.compressed = True
    return step, new_result


def compressed_steps_add(existing, new) -> list[tuple[PlanStep, StepResult]]:
    """past_steps 的分层压缩 reducer（替代裸 operator.add）

    策略：近处保真、远处保骨架——
      - 最近 steps_keep_full 条全量保留（FINISH 合成答案、路由看最近一步都依赖完整结果）
      - 更早的步骤降级为骨架（描述 + 成败 + 结果前 N 字符），不丢"做过什么/是否成功"的语义
    与 ChatMemory 的 LLM 摘要压缩不同：past_steps 生命周期仅限单次图执行，
    用确定性降级即可收敛体积，不值得为其额外付出 LLM 调用。
    """
    s = get_settings()
    combined = list(existing or []) + list(new or [])
    keep = s.steps_keep_full
    if len(combined) <= keep:
        return combined
    demoted = [_demote(step, result, s.steps_skeleton_chars)
               for step, result in combined[:-keep]]
    return demoted + list(combined[-keep:])


class PlanExecuteState(TypedDict, total=False):
    """Plan-and-Execute 全流程状态"""
    # 输入
    input: str                            # 用户消息
    chat_history: str                     # 对话历史摘要
    memory_id: int                        # 会话记忆ID
    user_id: int                          # 用户ID（RAG动态过滤等）

    # 计划与执行
    plan: Plan                            # 当前计划
    past_steps: Annotated[list[tuple[PlanStep, StepResult]], compressed_steps_add]  # 已执行步骤（追加式+分层压缩）
    last_result: Optional[StepResult]     # 最近一步的执行结果

    # 控制
    replan_count: int                     # 已重规划次数
    replan_reason: Optional[str]          # 本次重规划原因

    # 输出
    final_answer: str                     # 最终给用户的回答

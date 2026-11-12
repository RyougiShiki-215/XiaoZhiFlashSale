"""Plan-and-Execute 数据模型"""
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class StepType(str, Enum):
    """步骤类型"""
    RAG = "RAG"      # 知识库检索
    TOOL = "TOOL"    # 业务工具调用
    LLM = "LLM"      # 大模型推理
    SKILL = "SKILL"  # 技能包调用
    FINISH = "FINISH"  # 输出最终答案


class StepStatus(str, Enum):
    """步骤状态"""
    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class PlanStep(BaseModel):
    """单个执行步骤"""
    step_number: int = 0
    step_type: StepType = StepType.LLM
    description: str = ""
    status: StepStatus = StepStatus.PENDING

    # TOOL 步骤
    tool_name: Optional[str] = None
    tool_arguments: dict = Field(default_factory=dict)

    # RAG 步骤
    retrieval_query: Optional[str] = None

    # LLM 步骤
    llm_prompt: Optional[str] = None

    # SKILL 步骤
    skill_id: Optional[str] = None
    skill_prompt: Optional[str] = None

    # 执行结果
    execution_result: Optional[str] = None


class Plan(BaseModel):
    """执行计划"""
    user_objective: str = ""
    strategy: Optional[str] = None
    steps: list[PlanStep] = Field(default_factory=list)
    needs_replan: bool = False
    replan_reason: Optional[str] = None

    def next_pending_step(self) -> Optional[PlanStep]:
        for s in self.steps:
            if s.status == StepStatus.PENDING:
                return s
        return None

    def all_steps_completed(self) -> bool:
        return all(s.status != StepStatus.PENDING for s in self.steps)


class StepResult(BaseModel):
    """步骤执行结果"""
    step_number: int
    success: bool
    result: Optional[str] = None
    error_message: Optional[str] = None
    trigger_replan: bool = False
    replan_reason: Optional[str] = None
    compressed: bool = False  # 已被降级压缩（仅保留结果骨架，详见 state.py 的 reducer）

    @classmethod
    def ok(cls, step_number: int, result: str) -> "StepResult":
        return cls(step_number=step_number, success=True, result=result)

    @classmethod
    def fail(
        cls,
        step_number: int,
        error: str,
        trigger_replan: bool = False,
        replan_reason: Optional[str] = None,
    ) -> "StepResult":
        return cls(
            step_number=step_number,
            success=False,
            error_message=error,
            trigger_replan=trigger_replan,
            replan_reason=replan_reason,
        )

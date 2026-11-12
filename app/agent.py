"""小智智能体

流程：加载系统提示词 → 组装初始 State → plan_execute_graph.invoke(...)
"""
import logging
from pathlib import Path

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from app.config import BASE_DIR
from app.graph.builder import plan_execute_graph
from app.memory.chat_memory import ChatMemory, memory_provider
from app.memory.working_memory import clear_thread, make_thread_id

logger = logging.getLogger(__name__)

# 默认系统提示词
DEFAULT_SYSTEM_PROMPT = "你是闪购平台的智能客服小智，请用柔和礼貌的语气与用户交流。"

PROMPT_FILE = BASE_DIR / "prompts" / "shangou_prompt.txt"


def load_system_prompt() -> str:
    """加载系统提示词"""
    try:
        if PROMPT_FILE.exists():
            return PROMPT_FILE.read_text(encoding="utf-8").strip()
    except Exception as e:
        logger.warning("⚠️ 读取系统提示词文件失败，使用默认提示词: %s", e)
    return DEFAULT_SYSTEM_PROMPT


def _llm_summarize(messages: list[BaseMessage]) -> str:
    """短期记忆压缩用的 LLM 摘要（压缩旧对话为要点，供 ChatMemory.compact_if_needed 注入）"""
    from app.llm import create_chat_model

    text = "\n".join(
        f"[{'用户' if m.type == 'human' else '助手'}] {m.content}" for m in messages
    )
    resp = create_chat_model().invoke([
        SystemMessage(content=(
            "把以下客服对话历史压缩成简洁摘要，必须保留：用户口味/商家/商品偏好、"
            "关键事实（订单、优惠券、地址）、未完成事项。300字以内，用要点列出。"
        )),
        HumanMessage(content=text),
    ])
    return resp.content or ""


class XiaozhiAgent:
    """Plan-and-Execute 智能体"""

    def __init__(self):
        self.system_prompt = load_system_prompt()
        self.graph = plan_execute_graph

    def _prepare_turn(self, user_id: int, memory_id: int, user_message: str) -> tuple[ChatMemory, dict]:
        """一轮对话的公共准备：记忆写入/压缩 + 组装图初始状态"""
        # 1. 获取/创建记忆并写入系统提示词（仅首条）与用户消息
        memory: ChatMemory = memory_provider.get(memory_id)
        if not any(m.type == "system" for m in memory.messages):
            memory.add_system(self.system_prompt)
        memory.add_user(user_message)

        # 2. 短期记忆压缩：接近窗口阈值时把旧对话摘要化（注入 LLM 摘要，失败自动回退拼接）
        memory.summarize_fn = _llm_summarize
        memory.compact_if_needed()

        # 3. 组装图初始状态
        state: dict = {
            "input": user_message,
            "chat_history": memory.as_history_text(),
            "memory_id": memory_id,
            "user_id": user_id,
            "past_steps": [],
            "replan_count": 0,
            "final_answer": "",
        }
        return memory, state

    def chat(self, user_id: int, memory_id: int, user_message: str) -> str:
        """执行一次完整对话循环"""
        logger.info("🚀 启动 Plan-and-Execute 循环，memoryId=%s, userId=%s", memory_id, user_id)
        memory, state = self._prepare_turn(user_id, memory_id, user_message)

        # 运行图（recursion_limit 覆盖：计划多步+重规划场景默认25不够）
        # 每轮用唯一线程承载工作记忆（checkpoint），结束后必须在 finally 清除，防止残留在 Redis。
        thread_id = make_thread_id(memory_id)
        try:
            final_state = self.graph.invoke(
                state,
                config={"recursion_limit": 80,
                        "configurable": {"thread_id": thread_id}},
            )
            final_answer = final_state.get("final_answer") or "（系统无输出）"
        except Exception as e:
            logger.error("💥 图执行顶层异常: %s", e)
            final_answer = f"系统内部错误: {e}"
        finally:
            clear_thread(thread_id)

        # 写入 AI 记忆
        memory.add_ai(final_answer)
        logger.info("🏁 循环结束，最终答案长度: %d 字符", len(final_answer))
        return final_answer

    async def chat_stream(self, user_id: int, memory_id: int, user_message: str):
        """流式对话：异步生成器，逐事件产出（plan/step/token/replan/done 等）。

        事件（dict，含 type 字段）：
          plan_start / plan_ready / replan   —— 规划进度
          step_start / step_end              —— 步骤执行进度
          token                              —— FINISH 答案的增量文本
          node_update                        —— 图节点切换
          error / done                       —— 终止事件
        """
        logger.info("🚀 启动流式 Plan-and-Execute 循环，memoryId=%s, userId=%s", memory_id, user_id)
        memory, state = self._prepare_turn(user_id, memory_id, user_message)

        # 每轮唯一线程承载工作记忆（checkpoint）；成功与异常都在 finally 清理，防止残留在 Redis
        thread_id = make_thread_id(memory_id)
        final_answer = ""
        try:
            final_state: dict = {}
            # 三种流模式：custom=图内 emit_event 推送的事件；updates=节点切换；values=每步后的完整状态
            async for mode, chunk in self.graph.astream(
                state,
                config={"recursion_limit": 80,
                        "configurable": {"thread_id": thread_id}},
                stream_mode=["custom", "updates", "values"],
            ):
                if mode == "custom":
                    yield chunk
                elif mode == "values":
                    final_state = chunk or {}
                elif mode == "updates":
                    for node in (chunk or {}):
                        yield {"type": "node_update", "node": node}
            final_answer = final_state.get("final_answer") or "（系统无输出）"
        except Exception as e:
            logger.error("💥 流式图执行顶层异常: %s", e)
            final_answer = f"系统内部错误: {e}"
            yield {"type": "error", "message": str(e)}
        finally:
            clear_thread(thread_id)

        # 与 chat() 一致：最终答案写入记忆后再发 done
        memory.add_ai(final_answer)
        logger.info("🏁 流式循环结束，最终答案长度: %d 字符", len(final_answer))
        yield {"type": "done", "final_answer": final_answer}


# 全局单例（全局单例）
xiaozhi_agent = XiaozhiAgent()

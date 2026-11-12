"""会话记忆：短期记忆 + 压缩

实现说明：
  - MongoDB/Redis 存储 → 进程内字典（生产可替换为 Redis/Mongo 实现，接口不变）
  - 短期记忆压缩：消息数接近窗口阈值时，把较早的对话压缩成一条摘要消息，
    而不是直接截断丢弃 —— 保证超阈值的历史信息以摘要形式保留。
"""
import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.config import get_settings

logger = logging.getLogger(__name__)

# 摘要函数签名：旧消息列表 -> 摘要文本（由外部注入，通常用 LLM 生成）
SummarizeFn = Callable[[list[BaseMessage]], str]

# 摘要回退实现：LLM 不可用时，把旧消息压缩为要点式文本（每条截断），保证压缩不丢关键信息
_FALLBACK_LINE_LIMIT = 120


def _fallback_summarize(messages: list[BaseMessage]) -> str:
    lines = []
    for m in messages:
        role = "用户" if isinstance(m, HumanMessage) else "助手" if isinstance(m, AIMessage) else "系统"
        content = str(m.content).strip()
        if len(content) > _FALLBACK_LINE_LIMIT:
            content = content[:_FALLBACK_LINE_LIMIT] + "…"
        lines.append(f"[{role}] {content}")
    return "\n".join(lines)


@dataclass
class ChatMemory:
    """单会话短期记忆

    压缩策略：
      消息数（不含开头系统提示词）达到 compact_trigger 时触发压缩：
      [系统提示词] + [旧对话→摘要消息] + [最近 keep_recent 条原样保留]
    """
    memory_id: int
    messages: list[BaseMessage] = field(default_factory=list)
    compact_trigger: int = 32          # 接近窗口阈值时触发压缩
    keep_recent: int = 12              # 压缩后保留最近 N 条原文
    summarize_fn: Optional[SummarizeFn] = None   # 摘要函数（None 时回退为要点拼接）
    compact_count: int = 0             # 已压缩次数（可观测）

    # ---- 写入 ----
    def add(self, message: BaseMessage) -> None:
        self.messages.append(message)

    def add_system(self, text: str) -> None:
        self.messages.append(SystemMessage(content=text))

    def add_user(self, text: str) -> None:
        self.messages.append(HumanMessage(content=text))

    def add_ai(self, text: str) -> None:
        self.messages.append(AIMessage(content=text))

    # ---- 压缩 ----
    def compact_if_needed(self) -> bool:
        """消息数接近阈值时压缩旧对话为摘要；返回是否执行了压缩"""
        sys_count = 0
        for m in self.messages:
            if isinstance(m, SystemMessage):
                sys_count += 1
            else:
                break
        conversation = self.messages[sys_count:]
        if len(conversation) < self.compact_trigger:
            return False

        old, recent = conversation[: -self.keep_recent], conversation[-self.keep_recent:]
        summary = self._summarize(old)
        self.messages = (
            self.messages[:sys_count]
            + [SystemMessage(content=f"【此前对话摘要】\n{summary}")]
            + recent
        )
        self.compact_count += 1
        logger.info("🗜️ 会话%s短期记忆压缩：旧%d条→摘要1条，保留最近%d条原文（第%d次压缩）",
                    self.memory_id, len(old), len(recent), self.compact_count)
        return True

    def _summarize(self, old: list[BaseMessage]) -> str:
        if not self.summarize_fn:
            return _fallback_summarize(old)
        try:
            return self.summarize_fn(old)
        except Exception as e:
            logger.warning("⚠️ LLM摘要失败，回退要点拼接: %s", e)
            return _fallback_summarize(old)

    # ---- 读取 ----
    def recent(self, limit: Optional[int] = None) -> list[BaseMessage]:
        limit = limit or get_settings().max_recent_messages
        return self.messages[-limit:]

    def as_history_text(self, limit: Optional[int] = None) -> str:
        """转为 [角色] 文本 摘要"""
        lines = []
        for m in self.recent(limit):
            role = "系统" if isinstance(m, SystemMessage) else "用户" if isinstance(m, HumanMessage) else "助手"
            lines.append(f"[{role}] {m.content}")
        return "\n".join(lines)


class ChatMemoryProvider:
    """记忆提供者"""

    def __init__(self):
        self._store: dict[int, ChatMemory] = {}

    def get(self, memory_id: int) -> ChatMemory:
        if memory_id not in self._store:
            s = get_settings()
            self._store[memory_id] = ChatMemory(
                memory_id=memory_id,
                compact_trigger=s.memory_compact_trigger,
                keep_recent=s.memory_keep_recent,
            )
        return self._store[memory_id]


# 全局单例（全局单例）
memory_provider = ChatMemoryProvider()

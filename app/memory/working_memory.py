"""工作记忆（graph state）的存储边界：Redis 持久化 + 每轮删除

设计定位：
  工作记忆 = 图中 plan / past_steps / last_result 等过渡态，生命周期=一轮图执行。
  经 LangGraph Checkpointer 落在 Redis，线程（thread_id）即为一个工作单元；
  agent 每轮图结束后调用 clear_thread 删除该线程，实现"每轮结束自动清工作记忆"。

降级策略：
  - settings.state_store == "memory"（默认）→ 不挂 checkpointer，仅使用单次调用的内存态。
  - state_store == "redis" 但 Redis 不可达 → 尝试后回退内存并告警，不影响启动与运行。
  - 结果全局缓存，避免每次调用都重连探测。
"""
from __future__ import annotations

import logging
import uuid
from typing import Optional

from app.config import get_settings

logger = logging.getLogger(__name__)

_SAVER: Optional[object] = None       # 缓存已解析的 RedisSaver（None=未决）
_SKIP: bool = False                   # True=已判定不可用/降级，不再尝试


def make_thread_id(memory_id: int) -> str:
    """生成一个唯一的线程 ID（工作单元 id）。每轮对话一个，保证互不污染。"""
    return f"wm-{memory_id}-{uuid.uuid4().hex}"


def get_checkpointer():
    """按配置返回工作记忆 checkpointer（RedisSaver），否则返回 None（内存模式）。

    返回 None 时：图不挂 checkpointer，past_steps 等只在内存随单次 invoke 演进。
    """
    global _SAVER, _SKIP
    if _SKIP:
        return None
    if _SAVER is not None:
        return _SAVER

    s = get_settings()
    if s.state_store != "redis":
        _SKIP = True
        return None

    try:
        from langgraph.checkpoint.redis import RedisSaver
        from redis import Redis

        client = Redis.from_url(s.redis_url, socket_connect_timeout=2, socket_timeout=2)
        if not client.ping():
            raise ConnectionError("Redis ping 失败")
        _SAVER = RedisSaver(redis_client=client)
        logger.info("🧠 工作记忆启用 Redis checkpoint: %s", s.redis_url)
        return _SAVER
    except Exception as e:  # noqa: BLE001 - 连不上则降级，不影响主流程
        _SKIP = True
        logger.warning("⚠️ Redis 工作记忆不可用（%s），降级为内存模式", e)
        return None


def clear_thread(thread_id: str) -> None:
    """单轮结束后清理该线程的工作记忆（delete_thread）。

    Redis 模式下删除 Redis 中该线程所有 checkpoint；内存模式（无 checkpointer）为空操作。
    调用在 finally 中，保证图成功或失败都会清理。
    """
    saver = get_checkpointer()
    if saver is None:
        return
    try:
        saver.delete_thread(thread_id)
        logger.debug("🗑️ 已清除工作记忆线程 %s", thread_id)
    except Exception as e:  # noqa: BLE001 - 清理失败只告警
        logger.warning("⚠️ 清除工作记忆线程 %s 失败: %s", thread_id, e)
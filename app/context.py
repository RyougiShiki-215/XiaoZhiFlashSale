"""请求上下文

使用 contextvars 保存请求级数据（对 async 友好）。
工具函数内通过 get_current_user_id() 获取当前请求的 userId。
"""
import contextvars
from typing import Optional

# 当前请求的 userId
_current_user_id: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar("current_user_id", default=None)

# memoryId -> userId 映射
_memory_id_to_user_id: dict[int, int] = {}


def set_current_user_id(user_id: Optional[int]) -> None:
    _current_user_id.set(user_id)


def get_current_user_id() -> Optional[int]:
    return _current_user_id.get()


def set_memory_id_to_user_id(memory_id: int, user_id: int) -> None:
    _memory_id_to_user_id[memory_id] = user_id


def get_user_id_by_memory_id(memory_id: int) -> Optional[int]:
    return _memory_id_to_user_id.get(memory_id)

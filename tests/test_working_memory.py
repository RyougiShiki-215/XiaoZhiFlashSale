"""工作记忆（Redis checkpoint / 每轮删除）封装测试

覆盖：
  - make_thread_id 唯一且携带 memory_id
  - 内存模式（无 checkpointer）下 clear_thread 为空操作，不抛异常
  - Redis 不可达时 get_checkpointer 降级为 None（不影响内存模式运行）

运行：python tests\test_working_memory.py
"""
import sys
import unittest.mock as mk
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app.memory.working_memory as wm
from app.memory.working_memory import clear_thread, make_thread_id


def test_make_thread_id_unique():
    a = make_thread_id(1)
    b = make_thread_id(1)
    assert a != b, "同一 memory 的不同轮应生成不同线程 id"
    assert a.startswith("wm-1-"), "线程 id 应携带 memory_id 前缀，便于归因"


def test_clear_thread_noop_without_checkpointer():
    """内存模式：get_checkpointer 返回 None，clear_thread 必须静默结束"""
    with mk.patch.object(wm, "get_checkpointer", return_value=None):
        clear_thread("wm-9-abc")  # 不应抛


def test_redis_unreachable_degrades():
    """state_store=redis 但 Redis 不可达：get_checkpointer 应返回 None（降级内存）
    用必然连不上的地址，保证结果与外部 Redis 是否运行无关。"""
    class _FakeSettings:
        state_store = "redis"
        redis_url = "redis://127.0.0.1:6399/0"   # 无效端口，必失败

    wm._SAVER = None
    wm._SKIP = False
    with mk.patch.object(wm, "get_settings", return_value=_FakeSettings()):
        cp = wm.get_checkpointer()
    assert cp is None, "Redis 不可达时应降级返回 None"


def test_clear_thread_calls_delete_when_checkpointer_present():
    """有 checkpointer 时，clear_thread 应调用 delete_thread(thread_id)"""
    class _FakeSaver:
        def __init__(self): self.deleted = []
        def delete_thread(self, tid): self.deleted.append(tid)

    saver = _FakeSaver()
    with mk.patch.object(wm, "get_checkpointer", return_value=saver):
        clear_thread("wm-3-x")
    assert saver.deleted == ["wm-3-x"]


if __name__ == "__main__":
    test_make_thread_id_unique()
    test_clear_thread_noop_without_checkpointer()
    test_redis_unreachable_degrades()
    test_clear_thread_calls_delete_when_checkpointer_present()
    print("全部通过")
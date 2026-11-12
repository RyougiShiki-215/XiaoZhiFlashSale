"""ToolRunner 抽象：统一工具执行的 transport 边界

设计目标：让异常处置策略（重试/参数修复/熔断/重规划/ToolExecutionAbort）
与"工具从哪里来"解耦。策略层只消费 RawOutcome（含已分类的 ErrorKind），
而分类动作发生在异常诞生的那侧——

  LocalToolRunner（进程内）：在本地 catch 活异常 → classify_exception
  McpToolRunner   （MCP）：  service 端(见 mcp_server._call_tool)已分类，
                             客户端直接读取外壳中的 kind，无需活异常

这样 execute_tool_with_policy 可以同时服务两种 transport，处置逻辑零改动。
"""
from __future__ import annotations

import asyncio
import json
import logging
import shlex
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol

# 项目根目录（runner.py ← tools ← app ← project_root）
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])

from app.config import get_settings
from app.tools.errors import ErrorKind

logger = logging.getLogger(__name__)


@dataclass
class RawOutcome:
    """一次工具调用的原始结果（已携带分类信息）"""
    ok: bool
    kind: Optional[ErrorKind] = None
    result: Optional[str] = None
    error: Optional[str] = None


def _to_kind(raw: Any) -> Optional[ErrorKind]:
    if not raw:
        return None
    try:
        return ErrorKind(raw)
    except ValueError:
        return None


class ToolRunner(Protocol):
    """工具执行者的最小契约（transport 无关）"""

    def tool_names(self) -> list[str]: ...

    def tool_schema(self, name: str) -> str: ...

    def invoke(self, name: str, args: dict) -> RawOutcome: ...

    def close(self) -> None: ...


# ============ LocalToolRunner：进程内 BaseTool ============

class LocalToolRunner:
    """进程内执行：作用在 tool_map 的 BaseTool 上，catch 活异常后分类"""

    def __init__(self, tool_map: dict):
        self._map = tool_map

    def tool_names(self) -> list[str]:
        return list(self._map)

    def tool_schema(self, name: str) -> str:
        t = self._map.get(name)
        return json.dumps(t.args if t else {}, ensure_ascii=False) if t else ""

    def invoke(self, name: str, args: dict) -> RawOutcome:
        from app.tools.errors import classify_business_text, classify_exception

        tool = self._map.get(name)
        if tool is None:
            return RawOutcome(False, ErrorKind.TOOL_NOT_FOUND,
                              error=f"未知工具[{name}]")
        try:
            text = str(tool.invoke(args or {}))
        except Exception as e:  # noqa: BLE001 - ToolError 由 classify_exception 兜底
            return RawOutcome(False, classify_exception(e),
                              error=f"{type(e).__name__}: {e}")
        biz = classify_business_text(text)
        if biz is not None:
            return RawOutcome(False, biz, result=text,
                              error=f"工具返回业务失败({biz.value})")
        return RawOutcome(True, result=text)

    def close(self) -> None:
        pass


# ============ McpToolRunner：自研薄 MCP client（stdio） ============

class McpToolRunner:
    """经 MCP client 消费工具：读取 service 端结构化外壳中的 kind。

    自研而未用 langchain-mcp-adapters.load_mcp_tools 的原因：后者把工具结果
    压平为文本，丢掉了错误分类 kind；本 runner 直接解析外壳凭 kind 决策，
    从而与 LocalToolRunner 走同一套策略逻辑。

    同步 invoke 通过后台线程里的独立事件循环执行——可在 async 图节点内安全调用，
    不带环冲突。
    """

    def __init__(self, command: str = None, spawn_args: str = None, cwd: str = None):
        s = get_settings()
        self._command = command or s.mcp_server_command
        self._spawn_args = shlex.split(spawn_args if spawn_args is not None else s.mcp_server_args)
        self._cwd = cwd or _PROJECT_ROOT
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._session = None
        self._tools: dict[str, Any] = {}
        self._error: Optional[str] = None
        self._task: Optional[asyncio.Task] = None
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    # ---- 后台事件循环 ----
    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._task = self._loop.create_task(self._serve())
        try:
            self._loop.run_until_complete(self._task)
        except asyncio.CancelledError:
            pass  # close() 主动取消，正常退出
        except Exception as e:  # noqa: BLE001 - 记录连接失败，避免被吞
            self._error = f"{type(e).__name__}: {e}"
            logger.error("🔌 McpToolRunner 连接失败: %s", self._error)
            self._ready.set()

    async def _serve(self) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        server_params = StdioServerParameters(
            command=self._command, args=self._spawn_args, cwd=self._cwd)
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                listed = await session.list_tools()
                self._tools = {t.name: t for t in listed.tools}
                self._session = session
                self._ready.set()
                logger.info("🔌 McpToolRunner 已连接，工具数=%d", len(self._tools))
                while True:  # 保持会话
                    await asyncio.sleep(3600)

    def _wait_ready(self) -> None:
        if not self._ready.wait(timeout=10):
            raise RuntimeError("MCP 会话未就绪（超时）")
        if self._error:
            raise RuntimeError(f"MCP 连接失败: {self._error}")

    # ---- 同步接口 ----
    def tool_names(self) -> list[str]:
        self._wait_ready()
        return list(self._tools)

    def tool_schema(self, name: str) -> str:
        t = self._tools.get(name)
        return json.dumps((t.inputSchema or {}) if t else {}, ensure_ascii=False) if t else ""

    def invoke(self, name: str, args: dict) -> RawOutcome:
        self._wait_ready()
        if name not in self._tools:
            return RawOutcome(False, ErrorKind.TOOL_NOT_FOUND, error=f"未知工具[{name}]")
        fut = asyncio.run_coroutine_threadsafe(self._call(name, args or {}), self._loop)
        return fut.result(timeout=30)

    async def _call(self, name: str, args: dict) -> RawOutcome:
        res = await self._session.call_tool(name, args)
        text = ""
        if res.content:
            first = res.content[0]
            text = getattr(first, "text", "") or ""
        try:
            env = json.loads(text)
            return RawOutcome(
                ok=bool(env.get("ok")),
                kind=_to_kind(env.get("kind")),
                result=env.get("result"),
                error=env.get("error"),
            )
        except (ValueError, TypeError) as e:
            return RawOutcome(False, None, error=f"MCP 结果解析失败: {e}")

    def close(self) -> None:
        if self._task is not None:
            self._loop.call_soon_threadsafe(self._task.cancel)


# ============ 工厂 ============

_runner: Optional[ToolRunner] = None


def get_tool_runner() -> ToolRunner:
    """按 settings.tools_transport 返回全局运行器（懒加载单例）

    local → 进程内 BaseTool；mcp → 自研薄 MCP client（默认）
    """
    global _runner
    if _runner is not None:
        return _runner
    s = get_settings()
    if s.tools_transport == "local":
        from app.tools.registry import TOOL_MAP

        _runner = LocalToolRunner(TOOL_MAP)
    else:
        _runner = McpToolRunner()
    return _runner
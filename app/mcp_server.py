"""MCP Server：把现有业务工具暴露为标准 MCP 工具

设计原则：只做协议封装，不动任何工具内部逻辑与 agent 的错误治理。
  - 工具来源仍是 app.tools.registry.ALL_TOOLS（LangChain BaseTool 实例），
    通过 fastmcp 动态注册，保留每个工具的原始参数 JSON Schema 与中文描述。
  - agent 内部调用路径（ReAct 循环 + execute_tool_with_policy 的
    参数修复/静默重试/熔断/ToolExecutionAbort）完全不变，仍作用于本地 BaseTool；
    本 Server 供外部 MCP 客户端 / 未来多消费者调用同一批工具。

运行方式：
  python -m app.mcp_server --transport stdio    # 标准输入输出
  python -m app.mcp_server --transport http     # Streamable HTTP，默认 0.0.0.0:9100
"""
import argparse
import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

from app.skill.registry import skill_registry
from app.tools.registry import ALL_TOOLS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger(__name__)

# JSON Schema type → Python type hint（用于构建 fastmcp 可见的函数签名）
_TYPE_MAP: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _call_tool(tool, args: dict[str, Any]) -> Any:
    """执行工具并返回结构化外壳（服务端完成异常分类）

    外壳 {ok, kind, result, error}：
      - 客户端（自研薄 executor）据此做静默重试/参数修复/熔断/重规划决策，
        把 ValueError 语义无损地传过 MCP 进程边界
      - kind 为 ErrorKind 枚举值；ok=true 但文本含业务失败关键词时同样标记失败
    异常分类复用同包 errors.py 的 classify_*（分类逻辑仍在异常诞生的服务端）。
    """
    from app.tools.errors import classify_business_text, classify_exception

    try:
        text = str(tool.invoke(args))
    except Exception as e:  # noqa: BLE001 - ToolError 由 classify_exception 兜底识别
        kind = classify_exception(e)
        return {"ok": False, "kind": kind.value, "result": None,
                "error": f"{type(e).__name__}: {e}"}
    biz = classify_business_text(text)
    if biz is not None:
        return {"ok": False, "kind": biz.value, "result": text,
                "error": f"工具返回业务失败({biz.value}): {text if len(text) <= 200 else text[:200] + '...'}"}
    return {"ok": True, "kind": None, "result": text, "error": None}


def _build_wrapper(tool):
    """为单个 BaseTool 动态构造一个带精确参数签名的包装函数，供 fastmcp 注册。

    依据工具自带 JSON Schema 生成具名形参（必填/可选、类型），
    使 MCP 客户端看到的 inputSchema 与 LangChain schema 字段一一对应。
    _tool/_call_tool 通过 exec 的 globals 注入（不进函数签名），对 MCP 不可见。
    """
    schema = tool.args or {}
    # LangChain BaseTool.args 对简单 @tool 直接返回字段映射（无 properties 包装层）；
    # 结构化工具则返回 {"type":"object","properties":{...}}。两种形态都要兼容。
    props: dict = (schema.get("properties", {}) if "properties" in schema else schema)
    required: list = schema.get("required", []) or []

    param_defs: list[str] = []
    for pname, pmeta in props.items():
        pytype = _TYPE_MAP.get(pmeta.get("type"), str)
        ann = getattr(pytype, "__name__", "str")
        param_defs.append(f"{pname}: {ann}" if pname in required else f"{pname}: {ann} = None")

    src = "def _wrapped(" + ", ".join(param_defs) + "):\n"
    src += "    _args = {}\n"
    for pname in props:
        if pname in required:
            src += f"    _args[{pname!r}] = {pname}\n"
        else:
            src += f"    if {pname} is not None: _args[{pname!r}] = {pname}\n"
    src += "    return _call_tool(_tool, _args)"

    _globals = {"_tool": tool, "_call_tool": _call_tool}
    exec(src, _globals)
    fn = _globals["_wrapped"]
    fn.__name__ = tool.name
    fn.__doc__ = tool.description
    return fn


def build_mcp_server() -> FastMCP:
    """构建并注册全部业务工具的 FastMCP Server"""
    # skill_registry 在 MCP 子进程是独立的空实例，须先把它加载出来，
    # 否则 SKILL 步骤经 MCP 调 read_skill_resource 时永远报 "Skill 不存在"。
    skill_registry.load_all()
    server = FastMCP("xiaozhi-tools")
    for tool in ALL_TOOLS:
        fn = _build_wrapper(tool)
        server.add_tool(fn, name=tool.name, description=tool.description or tool.name)
        logger.info("已注册 MCP 工具: %s", tool.name)
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="闪购小智 MCP Server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio",
                        help="传输方式：stdio（默认）或 http（Streamable HTTP）")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9100)
    args = parser.parse_args()

    server = build_mcp_server()
    logger.info("🚀 MCP Server 启动，传输=%s，工具数=%d", args.transport, len(ALL_TOOLS))

    if args.transport == "http":
        server.run(transport="http", host=args.host, port=args.port)
    else:
        # stdio 模式：MCP 客户端（如 Claude/Cursor）通过子进程与本进程经标准输入输出通信
        server.run(transport="stdio")


if __name__ == "__main__":
    main()
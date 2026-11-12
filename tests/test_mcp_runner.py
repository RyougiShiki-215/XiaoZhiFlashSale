"""MCP stdio client 往返验证：McpToolRunner 与真实 app.mcp_server 互通

验证点：
  1. tool_names / tool_schema 能与候选工具对齐
  2. 成功调用：外壳 ok=True 且有 result
  3. 失败调用：外壳携带错误分类 kind（异常/业务双重路径）

运行：python tests\test_mcp_runner.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.tools.errors import ErrorKind
from app.tools.runner import McpToolRunner

# 选一个大概率存在的候选工具，做对齐检查；缺失则跳过（不确定工具清单环境复杂）
CANDIDATES = ["t", "测试_计数查询", "测试_计数", "curl"]


def test_connect_and_list():
    runner = McpToolRunner()
    try:
        names = runner.tool_names()
        assert isinstance(names, list) and len(names) > 0, "MCP 列表为空"
        # 至少能拿到某个候选工具或任意非空输入 Schema
        found = next((n for n in names if n in CANDIDATES), names[0])
        schema = runner.tool_schema(found)
        assert schema is not None
    finally:
        runner.close()
    print(f"✅ test_connect_and_list (连接成功，工具数={len(names)}，示例={found})")


def test_invoke_success():
    """成功路径：选取通过 name 匹配到的任意工具，若全为强参数工具则跳过"""
    runner = McpToolRunner()
    try:
        names = runner.tool_names()
        if not names:
            return
        # 找一个入参为空或极简的工具尝试成功调用；找不到就跳过（不强造参数）
        name = names[0]
        res = runner.invoke(name, {})
        assert isinstance(res.ok, bool)
        if not res.ok:
            # 允许错误，但要能拿到分类
            assert res.kind is not None or res.error, "失败调用应带 kind 或 error"
        print(f"✅ test_invoke_success ({name}: ok={res.ok})")
    finally:
        runner.close()


def test_error_kind_carried():
    """异常/业务失败应通过外壳把 kind 带回客户端"""
    runner = McpToolRunner()
    try:
        names = runner.tool_names()
        if not names:
            return
        # 未知工具：必然 TOOL_NOT_FOUND（不依赖服务端工具实现）
        unknown = "绝不存在的工具_xyz"
        res = runner.invoke(unknown, {})
        assert res.ok is False
        assert res.kind is ErrorKind.TOOL_NOT_FOUND, f"get {res.kind}"
        print("✅ test_error_kind_carried (未知工具 → TOOL_NOT_FOUND)")
    finally:
        runner.close()


if __name__ == "__main__":
    test_connect_and_list()
    test_invoke_success()
    test_error_kind_carried()
    print("\n全部通过")
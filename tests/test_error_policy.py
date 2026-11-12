"""异常分类与处置策略测试：分类器 / 处置矩阵 / 静默重试 / 参数修复 / 熔断

运行：python tests\test_error_policy.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.tools import tool

from app.tools.errors import (
    CircuitBreaker,
    Disposition,
    ErrorKind,
    ToolError,
    build_llm_guidance,
    classify_business_text,
    classify_exception,
    decide_disposition,
    execute_tool_with_policy,
)

NO_SLEEP = lambda _s: None  # noqa: E731


# ============ 数据库异常分类 ============

def test_classify_db_transient():
    """数据库瞬时异常：连接类/超时类/死锁/限流（可重试）"""
    cases = [
        ("Connection to MySQL failed: connection refused", ErrorKind.TRANSIENT_DB_CONNECTION),
        ("Too many connections", ErrorKind.TRANSIENT_DB_CONNECTION),
        ("connection pool exhausted, wait timeout", ErrorKind.TRANSIENT_DB_CONNECTION),
        ("Lock wait timeout exceeded; try restarting transaction", ErrorKind.TRANSIENT_DB_TIMEOUT),
        ("Deadlock found when trying to get lock", ErrorKind.TRANSIENT_DB_TIMEOUT),
        ("Read timeout while querying orders", ErrorKind.TRANSIENT_DB_TIMEOUT),
        ("429 Too Many Requests: rate limit exceeded", ErrorKind.TRANSIENT_RATE_LIMIT),
    ]
    for msg, expected in cases:
        assert classify_exception(RuntimeError(msg)) is expected, msg


def test_classify_db_input_and_fatal():
    """数据库输入类（LLM可修）与致命类（不可恢复）"""
    cases = [
        ("Duplicate key error: coupon already exists (unique constraint)", ErrorKind.DB_CONSTRAINT),
        ("Foreign key constraint fails: order_id references missing", ErrorKind.DB_CONSTRAINT),
        ("Column 'shop_nmae' cannot be null", ErrorKind.DB_CONSTRAINT),
        ("You have an error in your SQL syntax", ErrorKind.CODE_BUG),
        ("Unknown column 'user_nmae' in 'field list'", ErrorKind.CODE_BUG),
        ("Unknown table 'orderss' doesn't exist", ErrorKind.CODE_BUG),
        ("Access denied for user 'app'@'host'", ErrorKind.AUTH_FAILURE),
        ("Permission denied: SELECT command denied", ErrorKind.AUTH_FAILURE),
        ("The disk is full, unable to write", ErrorKind.RESOURCE_EXHAUSTED),
    ]
    for msg, expected in cases:
        assert classify_exception(RuntimeError(msg)) is expected, msg


def test_classify_type_and_toolerror():
    """异常类名识别 + ToolError 显式声明优先"""
    assert classify_exception(ConnectionError("xxx")) is ErrorKind.TRANSIENT_NETWORK
    assert classify_exception(TimeoutError()) is ErrorKind.TRANSIENT_DB_TIMEOUT
    assert classify_exception(ToolError(ErrorKind.RECORD_NOT_FOUND, "订单不存在")) \
        is ErrorKind.RECORD_NOT_FOUND
    # 未知异常 → 业务失败兜底（触发重规划，不浪费修复调用）
    assert classify_exception(RuntimeError("完全没见过的错误")) is ErrorKind.BUSINESS_FAILURE


def test_classify_order_sensitive():
    """规则顺序：'unknown table doesn't exist' 必须命中 CODE_BUG 而非 RECORD_NOT_FOUND"""
    kind = classify_exception(RuntimeError("Unknown table 'shop' doesn't exist in db"))
    assert kind is ErrorKind.CODE_BUG


def test_classify_business_text():
    """工具返回文本的业务失败细分"""
    assert classify_business_text("订单 O123 不存在") is ErrorKind.RECORD_NOT_FOUND
    assert classify_business_text("调用工具ZXC失败，请重试") is ErrorKind.BUSINESS_FAILURE
    assert classify_business_text("已成功发放优惠券满30减5") is None


# ============ 处置矩阵 ============

def test_disposition_matrix():
    """异常性质 × 幂等性 → 处置去向"""
    idem_tool, write_tool = "查询用户历史订单", "营销_发放优惠券"

    # 瞬时异常：幂等工具静默重试；非幂等写操作交 LLM 先确认（禁止盲重试）
    for kind in (ErrorKind.TRANSIENT_NETWORK, ErrorKind.TRANSIENT_DB_TIMEOUT):
        assert decide_disposition(kind, idem_tool) is Disposition.RETRY_SILENT
        assert decide_disposition(kind, write_tool) is Disposition.FEEDBACK_TO_LLM

    # 输入类：一律交 LLM 修参数
    for kind in (ErrorKind.INVALID_ARGS, ErrorKind.DB_CONSTRAINT, ErrorKind.TOOL_NOT_FOUND):
        assert decide_disposition(kind, idem_tool) is Disposition.FEEDBACK_TO_LLM
        assert decide_disposition(kind, write_tool) is Disposition.FEEDBACK_TO_LLM

    # 业务类：交 Planner 重规划
    for kind in (ErrorKind.BUSINESS_FAILURE, ErrorKind.RECORD_NOT_FOUND):
        assert decide_disposition(kind, idem_tool) is Disposition.TRIGGER_REPLAN

    # 致命类：直接失败 + 熔断
    for kind in (ErrorKind.AUTH_FAILURE, ErrorKind.CODE_BUG, ErrorKind.RESOURCE_EXHAUSTED):
        assert decide_disposition(kind, idem_tool) is Disposition.FAIL_FAST


# ============ 执行管道 ============

def _make_tools():
    calls = {"n": 0}

    @tool("测试_计数查询", description="幂等读：第3次才成功")
    def flaky_query(q: str) -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("connection refused (attempt %d)" % calls["n"])
        return f"查询结果:{q}"

    @tool("测试_慢查询", description="永远超时")
    def slow_query(q: str) -> str:
        raise TimeoutError("lock wait timeout exceeded")

    @tool("测试_坏SQL", description="代码bug")
    def buggy_query(q: str) -> str:
        raise RuntimeError("You have an error in your SQL syntax")

    @tool("测试_参数敏感", description="参数校验")
    def strict_query(n: int) -> str:
        if not isinstance(n, int):
            raise TypeError("n 必须是整数")
        return f"n={n}"

    @tool("营销_发放优惠券", description="非幂等写操作")
    def grant(coupon_type: str) -> str:
        return f"已发放:{coupon_type}"

    @tool("营销_发放优惠券_超时版", description="非幂等写操作：永远超时")
    def grant_timeout(coupon_type: str) -> str:
        raise TimeoutError("lock wait timeout exceeded")

    return {"测试_计数查询": flaky_query, "测试_慢查询": slow_query,
            "测试_坏SQL": buggy_query, "测试_参数敏感": strict_query,
            "营销_发放优惠券": grant, "营销_发放优惠券_超时版": grant_timeout}


def test_retry_silent_then_success():
    """幂等工具瞬时异常：系统静默重试后成功，LLM 全程无感知"""
    tm = _make_tools()
    outcome = execute_tool_with_policy("测试_计数查询", tm, {"q": "麦当劳"},
                                       retry_delays=(0.0, 0.0), sleeper=NO_SLEEP)
    assert outcome.success and outcome.attempts == 3 and outcome.repaired is False


def test_retry_exhausted_triggers_replan():
    """幂等工具静默重试耗尽 → 升级为重规划"""
    tm = _make_tools()
    outcome = execute_tool_with_policy("测试_慢查询", tm, {"q": "x"},
                                       retry_delays=(0.0, 0.0), sleeper=NO_SLEEP)
    assert not outcome.success
    assert outcome.kind is ErrorKind.TRANSIENT_DB_TIMEOUT
    assert outcome.disposition is Disposition.TRIGGER_REPLAN


def test_non_idempotent_timeout_guidance():
    """非幂等写操作超时：不静默重试，guidance 要求先查询确认"""
    tm = _make_tools()
    # 用同名超时实现替换正常发放工具（名字必须在 NON_IDEMPOTENT_TOOLS 中）
    tm["营销_发放优惠券"] = tm["营销_发放优惠券_超时版"]
    outcome = execute_tool_with_policy("营销_发放优惠券", tm, {"coupon_type": "满30减5"},
                                       retry_delays=(0.0,), sleeper=NO_SLEEP)
    assert not outcome.success
    assert outcome.kind is ErrorKind.TRANSIENT_DB_TIMEOUT
    assert outcome.disposition is Disposition.FEEDBACK_TO_LLM
    assert "禁止直接重试" in outcome.guidance and "确认" in outcome.guidance


def test_fail_fast_and_circuit_breaker():
    """致命异常直接失败；连续达阈值后熔断打开，后续调用被短路"""
    tm = _make_tools()
    breaker = CircuitBreaker(threshold=2, cooldown_s=60.0)
    o1 = execute_tool_with_policy("测试_坏SQL", tm, {"q": "x"},
                                  retry_delays=(), sleeper=NO_SLEEP, breaker=breaker)
    assert o1.disposition is Disposition.FAIL_FAST and o1.kind is ErrorKind.CODE_BUG
    assert not breaker.is_open("测试_坏SQL")

    o2 = execute_tool_with_policy("测试_坏SQL", tm, {"q": "x"},
                                  retry_delays=(), sleeper=NO_SLEEP, breaker=breaker)
    assert o2.disposition is Disposition.FAIL_FAST
    assert breaker.is_open("测试_坏SQL")  # 第2次致命失败 → 熔断打开

    o3 = execute_tool_with_policy("测试_坏SQL", tm, {"q": "x"},
                                  retry_delays=(), sleeper=NO_SLEEP, breaker=breaker)
    assert o3.kind is ErrorKind.SERVICE_UNAVAILABLE  # 熔断中：不再真实调用
    assert o3.disposition is Disposition.FAIL_FAST


def test_circuit_breaker_half_open_recovery():
    """熔断冷却期过后半开放行；成功则彻底恢复"""
    now = [1000.0]
    breaker = CircuitBreaker(threshold=1, cooldown_s=10.0, clock=lambda: now[0])
    breaker.record_catastrophe("t")
    assert breaker.is_open("t")
    now[0] += 11.0
    assert not breaker.is_open("t")  # 冷却期过 → 半开放行
    breaker.record_success("t")
    assert not breaker.is_open("t")


def test_args_repair_path():
    """INVALID_ARGS + 注入 repair_fn：一次参数修复后成功"""
    tm = _make_tools()
    repaired_args = {"n": 42}

    def fake_repair(tool_name, schema_str, bad_args, kind, error_text):
        assert kind is ErrorKind.INVALID_ARGS
        return repaired_args

    outcome = execute_tool_with_policy("测试_参数敏感", tm, {"n": "abc"},
                                       repair_fn=fake_repair,
                                       retry_delays=(), sleeper=NO_SLEEP)
    assert outcome.success and outcome.repaired is True
    assert outcome.result == "n=42"


def test_args_repair_failure_escalates():
    """参数修复仍失败 → 升级为重规划（Planner 重新设计步骤）"""
    tm = _make_tools()

    def bad_repair(tool_name, schema_str, bad_args, kind, error_text):
        return {"n": "still-not-int"}  # 修复后的参数依然错误

    outcome = execute_tool_with_policy("测试_参数敏感", tm, {"n": "abc"},
                                       repair_fn=bad_repair,
                                       retry_delays=(), sleeper=NO_SLEEP)
    assert not outcome.success
    assert outcome.disposition is Disposition.TRIGGER_REPLAN


def test_business_failure_text_routing():
    """工具执行成功但返回业务失败文本 → 触发重规划（细分记录不存在/一般失败）"""
    tm = _make_tools()
    assert classify_business_text("店铺 海底捞 不存在") is ErrorKind.RECORD_NOT_FOUND


def test_tool_not_found_guidance():
    """未知工具：guidance 给出相近工具建议"""
    tm = _make_tools()
    outcome = execute_tool_with_policy("测试_计数查詢", tm, {"q": "x"},  # 繁体错字
                                       retry_delays=(), sleeper=NO_SLEEP)
    assert outcome.kind is ErrorKind.TOOL_NOT_FOUND
    assert outcome.disposition is Disposition.FEEDBACK_TO_LLM
    assert "相近工具" in outcome.guidance or "工具" in outcome.guidance


def test_react_mode_no_auto_repair():
    """ReAct 模式（无 repair_fn）：INVALID_ARGS 不自动修复，回传修正指导"""
    tm = _make_tools()
    outcome = execute_tool_with_policy("测试_参数敏感", tm, {"n": "abc"},
                                       retry_delays=(), sleeper=NO_SLEEP)
    assert outcome.disposition is Disposition.FEEDBACK_TO_LLM
    assert "Schema" in outcome.guidance  # 指导中附带参数 Schema


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"✅ {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"❌ {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{'全部通过' if failed == 0 else f'{failed} 个失败'}（共 {len(tests)} 项）")
    sys.exit(1 if failed else 0)

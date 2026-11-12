"""Tool Calling 异常分类与处置策略（自研异常处理体系）

设计参考 JDBC 4.0 的 SQLTransientException / SQLNonTransientException 分类思想：
按"谁能修复异常"把错误分流到不同处理层，最小化交给 LLM 的异常量：

  异常性质                     处置（幂等工具 / 非幂等写操作）
  ─────────────────────────────────────────────────────────────────
  瞬时：网络/限流/DB连接/超时    系统静默重试 / 交LLM先确认（结果未知禁止盲重试）
  输入：参数错误/约束冲突        交LLM修参数（ReAct在环自纠 或 一次参数修复调用）
  业务：业务失败/记录不存在      交Planner重规划（改参数无用，需换路径）
  致命：认证/代码bug/资源耗尽    直接失败+熔断（重试与修复均无意义）

同一异常在不同执行模式下处置不同：
  - ReAct 循环内（LLM 在环）：FEEDBACK 类异常作为 ToolMessage 回传，LLM 下一轮自纠
  - Planner 指定的 TOOL 步骤（离线执行）：FEEDBACK 类先做一次 LLM 参数修复，失败再重规划
"""
from __future__ import annotations

import difflib
import json
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class ErrorKind(str, Enum):
    """异常性质分类（数据库异常参考 JDBC 异常层次细分）"""
    # ---- 瞬时：稍后重试可能成功 ----
    TRANSIENT_NETWORK = "TRANSIENT_NETWORK"              # 网络中断/连接重置
    TRANSIENT_RATE_LIMIT = "TRANSIENT_RATE_LIMIT"        # 限流/配额耗尽
    TRANSIENT_DB_CONNECTION = "TRANSIENT_DB_CONNECTION"  # DB连接失败/连接池耗尽
    TRANSIENT_DB_TIMEOUT = "TRANSIENT_DB_TIMEOUT"        # 查询超时/锁等待/死锁
    # ---- 输入：LLM 修正参数即可恢复 ----
    TOOL_NOT_FOUND = "TOOL_NOT_FOUND"                    # 工具名不存在/拼写错误
    INVALID_ARGS = "INVALID_ARGS"                        # 参数缺失/类型不匹配
    DB_CONSTRAINT = "DB_CONSTRAINT"                      # 唯一键/外键/非空约束冲突
    # ---- 业务：改参数没用，需要换路径 ----
    BUSINESS_FAILURE = "BUSINESS_FAILURE"                # 工具返回业务失败
    RECORD_NOT_FOUND = "RECORD_NOT_FOUND"                # 目标记录不存在
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"          # 下游长期不可用/熔断打开
    # ---- 致命：重试与修复均无意义 ----
    AUTH_FAILURE = "AUTH_FAILURE"                        # 认证/授权失败
    CODE_BUG = "CODE_BUG"                                # SQL语法/字段映射错误（开发者bug）
    RESOURCE_EXHAUSTED = "RESOURCE_EXHAUSTED"            # 磁盘满/内存不足


class Disposition(str, Enum):
    """处置策略：异常交给哪一层处理"""
    RETRY_SILENT = "RETRY_SILENT"          # 系统静默重试（LLM 无感知，零 token 开销）
    FEEDBACK_TO_LLM = "FEEDBACK_TO_LLM"    # 回传 LLM 自纠（修参数 / 先确认再行动）
    TRIGGER_REPLAN = "TRIGGER_REPLAN"      # 触发全局重规划（Planner 换路径）
    FAIL_FAST = "FAIL_FAST"                # 直接失败（+熔断计数，避免拖垮整条链路）


class ToolError(Exception):
    """工具主动抛出的带分类异常（业务工具可用它精确声明失败性质）"""

    def __init__(self, kind: ErrorKind, message: str = ""):
        super().__init__(message or kind.value)
        self.kind = kind


# ============ 分类规则 ============

# 异常类名 → 分类（langchain/pydantic 常见异常 + 标准库网络异常）
_TYPE_RULES: dict[str, ErrorKind] = {
    "ConnectionError": ErrorKind.TRANSIENT_NETWORK,
    "TimeoutError": ErrorKind.TRANSIENT_DB_TIMEOUT,
    "socket.timeout": ErrorKind.TRANSIENT_DB_TIMEOUT,
    "RateLimitError": ErrorKind.TRANSIENT_RATE_LIMIT,
    "AuthenticationError": ErrorKind.AUTH_FAILURE,
    "PermissionError": ErrorKind.AUTH_FAILURE,
    "ValidationError": ErrorKind.INVALID_ARGS,  # pydantic 参数校验失败
    "TypeError": ErrorKind.INVALID_ARGS,
    "KeyError": ErrorKind.INVALID_ARGS,
    "ValueError": ErrorKind.INVALID_ARGS,
}

# 错误消息关键词 → 分类（顺序敏感：先匹配具体的，再匹配一般的）
# 例："Unknown table 'orders' (doesn't exist)" 必须先命中 CODE_BUG，
#      而不能落到后面的 RECORD_NOT_FOUND("not found") 规则
_MSG_RULES: list[tuple[ErrorKind, tuple[str, ...]]] = [
    (ErrorKind.CODE_BUG, ("syntax error", "sql syntax", "unknown column", "unknown table",
                          "undefined column", "字段不存在", "表不存在")),
    (ErrorKind.AUTH_FAILURE, ("access denied", "permission denied", "authentication failed",
                              "unauthorized", "forbidden", "403", "401")),
    (ErrorKind.RESOURCE_EXHAUSTED, ("disk full", "disk is full", "out of memory",
                                    "resource exhausted", "磁盘满", "内存不足")),
    (ErrorKind.TRANSIENT_RATE_LIMIT, ("rate limit", "too many requests", "429", "quota exceeded", "限流")),
    (ErrorKind.TRANSIENT_DB_CONNECTION, ("connection refused", "connection reset", "connection closed",
                                         "too many connections", "connection pool", "无法连接", "连接池耗尽")),
    (ErrorKind.TRANSIENT_DB_TIMEOUT, ("deadlock", "lock wait timeout", "死锁", "锁等待超时")),
    (ErrorKind.DB_CONSTRAINT, ("duplicate key", "unique constraint", "already exists", "foreign key",
                               "integrity constraint", "cannot be null", "唯一约束", "重复发放", "已存在")),
    (ErrorKind.RECORD_NOT_FOUND, ("not found", "no rows", "no such record", "record missing",
                                  "不存在", "无号")),
    (ErrorKind.TRANSIENT_DB_TIMEOUT, ("timeout", "timed out", "超时")),
    (ErrorKind.TRANSIENT_NETWORK, ("connection", "network", "unreachable", "refused", "reset",
                                   "broken pipe", "网络异常")),
]


def classify_exception(exc: Exception) -> ErrorKind:
    """三层识别：ToolError 显式声明 > 异常类名 > 错误消息关键词"""
    if isinstance(exc, ToolError):
        return exc.kind
    for base in type(exc).__mro__:
        if base.__name__ in _TYPE_RULES:
            return _TYPE_RULES[base.__name__]
    msg = str(exc).lower()
    for kind, keywords in _MSG_RULES:
        if any(kw in msg for kw in keywords):
            return kind
    # 未知异常：按业务失败处理（触发重规划），采用统一兜底行为
    return ErrorKind.BUSINESS_FAILURE


# 工具返回文本中的业务失败关键词（扩展细分）
_BIZ_TEXT_RULES: list[tuple[ErrorKind, tuple[str, ...]]] = [
    (ErrorKind.RECORD_NOT_FOUND, ("不存在", "无号", "未找到")),
    (ErrorKind.BUSINESS_FAILURE, ("失败", "错误", "异常", "暂不支持", "无法")),
]


def classify_business_text(text: str) -> Optional[ErrorKind]:
    """工具执行成功但返回业务失败文本 → 细分性质（记录不存在 vs 一般业务失败）"""
    for kind, keywords in _BIZ_TEXT_RULES:
        if any(kw in text for kw in keywords):
            return kind
    return None


# ============ 处置矩阵 ============

TRANSIENT_KINDS = frozenset({
    ErrorKind.TRANSIENT_NETWORK,
    ErrorKind.TRANSIENT_RATE_LIMIT,
    ErrorKind.TRANSIENT_DB_CONNECTION,
    ErrorKind.TRANSIENT_DB_TIMEOUT,
})

# 基础处置：按异常性质决定交给谁
_KIND_DISPOSITION: dict[ErrorKind, Disposition] = {
    **{k: Disposition.RETRY_SILENT for k in TRANSIENT_KINDS},
    ErrorKind.TOOL_NOT_FOUND: Disposition.FEEDBACK_TO_LLM,
    ErrorKind.INVALID_ARGS: Disposition.FEEDBACK_TO_LLM,
    ErrorKind.DB_CONSTRAINT: Disposition.FEEDBACK_TO_LLM,
    ErrorKind.BUSINESS_FAILURE: Disposition.TRIGGER_REPLAN,
    ErrorKind.RECORD_NOT_FOUND: Disposition.TRIGGER_REPLAN,
    ErrorKind.SERVICE_UNAVAILABLE: Disposition.TRIGGER_REPLAN,
    ErrorKind.AUTH_FAILURE: Disposition.FAIL_FAST,
    ErrorKind.CODE_BUG: Disposition.FAIL_FAST,
    ErrorKind.RESOURCE_EXHAUSTED: Disposition.FAIL_FAST,
}

# 非幂等写操作清单（读操作天然幂等；写操作重试可能产生副作用，如重复发放优惠券）
NON_IDEMPOTENT_TOOLS: set[str] = {
    "发放优惠券",
    "营销_发放优惠券",
    "更新知识库",
    "保存用户行为喜好",
}


def is_idempotent(tool_name: str) -> bool:
    return tool_name not in NON_IDEMPOTENT_TOOLS


def decide_disposition(kind: ErrorKind, tool_name: str) -> Disposition:
    """处置决策 = 异常性质 + 工具幂等性

    关键规则：非幂等写操作遇到瞬时异常（超时/断连）时，请求可能已生效，
    盲目静默重试会造成重复发放 —— 必须交给 LLM 先查询确认再决定。
    """
    base = _KIND_DISPOSITION[kind]
    if base is Disposition.RETRY_SILENT and not is_idempotent(tool_name):
        return Disposition.FEEDBACK_TO_LLM
    return base


# ============ LLM 修正指导 ============

def build_llm_guidance(
    kind: ErrorKind,
    tool_name: str,
    schema_str: str,
    error_text: str,
    tool_names: list | None = None,
) -> str:
    """为 FEEDBACK_TO_LLM 类异常构造可执行的修正指导（而非裸错误信息）

    transport 无关：不依赖活工具对象，只读 schema 文本与工具名列表（来自 ToolRunner）。
    """
    if kind in TRANSIENT_KINDS and not is_idempotent(tool_name):
        return (
            f"工具[{tool_name}]调用超时/中断，结果未知（请求可能已生效）。"
            "禁止直接重试！请先调用对应的查询类工具确认是否已生效，再决定是否需要重做。"
            f"原始错误：{error_text}"
        )
    if kind is ErrorKind.TOOL_NOT_FOUND:
        similar = difflib.get_close_matches(tool_name, list(tool_names or []), n=3, cutoff=0.3)
        hint = f"相近工具：{similar}" if similar else f"全部工具：{list(tool_names or [])[:10]}"
        return f"工具[{tool_name}]不存在。{hint}。请从已注册工具中选择并重新调用。"
    if kind is ErrorKind.INVALID_ARGS:
        schema = schema_str or "未知"
        return (
            f"参数错误：{error_text}\n"
            f"工具[{tool_name}]参数 Schema：{schema}\n"
            "请严格按照 Schema 的字段名和类型修正参数后重新调用。"
        )
    if kind is ErrorKind.DB_CONSTRAINT:
        return (
            f"数据库约束冲突：{error_text}\n"
            "通常意味着数据已存在或引用无效（如优惠券重复发放、订单号已存在）。"
            "请调整参数（跳过重复项）或改为确认现状，不要原样重试。"
        )
    return f"工具[{tool_name}]调用失败（{kind.value}）：{error_text}"


# ============ 熔断器 ============

class CircuitBreaker:
    """工具级熔断：连续 N 次致命失败后短路 M 秒

    熔断期间该工具的直接调用被拒绝（SERVICE_UNAVAILABLE），
    避免 SQL bug / 认证失效类工具反复炸穿整条执行链路。
    """

    def __init__(self, threshold: int = 3, cooldown_s: float = 60.0, clock: Callable[[], float] = time.monotonic):
        self._threshold = threshold
        self._cooldown_s = cooldown_s
        self._clock = clock
        self._failures: dict[str, int] = {}
        self._opened_at: dict[str, float] = {}

    def record_catastrophe(self, tool_name: str) -> None:
        """致命失败计数（仅 FAIL_FAST 类），达到阈值打开熔断"""
        self._failures[tool_name] = self._failures.get(tool_name, 0) + 1
        if self._failures[tool_name] >= self._threshold:
            self._opened_at[tool_name] = self._clock()
            logger.warning("⚡ 工具[%s]连续%d次致命失败，熔断%gs", tool_name,
                           self._failures[tool_name], self._cooldown_s)

    def record_success(self, tool_name: str) -> None:
        self._failures.pop(tool_name, None)
        self._opened_at.pop(tool_name, None)

    def is_open(self, tool_name: str) -> bool:
        opened = self._opened_at.get(tool_name)
        if opened is None:
            return False
        if self._clock() - opened >= self._cooldown_s:  # 冷却期过，半开试探
            self._opened_at.pop(tool_name, None)
            self._failures[tool_name] = self._threshold - 1  # 再失败一次立即重新熔断
            return False
        return True


_global_breaker = CircuitBreaker()


# ============ 统一执行管道 ============

@dataclass
class ToolOutcome:
    """工具执行结果（含异常分类与处置信息）"""
    success: bool
    result: Optional[str] = None
    error: Optional[str] = None
    kind: Optional[ErrorKind] = None
    disposition: Optional[Disposition] = None
    guidance: Optional[str] = None   # FEEDBACK_TO_LLM 时的修正指导文本
    attempts: int = 1
    repaired: bool = False           # 是否经过参数修复后成功


def execute_tool_with_policy(
    tool_name: str,
    runner,
    arguments: dict,
    *,
    repair_fn: Callable | None = None,
    retry_delays: tuple[float, ...] = (0.1, 0.3),
    breaker: CircuitBreaker | None = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> ToolOutcome:
    """统一工具执行管道：分类 → 处置 → （静默重试 / 参数修复 / 重规划 / 熔断）

    runner 为 ToolRunner（LocalToolRunner / McpToolRunner），或兼容旧调用的 dict
    （此时自动包成 LocalToolRunner）。runner.invoke 返回已分类的 RawOutcome，
    故处置决策不依赖任何活异常——重试/修复/熔断/ToolExecutionAbort 语义保持不变。

    repair_fn(tool_name, schema_str, bad_args, kind, error_text) -> dict|None：
        离线执行（Planner 指定的 TOOL 步骤）时注入，做一次 LLM 参数修复；
        ReAct 循环内不注入（LLM 在环，直接回传 guidance 自纠）。
    """
    if isinstance(runner, dict):  # 兼容旧调用（dict 映射 → 进程内 LocalToolRunner）
        from app.tools.runner import LocalToolRunner

        runner = LocalToolRunner(runner)

    breaker = breaker or _global_breaker
    tool_names = runner.tool_names()

    if breaker.is_open(tool_name):
        return ToolOutcome(
            success=False, kind=ErrorKind.SERVICE_UNAVAILABLE,
            disposition=Disposition.FAIL_FAST,
            error=f"工具[{tool_name}]处于熔断冷却中，已跳过调用",
        )
    if tool_name not in tool_names:
        return ToolOutcome(
            success=False, kind=ErrorKind.TOOL_NOT_FOUND,
            disposition=Disposition.FEEDBACK_TO_LLM,
            error=f"未知工具[{tool_name}]",
            guidance=build_llm_guidance(ErrorKind.TOOL_NOT_FOUND, tool_name, "",
                                        f"未知工具[{tool_name}]", tool_names),
        )

    schema = runner.tool_schema(tool_name)

    # ---- 第 1 层：执行 + 瞬时异常静默重试 ----
    kind: ErrorKind = ErrorKind.BUSINESS_FAILURE
    err = ""
    attempt = 0
    for attempt in range(len(retry_delays) + 1):
        if attempt:
            sleeper(retry_delays[attempt - 1])
        ro = runner.invoke(tool_name, arguments or {})
        if ro.ok:
            breaker.record_success(tool_name)
            return ToolOutcome(success=True, result=ro.result, attempts=attempt + 1)
        kind = ro.kind or ErrorKind.BUSINESS_FAILURE
        err = ro.error or "未知错误"
        if (kind in TRANSIENT_KINDS and is_idempotent(tool_name)
                and attempt < len(retry_delays)):
            logger.warning("🔄 工具[%s]瞬时异常(%s)，第%d次静默重试",
                           tool_name, kind.value, attempt + 1)
            continue
        break

    disp = decide_disposition(kind, tool_name)
    if disp is Disposition.RETRY_SILENT:
        disp = Disposition.TRIGGER_REPLAN  # 幂等工具静默重试耗尽 → 换路径

    if disp is Disposition.FAIL_FAST:
        breaker.record_catastrophe(tool_name)
        return ToolOutcome(success=False, kind=kind, disposition=disp, error=err,
                           attempts=attempt + 1)

    # ---- 第 2 层：LLM 参数修复（仅输入类异常 + 离线模式注入 repair_fn）----
    if (disp is Disposition.FEEDBACK_TO_LLM
            and kind in (ErrorKind.INVALID_ARGS, ErrorKind.DB_CONSTRAINT)
            and repair_fn is not None):
        try:
            fixed = repair_fn(tool_name, schema, arguments, kind, err)
            if fixed:
                ro2 = runner.invoke(tool_name, fixed)
                if ro2.ok:
                    breaker.record_success(tool_name)
                    return ToolOutcome(success=True, result=ro2.result,
                                       attempts=attempt + 2, repaired=True)
                kind = ro2.kind or kind
                err = ro2.error or err
        except Exception as e:
            logger.warning("⚠️ 工具[%s]参数修复失败: %s", tool_name, e)
        # 修复失败 → 升级为重规划（Planner 重新设计该步骤）
        return ToolOutcome(success=False, kind=kind, disposition=Disposition.TRIGGER_REPLAN,
                           error=err, attempts=attempt + 1)

    # ---- 第 3 层：回传 LLM 自纠（ReAct 在环）或重规划 ----
    return ToolOutcome(
        success=False, kind=kind, disposition=disp, error=err,
        guidance=build_llm_guidance(kind, tool_name, schema, err, tool_names),
        attempts=attempt + 1,
    )

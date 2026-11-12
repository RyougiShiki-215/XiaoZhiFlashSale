"""冒烟测试：不依赖真实 LLM API Key，验证模块导入与图结构

运行：python tests/test_smoke.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_imports():
    """所有模块可导入"""
    from app.config import get_settings
    from app.context import set_current_user_id
    from app.models import Plan, PlanStep, StepResult, StepStatus, StepType

    assert get_settings() is not None


def test_models():
    """Plan 模型行为"""
    from app.models import Plan, PlanStep, StepStatus, StepType

    plan = Plan(
        steps=[
            PlanStep(step_number=1, step_type=StepType.RAG, description="检索"),
            PlanStep(step_number=2, step_type=StepType.FINISH, description="输出"),
        ]
    )
    assert plan.next_pending_step().step_type == StepType.RAG
    assert not plan.all_steps_completed()
    plan.steps[0].status = StepStatus.COMPLETED
    assert plan.next_pending_step().step_type == StepType.FINISH


def test_skill_registry():
    """Skill 注册表：元数据加载 + 关键词匹配 + 懒加载正文"""
    from app.skill.registry import skill_registry

    skill_registry.load_all()
    assert "smart-marketing" in skill_registry.list_ids(), "smart-marketing skill 应被加载"

    matched = skill_registry.match_by_keywords("推荐点餐优惠")
    assert any(m.skill_id == "smart-marketing" for m in matched), "触发词应命中 smart-marketing"

    content = skill_registry.ensure_content_loaded("smart-marketing")
    assert content is not None and "智能营销技能" in content.markdown_body

    resources = skill_registry.list_resource_paths("smart-marketing")
    assert any("cuisine-mapping.md" in p for p in resources), "资源文件应被列出"


def test_tools_registry():
    """工具注册表：营销工具+闪购工具+Skill工具均注册且无重名冲突"""
    from app.tools.registry import ALL_TOOLS, TOOL_MAP, render_tools_as_text

    assert len(ALL_TOOLS) == len(TOOL_MAP), "工具名不应重复"
    assert "营销_发放优惠券" in TOOL_MAP
    assert "查询用户历史订单" in TOOL_MAP
    assert "读取Skill资源文件" in TOOL_MAP
    text = render_tools_as_text()
    assert "[TOOL] 营销_发放优惠券" in text


def test_mock_tools():
    """工具直接调用（不经过 LLM）"""
    from app.tools.marketing_tools import get_avg_order_amount, query_available_coupons
    from app.tools.shangou_tools import do_zxc, query_orders

    assert get_avg_order_amount.invoke({}) == "42.50"
    assert "C20001" in query_available_coupons.invoke({"merchant_ids": "M001"})
    assert "失败" in do_zxc.invoke({})  # ZXC 返回业务失败（用于验证重规划）
    assert "麦当劳" in query_orders.invoke({})


def test_rag_filters():
    """动态过滤器逻辑（不触发真实向量检索）"""
    from app.rag.retriever import RetrievalContext, shop_filter, xiaozhi_main_filter

    ctx = RetrievalContext(user_id=123)
    f = xiaozhi_main_filter(ctx)
    assert f({"doc_type": "common"}) is True          # 公共知识命中
    assert f({"userId": "123"}) is True               # 私人知识命中
    assert f({"userId": "999"}) is False              # 他人私人知识不命中
    assert f({"shop_name": "海底捞火锅"}) is True      # 北京附近商家命中
    assert f({"shop_name": "广州酒家"}) is False       # 非附近商家不命中

    sf = shop_filter(ctx)
    assert sf({"shop_name": "麦当劳"}) is True
    assert sf({"doc_type": "common"}) is False


def test_graph_structure():
    """LangGraph 图编译成功且节点齐全"""
    from app.graph.builder import build_graph

    graph = build_graph()
    node_names = set(graph.get_graph().nodes.keys())
    for expected in ("plan", "execute_step", "replan", "finalize"):
        assert expected in node_names, f"图缺少节点 {expected}"


def test_routing(monkeypatch):
    """条件边路由逻辑"""
    from app.graph.builder import route_after_step
    from app.graph.planner import PlanValidityDecision
    from app.models import Plan, PlanStep, StepResult, StepStatus, StepType

    plan = Plan(steps=[
        PlanStep(step_number=1, step_type=StepType.TOOL, description="t1"),
        PlanStep(step_number=2, step_type=StepType.FINISH, description="f"),
    ])

    # 成功且还有步骤 → 计划校验通过（None 视为有效）→ 继续执行
    monkeypatch.setattr("app.graph.builder.check_plan_validity", lambda state: None)
    state = {"plan": plan, "replan_count": 0,
             "past_steps": [(plan.steps[0], StepResult.ok(1, "ok"))],
             "last_result": StepResult.ok(1, "ok")}
    assert route_after_step(state) == "execute_step"

    # 成功步骤但校验判定剩余计划不适用 → 重规划
    monkeypatch.setattr(
        "app.graph.builder.check_plan_validity",
        lambda state: PlanValidityDecision(plan_is_valid=False, reason="执行中发现前提已变化"),
    )
    state = {"plan": plan, "replan_count": 0,
             "past_steps": [(plan.steps[0], StepResult.ok(1, "ok"))],
             "last_result": StepResult.ok(1, "ok")}
    assert route_after_step(state) == "replan"

    # 成功且还有步骤，但重规划已耗尽 → 不再校验，继续执行
    monkeypatch.setattr(
        "app.graph.builder.check_plan_validity",
        lambda state: PlanValidityDecision(plan_is_valid=False, reason="x"),
    )
    state = {"plan": plan, "replan_count": 3,
             "past_steps": [(plan.steps[0], StepResult.ok(1, "ok"))],
             "last_result": StepResult.ok(1, "ok")}
    assert route_after_step(state) == "execute_step"

    # 失败且有重规划额度 → 重规划
    state = {"plan": plan, "replan_count": 0,
             "past_steps": [(plan.steps[0], StepResult.fail(1, "boom"))],
             "last_result": StepResult.fail(1, "boom")}
    assert route_after_step(state) == "replan"

    # 失败但重规划耗尽 → 立即 finalize（关键修复行为）
    state = {"plan": plan, "replan_count": 3,
             "past_steps": [(plan.steps[0], StepResult.fail(1, "boom"))],
             "last_result": StepResult.fail(1, "boom")}
    assert route_after_step(state) == "finalize"

    # FINISH 成功 → finalize（不触发计划校验）
    plan.steps[0].status = StepStatus.COMPLETED
    finish_step = plan.steps[1]
    state = {"plan": plan, "replan_count": 0,
             "past_steps": [(finish_step, StepResult.ok(2, "最终答案"))],
             "last_result": StepResult.ok(2, "最终答案")}
    assert route_after_step(state) == "finalize"


def test_memory():
    """会话记忆"""
    from app.memory.chat_memory import memory_provider

    mem = memory_provider.get(999)
    mem.add_user("你好")
    mem.add_ai("你好，我是小智")
    history = mem.as_history_text()
    assert "[用户] 你好" in history
    assert "[助手] 你好，我是小智" in history


def test_clean_think_tag():
    from app.graph.executor import clean_think_tag

    assert clean_think_tag("<think>推理</think>答案") == "答案"
    assert clean_think_tag(None) is None


# ============ 短期记忆压缩 ============

def test_chat_memory_compact():
    """接近阈值触发压缩：旧对话→摘要，保留系统提示词与最近原文，不直接截断"""
    from langchain_core.messages import SystemMessage

    from app.memory.chat_memory import ChatMemory

    mem = ChatMemory(memory_id=1, compact_trigger=10, keep_recent=4, summarize_fn=None)
    mem.add_system("系统提示词")
    for i in range(12):
        mem.add_user(f"用户消息{i}")
        mem.add_ai(f"助手回答{i}")

    assert mem.compact_if_needed() is True
    # 结构：1条系统提示词 + 1条摘要 + 最近4条原文
    assert len(mem.messages) == 6
    assert isinstance(mem.messages[0], SystemMessage) and mem.messages[0].content == "系统提示词"
    assert "此前对话摘要" in mem.messages[1].content
    assert "用户消息9" in mem.messages[1].content       # 旧信息以摘要保留（回退拼接含原文）
    assert mem.messages[-1].content == "助手回答11"      # 最近原文未被截断
    assert mem.compact_count == 1
    # 压缩后低于阈值不再触发
    assert mem.compact_if_needed() is False


def test_chat_memory_compact_llm_summary():
    """注入摘要函数时优先用 LLM 摘要，异常自动回退拼接"""
    from app.memory.chat_memory import ChatMemory

    mem = ChatMemory(memory_id=2, compact_trigger=6, keep_recent=2,
                     summarize_fn=lambda msgs: "LLM生成的摘要")
    mem.add_system("sys")
    for i in range(8):
        mem.add_user(f"msg{i}")
    assert mem.compact_if_needed() is True
    assert "LLM生成的摘要" in mem.messages[1].content

    # 摘要函数抛异常 → 回退要点拼接，不丢消息
    mem2 = ChatMemory(memory_id=3, compact_trigger=6, keep_recent=2,
                      summarize_fn=lambda msgs: (_ for _ in ()).throw(RuntimeError("boom")))
    for i in range(8):
        mem2.add_user(f"msg{i}")
    assert mem2.compact_if_needed() is True
    assert "msg0" in mem2.messages[0].content  # 无系统提示词，摘要在首位


# ============ 长期记忆：语义冲突检测更新 ============

class _BagOfCharsEmbeddings:
    """确定性假 Embedding（字符袋向量）：字符重合度高→相似度高，无需真实 API"""

    def _v(self, text: str) -> list[float]:
        import numpy as np

        v = np.zeros(512)
        for ch in text:
            v[ord(ch) % 512] += 1
        n = float(np.linalg.norm(v))
        return (v / n if n else v).tolist()

    def embed_documents(self, texts):
        return [self._v(t) for t in texts]

    def embed_query(self, text):
        return self._v(text)


def test_long_term_memory_upsert():
    """语义相似的旧画像被覆盖更新，不同维度画像共存"""
    from app.memory.long_term_memory import LongTermMemoryStore
    from app.rag.store import InMemoryVectorStore

    store = InMemoryVectorStore(_BagOfCharsEmbeddings())  # 不播种演示知识
    ltm = LongTermMemoryStore(similarity_threshold=0.75, store=store)

    # 1. 新增画像
    r1 = ltm.upsert_profile(123, "口味偏好：爱吃辣")
    assert r1["action"] == "inserted"

    # 2. 语义相似的新画像 → 覆盖旧画像（爱吃辣 vs 爱吃辣+常点川菜，字符重合度高）
    r2 = ltm.upsert_profile(123, "口味偏好：爱吃辣，常点川菜")
    assert r2["action"] == "updated"
    assert len(r2["replaced"]) == 1 and "爱吃辣" in r2["replaced"][0]["text"]

    # 3. 不同维度画像 → 共存（工作信息与口味偏好无字符/语义重合）
    r3 = ltm.upsert_profile(123, "工作日在中关村附近上班")
    assert r3["action"] == "inserted"

    # 4. 该用户画像共2条（覆盖1次），其他用户隔离
    assert len(ltm.retrieve_profiles(123, "画像")) == 2
    assert len(ltm.retrieve_profiles(456, "画像")) == 0

    # 5. 口味更新画像覆盖旧口味画像（同一画像维度），不与工作画像冲突
    r4 = ltm.upsert_profile(123, "口味偏好：爱吃辣改为爱吃清淡，常点川菜")
    assert r4["action"] == "updated" and "川菜" in r4["replaced"][0]["text"]
    assert len(ltm.retrieve_profiles(123, "画像")) == 2


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"✅ {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"❌ {t.__name__}: {e}")
    print(f"\n{'全部通过' if failed == 0 else f'{failed} 个失败'}（共 {len(tests)} 项）")
    sys.exit(1 if failed else 0)

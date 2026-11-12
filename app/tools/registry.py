"""工具注册表"""
from langchain_core.tools import BaseTool

from app.tools import marketing_tools, shangou_tools, skill_tools

# 全部业务工具（顺序即 Planner 提示词中的展示顺序）
ALL_TOOLS: list[BaseTool] = [
    marketing_tools.query_user_history_orders,
    marketing_tools.rag_query_user_profile,
    marketing_tools.query_nearby_merchants,
    marketing_tools.query_available_coupons,
    marketing_tools.grant_coupon,
    marketing_tools.get_avg_order_amount,
    shangou_tools.search_web,
    shangou_tools.insert_knowledge,
    shangou_tools.recommend,
    shangou_tools.query_shop_info,
    shangou_tools.query_orders,
    shangou_tools.query_orders_money,
    shangou_tools.query_tel,
    shangou_tools.send_coupons,
    shangou_tools.save_user_profile,
    shangou_tools.do_abc,
    shangou_tools.do_fgh,
    shangou_tools.do_zxc,
    shangou_tools.do_xyz,
    skill_tools.list_skill_resources,
    skill_tools.read_skill_resource,
]

# 工具名 → 工具实例
TOOL_MAP: dict[str, BaseTool] = {t.name: t for t in ALL_TOOLS}


def render_tools_as_text() -> str:
    """渲染工具描述给 Planner"""
    lines = []
    for i, t in enumerate(ALL_TOOLS, 1):
        args = ", ".join(
            f"{k}:{'str' if v.get('type') in ('string', None) else v.get('type')}"
            for k, v in (t.args or {}).get("properties", {}).items()
        )
        lines.append(f"{i}. [TOOL] {t.name}\n     说明: {t.description}\n     参数: {args or '无'}")
    return "\n".join(lines) if lines else "（无可用工具）"

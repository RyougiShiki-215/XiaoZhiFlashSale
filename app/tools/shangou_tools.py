"""闪购业务工具

实现说明：
  - 工具定义 → langchain @tool 装饰器（中文工具名保持一致，便于提示词复用）
  - 用户身份注入 → 通过 app.context（contextvars）获取当前 userId，
    与请求上下文中的用法一致
"""
import json
import logging

from langchain_core.tools import tool

from app import context
from app.memory.long_term_memory import long_term_memory
from app.mock import db
from app.rag.retriever import RetrievalContext, shop_retriever
from app.rag.store import get_vector_store

logger = logging.getLogger(__name__)

# 业务失败关键词
_FAILURE_KEYWORDS = ("失败", "无号", "不存在", "错误")


def _current_user_id() -> int:
    uid = context.get_current_user_id()
    if uid is None:
        uid = 123
    return uid


@tool(description="当需要联网搜索时使用此工具")
def search_web(query: str) -> str:
    """联网搜索"""
    try:
        from tavily import TavilyClient

        from app.config import get_settings

        api_key = get_settings().tavily_api_key
        if not api_key:
            return "搜索失败：未配置 TAVILY_API_KEY"
        client = TavilyClient(api_key=api_key)
        resp = client.search(query=query, max_results=3)
        lines = [f"{r['title']}: {r['content']}" for r in resp.get("results", [])]
        return "\n".join(lines) if lines else "（未搜索到相关结果）"
    except Exception as e:
        return f"搜索失败：{e}"


@tool(description="当需要向知识库插入新的知识时使用此工具")
def insert_knowledge(text: str) -> str:
    """向向量库写入新知识"""
    get_vector_store().add([text], [{"doc_type": "common"}])
    return "知识已成功写入知识库"


@tool(description="调用该工具可以获取用来给用户推荐商品的信息，包括：用户历史订单、用户画像、附近商家信息。工具需要输入用来检索推荐商品的信息。")
def recommend(query: str) -> str:
    """推荐商品信息"""
    user_id = _current_user_id()
    location = db.get_location_by_user_id(user_id)
    shops = db.select_shop(location)
    orders = db.select_user_orders(user_id)
    return f"用户画像：（暂无）；用户历史订单：{json.dumps(orders, ensure_ascii=False)}；附近商家信息：{json.dumps(shops, ensure_ascii=False)}"


@tool(description="调用该工具可以检索附近商家信息")
def query_shop_info(shop_query: str = "") -> str:
    """检索附近商家"""
    user_id = _current_user_id()
    ctx = RetrievalContext(user_id=user_id, memory_id=None)
    shops = db.select_shop(db.get_location_by_user_id(user_id))

    answers = []
    for shop_name in shops:
        info = shop_retriever.retrieve(f"商家描述 {shop_name}", ctx)
        dishes = shop_retriever.retrieve(f"菜品信息 {shop_name}", ctx)
        answers.append(
            f"商家名称：{shop_name}；"
            f"商家信息：{json.dumps([c.text for c in info.chunks], ensure_ascii=False)}；"
            f"菜品信息：{json.dumps([c.text for c in dishes.chunks], ensure_ascii=False)}"
        )
    return json.dumps(answers, ensure_ascii=False)


@tool(description="调用该工具可以查询该用户历史订单,返回用户历史订单里涉及的店铺。")
def query_orders() -> str:
    """查询历史订单"""
    orders = db.select_user_orders(_current_user_id())
    return json.dumps(orders, ensure_ascii=False)


@tool(description="调用该工具可以获取用户的历史订单平均金额")
def query_orders_money() -> str:
    """历史订单平均金额"""
    return "23"


@tool(description="根据用户提供的店铺名称(以shopName参数输入进工具)，调用该工具可以查询对应店铺电话")
def query_tel(shop_name: str) -> str:
    """查询店铺电话"""
    logger.info("查询店铺电话：%s", shop_name)
    return "10086"


@tool(description="给用户发放优惠券。你自行判断给用户该发放哪一类型优惠券，同时说明为什么发放这一类型的优惠券")
def send_coupons(coupon_type: str) -> str:
    """发放优惠券（CouponTypeEnum → 字符串枚举更灵活）"""
    logger.info("发放优惠券面值：%s", coupon_type)
    return f"已成功发放优惠券：{coupon_type}"


@tool(description="当判断出用户的大致行为喜好时，调用该工具保存用户画像，重点关注用户喜欢的店铺和商品。")
def save_user_profile(profile: str) -> str:
    """保存用户画像（长期记忆 upsert：语义相似的过时画像会被覆盖更新）"""
    user_id = _current_user_id()
    result = long_term_memory.upsert_profile(user_id, profile)
    if result["action"] == "updated":
        replaced = "；".join(r["text"][:50] for r in result["replaced"])
        return f"用户画像已覆盖更新（检测到{len(result['replaced'])}条过时画像并替换，旧画像：{replaced}）"
    return "用户画像已保存"


@tool(description="当用户让你调用工具ABC时调用该工具")
def do_abc() -> str:
    """示例工具 ABC"""
    return "调用工具ABC成功"


@tool(description="当工具ABC调用失败时，调用该工具")
def do_fgh() -> str:
    """示例工具 FGH"""
    return "调用工具FGH成功"


@tool(description="当用户让你调用工具ZXC时调用该工具,如果返回调用失败，则重新再调用该工具,最多调用3次")
def do_zxc() -> str:
    """示例工具 ZXC"""
    return "调用工具ZXC失败，请重新调用工具ZXC"


@tool(description="当用户让你调用工具XYZ时调用该工具。如果返回调用失败的话则换个输入参数重新调用，最大重试次数为4")
def do_xyz(input: str) -> str:
    """示例工具 XYZ"""
    return f"调用工具XYZ成功，输入参数：{input}"

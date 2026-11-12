"""智能营销 Skill 配套工具"""
import json
import logging
import time
from datetime import date, timedelta

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool(description="【营销Skill第一步专用】查询当前用户的历史订单明细，用于分析用户常点的店铺、品类和消费档次。返回订单列表（含店铺名、菜品品类、实付金额、下单时间）。")
def query_user_history_orders(limit: int = 5) -> str:
    orders = [
        {"orderId": "O20240810001", "shopName": "麦当劳(中关村店)", "categories": ["汉堡", "西式快餐"], "items": ["麦辣鸡腿堡", "中薯条", "可乐"], "totalAmount": 38.5, "payTime": "2024-08-10 12:15:00"},
        {"orderId": "O20240808003", "shopName": "半天妖烤鱼(五道口店)", "categories": ["川菜", "烤鱼"], "items": ["香辣烤鱼2.5斤", "米饭×2"], "totalAmount": 128.0, "payTime": "2024-08-08 19:32:00"},
        {"orderId": "O20240805012", "shopName": "麻辣王子麻辣香锅", "categories": ["川菜", "香锅"], "items": ["微辣香锅(牛肉+午餐肉+素菜)"], "totalAmount": 46.8, "payTime": "2024-08-05 12:40:00"},
        {"orderId": "O20240802005", "shopName": "肯德基(学院路店)", "categories": ["汉堡", "西式快餐"], "items": ["新奥尔良烤鸡腿堡", "老北京鸡肉卷"], "totalAmount": 42.0, "payTime": "2024-08-02 13:05:00"},
        {"orderId": "O20240728009", "shopName": "正宗兰州牛肉面", "categories": ["面食", "西北菜"], "items": ["牛肉拉面(二细)", "卤蛋"], "totalAmount": 23.0, "payTime": "2024-07-28 08:20:00"},
    ]
    size = 5 if (not limit or limit <= 0) else min(limit, 20)
    return json.dumps(orders[:size], ensure_ascii=False)


@tool(description="【营销Skill第一步专用】通过RAG检索当前用户的画像信息，包括：口味偏好(辣/清淡/甜等)、忌口、饮食规律(早/中/晚餐习惯)、消费档次(客单价区间)、常点菜系TOP3。返回结构化画像描述。")
def rag_query_user_profile(query: str = "") -> str:
    return (
        "【用户画像ID=U12345】\n"
        "- 核心偏好标签：[爱吃辣, 川菜党, 食肉族, 高频午餐点外卖]\n"
        "- 常点菜系TOP3：① 川菜(45%) ② 西式快餐(30%) ③ 面食(15%)\n"
        "- 口味偏好：重口味 > 微辣 > 中辣 > 麻辣；不爱吃甜口、不挑食于香菜\n"
        "- 明确忌口：海鲜(过敏)、芹菜(不喜欢)\n"
        "- 消费档次：历史客单价区间 23-128 元，中位数约 42 元，属于【中档】\n"
        "- 饮食规律：工作日 11:50-12:30 下单最频繁(午餐)，周末偶有晚餐外卖\n"
        "- 品牌忠诚度：麦当劳、肯德基、半天妖 三个品牌复购率最高\n"
        "- 近30天订单数：12 单，月均消费约 560 元，属于【活跃用户·中等价值】"
    )


@tool(description="【营销Skill第一步专用】查询用户当前位置3-5公里内可配送的闪购商家清单，用于后续推荐。返回商家ID、名称、菜系、评分、月销量、配送费、主招牌菜TOP2。")
def query_nearby_merchants(location: str = "", max_count: int = 6) -> str:
    merchants = [
        {"merchantId": "M001", "name": "川香小厨(中关村旗舰店)", "cuisine": "川菜", "rating": 4.8, "monthlySales": 5200, "deliveryFee": 3, "minOrder": 20, "distance": "1.2km", "signatureDishes": ["麻婆豆腐套餐¥22", "香辣水煮鱼¥58"]},
        {"merchantId": "M002", "name": "麦当劳(中关村大街店)", "cuisine": "西式快餐", "rating": 4.7, "monthlySales": 8800, "deliveryFee": 5, "minOrder": 25, "distance": "0.8km", "signatureDishes": ["麦辣鸡腿堡三件套¥35", "板烧鸡腿堡套餐¥38"]},
        {"merchantId": "M003", "name": "辣妹子串串香(五道口店)", "cuisine": "川菜/串串", "rating": 4.6, "monthlySales": 3200, "deliveryFee": 0, "minOrder": 30, "distance": "2.1km", "signatureDishes": ["串串自助¥49/位", "冒鸭血¥12"]},
        {"merchantId": "M004", "name": "半天妖烤鱼(学院路店)", "cuisine": "川菜/烤鱼", "rating": 4.9, "monthlySales": 6500, "deliveryFee": 6, "minOrder": 88, "distance": "2.8km", "signatureDishes": ["香辣烤鱼2.5斤套餐¥138", "蒜香烤鱼2斤¥108"]},
        {"merchantId": "M005", "name": "肯德基(清华园店)", "cuisine": "西式快餐", "rating": 4.6, "monthlySales": 7500, "deliveryFee": 4, "minOrder": 30, "distance": "1.5km", "signatureDishes": ["新奥尔良烤鸡腿堡套餐¥36", "香辣鸡翅5块¥29"]},
        {"merchantId": "M006", "name": "兰州正宗牛肉面", "cuisine": "面食/西北", "rating": 4.3, "monthlySales": 2100, "deliveryFee": 2, "minOrder": 15, "distance": "1.0km", "signatureDishes": ["牛肉拉面¥18", "牛肉凉面¥20"]},
    ]
    return json.dumps(merchants[: max_count if max_count > 0 else 6], ensure_ascii=False)


@tool(description="【营销Skill第三步专用】查询当前AI有权限发放、且适用于指定商家的所有优惠券清单。必须传入第二步中推荐的商家ID或商家名称列表（逗号分隔），工具会过滤出只在这些商家可用的券。返回字段：couponId、couponType(满减券/折扣券/无门槛)、name、threshold(门槛金额)、discount(减免额)、discountRate(折扣率)、applicableMerchants、expireDays、stock。")
def query_available_coupons(merchant_ids: str = "", merchant_names: str = "") -> str:
    coupons = [
        {"couponId": "C20001", "couponType": "满减券", "name": "满20减5通用券", "threshold": 20.0, "discount": 5.0, "discountRate": None, "applicableMerchants": ["M001", "M002", "M005", "M006", "川香小厨", "麦当劳", "肯德基", "兰州牛肉面"], "expireDays": 7, "stock": 120, "aiGrantable": True},
        {"couponId": "C20002", "couponType": "折扣券", "name": "川香小厨专属8折券", "threshold": 0.0, "discount": None, "discountRate": 0.8, "applicableMerchants": ["M001", "川香小厨"], "expireDays": 3, "stock": 50, "aiGrantable": True},
        {"couponId": "C20003", "couponType": "满减券", "name": "满30减10川菜券", "threshold": 30.0, "discount": 10.0, "discountRate": None, "applicableMerchants": ["M001", "M003", "M004", "川香小厨", "辣妹子串串香", "半天妖烤鱼"], "expireDays": 15, "stock": 200, "aiGrantable": True},
        {"couponId": "C20004", "couponType": "满减券", "name": "满88减20烤鱼券", "threshold": 88.0, "discount": 20.0, "discountRate": None, "applicableMerchants": ["M004", "半天妖烤鱼"], "expireDays": 10, "stock": 35, "aiGrantable": True},
        {"couponId": "C20005", "couponType": "无门槛券", "name": "新人3元无门槛券", "threshold": 0.0, "discount": 3.0, "discountRate": None, "applicableMerchants": ["全部商家"], "expireDays": 5, "stock": 0, "aiGrantable": True},
        {"couponId": "C20006", "couponType": "满减券", "name": "满50减15高客单券", "threshold": 50.0, "discount": 15.0, "discountRate": None, "applicableMerchants": ["M001", "M002", "M003", "M004", "M005"], "expireDays": 7, "stock": 80, "aiGrantable": True},
        {"couponId": "C90099", "couponType": "满减券", "name": "运营专属满100减50券(AI无权限)", "threshold": 100.0, "discount": 50.0, "discountRate": None, "applicableMerchants": ["全部商家"], "expireDays": 1, "stock": 3, "aiGrantable": False},
    ]
    return json.dumps(coupons, ensure_ascii=False)


@tool(description="【营销Skill第四步专用】按优惠券ID给当前用户发放指定优惠券。发放前必须先调用『营销_查询可发放优惠券列表』拿到候选，且couponId必须来自候选列表；同时必须传入选券理由reason。发放成功返回领取记录ID。")
def grant_coupon(coupon_id: str, coupon_name: str, reason: str) -> str:
    grant_id = f"GRANT{int(time.time() * 1000) % 100000000:08d}"
    record = {
        "success": True,
        "grantId": grant_id,
        "couponId": coupon_id,
        "couponName": coupon_name,
        "userId": "U12345",
        "validFrom": str(date.today()),
        "validTo": str(date.today() + timedelta(days=7)),
        "reason": reason,
        "message": "优惠券已成功发放至用户卡包，可在 '我的-优惠券' 中查看",
    }
    return json.dumps(record, ensure_ascii=False)


@tool(description="【营销Skill辅助】获取当前用户近N单的平均客单价，作为第四步选择优惠券门槛的参考基准。返回平均金额(元，保留2位小数)。")
def get_avg_order_amount(recent_orders: int = 5) -> str:
    return "42.50"

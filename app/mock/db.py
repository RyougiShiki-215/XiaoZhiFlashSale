"""模拟数据层

通过 MySQL 查询商家/用户位置；此处用模拟数据保持业务闭环可运行。
"""
from app import context

# 模拟商家库：位置 -> 附近商家列表
_SHOP_DB: dict[str, list[str]] = {
    "北京": ["海底捞火锅", "麦当劳", "肯德基", "半天妖烤鱼", "麻辣王子麻辣烫"],
    "上海": ["海底捞火锅", "老乡鸡", "肯德基", "小杨生煎"],
    "广州": ["广州酒家", "麦当劳", "点都德", "半天妖烤鱼"],
}

# 模拟用户位置：userId -> location
_USER_LOCATION: dict[int, str] = {123: "北京", 456: "上海"}

# 模拟历史订单：userId -> 店铺列表
_USER_ORDERS: dict[int, list[str]] = {
    123: ["麦当劳", "肯德基", "半天妖", "麻辣王子"],
    456: ["老乡鸡", "小杨生煎", "肯德基"],
}


def get_location_by_user_id(user_id: int | None) -> str:
    if user_id is None:
        return "北京"
    return _USER_LOCATION.get(user_id, "北京")


def select_shop(location: str) -> list[str]:
    """按位置查询附近商家"""
    return _SHOP_DB.get(location, ["麦当劳", "肯德基"])


def select_user_orders(user_id: int | None) -> list[str]:
    """查询用户历史订单"""
    return _USER_ORDERS.get(user_id or 123, [])

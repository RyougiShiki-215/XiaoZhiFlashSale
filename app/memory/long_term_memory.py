"""长期记忆：用户画像沉淀 + 冲突处理

写入流程（upsert）：
  1. 在该 userId 的旧画像中做向量语义相似检索
  2. 相似度 >= 阈值 → 判定为同一画像维度的过时信息，删除旧画像后写入新画像（覆盖更新）
  3. 无相似命中 → 视为新维度画像，直接插入
这样避免"爱吃辣"与"口味变清淡"两条矛盾画像共存，检索时互相污染。
"""
import logging
from typing import Optional
from uuid import uuid4

from app.config import get_settings
from app.rag.store import Match, get_vector_store

logger = logging.getLogger(__name__)


class LongTermMemoryStore:
    """长期记忆存储：以向量库为载体，按语义相似度做画像的冲突检测与覆盖更新"""

    def __init__(self, similarity_threshold: float | None = None, store=None):
        self._threshold = similarity_threshold
        self._store = store  # 惰性：为 None 时首次写入/检索再获取全局向量库

    @property
    def threshold(self) -> float:
        return self._threshold if self._threshold is not None else get_settings().profile_conflict_threshold

    @property
    def store(self):
        return self._store if self._store is not None else get_vector_store()

    def upsert_profile(self, user_id: int, text: str) -> dict:
        """写入画像：语义相似的旧画像被覆盖，不同维度画像共存"""
        uid = str(user_id)

        # 1. 语义相似检索该用户的旧画像（只在同一 userId 内比较）
        matched: list[Match] = self.store.search(
            text,
            k=5,
            min_score=0.0,
            metadata_filter=lambda m: m.get("doc_type") == "profile" and str(m.get("userId", "")) == uid,
        )
        conflicts = [m for m in matched if m.score >= self.threshold]

        # 2. 覆盖更新：删除过时画像，写入新画像
        if conflicts:
            removed = self.store.delete([m.id for m in conflicts])
            logger.info("🔄 长期记忆冲突处理：userId=%s 检测到%d条语义过时画像（最高相似度%.3f），已覆盖",
                        user_id, removed, conflicts[0].score)

        new_id = f"profile-{uid}-{uuid4().hex[:8]}"
        self.store.add([text], [{"userId": uid, "doc_type": "profile"}], ids=[new_id])

        action = "updated" if conflicts else "inserted"
        logger.info("💾 用户画像已%s userId=%s: %s", "覆盖更新" if conflicts else "新增", user_id, text[:100])
        return {
            "action": action,
            "replaced": [{"id": m.id, "score": round(m.score, 3), "text": m.text} for m in conflicts],
            "new_id": new_id,
        }

    def retrieve_profiles(self, user_id: int, query: str, top_k: int = 3) -> list[Match]:
        """检索该用户的画像（供 RAG 注入个性化上下文）"""
        uid = str(user_id)
        return self.store.search(
            query,
            k=top_k,
            min_score=0.0,
            metadata_filter=lambda m: m.get("doc_type") == "profile" and str(m.get("userId", "")) == uid,
        )


# 全局单例（惰性持有向量库，导入期零开销）
long_term_memory = LongTermMemoryStore()

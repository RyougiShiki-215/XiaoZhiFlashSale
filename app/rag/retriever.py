"""检索器

核心特性：动态过滤器 —— 根据每次请求的 RetrievalContext（userId/memoryId/location）
实时构建元数据过滤条件。
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from app import context
from app.config import get_settings
from app.rag.kwd_store import get_kwd_store
from app.rag.reranker import get_reranker
from app.mock import db
from app.rag.store import Match, get_vector_store

logger = logging.getLogger(__name__)


@dataclass
class RetrievalContext:
    """检索上下文"""
    memory_id: Optional[int] = None
    user_id: Optional[int] = None
    location: Optional[str] = None


@dataclass
class RetrievalResult:
    """检索结果"""
    query: str = ""
    chunks: list[Match] = field(default_factory=list)
    cost_ms: int = 0

    def to_context_text(self) -> str:
        """把命中块转为可注入 LLM 的上下文文本"""
        if not self.chunks:
            return "（知识库未检索到相关内容）"
        lines = []
        for i, c in enumerate(self.chunks, 1):
            lines.append(f"[{i}] (score={c.score:.3f}) {c.text}")
        return "\n".join(lines)


def _resolve_user_id(ctx: RetrievalContext | None) -> int:
    """解析用户ID"""
    if ctx and ctx.user_id is not None:
        return ctx.user_id
    if ctx and ctx.memory_id is not None:
        uid = context.get_user_id_by_memory_id(ctx.memory_id)
        if uid is not None:
            return uid
    return 123


def _resolve_shops(ctx: RetrievalContext | None, user_id: int) -> list[str]:
    location = (ctx.location if ctx and ctx.location else db.get_location_by_user_id(user_id))
    return db.select_shop(location)


# ---------------- 动态过滤器工厂 ----------------

def xiaozhi_main_filter(ctx: RetrievalContext | None) -> Callable[[dict], bool]:
    """小智主知识库过滤：用户私人知识 OR (公共知识 OR 附近商家)"""
    user_id = _resolve_user_id(ctx)
    shops = _resolve_shops(ctx, user_id)

    def match(meta: dict) -> bool:
        private = str(meta.get("userId", "")) == str(user_id)
        common = meta.get("doc_type") == "common"
        nearby = meta.get("shop_name") in shops
        return private or (common or nearby)

    return match


def shop_filter(ctx: RetrievalContext | None) -> Callable[[dict], bool]:
    """商家知识库过滤：仅附近商家"""
    user_id = _resolve_user_id(ctx)
    shops = _resolve_shops(ctx, user_id)

    def match(meta: dict) -> bool:
        return meta.get("shop_name") in shops

    return match


def user_profile_filter(ctx: RetrievalContext | None) -> Callable[[dict], bool]:
    """用户画像知识库过滤"""
    user_id = _resolve_user_id(ctx)
    shops = _resolve_shops(ctx, user_id)

    def match(meta: dict) -> bool:
        return meta.get("doc_type") == "common" or meta.get("shop_name") in shops

    return match


def _rrf_fuse(*groups: list[Match], k: int = 60, top_n: int = 10) -> list[Match]:
    """RRF 融合多路检索结果：以 doc id 归并，score = Σ 1/(k + rank)。

    对向量/BM25 两路分数量纲不敏感，天然去重；展示仍取首次出现的原文与元数据。
    """
    scores: dict[str, float] = {}
    keep: dict[str, Match] = {}
    for results in groups:
        for rank, m in enumerate(results, 1):
            scores[m.id] = scores.get(m.id, 0.0) + 1.0 / (k + rank)
            if m.id not in keep:
                keep[m.id] = m
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return [keep[key] for key, _ in ranked]


class EmbeddingRetriever:
    """检索器

    混合检索：ES BM25 关键词 + 向量 ANN，RRF 融合（ES 未配置时退化为纯向量）；
    可选再经 CrossEncoder 重排精排（未配置时保持 RRF 顺序）。
    """

    def __init__(
        self,
        max_results: int | None = None,
        min_score: float | None = None,
        dynamic_filter_func: Callable[[RetrievalContext], Callable[[dict], bool]] | None = None,
    ):
        s = get_settings()
        self.max_results = max_results or s.rag_max_results
        self.min_score = min_score if min_score is not None else s.rag_min_score
        self.dynamic_filter_func = dynamic_filter_func

    @property
    def store(self):
        """懒获取向量库（避免导入期实例化/网络调用）"""
        return get_vector_store()

    def retrieve(self, query: str, ctx: RetrievalContext | None = None) -> RetrievalResult:
        start = time.time()
        try:
            metadata_filter = None
            if self.dynamic_filter_func is not None and ctx is not None:
                metadata_filter = self.dynamic_filter_func(ctx)

            # 一路：向量 ANN（Milvus/内存）
            vector_chunks = self.store.search(
                query,
                k=self.max_results,
                min_score=self.min_score,
                metadata_filter=metadata_filter,
            )

            reranker = get_reranker()
            # 二路：ES BM25 关键词（未配置/不可用时退化为纯向量）
            kwd = get_kwd_store()
            if kwd is not None:
                kwd_chunks = kwd.search(
                    query,
                    k=self.max_results,
                    min_score=0.0,  # BM25 分数量纲与向量无关，阈值交由 RRF 取秩
                    metadata_filter=metadata_filter,
                )
                # 若无重排，RRF 直接收敛到 max_results；有重排则保留稍宽候选交给精排
                fuse_n = reranker.top_n if reranker is not None else self.max_results
                chunks = _rrf_fuse(vector_chunks, kwd_chunks, top_n=fuse_n)
                logger.info("🔀 混合检索 query=%r 向量%d条+关键词%d条→融合%d条",
                            query[:50], len(vector_chunks), len(kwd_chunks), len(chunks))
            else:
                chunks = vector_chunks

            # 重排精排：对融合候选按 query 与文本相关度重新打分排序
            if reranker is not None and chunks:
                chunks = reranker.rerank(query, chunks)[: self.max_results]

            cost = int((time.time() - start) * 1000)
            logger.info("✅ RAG检索完成 query=%r 命中%d条 耗时%dms", query[:50], len(chunks), cost)
            return RetrievalResult(query=query, chunks=chunks, cost_ms=cost)
        except Exception as e:  # 检索失败不阻断对话
            cost = int((time.time() - start) * 1000)
            logger.error("❌ RAG检索失败(%dms): %s", cost, e)
            return RetrievalResult(query=query, chunks=[], cost_ms=cost)


# 预装配检索器
main_retriever = EmbeddingRetriever(dynamic_filter_func=xiaozhi_main_filter)
shop_retriever = EmbeddingRetriever(dynamic_filter_func=shop_filter)
user_profile_retriever = EmbeddingRetriever(dynamic_filter_func=user_profile_filter)

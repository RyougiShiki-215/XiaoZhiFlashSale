"""Elasticsearch 关键词检索库（混合检索的 BM25 分支）

与向量库共用同一套文本/元数据/主键（doc id 与向量侧一致，便于 RRF 归并去重）。
只做倒排索引检索，不参与 embedding；写入由向量库的 add/delete 链路同步调用。
ES 未配置 / 不可用时 get_kwd_store() 返回 None，上层优雅回退为纯向量检索。
"""
import logging
from typing import Any, Callable, Optional

from app.rag.store import Match

logger = logging.getLogger(__name__)

_kwd_store: Optional[object] = None


def get_kwd_store():
    """单例获取 ES 关键词库；未配置 ES 端点时返回 None。

    仅在显式配置了 es_uri（或 es_host+es_port）时才建立索引，避免无 ES 环境下误连。
    """
    global _kwd_store
    if _kwd_store is not None:
        return _kwd_store

    from app.config import get_settings
    s = get_settings()
    uri = s.es_uri
    if not uri:
        if not s.es_host:
            return None
        uri = f"http://{s.es_host}:{s.es_port}"
    try:
        _kwd_store = KwdStore(uri=uri, index=s.es_index, analyzer=s.es_analyzer, timeout=s.es_timeout)
        logger.info("🔍 ES 关键词库就绪 (%s/%s)", uri, s.es_index)
    except Exception as e:  # 建不了索引则以纯向量兜底
        logger.warning("⚠️ ES 不可用，关键词检索停用（仅向量检索）: %s", e)
        _kwd_store = None
    return _kwd_store


class KwdStore:
    """基于 ES 的 BM25 关键词检索，接口与向量库对齐（add/delete/search → list[Match]）"""

    def __init__(self, uri: str, index: str = "xiaozhi", analyzer: str = "standard", timeout: int = 5):
        from elasticsearch import Elasticsearch  # 延迟导入，避免无依赖时模块不可用

        # elasticsearch 9.x 用 request_timeout；8.x 用 timeout（兼容两者）
        try:
            self._client = Elasticsearch(uri, request_timeout=timeout, max_retries=1)
        except TypeError:
            self._client = Elasticsearch(uri, timeout=timeout, max_retries=1)
        self._index = index
        self._analyzer = analyzer
        self._ensure_index()

    def _ensure_index(self) -> None:
        """索引不存在则创建：text 用指定 analyzer（默认 standard；中文可换 ik_max_word 需装插件）"""
        if self._client.indices.exists(index=self._index):
            return
        body = {"mappings": {"properties": {"text": {"type": "text", "analyzer": self._analyzer}}}}
        self._client.indices.create(index=self._index, body=body)

    def add(self, texts: list[str], metadatas: list[dict] | None = None, ids: list[str] | None = None) -> None:
        """按指定 id（或自动生成）写入倒排索引；元数据入 _source 供过滤/回传"""
        from elasticsearch import helpers  # noqa: 延迟导入

        actions = []
        for i, text in enumerate(texts):
            did = ids[i] if ids else f"doc-{abs(hash(text))}-{i}"
            doc = {**(metadatas[i] if metadatas else {}), "text": text}
            actions.append({"_op_type": "index", "_index": self._index, "_id": did, "_source": doc})
        helpers.bulk(self._client, actions, raise_on_error=False, refresh=True)

    def delete(self, ids: list[str]) -> int:
        """按 id 批量删除，返回请求删除数"""
        from elasticsearch import helpers  # noqa: 延迟导入

        if not ids:
            return 0
        actions = [{"_op_type": "delete", "_index": self._index, "_id": i} for i in ids]
        helpers.bulk(self._client, actions, raise_on_error=False, refresh=True)
        return len(ids)

    def search(
        self,
        query: str,
        k: int = 10,
        min_score: float = 0.0,
        metadata_filter: Callable[[dict], bool] | None = None,
    ) -> list[Match]:
        """BM25 关键词检索：match text + 候选池查询端按 metadata_filter 过滤。

        注：BM25 分数与向量相似度量纲无关，min_score 视为不设阈值，交由上层 RRF 归并取秩。
        """
        body = {"query": {"match": {"text": {"query": query, "analyzer": self._analyzer}}}}
        resp = self._client.search(index=self._index, body=body, size=max(k * 10, 128))
        matches: list[Match] = []
        for hit in resp["hits"]["hits"]:
            src = hit.get("_source") or {}
            meta = {kk: vv for kk, vv in src.items() if kk != "text"}
            if metadata_filter is not None and not metadata_filter(meta):
                continue
            matches.append(Match(text=src.get("text", ""), score=float(hit.get("_score", 0.0)),
                                 metadata=meta, id=hit.get("_id", "")))
            if len(matches) >= k:
                break
        return matches
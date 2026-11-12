"""向量库封装

提供两种实现：
  1. InMemoryVectorStore：numpy 余弦相似度 + 元数据过滤（零外部依赖，开发兜底）
  2. MilvusVectorStore：配置了 MILVUS_URI 时启用（COSINE ANN + 动态字段元数据过滤）

统一接口约定：
  - add(texts, metadatas, ids) → 向量化并入库
  - search(query, k, filter)  → 相似度检索 + 元数据过滤
"""
import os
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
from langchain_openai import OpenAIEmbeddings

from app.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class Match:
    """单条检索命中"""
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = ""


class InMemoryVectorStore:
    """内存向量库：余弦相似度 + 简单元数据过滤（AND 语义）"""

    def __init__(self, embeddings: OpenAIEmbeddings, seed_fn: Callable | None = None, kwd_sink: Any | None = None):
        self._embeddings = embeddings
        self._texts: list[str] = []
        self._metadatas: list[dict] = []
        self._vectors: list[np.ndarray] = []
        self._ids: list[str] = []
        self._seed_fn = seed_fn
        self._seeded = False
        self._kwd = kwd_sink  # 可选：ES 关键词库，写入时同步

    def _sync_kwd(self, fn: Callable) -> None:
        """ES 同步失败只告警，不回滚向量主流程"""
        if self._kwd is None:
            return
        try:
            fn(self._kwd)
        except Exception as e:
            logger.warning("⚠️ ES 关键词库同步失败（数据可能不同步）: %s", e)

    def _ensure_seeded(self) -> None:
        """首次使用时才播种演示知识（避免导入期触发真实 Embedding 调用）"""
        if self._seeded:
            return
        self._seeded = True  # 先置位防递归
        if self._seed_fn is not None:
            try:
                self._seed_fn(self)
            except Exception as e:
                logger.warning("⚠️ 演示知识入库失败（检索将返回空结果）: %s", e)

    def add(self, texts: list[str], metadatas: list[dict] | None = None, ids: list[str] | None = None) -> None:
        self._ensure_seeded()
        if ids is None:  # 统一 id 生成，保证与 ES 分支一致（RRF 去重依赖同 id）
            ids = [f"doc-{abs(hash(text))}-{i}" for i, text in enumerate(texts)]
        vecs = np.array(self._embeddings.embed_documents(texts))
        for i, text in enumerate(texts):
            self._texts.append(text)
            self._metadatas.append(metadatas[i] if metadatas else {})
            self._vectors.append(vecs[i])
            self._ids.append(ids[i])
        self._sync_kwd(lambda k: k.add(texts, metadatas, ids))

    def delete(self, ids: list[str]) -> int:
        """按 id 删除向量（长期记忆覆盖更新用）；返回实际删除数"""
        id_set = set(ids)
        keep = [i for i, did in enumerate(self._ids) if did not in id_set]
        removed = len(self._ids) - len(keep)
        if removed:
            self._texts = [self._texts[i] for i in keep]
            self._metadatas = [self._metadatas[i] for i in keep]
            self._vectors = [self._vectors[i] for i in keep]
            self._ids = [self._ids[i] for i in keep]
        if ids:
            self._sync_kwd(lambda k: k.delete(ids))
        return removed

    def query(
        self,
        doc_id: str | None = None,
        shop_name: str | None = None,
        limit: int = 100,
    ) -> list[Match]:
        """按元数据（AND 语义）返回匹配的块，供列出/按文档删除用"""
        self._ensure_seeded()
        out: list[Match] = []
        for text, meta, uid in zip(self._texts, self._metadatas, self._ids):
            if doc_id is not None and meta.get("doc_id") != doc_id:
                continue
            if shop_name is not None and meta.get("shop_name") != shop_name:
                continue
            out.append(Match(text=text, score=0.0, metadata=meta, id=uid))
            if len(out) >= limit:
                break
        return out

    def search(
        self,
        query: str,
        k: int = 10,
        min_score: float = 0.0,
        metadata_filter: Callable[[dict], bool] | None = None,
    ) -> list[Match]:
        self._ensure_seeded()
        if not self._texts:
            return []
        q = np.array(self._embeddings.embed_query(query))

        matrix = np.array(self._vectors)
        # 余弦相似度
        sim = matrix @ q / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(q) + 1e-10)

        # 按分数倒序，过滤分数与元数据
        order = np.argsort(-sim)
        matches: list[Match] = []
        for idx in order:
            score = float(sim[idx])
            if score < min_score:
                continue
            meta = self._metadatas[idx]
            if metadata_filter is not None and not metadata_filter(meta):
                continue
            matches.append(Match(text=self._texts[idx], score=score, metadata=meta, id=self._ids[idx]))
            if len(matches) >= k:
                break
        return matches


class MilvusVectorStore:
    """Milvus 向量库

    配置项（传给 MilvusVectorStore 的 uri / token / collection / dim，参考 get_settings）：
      uri           本地 Milvus Lite 库文件路径（如 ./data/milvus_lite.db）或远程地址（http(s)）
      token         访问令牌（Zilliz Cloud 必填；本地部署一般为空）
      collection    集合名，默认 xiaozhi
      dim           向量维度，需与所选 embedding 模型一致，默认 1024

    元数据过滤沿用上层统一的 Callable 语义（与 InMemoryVectorStore 一致）：
      add 写入时把 text 与业务元数据一并作为动态字段入库，
      search 先用 COSINE 度量做 ANN，再对候选池按 metadata_filter 应用方函数过滤。
    """

    def __init__(
        self,
        embeddings: OpenAIEmbeddings,
        uri: str,
        token: str = "",
        collection: str = "xiaozhi",
        dim: int = 1024,
        seed_fn: Callable | None = None,
        kwd_sink: Any | None = None,
    ):
        from pymilvus import MilvusClient  # noqa: 延迟导入

        # Milvus Lite 本地库文件要求父目录存在，否则连接报 "Open local milvus failed, dir: ... not exists"
        if uri and "://" not in uri:
            _parent = os.path.dirname(uri)
            if _parent:
                os.makedirs(_parent, exist_ok=True)

        self._embeddings = embeddings
        self._collection = collection
        self._seed_fn = seed_fn
        self._seeded = False
        self._kwd = kwd_sink  # 可选：ES 关键词库，写入时同步
        self._client = MilvusClient(uri=uri, token=token) if token else MilvusClient(uri=uri)
        self._ensure_collection(dim=dim)

    def _sync_kwd(self, fn: Callable) -> None:
        """ES 同步失败只告警，不回滚向量主流程"""
        if self._kwd is None:
            return
        try:
            fn(self._kwd)
        except Exception as e:
            logger.warning("⚠️ ES 关键词库同步失败（数据可能不同步）: %s", e)

    def _ensure_collection(self, dim: int) -> None:
        """集合不存在则创建：VarChar 主键 + 动态字段（text 与业务元数据）+ COSINE 索引"""
        from pymilvus import DataType  # noqa: 延迟导入

        if self._client.has_collection(self._collection):
            # Milvus Lite 持久化库重启后集合存在但处于 released 状态，
            # add/search/query 前必须显式 load 才可访问。
            self._client.load_collection(self._collection)
            return
        schema = self._client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=256)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
        index_params = self._client.prepare_index_params()
        # Windows/milvus-lite 下 faiss AUTOINDEX 写磁盘 .idx 会偶发失败（FileIOWriter could not open）
        # 演示数据量很小，改用 FLAT（全量内积/余弦）更稳定。
        index_params.add_index(field_name="vector", index_type="FLAT", metric_type="COSINE")
        self._client.create_collection(collection_name=self._collection, schema=schema, index_params=index_params)
        self._client.load_collection(self._collection)
        logger.info("📚 Milvus 集合已创建: %s (dim=%d)", self._collection, dim)

    def _ensure_seeded(self) -> None:
        """首次使用且集合为空时才播种演示知识（持久化库避免重复写入）"""
        if self._seeded:
            return
        self._seeded = True  # 先置位防递归
        if self._seed_fn is None:
            return
        try:
            existing = self._client.query(collection_name=self._collection, output_fields=["id"], limit=1)
            if existing:
                return
            self._seed_fn(self)
        except Exception as e:
            logger.warning("⚠️ 演示知识入库失败（检索将返回空结果）: %s", e)

    def add(self, texts: list[str], metadatas: list[dict] | None = None, ids: list[str] | None = None) -> None:
        self._ensure_seeded()
        if ids is None:  # 统一 id 生成，保证与 ES 分支一致（RRF 去重依赖同 id）
            ids = [f"doc-{abs(hash(text))}-{i}" for i, text in enumerate(texts)]
        vecs = self._embeddings.embed_documents(texts)
        data = []
        for i, text in enumerate(texts):
            data.append({"id": ids[i], "vector": vecs[i], "text": text, **(metadatas[i] if metadatas else {})})
        self._client.upsert(collection_name=self._collection, data=data)
        self._sync_kwd(lambda k: k.add(texts, metadatas, ids))

    def delete(self, ids: list[str]) -> int:
        """按 id 删除向量；返回实际请求删除数"""
        if ids:
            self._client.delete(collection_name=self._collection, ids=list(ids))
            self._sync_kwd(lambda k: k.delete(ids))
        return len(ids)

    def query(
        self,
        doc_id: str | None = None,
        shop_name: str | None = None,
        limit: int = 100,
    ) -> list[Match]:
        """按动态字段（AND 语义）过滤返回匹配的块，供列出/按文档删除用"""
        conds = []
        if doc_id is not None:
            conds.append(f'doc_id == "{doc_id}"')
        if shop_name is not None:
            conds.append(f'shop_name == "{shop_name}"')
        expr = " and ".join(conds) if conds else 'doc_id != ""'
        resp = self._client.query(
            collection_name=self._collection,
            filter=expr,
            output_fields=["*"],
            limit=limit,
        )
        out: list[Match] = []
        for e in resp or []:
            entity = {k: v for k, v in e.items()}
            meta = {kk: vv for kk, vv in entity.items() if kk not in ("id", "vector", "text")}
            out.append(Match(text=entity.get("text", ""), score=0.0,
                             metadata=meta, id=str(entity.get("id", ""))))
        return out

    def search(
        self,
        query: str,
        k: int = 10,
        min_score: float = 0.0,
        metadata_filter: Callable[[dict], bool] | None = None,
    ) -> list[Match]:
        self._ensure_seeded()
        q = self._embeddings.embed_query(query)
        # Milvus 端做 ANN，取较宽的候选池，再按业务过滤器在查询端收窄（保序 + 元数据过滤）
        pool = max(k * 20, 256)
        resp = self._client.search(
            collection_name=self._collection,
            data=[q],
            limit=pool,
            output_fields=["*"],
        )
        matches: list[Match] = []
        for hit in resp[0] if resp else []:
            score = float(hit.get("distance", 0.0))  # COSINE 度量下 distance 即余弦相似度
            if score < min_score:
                continue
            entity = hit.get("entity") or {}
            meta = {kk: vv for kk, vv in entity.items() if kk not in ("text", "vector")}
            if metadata_filter is not None and not metadata_filter(meta):
                continue
            matches.append(Match(text=entity.get("text", ""), score=score, metadata=meta, id=str(hit.get("id", ""))))
            if len(matches) >= k:
                break
        return matches


_vector_store: Optional[object] = None


def get_vector_store():
    """单例获取向量库（Milvus Lite → 远程 Milvus → 内存，按配置优先级；懒播种）"""
    global _vector_store
    if _vector_store is not None:
        return _vector_store

    s = get_settings()
    embeddings = OpenAIEmbeddings(
        api_key=s.dashscope_api_key,
        base_url=s.dashscope_base_url,
        model=s.dashscope_embedding_model,
        # DeepSeek/DashScope 兼容端不支持 langchain 走 tiktoken token-id 形式输入
        # （会报 input.contents neither str nor list of str），关闭长度探测以发送纯文本。
        check_embedding_ctx_length=False,
    )
    # ES 关键词库作为写入 sink（配置了 ES 端点才启用，否则 None → 仅向量，纯向量回退）
    from app.rag.kwd_store import get_kwd_store
    kwd = get_kwd_store()
    if s.milvus_lite_db:
        logger.info("📦 使用 Milvus Lite 向量库 (%s)", s.milvus_lite_db)
        _vector_store = MilvusVectorStore(
            embeddings,
            uri=s.milvus_lite_db,
            collection=s.milvus_collection,
            dim=s.milvus_vector_dim,
            seed_fn=_seed_demo_knowledge,
            kwd_sink=kwd,
        )
    elif s.milvus_uri:
        logger.info("📦 使用远程 Milvus 向量库 (%s/%s)", s.milvus_uri, s.milvus_collection)
        _vector_store = MilvusVectorStore(
            embeddings,
            uri=s.milvus_uri,
            token=s.milvus_token,
            collection=s.milvus_collection,
            dim=s.milvus_vector_dim,
            seed_fn=_seed_demo_knowledge,
            kwd_sink=kwd,
        )
    else:
        logger.info("📦 使用内存向量库（如需 Milvus 请设置 MILVUS_LITE_DB 或 MILVUS_URI）")
        _vector_store = InMemoryVectorStore(embeddings, seed_fn=_seed_demo_knowledge, kwd_sink=kwd)
    return _vector_store


def _seed_demo_knowledge(store: InMemoryVectorStore) -> None:
    """内置演示知识（模拟已导入的商家文档），让 RAG 开箱可用"""
    docs = [
        ("闪购平台支持在线点餐、订单查询、优惠券发放、店铺推荐等服务。", {"doc_type": "common"}),
        ("满减优惠券：满30减5，满50减10，每周三发放，有效期7天。", {"doc_type": "common"}),
        ("海底捞火锅：以服务著称的连锁火锅店，招牌菜品有捞派滑牛肉、虾滑，人均消费120元。", {"shop_name": "海底捞火锅", "doc_type": "shop"}),
        ("麦当劳：全球连锁快餐品牌，招牌有巨无霸、麦辣鸡腿堡，人均消费35元，营业时间07:00-23:00，电话10086。", {"shop_name": "麦当劳", "doc_type": "shop"}),
        ("肯德基：炸鸡快餐连锁，招牌有吮指原味鸡、香辣鸡腿堡，人均消费40元，营业时间07:00-22:00。", {"shop_name": "肯德基", "doc_type": "shop"}),
        ("半天妖烤鱼：主打青花椒烤鱼和二合一烤鱼，人均消费70元，营业时间11:00-21:30。", {"shop_name": "半天妖烤鱼", "doc_type": "shop"}),
        ("麻辣王子麻辣烫：平价麻辣烫连锁，人均消费25元，支持自选配菜。", {"shop_name": "麻辣王子麻辣烫", "doc_type": "shop"}),
        ("优惠券使用规则：每单限用一张，不可叠加，下单时在结算页勾选使用。", {"doc_type": "common"}),
    ]
    store.add([d[0] for d in docs], [d[1] for d in docs])
    logger.info("🌱 已内置 %d 条演示知识", len(docs))

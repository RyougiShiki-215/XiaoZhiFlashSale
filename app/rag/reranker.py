"""重排器（rerank）：对混合检索(RRF)后的候选做精排

调用阿里云百炼（DashScope）的 text-rerank API（同步 HTTP），
对 query 与每个候选文本联合打分重排。API 调用失败时自动回退为入参顺序，不阻断检索。

- 模型名、端点 URL、工作空间均可通过配置指定（适配自定义网关）。
- 鉴权复用 dashscope_api_key。
"""
import logging
from typing import Optional

import httpx

from app.rag.store import Match

logger = logging.getLogger(__name__)

_reranker: Optional["DashScopeReranker"] = None

# 默认：阿里云百炼 text-rerank 原生 OpenAPI 端点
_DEFAULT_RERANK_URL = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text_rerank"


class DashScopeReranker:
    """基于阿里云百炼 text-rerank 的重排器"""

    def __init__(
        self,
        model: str = "qwen3.7-text-rerank",
        top_n: int = 10,
        api_key: str = "",
        url: str = "",
        workspace: str = "",
    ):
        self.model = model
        self.top_n = top_n
        self.api_key = api_key
        self.url = url or _DEFAULT_RERANK_URL
        self.workspace = workspace

    def rerank(self, query: str, matches: list[Match]) -> list[Match]:
        """对候选按 query 相关度打分并降序，返回前 top_n 条（原地更新 Match.score）"""
        if not matches:
            return matches
        try:
            body = {
                "model": self.model,
                "input": {"query": query, "documents": [m.text for m in matches]},
                "parameters": {
                    "top_n": max(self.top_n, len(matches)),
                    "instruct": "Given a web search query, retrieve relevant passages that answer the query.",
                },
            }
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            if self.workspace:
                headers["X-DashScope-Workspace"] = self.workspace
            resp = httpx.post(self.url, headers=headers, json=body, timeout=60)
            resp.raise_for_status()
            results = resp.json()["output"]["results"]
            ordered = sorted(results, key=lambda r: r.get("relevance_score", 0.0), reverse=True)
            out: list[Match] = []
            for r in ordered:
                idx = r.get("index")
                if not isinstance(idx, int) or idx >= len(matches):
                    continue
                m = matches[idx]
                m.score = float(r.get("relevance_score", 0.0))
                out.append(m)
                if len(out) >= self.top_n:
                    break
            return out
        except Exception as e:  # noqa: BLE001 - 重排失败不阻断检索，沿用 RRF 原序
            logger.warning("⚠️ DashScope 重排失败，采用 RRF 原序: %s", e)
            return matches


def get_reranker() -> "DashScopeReranker | None":
    """单例获取重排器；未开启或初始化失败时返回 None（调用方采用 RRF 原序）"""
    global _reranker
    if _reranker is not None:
        return _reranker

    from app.config import get_settings
    s = get_settings()
    if not s.rag_rerank_enabled:
        return None
    try:
        _reranker = DashScopeReranker(
            model=s.rag_rerank_model,
            top_n=s.rag_rerank_topk,
            api_key=s.dashscope_api_key,
            url=s.rag_rerank_url,
            workspace=s.rag_rerank_workspace,
        )
        logger.info("🔀 重排器就绪 (DashScope %s, top_n=%d, url=%s)",
                    _reranker.model, _reranker.top_n, _reranker.url)
    except Exception as e:  # noqa: BLE001 - 初始化失败自动降级
        logger.warning("⚠️ 重排器初始化失败，采用 RRF 原序: %s", e)
        _reranker = None
    return _reranker


# 便于测试/调用方重设
def reset_reranker() -> None:
    global _reranker
    _reranker = None
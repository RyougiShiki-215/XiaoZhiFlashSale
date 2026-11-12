"""全局配置"""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- Web ----
    port: int = 8000

    # ---- LLM 提供商选择：deepseek / dashscope / ollama ----
    llm_provider: str = "deepseek"
    llm_temperature: float = 0.7

    # ---- DeepSeek（OpenAI 兼容）----
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"

    # ---- DashScope 阿里百炼（OpenAI 兼容模式）----
    dashscope_api_key: str = ""
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dashscope_chat_model: str = "qwen-plus"
    dashscope_embedding_model: str = "text-embedding-v3"

    # ---- Ollama ----
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "deepseek-r1:8b"

    # ---- Milvus 向量库 ----
    # 注意：不用 MILVUS_URI 这个变量名承载 Lite 文件路径——pymilvus 导入时会读取
    # 同名环境变量作为其全局服务地址并校验必须是 http(s)，文件路径会导致 ImportError。
    # 因此 Lite 用独立的 MILVUS_LITE_DB（本地库文件），远程服务才用 MILVUS_URI。
    milvus_lite_db: str = ""          # 设置后启用 Milvus Lite（本地 .db 文件，免 Docker）
    milvus_uri: str = ""              # 远程 Milvus / Zilliz Cloud 地址（http(s)，可选）
    milvus_token: str = ""            # Zilliz Cloud 等托管服务需要
    milvus_collection: str = "xiaozhi"
    milvus_vector_dim: int = 1024     # 与 text-embedding-v3 默认维度对齐

    # ---- Elasticsearch 关键词检索（混合检索 BM25 分支，可选）----
    es_uri: str = ""                  # 完整地址，优先；如 http://es-host:9200 / 云 ES
    es_host: str = ""                 # 未填 es_uri 时，由 es_host:es_port 组合
    es_port: int = 9200
    es_index: str = "xiaozhi"
    es_analyzer: str = "standard"     # 中文可换 ik_max_word（需安装 ik 分词插件）
    es_timeout: int = 5

    # ---- 重排（rerank，可选）：对混合检索候选做精排 ----
    rag_rerank_enabled: bool = False          # 开启后对混合检索候选做精排（调用阿里云百炼 text-rerank API）
    rag_rerank_model: str = "qwen3.7-text-rerank"  # 百炼 rerank 模型名
    rag_rerank_topk: int = 20                 # 参与重排后保留的候选数上限
    rag_rerank_url: str = ""                  # 空→默认百炼原生 rerank 端点；自定义网关可指定完整 URL
    rag_rerank_workspace: str = ""            # 模型若部署在专用工作空间需填（X-DashScope-Workspace）

    # ---- Tavily 联网搜索 ----
    tavily_api_key: str = ""

    # ---- Skill ----
    skills_roots: str = "./skills"

    # ---- Plan-and-Execute 行为参数----
    max_replan_count: int = 2          # 最大重规划次数（最后一次机会强制收敛为 FINISH，避免反复检索导致超时）
    max_recent_messages: int = 20      # 参与规划的最大对话历史条数
    max_react_turns: int = 10           # 步骤内迷你 ReAct 最大轮次
    plan_check_enabled: bool = True    # 步骤成功后校验剩余计划是否仍合适（发现不适用→重规划换路径）
    plan_check_provider: str = ""      # 校验所用 provider（空=跟随主 LLM_PROVIDER；填 dashscope 等）
    plan_check_model: str = ""         # 校验用轻量模型名（空=复用主模型）；如 dashscope 可填 qwen-turbo
    rag_max_results: int = 10
    rag_min_score: float = 0.6

    # ---- 商家 PDF 入库的图片内容解析（多模态理解 LLM，非传统 OCR）----
    rag_ocr_enabled: bool = True          # 关闭后 PDF 中的图片不解析，仅入库文字与表格
    rag_ocr_model: str = "qwen3-vl-plus"  # 图片解析/多模态 LLM 模型名（经 OpenAi 兼容端点，qwen-vl 系列）

    # ---- 工具消费方式：local（进程内 BaseTool）| mcp（经自研薄 MCP client 调用）----
    tools_transport: str = "mcp"
    mcp_server_command: str = "python"
    mcp_server_args: str = "-m app.mcp_server --transport stdio"

    # ---- 工作记忆（graph state）存储：redis（持久到 Redis 并每轮删除）| memory（仅内存）----
    # 工作记忆 = 图中 plan/past_steps/last_result 等过渡态；每轮图结束后由 agent 删 thread 清理。
    state_store: str = "memory"   # 设 redis 需保证 REDIS_URL 可达，否则自动降级为内存
    redis_url: str = "redis://localhost:6379/0"

    # ---- 短期记忆压缩（接近窗口阈值时摘要压缩，避免直接截断）----
    memory_compact_trigger: int = 32   # 会话消息数达到该值触发压缩
    memory_keep_recent: int = 12       # 压缩后保留最近 N 条原文

    # ---- past_steps 分层压缩（图内步骤历史：近处保真，远处保骨架）----
    steps_keep_full: int = 8           # 最近 N 条步骤结果全量保留
    steps_skeleton_chars: int = 200    # 更早步骤降级后保留的结果字符数

    # ---- 长期记忆（用户画像）冲突处理 ----
    profile_conflict_threshold: float = 0.75  # 语义相似度超过该值视为同一画像维度→覆盖更新

    @property
    def skills_root_list(self) -> list[Path]:
        """Skill 根目录列表（逗号分隔）"""
        roots = []
        for p in self.skills_roots.replace(";", ",").split(","):
            p = p.strip()
            if p:
                roots.append((BASE_DIR / p).resolve() if not Path(p).is_absolute() else Path(p))
        return roots


@lru_cache
def get_settings() -> Settings:
    return Settings()

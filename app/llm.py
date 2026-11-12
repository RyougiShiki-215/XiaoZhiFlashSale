"""LLM 工厂

统一通过 OpenAI 兼容接口接入 DeepSeek / DashScope，Ollama 走本地接口。
"""
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from app.config import get_settings


def create_chat_model(provider: str | None = None, model: str | None = None) -> BaseChatModel:
    """按配置创建聊天模型实例；model 非空时覆盖当前 provider 的默认模型名"""
    s = get_settings()
    provider = (provider or s.llm_provider).lower()

    if provider == "deepseek":
        return ChatOpenAI(
            api_key=s.deepseek_api_key,
            base_url=s.deepseek_base_url,
            model=model or s.deepseek_model,
            temperature=s.llm_temperature,
        )
    if provider == "dashscope":
        return ChatOpenAI(
            api_key=s.dashscope_api_key,
            base_url=s.dashscope_base_url,
            model=model or s.dashscope_chat_model,
            temperature=s.llm_temperature,
        )
    if provider == "ollama":
        # Ollama 原生支持 OpenAI 兼容端点 /v1
        return ChatOpenAI(
            api_key="ollama",
            base_url=s.ollama_base_url.rstrip("/") + "/v1",
            model=model or s.ollama_model,
            temperature=s.llm_temperature,
        )
    raise ValueError(f"未知的 LLM_PROVIDER: {provider}（可选 deepseek/dashscope/ollama）")

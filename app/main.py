"""Web 入口

接口契约：
  POST /xiaozhi/chat   body: {"memoryId": 1, "message": "..."}   → text/plain; charset=utf-8

扩展接口：
  POST /xiaozhi/chat/stream  同请求体 → SSE 流式响应（plan/step/token/done 事件）
"""
import json
import logging
import re
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import context
from app.skill.registry import skill_registry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger(__name__)


class ChatForm(BaseModel):
    """对话请求体"""
    memoryId: int = 1
    message: str = ""


app = FastAPI(title="闪购小智 (python-langgraph)", version="1.0.0")

# 启动时加载 Skill 注册表
skill_registry.load_all()

# 静态前端（内置简单聊天页面，同源访问避免 CORS）
_STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.post("/xiaozhi/chat", response_class=PlainTextResponse)
def chat(chat_form: ChatForm) -> str:
    """对话接口"""
    logger.info("📨 收到 /xiaozhi/chat 请求, memoryId=%s, message=%s",
                chat_form.memoryId,
                chat_form.message[:80] + "..." if len(chat_form.message) > 80 else chat_form.message)

    # 模拟登录用户
    user_id = 123

    # 建立 memoryId → userId 映射 + 设置当前请求上下文
    context.set_memory_id_to_user_id(chat_form.memoryId, user_id)
    context.set_current_user_id(user_id)

    try:
        # 延迟导入避免启动时强依赖 LLM 配置
        from app.agent import xiaozhi_agent

        answer = xiaozhi_agent.chat(user_id, chat_form.memoryId, chat_form.message)

        # 清理思考标签（兼容 deepseek-r1 等推理模型输出）
        answer = re.sub(r"<think>[\s\S]*?</think>", "", answer or "").strip()

        logger.info("📤 /xiaozhi/chat 响应, 长度: %d 字符", len(answer))
        return answer
    finally:
        context.set_current_user_id(None)


@app.post("/xiaozhi/chat/stream")
async def chat_stream(chat_form: ChatForm) -> StreamingResponse:
    """流式对话接口（SSE）：逐事件推送规划/步骤进度与最终答案的 token 增量。

    事件格式（每条一个 SSE data 帧，JSON）：
      {"type": "plan_start"|"plan_ready"|"replan", ...}   规划进度
      {"type": "step_start"|"step_end", ...}              步骤进度
      {"type": "node_update", "node": "plan|execute_step|..."}
      {"type": "token", "content": "增量文本"}            最终答案逐 token
      {"type": "error"|"done", ...}                       终止事件
    """
    logger.info("📨 收到 /xiaozhi/chat/stream 请求, memoryId=%s, message=%s",
                chat_form.memoryId,
                chat_form.message[:80] + "..." if len(chat_form.message) > 80 else chat_form.message)

    user_id = 123
    context.set_memory_id_to_user_id(chat_form.memoryId, user_id)
    context.set_current_user_id(user_id)

    from app.agent import xiaozhi_agent

    async def event_gen():
        try:
            async for event in xiaozhi_agent.chat_stream(user_id, chat_form.memoryId, chat_form.message):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        finally:
            context.set_current_user_id(None)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/merchant/pdf/import")
async def merchant_pdf_import(
    file: UploadFile = File(...),
    shop_name: str = Form(...),
    doc_type: str = Form("shop"),
):
    """商家 PDF 上传入库 RAG 知识库。

    支持含文字 / 图片 / 表格的 PDF：解析后切块写入向量库（图片文字用 OCR 识别）。
    """
    logger.info("📄 收到商家PDF导入 file=%s shop_name=%s doc_type=%s", file.filename, shop_name, doc_type)
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="仅支持 PDF 文件")
    if not shop_name.strip():
        raise HTTPException(status_code=400, detail="shop_name 不能为空")

    from app.rag.pdf_ingest import ingest_pdf

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="上传文件为空")
    try:
        result = ingest_pdf(data, filename=file.filename, shop_name=shop_name.strip(), doc_type=doc_type)
    except Exception as e:
        logger.exception("❌ PDF 导入失败 shop=%s file=%s", shop_name, file.filename)
        raise HTTPException(status_code=400, detail=f"PDF 解析失败: {e}")

    return {
        "doc_id": result.doc_id,
        "source": result.source,
        "shop_name": result.shop_name,
        "doc_type": result.doc_type,
        "pages": result.pages,
        "chunks": result.chunks,
        "text_chars": result.text_chars,
        "ocr_images": result.ocr_images,
    }


@app.get("/merchant/pdf/docs")
def merchant_pdf_list(shop_name: str = ""):
    """按商家列出已导入文档（每个 PDF 一个条目，含块数）"""
    from app.rag.store import get_vector_store

    store = get_vector_store()
    matches = store.query(shop_name=shop_name) if shop_name else []
    # 按 doc_id 分组：source/doc_type 取组内首条，imported_at 取最大(最新导入)
    docs: dict[str, dict] = {}
    for m in matches:
        meta = m.metadata
        did = meta.get("doc_id", "")
        if not did:
            continue
        d = docs.setdefault(did, {
            "doc_id": did,
            "source": meta.get("source", ""),
            "shop_name": meta.get("shop_name", shop_name),
            "doc_type": meta.get("doc_type", ""),
            "chunk_count": 0,
            "imported_at": meta.get("imported_at", ""),
        })
        d["chunk_count"] += 1
        if meta.get("imported_at", "") > d["imported_at"]:
            d["imported_at"] = meta["imported_at"]
    return {"shop_name": shop_name, "total_chunks": len(matches), "documents": list(docs.values())}


@app.delete("/merchant/pdf/docs/{doc_id}")
def merchant_pdf_delete(doc_id: str):
    """按 doc_id 删除已导入 PDF 对应的所有向量块"""
    from app.rag.store import get_vector_store

    store = get_vector_store()
    matches = store.query(doc_id=doc_id)
    ids = [m.id for m in matches if m.id]
    deleted = store.delete(ids) if ids else 0
    logger.info("🗑️ 删除商家PDF文档 doc_id=%s 块%d", doc_id, deleted)
    return {"doc_id": doc_id, "deleted": deleted}


@app.get("/xiaozhi/health")
def health():
    """健康检查 + 模块自检"""
    return {
        "status": "ok",
        "skills": skill_registry.list_ids(),
        "tools": __import__("app.tools.registry", fromlist=["TOOL_MAP"]).TOOL_MAP and
                 list(__import__("app.tools.registry", fromlist=["TOOL_MAP"]).TOOL_MAP),
    }

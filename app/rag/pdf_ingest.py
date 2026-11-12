"""商家 PDF 导入 RAG 知识库（解析 + 切块 + 入库）

把商家提供的 PDF（可能含文字 / 图片 / 表格）转成可检索的文本块，复用
get_vector_store().add() 统一写入向量库（Milvus / 内存，可选同步 ES）。

流程：
  PDF bytes → 逐页提取文字层 + 表格(转 Markdown) + 用多模态 LLM 解析页面图片
           → 分页拼装文本 → 切块(带重叠) → 打 metadata → add() 入库

metadata 与 retriever.py 的过滤约定对齐：doc_type、shop_name、source、doc_id、page。

说明：这里的"图片解析"走的是 qwen-mt-image-2.0 这类多模态理解大模型（非传统 OCR），
由模型理解并转述图片中的内容；失败时对该图跳过，不阻断整页入库。
"""
import base64
import io
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime as _dt
from typing import Optional

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.rag.store import get_vector_store

logger = logging.getLogger(__name__)


@dataclass
class PageContent:
    """单页解析结果"""
    page_no: int
    text: str = ""          # 文字层
    table_md: str = ""      # 表格（Markdown）
    ocr_text: str = ""      # 页面图片由多模态 LLM 解析出的文本


@dataclass
class IngestResult:
    """一次 PDF 导入的结果"""
    doc_id: str
    source: str
    shop_name: str
    doc_type: str
    pages: int = 0
    chunks: int = 0
    text_chars: int = 0
    ocr_images: int = 0


# 图片解析的最小尺寸阈值（过滤图标/装饰小图）
_MIN_IMG_W = 120
_MIN_IMG_H = 120
# 文字层足够稀薄则即使无明显大图也对整页渲染并交给多模态 LLM 解析（覆盖扫描版/纯图文档）
_THIN_TEXT = 60
# 图片解析重试次数
_OCR_RETRIES = 1


def _page_semantics(page_no: int, text: str, table_md: str, ocr_text: str) -> PageContent:
    return PageContent(page_no=page_no, text=text, table_md=table_md, ocr_text=ocr_text)


def _make_ocr_llm(settings) -> Optional[ChatOpenAI]:
    """按配置创建图片解析用的多模态 LLM 客户端（DashScope OpenAi 兼容端点）"""
    if not settings.rag_ocr_enabled:
        return None
    if not settings.dashscope_base_url or not settings.dashscope_api_key:
        logger.warning("⚠️ 未配置 DashScope 端点，PDF 图片解析将跳过（仅入库文字与表格）")
        return None
    return ChatOpenAI(
        api_key=settings.dashscope_api_key,
        base_url=settings.dashscope_base_url,
        model=settings.rag_ocr_model,
        temperature=0,
    )


def _png_to_data_url(png_bytes: bytes) -> str:
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{b64}"


def ocr_image(ocr_llm: Optional[ChatOpenAI], png_bytes: bytes) -> str:
    """用多模态 LLM 理解 PNG 图片内容并转述为文本，失败返回空串（不阻断入库）"""
    if ocr_llm is None:
        return ""
    if not png_bytes:
        return ""
    data_url = _png_to_data_url(png_bytes)
    msg = HumanMessage(content=[
        {"type": "text",
         "text": "这是一张商家文档（如菜单/宣传单）中的图片，请理解并完整输出图片里的文字与信息，"
                 "按阅读顺序整理成文本；若图片无文字或无法解析则仅回复“无”。不要添加任何解释。"},
        {"type": "image_url", "image_url": {"url": data_url}},
    ])
    last_err: Exception | None = None
    for attempt in range(_OCR_RETRIES + 1):
        try:
            resp = ocr_llm.invoke([msg])
            text = (resp.content or "") if isinstance(resp.content, str) else str(resp.content or "")
            return text.strip() if text.strip() and text.strip() != "无" else ""
        except Exception as e:  # 图片解析失败不阻断入库
            last_err = e
            logger.warning("⚠️ 图片解析失败(第%d次): %s", attempt + 1, e)
    return ""


def extract_pdf(data: bytes) -> list[PageContent]:
    """逐页提取文字层、表格(Markdown) 与图片内容"""
    try:  # PyMuPDF 提供 fitz 别名；新版本建议直接用 pymupdf
        import pymupdf as fitz  # noqa
    except ImportError:
        import fitz  # noqa

    settings = get_settings()
    ocr_llm = _make_ocr_llm(settings)

    pages: list[PageContent] = []
    doc = fitz.open(stream=data, filetype="pdf")
    try:
        for i, page in enumerate(doc, 1):
            page_text = page.get_text("text").strip()

            # 表格 → Markdown
            table_md = ""
            try:
                tabs = page.find_tables()
                if tabs.tables:
                    table_md = "\n".join(t.to_markdown() for t in tabs.tables)
            except Exception as e:
                logger.debug("表格解析失败(page %d): %s", i, e)

            # 是否需要图片 OCR：有明显大图 或 文字层稀薄（扫描版）
            need_ocr = False
            try:
                for img_info in page.get_images(full=True) or []:
                    w, h = img_info[2], img_info[3]  # (xref, smask, w, h, bpc, cs, ...)
                    if w >= _MIN_IMG_W and h >= _MIN_IMG_H:
                        need_ocr = True
                        break
            except Exception:
                pass
            if not need_ocr and len(page_text) < _THIN_TEXT:
                need_ocr = True

            ocr_text = ""
            ocr_images = 0
            if need_ocr and ocr_llm is not None:
                pix = page.get_pixmap(matrix=fitz.Matrix(150 / 72, 150 / 72), alpha=False)  # ~150 DPI
                png = pix.tobytes("png")
                if png:
                    ocr_text = ocr_image(ocr_llm, png)
                    ocr_images = 1

            pages.append(_page_semantics(page_no=i, text=page_text, table_md=table_md, ocr_text=ocr_text))
            logger.debug("解析 page %d: 文字%d字 表格%d chars OCR%d chars",
                         i, len(page_text), len(table_md), len(ocr_text))
    finally:
        doc.close()
    return pages


def chunk_text(text: str, size: int = 350, overlap: int = 50) -> list[str]:
    """按字符数切块，带重叠防止跨块语义断裂"""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    step = max(size - overlap, 1)
    return [text[i:i + size] for i in range(0, len(text), step) if text[i:i + size].strip()]


def ingest_pdf(data: bytes, filename: str, shop_name: str, doc_type: str = "shop") -> IngestResult:
    """解析 PDF 并入库，返回导入统计"""
    pages = extract_pdf(data)
    doc_id = uuid.uuid4().hex
    now = _dt.now().isoformat(timespec="seconds")

    chunks: list[str] = []
    metadatas: list[dict] = []
    total_chars = 0
    ocr_images = 0

    for pc in pages:
        # 拼接该页全部内容（顺序：文字层、表格、图片 OCR）
        page_text = "\n".join(part for part in (pc.text, pc.table_md, pc.ocr_text) if part.strip())
        if pc.ocr_text:
            ocr_images += 1
        total_chars += len(page_text)
        for piece in chunk_text(page_text):
            chunks.append(piece)
            metadatas.append({
                "doc_type": doc_type,
                "shop_name": shop_name,
                "source": filename,
                "doc_id": doc_id,
                "page": pc.page_no,
                "imported_at": now,
            })

    store = get_vector_store()
    store.add(chunks, metadatas)

    logger.info("📥 PDF 入库 done  doc_id=%s source=%s shop=%s 页%d→块%d text_chars=%d ocr图%d",
                doc_id, filename, shop_name, len(pages), len(chunks), total_chars, ocr_images)

    return IngestResult(
        doc_id=doc_id,
        source=filename,
        shop_name=shop_name,
        doc_type=doc_type,
        pages=len(pages),
        chunks=len(chunks),
        text_chars=total_chars,
        ocr_images=ocr_images,
    )
# 闪购小智

基于 **LangGraph Plan-and-Execute** 的闪购/外卖场景智能客服。用户一句话提问后，系统会先拆解任务、再逐步调用工具或检索知识库，最终生成个性化回答，并支持流式输出规划与执行过程。

## 核心特性

- **Plan-and-Execute 智能体**：Planner 生成结构化计划，Executor 逐步执行；步骤失败或剩余计划不再适用时自动重规划，最多 2 次后强制收敛输出，避免无限循环。
- **步骤内迷你 ReAct**：TOOL / RAG / LLM / SKILL / FINISH 五类步骤均可在执行中自主补查工具，最多 3 轮。
- **混合检索 RAG**：Milvus 向量 ANN + Elasticsearch BM25 关键词检索，RRF 融合排序，可选 CrossEncoder 精排；ES 不可用时自动退化为纯向量检索。
- **三层渐进式 Skill**：启动只加载元数据 → Planner 规划仅看描述 → 执行时按需读取 `skill.md` 正文与资源文件，内置「智能营销」技能。
- **三层记忆体系**：
  - 短期记忆：会话历史 + 接近窗口阈值时自动摘要压缩；
  - 长期记忆：用户画像向量化沉淀，语义冲突时覆盖更新；
  - 工作记忆：基于 LangGraph Checkpointer（可选 Redis），每轮结束自动清理。
- **商家知识入库**：支持含文字、表格、图片的 PDF 上传，多模态模型解析图片内容后切块写入向量库，并提供文档列表与删除接口。
- **MCP 工具调用**：工具既可在进程内直接调用，也可通过自研薄 MCP Client 经 stdio 调用独立 MCP Server。
- **多模型接入**：DeepSeek / 阿里百炼 DashScope（OpenAI 兼容）/ Ollama 本地模型可切换。
- **内置调试页面**：同源静态聊天页，支持普通对话与 SSE 流式对话。

## 技术栈

| 模块 | 选型 |
| --- | --- |
| 智能体编排 | LangGraph、LangChain Core |
| Web 服务 | FastAPI、Uvicorn |
| 向量库 | Milvus / Milvus Lite（默认内存库兜底） |
| 关键词检索 | Elasticsearch 8.x（可选） |
| 重排 | DashScope Text-Rerank（可选） |
| 配置 | pydantic-settings + `.env` |
| PDF 解析 | PyMuPDF + 多模态视觉模型 |
| 联网搜索 | Tavily（可选） |

## 目录结构

```text
.
├── app/
│   ├── agent.py              # 智能体入口：提示词装配、对话与流式对话
│   ├── config.py             # 全局配置（pydantic-settings）
│   ├── context.py            # 请求级上下文（contextvars）
│   ├── llm.py                # LLM 工厂：DeepSeek / DashScope / Ollama
│   ├── main.py               # FastAPI 入口与业务接口
│   ├── mcp_server.py         # MCP 工具服务端
│   ├── models.py             # Plan / PlanStep / StepResult 等数据模型
│   ├── graph/                # Plan-and-Execute 图：planner / executor / builder / state
│   ├── memory/               # 短期记忆 / 长期记忆 / 工作记忆
│   ├── rag/                  # 向量库、混合检索、重排、PDF 入库
│   ├── skill/                # Skill 注册表与三层渐进式加载
│   ├── tools/                # 业务工具、工具注册表、异常处置、MCP 调用
│   ├── mock/db.py            # 本地模拟数据（商家 / 订单 / 优惠券）
│   └── static/index.html     # 内置聊天页面
├── prompts/                  # 系统提示词
├── skills/smart-marketing/   # 智能营销技能（示例、参考资料、优惠券规则）
├── tests/                    # 冒烟 / 记忆 / 工具异常 / MCP 测试
├── data/                     # Milvus Lite 本地数据（运行时生成，不提交）
├── logs/                     # 运行日志（运行时生成，不提交）
├── .env.example              # 环境变量模板
├── requirements.txt
├── run.py                    # 启动入口
├── start.ps1                 # Windows 一键启动（应用 + 可选 Elasticsearch）
└── start.cmd                 # start.ps1 的 cmd 包装
```

## 快速开始

要求 Python 3.10+。

```bash
# 1. 安装依赖
python -m venv .venv
.venv\Scripts\activate          # macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt

# 2. 准备配置
copy .env.example .env          # macOS / Linux: cp .env.example .env
# 编辑 .env，至少填入所选 LLM 提供商的 API Key

# 3. 启动
python run.py
```

启动后访问 <http://localhost:8000> 打开内置聊天页面。

Windows 下也可以直接双击 `start.cmd` 或执行：

```powershell
.\start.ps1          # 启动应用（可选自动安装并启动本地 Elasticsearch）
.\start.ps1 status   # 查看运行状态
.\start.ps1 stop     # 停止应用与 ES
```

## 配置说明

| 环境变量 | 说明 |
| --- | --- |
| `LLM_PROVIDER` | `deepseek` / `dashscope` / `ollama` |
| `DEEPSEEK_API_KEY` | DeepSeek API Key（OpenAI 兼容） |
| `DASHSCOPE_API_KEY` | 阿里百炼 API Key，Embedding / Rerank / PDF 图片解析依赖 |
| `DASHSCOPE_EMBEDDING_MODEL` | 向量模型，默认 `text-embedding-v3`（维度 1024） |
| `MILVUS_LITE_DB` | 本地 Milvus Lite 数据库路径，设置后免 Docker 运行 |
| `MILVUS_URI` | 远程 Milvus / Zilliz Cloud 地址（与 Lite 二选一） |
| `ES_URI` | Elasticsearch 地址；不填则纯向量检索 |
| `RAG_RERANK_ENABLED` | 是否开启检索结果重排 |
| `TAVILY_API_KEY` | 联网搜索（可选） |
| `TOOLS_TRANSPORT` | `mcp`（默认，走 MCP Server）或 `local`（进程内调用） |
| `STATE_STORE` | `memory`（默认）或 `redis`（工作记忆持久化） |

> 提示：Milvus Lite 数据库文件同一时间只能被一个进程占用，请勿同时启动多个服务实例。

## 接口一览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/` | 内置聊天页面 |
| `POST` | `/xiaozhi/chat` | 对话接口，返回纯文本答案 |
| `POST` | `/xiaozhi/chat/stream` | SSE 流式对话，推送规划、步骤进度与 token 增量 |
| `POST` | `/merchant/pdf/import` | 上传商家 PDF 入库（multipart：`file`、`shop_name`、`doc_type`） |
| `GET` | `/merchant/pdf/docs` | 按商家查看已导入文档 |
| `DELETE` | `/merchant/pdf/docs/{doc_id}` | 删除指定文档的全部向量块 |
| `GET` | `/xiaozhi/health` | 健康检查与 Skill / 工具自检 |

请求示例：

```bash
curl -X POST http://localhost:8000/xiaozhi/chat \
  -H "Content-Type: application/json" \
  -d "{\"memoryId\":1,\"message\":\"帮我推荐附近的餐厅，并用上我的优惠券\"}"
```

## 测试

无需真实 API Key 即可运行冒烟测试：

```bash
python tests/test_smoke.py
python tests/test_working_memory.py
python tests/test_error_policy.py
python tests/test_mcp_runner.py
```

## 运行须知

- `data/`、`logs/`、`.env` 已在 `.gitignore` 中排除，不会被提交。
- Elasticsearch、Redis、重排服务均为可选组件，未配置或连接失败时会自动降级，不影响主流程启动。
- 首次启动会自动创建 Milvus Lite 数据库并写入少量演示知识，便于开箱体验 RAG 流程。

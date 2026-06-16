# 论文研读 RAG 助手 — 项目指南（CLAUDE.md）

> 导入研究领域的论文（arXiv / PDF）与个人笔记，构建可问答的知识库。
> 提问得到**带精确引用溯源（哪篇 / 哪节 / 哪页）的答案**，支持多篇对比。
> 定位：**生产级 RAG + 评测驱动**——不止"能跑"，而是用评测集量化每一处改进。

---

## 这个项目要证明什么（简历定位）

本项目是作者的**第二个**作品，与第一个项目刻意互补、不重复：

| | 项目一 · 痛点挖掘系统 | 项目二 · 本项目 |
|---|---|---|
| 主打 | **多 Agent 编排**（LangGraph / 反思环） | **生产级 RAG + 评测驱动** |
| 关键词 | Agent 协作、任务规划、记忆 | 切分 · 混合检索 · 重排 · 引用溯源 · Evals · 工程化 |

> 故事线：研究生每天啃大量论文，痛点是"跨多篇找信息慢、出处记混、写综述要反复对比"。
> 于是做了这个 RAG 助手，并**建评测集量化改进**（例：加混合检索+重排后忠实度 X→Y）。

---

## 架构

```
【离线 · 入库 ingest】
论文 PDF / arXiv / 个人笔记
   → 解析 loader（PyMuPDF，保留页码/章节结构）
   → 切分 chunker（按结构递归切 + 元数据 paper_id/section/page）
   → Embedding（阿里百炼 text-embedding-v3，API）
   → 写入 Qdrant 向量库（P2 加 BM25 稀疏索引）

【在线 · 问答 retrieve → generate】
用户问题
   → 混合检索 hybrid（BM25 关键词 + 向量语义，RRF 融合）
   → 重排序 rerank（bge-reranker-v2-m3）
   → 带【引用溯源】的 LLM 作答（流式输出 SSE）
   → （可多篇对比）

【评测 eval · 重头戏】
评测集（问题 + 标准答案 + 应命中来源）
   → RAGAS 风格指标：忠实度 / 答案相关性 / 上下文召回
   → 消融对比实验（纯向量 vs 混合 vs +重排；不同 chunk 大小）→ 出对比表

【工程化】单元测试(pytest) + CI/CD(GitHub Actions) + Docker
【服务】FastAPI（REST + SSE 流式）+ Streamlit 界面
```

---

## 技术栈（刻意覆盖项目一没碰、招聘 JD 又高频的点）

| 层 | 技术 | 为什么这样选 |
|---|---|---|
| 语言 | **Python** | 公共底座 |
| 文档解析 | **PyMuPDF**（fitz） | 解析 PDF，保留页码/章节，供引用溯源 |
| 文本切分 | 结构感知递归切分（含 metadata） | 项目一没有正经 Chunking，这里补上 |
| Embedding | **阿里百炼 text-embedding-v3**（API） | 全链路走 API，不在本地下载模型；与 LLM 同一 key |
| 向量库 | **Qdrant** | JD 点名；原生支持混合检索；换库证明广度 |
| 检索 | **混合检索** BM25 + 向量 + **RRF 融合** | sparse 用 Qdrant 原生 BM25（非本地神经稀疏） |
| 重排 | **百炼 gte-rerank-v2**（API） | 项目一没有；显著提升 top 命中 |
| 生成 LLM | 通义千问 `qwen-plus`（默认，复用已有 key）/ 可换 DeepSeek | 带引用作答 + 流式 |
| 评测 | **RAGAS 风格指标 + 消融实验** | 项目一最大短板，本项目核心亮点 |
| 服务 | **FastAPI**（REST + **SSE 流式**） | 补"流式处理" |
| 前端 | **Streamlit**（聊天界面 + 引用展示） | 应用开发方向，快速出活 |
| 工程化 | **pytest** + **GitHub Actions** CI + **Docker / docker-compose** | 补"能交付"短板 |
| 数据库 | SQLite（会话/反馈记录，可选） | 轻量持久化 |

> 刻意**不引入多 Agent**，保持"纯 RAG"定位，避免与项目一重复（至多做轻量 query 改写）。

---

## 目录结构（规划，随 P 阶段逐步落地）

```
main.py                       # CLI 入口：导入论文 / 提问 / 跑评测
app.py                        # Streamlit 聊天界面
src/
├── ingest/                   # 离线入库
│   ├── loader.py             # PDF/arXiv 解析（PyMuPDF，出页码/章节）
│   ├── chunker.py            # 结构化切分 + 元数据
│   └── indexer.py            # Embedding + 写入 Qdrant
├── retrieve/                 # 在线检索
│   ├── store.py              # Qdrant 封装（向量 + BM25）
│   ├── hybrid.py             # BM25 + 向量 + RRF 融合
│   └── reranker.py           # bge-reranker 重排
├── generate/
│   ├── llm.py                # LLM 客户端（流式）
│   └── answerer.py           # 带引用溯源作答
├── eval/                     # 评测（重头戏）
│   ├── dataset.py            # 评测集（问题/标准答案/应命中来源）
│   ├── metrics.py            # RAGAS 指标实现
│   └── run_eval.py           # 消融对比实验，出对比表
├── prompts/                  # 提示词模板（引用作答 / query 改写）
├── embeddings.py             # BGE-m3 封装
├── api.py                    # FastAPI（REST + SSE 流式）
└── config.py                 # 配置（模型、阈值、路径）
tests/                        # pytest 单元测试
data/                         # 论文 PDF（gitignore）
.github/workflows/ci.yml      # CI/CD：自动跑测试
docker-compose.yml            # Qdrant + 后端一键起
Dockerfile
.env.example                  # 环境变量样例
requirements.txt
```

---

## 运行（规划）

```bash
# 1. 起向量库 + 装依赖
docker compose up -d qdrant
pip install -r requirements.txt
cp .env.example .env          # 填 DASHSCOPE_API_KEY 等

# 2. 导入论文
python main.py ingest ./data/papers/

# 3. 提问
python main.py ask "这几篇里 attention 的复杂度分别是多少？"

# 4. 跑评测（出消融对比表）
python main.py eval

# 5. 起界面
streamlit run app.py
```

---

## 约定 / 注意

- 每个 chunk 必带元数据 `{paper_id, title, section, page}`，**引用溯源全靠它**。
- 检索链路统一：`hybrid → rerank → answer`，参数（k、阈值、权重）集中放 `config.py`。
- 所有提示词走 `src/prompts`，勿在代码里写死。
- LLM / Embedding / 重排统一走百炼，key 环境变量 `DASHSCOPE_API_KEY`（同一个 key）。
- 全链路 API，不在本地下载模型；离线复跑需联网调用百炼。
- 运行时产物（`data/` `*.db` `qdrant_storage/` 模型缓存）已 gitignore，勿提交。
- 每个改进都要**先有评测、再下结论**，对比表进 README（这是本项目的灵魂）。

---

## 进度

- [x] **P0 骨架**：目录 + 依赖 + Docker(Qdrant) + PDF/arXiv 导入与解析（出页码/章节）
- [x] **P1 MVP**：切分 + Embedding + Qdrant 入库 + 基础向量检索 + LLM 作答（端到端跑通）
- [x] **P2 检索增强**：混合检索（BM25+向量+RRF）+ 重排序 + **引用溯源**
- [x] **P3 评测驱动**：评测集（含页级 gold）+ RAGAS 风格指标 + 确定性检索指标 + **消融出表**（框架完成；可信增益待扩语料+核准 gold 后复跑）
- [x] **P4 产品化**：SSE 流式输出（FastAPI）+ Streamlit 聊天界面（引用展示）+ 多文档对比
- [x] **P5 工程化**：pytest 单测 + GitHub Actions CI + Docker 封装 + README + 简历段

> 每个 Phase 完成后：更新本进度 + 提交并推送 GitHub。

---

## 技术栈 ↔ 招聘需求覆盖（need.md 对照）

两个项目合起来覆盖情况：

- **必备**：Python ✅ · Agent(项目一) ✅ · **RAG 深度(本项目)** ✅ · Prompt ✅ · LLM API ✅
- **加分**：**Evals(本项目)** ✅ · **单测/CI/CD/Docker(本项目)** ✅ · 向量库广度(Qdrant) ✅ · 流式 ✅ · 前端(FastAPI+Streamlit) ✅
- 暂不覆盖：Go / 微调·RLHF / 低代码工具（按目标岗位后续再补）

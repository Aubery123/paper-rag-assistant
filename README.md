# 论文研读 RAG 助手

导入研究领域的论文（PDF / arXiv）与笔记，构建可问答知识库。提问得到**带精确引用溯源（哪篇 / 哪节 / 哪页）**的答案，支持多篇对比。

定位：**生产级 RAG + 评测驱动**——不止"能跑"，而是用评测集量化每一处改进。

## 技术栈

| 层 | 选型 |
|---|---|
| 文档解析 | PyMuPDF（保留页码 / 章节结构） |
| 切分 | 结构感知按页切分 + 元数据 `{paper_id, title, section, page}` |
| Embedding | 阿里百炼 `text-embedding-v3`（API） |
| 向量库 | **Qdrant**（Docker） |
| 检索 | **混合检索**：BM25（rank-bm25）+ 向量 + **RRF 融合** |
| 重排 | 百炼 `gte-rerank-v2`（API） |
| 生成 | 百炼 `qwen-plus`，带引用溯源作答（支持流式） |
| 评测 | RAGAS 风格指标 + 确定性检索指标 + 消融对比 |
| 服务 / 前端 | FastAPI（REST + SSE）· Streamlit（规划中） |
| 工程化 | pytest · GitHub Actions · Docker（规划中） |

## 快速开始

```bash
docker compose up -d qdrant          # 起向量库
pip install -r requirements.txt
cp .env.example .env                 # 填 DASHSCOPE_API_KEY（百炼）

python main.py ingest ./data/papers/                 # 导入论文
python main.py ask "FLASH 在服务器端如何做漂移感知优化？"   # 提问（带引用）
python main.py eval --limit 2                         # 跑评测（测试用，限 2 题）

uvicorn src.api:app --reload                         # 起 API（/ask、/ask/stream SSE）
streamlit run app.py                                 # 起聊天界面（引用展示 + 多篇对比）
```

## 评测（项目核心）

评测集（`src/eval/eval_set.json`）：问题 + 参考答案 + **页级金标准**（答案所在 篇/页）。
消融对比 4 种检索配置：`纯向量 / +重排 / 混合 / 混合+重排`，分**检索层**（确定性、无 LLM）与**生成层**（RAGAS 风格 LLM 判分）两组指标。

> ⚠️ **当前为测试阶段的初步结果**：评测语料仅 5 篇且主题高度同质（联邦学习 + 概念漂移），
> 纯向量检索已接近天花板；页级 gold 由章节结构标注、尚待人工核准。
> 因此下表**暂未体现混合检索 / 重排的预期增益**——这本身是评测揭示的结论（小而同质语料下增益有限）。
> 真实增益需扩大语料 + 核准 gold 后复跑，详见「后续」。

**检索层（页级金标准，10 题）**

| 配置 | 页级 Recall@5 | 页级 Recall@10 | 页级 MRR |
|---|---|---|---|
| 纯向量 (dense) | 0.475 | 0.592 | 0.600 |
| 纯向量+重排 | 0.425 | 0.525 | 0.492 |
| 混合 (hybrid) | 0.450 | 0.508 | 0.575 |
| 混合+重排 (full) | 0.433 | 0.467 | 0.451 |

**生成层（RAGAS 风格，节选 2 题）**

| 配置 | 上下文召回 | 忠实度 | 答案相关性 |
|---|---|---|---|
| 纯向量 (dense) | 1.000 | 0.962 | 0.894 |
| 纯向量+重排 | 1.000 | 0.833 | 0.882 |
| 混合 (hybrid) | 1.000 | 0.643 | 0.855 |
| 混合+重排 (full) | 1.000 | 1.000 | 0.868 |

后续：① 扩充评测语料至 20~30 篇异质论文；② 人工核准页级 gold；③ 全量复跑出可信对比表。

## 进度

- [x] P0 骨架 · [x] P1 MVP（端到端） · [x] P2 检索增强（混合+重排+引用）
- [x] P3 评测驱动（框架完成，结果待语料扩充后复跑）
- [x] P4 产品化（FastAPI SSE + Streamlit 聊天界面 + 多文档对比）
- [ ] P5 工程化（CI + Docker + 简历段）

"""FastAPI 服务（REST + SSE 流式）。

端点：
- GET  /health        健康检查 + 库内向量数
- POST /ask           非流式：返回 {answer, sources}
- POST /ask/stream    SSE 流式：先逐 token 推送答案，末尾推送 sources

启动：uvicorn src.api:app --reload
"""

from __future__ import annotations

import json

from fastapi import FastAPI
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from src.generate.answerer import answer, answer_stream
from src.retrieve.store import SearchHit, VectorStore

app = FastAPI(title="论文研读 RAG 助手", version="0.1.0")


class AskRequest(BaseModel):
    question: str
    papers: list[str] | None = None  # 限定 paper_id（多篇对比 / 限定来源）
    mode: str | None = None          # dense | hybrid
    use_rerank: bool | None = None


def _sources_json(hits: list[SearchHit]) -> list[dict]:
    out = []
    for i, h in enumerate(hits, start=1):
        p = h.payload
        out.append({
            "index": i,
            "paper_id": p.get("paper_id"),
            "title": p.get("title"),
            "section": p.get("section"),
            "page": p.get("page"),
            "score": round(h.score, 4),
        })
    return out


@app.get("/health")
def health() -> dict:
    try:
        count = VectorStore().count()
    except Exception as e:
        return {"status": "degraded", "error": str(e)}
    return {"status": "ok", "vectors": count}


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    res = answer(req.question, paper_ids=req.papers, mode=req.mode, use_rerank=req.use_rerank)
    return {"answer": res.answer, "sources": _sources_json(res.sources)}


@app.post("/ask/stream")
async def ask_stream(req: AskRequest):
    stream, hits = answer_stream(
        req.question, paper_ids=req.papers, mode=req.mode, use_rerank=req.use_rerank
    )

    async def event_gen():
        for delta in stream:  # 逐 token
            yield {"event": "token", "data": delta}
        yield {"event": "sources", "data": json.dumps(_sources_json(hits), ensure_ascii=False)}
        yield {"event": "done", "data": ""}

    return EventSourceResponse(event_gen())

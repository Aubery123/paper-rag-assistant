"""重排序（阿里百炼 gte-rerank-v2，原生 API）。

对 hybrid 召回的候选用 cross-encoder 式重排打分，提升 top 命中精度。
百炼重排是 DashScope 原生接口（非 OpenAI 兼容端点），用 httpx 直连。
"""

from __future__ import annotations

import httpx

from src.config import settings
from src.retrieve.store import SearchHit


def rerank(
    query: str, hits: list[SearchHit], top_n: int | None = None
) -> list[SearchHit]:
    """对候选 hits 重排，返回按相关度降序的 top_n（score 替换为重排分）。"""
    if not hits:
        return []
    top_n = top_n or settings.rerank_top_k
    documents = [h.payload.get("text", "") for h in hits]

    resp = httpx.post(
        settings.rerank_url,
        headers={
            "Authorization": f"Bearer {settings.dashscope_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": settings.reranker_model,
            "input": {"query": query, "documents": documents},
            "parameters": {"return_documents": False, "top_n": top_n},
        },
        timeout=30.0,
    )
    resp.raise_for_status()
    results = resp.json()["output"]["results"]

    reranked: list[SearchHit] = []
    for r in results:
        hit = hits[r["index"]]
        reranked.append(SearchHit(score=r["relevance_score"], payload=hit.payload))
    return reranked


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from src.retrieve.hybrid import hybrid_search

    q = sys.argv[1] if len(sys.argv) > 1 else "FLASH 如何在服务器端做漂移感知优化？"
    cands = hybrid_search(q, top_k=settings.retrieve_top_k)
    print(f"Q: {q}\n--- 重排前 (hybrid top{len(cands)}) ---")
    for i, h in enumerate(cands[:8], 1):
        print(f"  [{i}] {h.payload['paper_id']:12} p{h.payload['page']:<3} {h.score:.4f}")
    print("--- 重排后 ---")
    for i, h in enumerate(rerank(q, cands), 1):
        print(f"  [{i}] {h.payload['paper_id']:12} p{h.payload['page']:<3} {h.score:.4f} | {h.payload.get('section','')[:30]}")

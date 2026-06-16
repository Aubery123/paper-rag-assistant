"""混合检索：BM25 关键词 + 向量语义 + RRF 融合。

- 向量路：百炼 embedding + Qdrant 余弦检索（语义相似）；
- 关键词路：rank-bm25 在全量 chunk 上做 BM25（精确术语命中，如 FedProx/GAT/概念漂移英文词）；
- 融合：RRF（Reciprocal Rank Fusion），按各路排名倒数加权，融合分 = Σ 1/(rrf_k + rank)。

BM25 索引在进程内按 collection 懒加载并缓存（论文级语料量很小，重建成本低）。
参数（top_k、rrf_k）取自 config。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from src.config import settings
from src.embeddings import get_embedder
from src.retrieve.store import SearchHit, VectorStore

_TOKEN_RE = re.compile(r"[A-Za-z0-9一-鿿]+")


def _tokenize(text: str) -> list[str]:
    """简单分词：英文/数字按词，中文按单字（与英文语料混检足够用）。"""
    tokens: list[str] = []
    for tok in _TOKEN_RE.findall(text.lower()):
        if re.fullmatch(r"[一-鿿]+", tok):
            tokens.extend(list(tok))  # 中文逐字
        else:
            tokens.append(tok)
    return tokens


@dataclass
class _BM25Index:
    bm25: BM25Okapi
    payloads: list[dict]


_cache: dict[str, _BM25Index] = {}


def _get_bm25(store: VectorStore) -> _BM25Index:
    """懒加载并缓存某 collection 的 BM25 索引。"""
    key = store.collection
    if key not in _cache:
        payloads = store.scroll_all()
        corpus = [_tokenize(p.get("text", "")) for p in payloads]
        _cache[key] = _BM25Index(bm25=BM25Okapi(corpus), payloads=payloads)
    return _cache[key]


def invalidate_cache() -> None:
    """重新入库后调用，使 BM25 缓存失效。"""
    _cache.clear()


def _rrf(rankings: list[list[str]], rrf_k: int) -> dict[str, float]:
    """RRF 融合：输入若干条已排序的 key 列表（best first），输出 key→融合分。"""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking):
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
    return scores


def hybrid_search(
    question: str,
    top_k: int | None = None,
    paper_ids: list[str] | None = None,
    recall_k: int | None = None,
) -> list[SearchHit]:
    """混合检索：返回 RRF 融合后的 top_k 命中。

    recall_k：每路召回数量（默认 settings.retrieve_top_k）。
    top_k：融合后返回数量（默认 settings.retrieve_top_k）。
    """
    recall = recall_k or settings.retrieve_top_k
    final_k = top_k or settings.retrieve_top_k
    store = VectorStore()

    # --- 向量路 ---
    qvec = get_embedder().encode_dense([question])[0]
    dense_hits = store.search_dense(qvec, top_k=recall, paper_ids=paper_ids)
    dense_rank = [h.payload["chunk_id"] for h in dense_hits]

    # --- BM25 路 ---
    idx = _get_bm25(store)
    pid_set = set(paper_ids) if paper_ids else None
    scored = list(zip(idx.bm25.get_scores(_tokenize(question)), idx.payloads))
    scored.sort(key=lambda x: x[0], reverse=True)
    bm25_rank, n = [], 0
    for score, payload in scored:
        if score <= 0:
            break
        if pid_set and payload.get("paper_id") not in pid_set:
            continue
        bm25_rank.append(payload["chunk_id"])
        n += 1
        if n >= recall:
            break

    # --- RRF 融合 ---
    fused = _rrf([dense_rank, bm25_rank], settings.rrf_k)

    # payload 映射（两路并集）
    pmap = {h.payload["chunk_id"]: h.payload for h in dense_hits}
    for payload in idx.payloads:
        pmap.setdefault(payload["chunk_id"], payload)

    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:final_k]
    return [SearchHit(score=s, payload=pmap[cid]) for cid, s in ranked]


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    q = sys.argv[1] if len(sys.argv) > 1 else "联邦学习中的概念漂移有哪些处理方法？"
    print("Q:", q, "\n")
    for i, h in enumerate(hybrid_search(q, top_k=8), 1):
        p = h.payload
        print(f"[{i}] {p['paper_id']:12} p{p['page']:<3} score={h.score:.4f} | {p.get('section','')[:30]}")

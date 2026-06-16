"""RAGAS 风格指标（自实现，全程走百炼 API）。

- context_recall：应命中论文中，被检索到的比例（确定性计算，无需 LLM）；
- faithfulness：答案的原子陈述中被检索上下文支持的比例（qwen 判分）；
- answer_relevancy：由答案反推问题，与原问题的平均余弦相似度（qwen 生成 + embedding）。

每个指标返回 [0,1]；数据集层面取均值。
"""

from __future__ import annotations

import json
import re

import numpy as np

from src.embeddings import get_embedder
from src.generate.llm import LLMClient
from src.prompts.eval import (
    FAITHFULNESS_SYSTEM,
    FAITHFULNESS_USER,
    RELEVANCY_SYSTEM,
    RELEVANCY_USER,
)
from src.retrieve.store import SearchHit


def _parse_json(text: str) -> dict:
    """从 LLM 输出中稳健提取 JSON 对象。"""
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {}


# ============ 检索层指标（确定性，无 LLM，篇级判据） ============

def recall_at_k(hits: list[SearchHit], expected_paper_ids: list[str], k: int) -> float:
    """Recall@k：应命中论文中，在前 k 个命中里出现的比例。"""
    if not expected_paper_ids:
        return 1.0
    topk_papers = {h.payload.get("paper_id") for h in hits[:k]}
    covered = sum(1 for p in expected_paper_ids if p in topk_papers)
    return covered / len(expected_paper_ids)


def hit_at_k(hits: list[SearchHit], expected_paper_ids: list[str], k: int) -> float:
    """Hit@k：前 k 个命中里是否至少出现一篇应命中论文（1/0）。"""
    if not expected_paper_ids:
        return 1.0
    topk_papers = {h.payload.get("paper_id") for h in hits[:k]}
    return 1.0 if any(p in topk_papers for p in expected_paper_ids) else 0.0


def mrr(hits: list[SearchHit], expected_paper_ids: list[str]) -> float:
    """MRR：第一个命中应命中论文的位次的倒数（衡量排序质量）。"""
    if not expected_paper_ids:
        return 1.0
    exp = set(expected_paper_ids)
    for rank, h in enumerate(hits, start=1):
        if h.payload.get("paper_id") in exp:
            return 1.0 / rank
    return 0.0


def _hit_pair(h: SearchHit) -> tuple[str, int]:
    return (h.payload.get("paper_id"), int(h.payload.get("page", -1)))


def page_recall_at_k(hits: list[SearchHit], gold_pages: set, k: int) -> float:
    """页级 Recall@k：gold (篇,页) 中在前 k 个命中里出现的比例。"""
    if not gold_pages:
        return 1.0
    topk = {_hit_pair(h) for h in hits[:k]}
    covered = sum(1 for g in gold_pages if g in topk)
    return covered / len(gold_pages)


def page_mrr(hits: list[SearchHit], gold_pages: set) -> float:
    """页级 MRR：第一个命中 gold (篇,页) 的位次倒数（衡量精确定位与排序）。"""
    if not gold_pages:
        return 1.0
    for rank, h in enumerate(hits, start=1):
        if _hit_pair(h) in gold_pages:
            return 1.0 / rank
    return 0.0


# ============ 生成层指标（RAGAS 风格，LLM-as-judge） ============

def context_recall(expected_paper_ids: list[str], hits: list[SearchHit]) -> float:
    """上下文召回（篇级）：应命中论文中被检索到的比例。"""
    if not expected_paper_ids:
        return 1.0
    got = {h.payload.get("paper_id") for h in hits}
    covered = sum(1 for p in expected_paper_ids if p in got)
    return covered / len(expected_paper_ids)


def faithfulness(answer: str, hits: list[SearchHit], llm: LLMClient | None = None) -> float:
    """忠实度：答案原子陈述被上下文支持的比例。"""
    if not answer.strip() or not hits:
        return 0.0
    llm = llm or LLMClient()
    context = "\n\n".join(h.payload.get("text", "") for h in hits)
    raw = llm.chat(
        [
            {"role": "system", "content": FAITHFULNESS_SYSTEM},
            {"role": "user", "content": FAITHFULNESS_USER.format(context=context, answer=answer)},
        ],
        temperature=0.0,
    )
    stmts = _parse_json(raw).get("statements", [])
    if not stmts:
        return 0.0
    supported = sum(1 for s in stmts if s.get("supported"))
    return supported / len(stmts)


def answer_relevancy(question: str, answer: str, llm: LLMClient | None = None) -> float:
    """答案相关性：反推问题与原问题的平均余弦相似度。"""
    if not answer.strip():
        return 0.0
    llm = llm or LLMClient()
    raw = llm.chat(
        [
            {"role": "system", "content": RELEVANCY_SYSTEM},
            {"role": "user", "content": RELEVANCY_USER.format(answer=answer)},
        ],
        temperature=0.0,
    )
    gen_qs = _parse_json(raw).get("questions", [])
    if not gen_qs:
        return 0.0
    embedder = get_embedder()
    vecs = embedder.encode_dense([question] + gen_qs)  # 已归一化
    q0, gens = vecs[0], vecs[1:]
    sims = [float(np.dot(q0, g)) for g in gens]  # 归一化向量点积=余弦
    return float(np.mean(sims))

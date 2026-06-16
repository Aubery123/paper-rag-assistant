"""带引用溯源作答。

流程：问题 → embedding → 向量检索 top_k → 拼装带编号资料 → LLM 作答。
返回答案文本与来源列表（含 paper_id/section/page），供引用展示。

P1 为基础向量检索；P2 接入 hybrid → rerank 后，仅需替换这里的检索调用。
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from src.config import settings
from src.embeddings import get_embedder
from src.generate.llm import LLMClient
from src.prompts.answer import ANSWER_SYSTEM, ANSWER_USER, CONTEXT_ITEM
from src.retrieve.store import SearchHit, VectorStore


@dataclass
class AnswerResult:
    """作答结果：答案 + 来源。"""

    answer: str
    sources: list[SearchHit]


def _retrieve(question: str, top_k: int, paper_ids: list[str] | None) -> list[SearchHit]:
    qvec = get_embedder().encode_dense([question])[0]
    return VectorStore().search_dense(qvec, top_k=top_k, paper_ids=paper_ids)


def _build_messages(question: str, hits: list[SearchHit]) -> list[dict]:
    items = []
    for i, h in enumerate(hits, start=1):
        p = h.payload
        items.append(
            CONTEXT_ITEM.format(
                n=i,
                title=p.get("title", ""),
                section=p.get("section", "") or "正文",
                page=p.get("page", "?"),
                text=p.get("text", "").strip(),
            )
        )
    context = "\n\n".join(items)
    return [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user", "content": ANSWER_USER.format(context=context, question=question)},
    ]


def answer(
    question: str, top_k: int | None = None, paper_ids: list[str] | None = None
) -> AnswerResult:
    """非流式作答。"""
    hits = _retrieve(question, top_k or settings.rerank_top_k, paper_ids)
    if not hits:
        return AnswerResult(answer="知识库为空或未检索到相关内容。", sources=[])
    text = LLMClient().chat(_build_messages(question, hits))
    return AnswerResult(answer=text, sources=hits)


def answer_stream(
    question: str, top_k: int | None = None, paper_ids: list[str] | None = None
) -> tuple[Iterator[str], list[SearchHit]]:
    """流式作答：返回 (文本增量迭代器, 来源列表)。"""
    hits = _retrieve(question, top_k or settings.rerank_top_k, paper_ids)
    if not hits:
        return iter(["知识库为空或未检索到相关内容。"]), []
    return LLMClient().chat_stream(_build_messages(question, hits)), hits


def format_sources(hits: list[SearchHit]) -> str:
    """把来源列表格式化为可读的引用清单。"""
    lines = []
    for i, h in enumerate(hits, start=1):
        p = h.payload
        lines.append(
            f"[{i}] 《{p.get('title','')}》 · {p.get('section','') or '正文'} "
            f"· 第{p.get('page','?')}页 (score={h.score:.3f})"
        )
    return "\n".join(lines)

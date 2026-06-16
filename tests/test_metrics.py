"""评测指标单测：确定性检索指标（不依赖 LLM/网络）。"""

from src.eval.metrics import (
    context_recall,
    hit_at_k,
    mrr,
    page_mrr,
    page_recall_at_k,
    recall_at_k,
)
from src.retrieve.store import SearchHit


def _hit(paper_id: str, page: int = 1, score: float = 1.0) -> SearchHit:
    return SearchHit(score=score, payload={"paper_id": paper_id, "page": page})


def test_recall_and_hit_at_k():
    hits = [_hit("A", 1), _hit("B", 2), _hit("C", 3)]
    assert recall_at_k(hits, ["A", "B"], k=3) == 1.0
    assert recall_at_k(hits, ["A", "D"], k=3) == 0.5
    assert hit_at_k(hits, ["D"], k=3) == 0.0
    assert hit_at_k(hits, ["C"], k=3) == 1.0
    # 截断到 k=1 只看第一个
    assert recall_at_k(hits, ["B"], k=1) == 0.0


def test_mrr_paper_level():
    hits = [_hit("X", 1), _hit("A", 2)]
    assert mrr(hits, ["A"]) == 0.5      # A 在第 2 位
    assert mrr(hits, ["X"]) == 1.0      # X 在第 1 位
    assert mrr(hits, ["Z"]) == 0.0      # 未命中


def test_page_level_metrics():
    hits = [_hit("A", 5), _hit("A", 3), _hit("B", 2)]
    gold = {("A", 3), ("B", 2)}
    # top2 含 (A,5),(A,3) → 命中 (A,3)，未命中 (B,2)
    assert page_recall_at_k(hits, gold, k=2) == 0.5
    assert page_recall_at_k(hits, gold, k=3) == 1.0
    # 第一个命中 gold 的是 (A,3) 在第 2 位
    assert page_mrr(hits, gold) == 0.5


def test_context_recall():
    hits = [_hit("A"), _hit("B")]
    assert context_recall(["A", "B"], hits) == 1.0
    assert context_recall(["A", "C"], hits) == 0.5
    assert context_recall([], hits) == 1.0  # 无期望时视为满分

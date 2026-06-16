"""检索辅助逻辑单测：RRF 融合 + 分词（纯逻辑，不依赖网络）。"""

from src.retrieve.hybrid import _rrf, _tokenize


def test_tokenize_mixed_zh_en():
    toks = _tokenize("FedProx 联邦学习 GAT")
    assert "fedprox" in toks      # 英文小写整词
    assert "gat" in toks
    assert "联" in toks and "邦" in toks  # 中文逐字


def test_rrf_basic_ordering():
    # 两路排名：A 在两路都靠前 → 融合分最高
    dense = ["A", "B", "C"]
    bm25 = ["A", "C", "D"]
    scores = _rrf([dense, bm25], rrf_k=60)
    assert scores["A"] == max(scores.values())
    # 同时出现在两路的项，分数高于只出现一路的
    assert scores["A"] > scores["B"]
    assert scores["C"] > scores["D"]


def test_rrf_rank_weighting():
    # 单路，排名越靠前分越高
    scores = _rrf([["X", "Y", "Z"]], rrf_k=60)
    assert scores["X"] > scores["Y"] > scores["Z"]

"""混合检索：BM25 关键词 + 向量语义 + RRF 融合。

职责：并行跑稀疏与稠密检索，用 RRF（Reciprocal Rank Fusion）融合排名。

TODO(P2): 实现 BM25 + 向量双路召回 + RRF 融合，参数（k、权重）取自 config。
"""

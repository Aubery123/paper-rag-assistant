"""Embedding + 写入 Qdrant。

职责：调用 embeddings 生成向量（及 BM25 稀疏表示），连同元数据写入 Qdrant collection。

TODO(P1): 批量 embedding + upsert 到 Qdrant（dense + sparse 向量）。
"""

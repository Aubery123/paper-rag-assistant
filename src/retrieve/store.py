"""Qdrant 封装（向量 + BM25 稀疏索引）。

职责：collection 创建/管理，dense + sparse 向量的 upsert 与查询。

TODO(P1): 封装 QdrantClient，建 collection（含稀疏向量配置），提供 upsert / search。
"""

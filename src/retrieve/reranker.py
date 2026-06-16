"""重排序（bge-reranker-v2-m3）。

职责：对 hybrid 召回的候选用 cross-encoder 精排，提升 top 命中。

TODO(P2): 加载 bge-reranker-v2-m3，对 (query, chunk) 打分并重排，截取 top_k。
"""

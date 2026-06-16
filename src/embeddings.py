"""文本嵌入封装（阿里百炼 Embedding API）。

走百炼 OpenAI 兼容接口（与 LLM 同一 key / base_url），默认 text-embedding-v3。
不在本地加载模型——纯 API 调用。

百炼兼容接口单次 input 最多 10 条，这里自动分批。
"""

from __future__ import annotations

import numpy as np
from openai import OpenAI

from src.config import settings


class Embedder:
    """百炼文本嵌入客户端。"""

    def __init__(self, model: str | None = None, dim: int | None = None):
        self.model = model or settings.embedding_model
        self.dim = dim or settings.embedding_dim
        self.client = OpenAI(
            api_key=settings.dashscope_api_key, base_url=settings.llm_base_url
        )

    def encode_dense(
        self, texts: list[str], batch_size: int | None = None
    ) -> np.ndarray:
        """编码为稠密向量，返回 (n, dim) float32 数组。自动分批。"""
        bs = batch_size or settings.embedding_batch
        vecs: list[list[float]] = []
        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            resp = self.client.embeddings.create(
                model=self.model,
                input=batch,
                dimensions=self.dim,
                encoding_format="float",
            )
            # 按 index 排序以确保顺序与输入一致
            for item in sorted(resp.data, key=lambda d: d.index):
                vecs.append(item.embedding)
        return np.asarray(vecs, dtype=np.float32)


_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    """进程内单例。"""
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder

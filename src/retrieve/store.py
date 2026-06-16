"""Qdrant 封装（向量 + BM25 稀疏索引）。

collection 同时配置命名向量：
- "dense"：BGE-m3 稠密向量（COSINE），P1 向量检索用；
- "sparse"：稀疏向量，P2 混合检索（BM25 替身）用，现在先建好不影响 P1。

point id 由 chunk_id 派生（UUID5），保证可幂等重写（重复 ingest 不产生重复点）。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from qdrant_client import QdrantClient, models

from src.config import settings

_NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")


def _point_id(chunk_id: str) -> str:
    """由 chunk_id 生成稳定 UUID（幂等 upsert）。"""
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


@dataclass
class SearchHit:
    """检索命中结果。"""

    score: float
    payload: dict  # 含 text 与 {paper_id,title,section,page,chunk_id}


class VectorStore:
    """Qdrant collection 的薄封装。"""

    def __init__(self, url: str | None = None, collection: str | None = None):
        self.collection = collection or settings.qdrant_collection
        self.client = QdrantClient(url=url or settings.qdrant_url)

    def ensure_collection(self, recreate: bool = False) -> None:
        """创建 collection（含 dense + sparse 向量配置）。recreate=True 先删后建。"""
        exists = self.client.collection_exists(self.collection)
        if exists and not recreate:
            return
        if exists and recreate:
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                "dense": models.VectorParams(
                    size=settings.embedding_dim, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
        )

    def upsert(self, payloads: list[dict], dense_vecs, sparse=None) -> int:
        """写入一批点。payloads[i] 对应 dense_vecs[i]（可选 sparse[i]）。"""
        points = []
        for i, payload in enumerate(payloads):
            vector: dict = {"dense": dense_vecs[i].tolist()}
            if sparse is not None:
                sv = sparse[i]
                vector["sparse"] = models.SparseVector(
                    indices=list(sv.keys()), values=list(sv.values())
                )
            points.append(
                models.PointStruct(
                    id=_point_id(payload["chunk_id"]), vector=vector, payload=payload
                )
            )
        self.client.upsert(collection_name=self.collection, points=points)
        return len(points)

    def search_dense(
        self, query_vec, top_k: int | None = None, paper_ids: list[str] | None = None
    ) -> list[SearchHit]:
        """稠密向量检索；可按 paper_id 过滤（多篇对比/限定来源）。"""
        flt = None
        if paper_ids:
            flt = models.Filter(
                must=[
                    models.FieldCondition(
                        key="paper_id", match=models.MatchAny(any=paper_ids)
                    )
                ]
            )
        res = self.client.query_points(
            collection_name=self.collection,
            query=query_vec.tolist() if hasattr(query_vec, "tolist") else query_vec,
            using="dense",
            limit=top_k or settings.retrieve_top_k,
            with_payload=True,
            query_filter=flt,
        )
        return [SearchHit(score=p.score, payload=p.payload) for p in res.points]

    def count(self) -> int:
        return self.client.count(self.collection).count

"""Embedding + 写入 Qdrant。

把离线入库串起来：load(PDF) → chunk → 百炼 Embedding API → upsert 到 Qdrant。
"""

from __future__ import annotations

from pathlib import Path

from src.embeddings import get_embedder
from src.ingest.chunker import chunk_papers
from src.ingest.loader import load_dir, load_pdf
from src.retrieve.store import VectorStore


def ingest(source: str | Path, recreate: bool = True) -> dict:
    """导入论文到向量库。

    source 为 PDF 文件或包含 PDF 的目录。recreate=True 时重建 collection。
    返回统计字典。
    """
    source = Path(source)
    papers = [load_pdf(source)] if source.suffix.lower() == ".pdf" else load_dir(source)
    if not papers:
        raise FileNotFoundError(f"未找到 PDF：{source}")

    chunks = chunk_papers(papers)
    texts = [c.text for c in chunks]

    embedder = get_embedder()
    dense_vecs = embedder.encode_dense(texts)

    store = VectorStore()
    store.ensure_collection(recreate=recreate)
    n = store.upsert([c.to_payload() for c in chunks], dense_vecs)

    return {
        "papers": len(papers),
        "chunks": len(chunks),
        "upserted": n,
        "collection_count": store.count(),
        "paper_ids": [p.paper_id for p in papers],
    }


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "data/papers"
    print("开始导入：", target)
    stats = ingest(target)
    print("完成：", stats)

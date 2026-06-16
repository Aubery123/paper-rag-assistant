"""结构感知递归切分 + 元数据。

把 loader 产出的逐页文本切成检索粒度的 chunk。设计要点：

- **按页切分**：每个 chunk 只归属一页，使引用页码绝对精确（贴合"引用溯源"目标）；
- **递归分隔符**：段落(\\n\\n) → 行(\\n) → 句(. ) → 字符，尽量在自然边界断开；
- 每个 chunk 继承所在页的 {paper_id, title, section, page}，并附唯一 chunk_id；
- 相邻 chunk 间带字符级重叠（chunk_overlap）以保上下文；过短尾块并入前块。

参数取自 config.settings（chunk_size / chunk_overlap）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.config import settings
from src.ingest.loader import ParsedPaper, PageContent

_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass
class Chunk:
    """检索粒度的文本块（含引用溯源元数据）。"""

    chunk_id: str
    paper_id: str
    title: str
    section: str
    page: int
    text: str

    def to_payload(self) -> dict:
        """写入向量库时的 payload（含正文与元数据）。"""
        return {
            "chunk_id": self.chunk_id,
            "paper_id": self.paper_id,
            "title": self.title,
            "section": self.section,
            "page": self.page,
            "text": self.text,
        }


def _recursive_split(text: str, size: int, seps: list[str]) -> list[str]:
    """递归按分隔符切分，使每片尽量 <= size。返回非空片段（不含分隔符）。"""
    sep, rest = seps[0], seps[1:]
    if sep == "":  # 最底层：硬切字符
        return [text[i : i + size] for i in range(0, len(text), size)]

    pieces: list[str] = []
    for part in text.split(sep):
        part = part.strip()
        if not part:
            continue
        if len(part) <= size or not rest:
            pieces.append(part)
        else:
            pieces.extend(_recursive_split(part, size, rest))
    return pieces


def _pack(pieces: list[str], size: int, overlap: int, joiner: str = " ") -> list[str]:
    """贪心打包片段到 <= size 的 chunk，并在相邻 chunk 间加字符级重叠。"""
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for piece in pieces:
        add = len(piece) + (len(joiner) if cur else 0)
        if cur and cur_len + add > size:
            chunks.append(joiner.join(cur))
            cur, cur_len = [], 0
        cur.append(piece)
        cur_len += len(piece) + (len(joiner) if len(cur) > 1 else 0)
    if cur:
        chunks.append(joiner.join(cur))

    if not chunks:
        return chunks

    # 过短尾块并入前一块
    if len(chunks) > 1 and len(chunks[-1]) < size * 0.3:
        chunks[-2] = chunks[-2] + joiner + chunks[-1]
        chunks.pop()

    # 加重叠：把前一块末尾 overlap 个字符（按词边界回退）前置到当前块
    if overlap > 0 and len(chunks) > 1:
        out = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-overlap:]
            tail = tail[tail.find(" ") + 1 :] if " " in tail else tail  # 不从半个词开始
            out.append((tail + joiner + chunks[i]).strip())
        chunks = out
    return chunks


def _chunk_page(pc: PageContent, size: int, overlap: int) -> list[str]:
    """切分单页文本。"""
    text = re.sub(r"[ \t]+", " ", pc.text).strip()
    if not text:
        return []
    pieces = _recursive_split(text, size, _SEPARATORS)
    return _pack(pieces, size, overlap)


def chunk_paper(
    paper: ParsedPaper,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """把一篇论文切成 Chunk 列表（按页，继承页的元数据）。"""
    size = chunk_size or settings.chunk_size
    overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

    chunks: list[Chunk] = []
    idx = 0
    for pc in paper.pages:
        for piece in _chunk_page(pc, size, overlap):
            chunks.append(
                Chunk(
                    chunk_id=f"{paper.paper_id}::{idx}",
                    paper_id=paper.paper_id,
                    title=paper.title,
                    section=pc.section,
                    page=pc.page,
                    text=piece,
                )
            )
            idx += 1
    return chunks


def chunk_papers(papers: list[ParsedPaper], **kw) -> list[Chunk]:
    """批量切分多篇论文。"""
    out: list[Chunk] = []
    for p in papers:
        out.extend(chunk_paper(p, **kw))
    return out


if __name__ == "__main__":
    # 快速检查：python -m src.ingest.chunker [path]
    import sys
    from src.ingest.loader import load_pdf, load_dir
    from pathlib import Path

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows 控制台兜底

    target = sys.argv[1] if len(sys.argv) > 1 else "data/papers"
    tp = Path(target)
    papers = [load_pdf(tp)] if tp.suffix == ".pdf" else load_dir(tp)
    for paper in papers:
        chunks = chunk_paper(paper)
        sizes = [len(c.text) for c in chunks]
        avg = sum(sizes) / len(sizes) if sizes else 0
        print(
            f"{paper.paper_id:14} pages={paper.n_pages:3} chunks={len(chunks):3} "
            f"avg_len={avg:6.0f} min={min(sizes, default=0)} max={max(sizes, default=0)}"
        )
    # 抽样展示一篇的前两个 chunk
    if papers:
        sample = chunk_paper(papers[0])[:2]
        for c in sample:
            print(f"\n--- {c.chunk_id} | p{c.page} | [{c.section[:30]}] ---")
            print(c.text[:300])

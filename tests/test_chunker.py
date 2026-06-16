"""chunker 单测（纯逻辑，不依赖网络/模型）。"""

from src.config import settings
from src.ingest.chunker import _pack, _recursive_split, chunk_paper
from src.ingest.loader import PageContent, ParsedPaper


def _make_paper(text: str, pages: int = 1) -> ParsedPaper:
    pcs = [
        PageContent(paper_id="P", title="T", section=f"S{i}", page=i, text=text)
        for i in range(1, pages + 1)
    ]
    return ParsedPaper(paper_id="P", title="T", n_pages=pages, pages=pcs)


def test_recursive_split_respects_size():
    text = "。".join(f"句子{i}" * 20 for i in range(20))
    pieces = _recursive_split(text, 100, ["\n\n", "\n", "。", " ", ""])
    # 递归切分后，绝大多数片段不超过 size（含分隔后的硬切兜底）
    assert all(len(p) <= 100 for p in pieces)


def test_pack_overlap_and_size():
    pieces = [f"piece{i}" * 10 for i in range(10)]  # 每片 60 字符
    chunks = _pack(pieces, size=200, overlap=20)
    assert len(chunks) >= 2
    # 重叠：从第二块起，开头应包含上一块尾部内容
    assert chunks[0][-10:] in chunks[1] or len(chunks) == 1


def test_chunk_paper_metadata_and_pages():
    paper = _make_paper("这是一段测试正文。" * 100, pages=3)
    chunks = chunk_paper(paper, chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 0
    # 元数据齐全 + chunk_id 唯一
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    for c in chunks:
        assert c.paper_id == "P" and c.title == "T"
        assert 1 <= c.page <= 3
        assert c.section.startswith("S")
        assert set(c.to_payload()) == {"chunk_id", "paper_id", "title", "section", "page", "text"}


def test_empty_page_skipped():
    paper = _make_paper("", pages=2)
    assert chunk_paper(paper) == []

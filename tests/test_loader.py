"""loader 测试：用 data/papers 下任一 PDF 验证解析；无 PDF 则自动跳过（CI 无样本）。"""

from pathlib import Path

import pytest

from src.config import DATA_DIR
from src.ingest.loader import ParsedPaper, load_pdf

_PDFS = sorted((DATA_DIR / "papers").glob("*.pdf")) if (DATA_DIR / "papers").exists() else []


@pytest.mark.skipif(not _PDFS, reason="data/papers 下无 PDF 样本（CI 环境跳过）")
def test_load_pdf_basic():
    paper: ParsedPaper = load_pdf(_PDFS[0])

    # 基本结构
    assert paper.n_pages > 0
    assert len(paper.pages) == paper.n_pages
    assert paper.title and paper.title != ""

    # 页码连续、从 1 开始
    assert [p.page for p in paper.pages] == list(range(1, paper.n_pages + 1))

    # 每页都带齐引用溯源所需元数据
    for pc in paper.pages:
        meta = pc.metadata()
        assert set(meta) == {"paper_id", "title", "section", "page"}
        assert meta["paper_id"] == paper.paper_id

    # 至少有页解析出非空文本
    assert any(p.text.strip() for p in paper.pages)

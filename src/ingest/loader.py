"""PDF / arXiv 解析（PyMuPDF）。

职责：把论文 PDF 解析为带页码、章节结构的逐页文本，供下游切分与引用溯源使用。
每条产出携带元数据：{paper_id, title, section, page}。

章节识别策略（两级兜底）：
1. 优先用 PDF 内置目录 `get_toc()`，建立「页 → 当前章节」映射（多数论文有）；
2. 无 TOC 时，用正则在正文里识别章节标题行（如 "3 Method" / "3.1 ..."）兜底。

标题解析优先级：TOC 一级条目 > PDF 元数据 title > 首页最大字号文本行 > 文件名。

注意（P0 已知局限）：章节按「页起始处生效的章节」归属，同一页内跨章节的情形
暂按整页归到该页起始章节；更细的章节边界留待 chunker(P1) 结合页内标题位置细化。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF


# ---- 章节标题正则（无 TOC 时兜底）----
# 形如 "3 Method" / "3.1 Weight Normalization" / "IV. Experiments"
_NUM_HEADING_RE = re.compile(
    r"^\s*((?:\d+\.)*\d+|[IVXLC]+)\.?\s+([A-Z][A-Za-z][\w \-:/]{1,70})\s*$"
)
# 常见无编号章节名
_NAMED_HEADINGS = {
    "abstract", "introduction", "related work", "background", "method",
    "methods", "methodology", "approach", "experiments", "evaluation",
    "results", "discussion", "conclusion", "conclusions", "references",
    "appendix", "acknowledgments", "acknowledgements",
}


@dataclass
class PageContent:
    """单页解析结果。"""

    paper_id: str
    title: str
    section: str  # 该页起始处生效的章节标题；未知为 ""
    page: int     # 1-indexed 页码
    text: str

    def metadata(self) -> dict:
        """供下游 chunk 复用的元数据。"""
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "section": self.section,
            "page": self.page,
        }


@dataclass
class ParsedPaper:
    """整篇论文解析结果。"""

    paper_id: str
    title: str
    n_pages: int
    pages: list[PageContent] = field(default_factory=list)


def _toc_structure(toc: list) -> tuple[bool, int]:
    """判断 TOC 结构。

    返回 (single_root, section_min_level)：
    - 恰有 1 个一级条目 → 它是论文标题，章节从 level>=2 起；
    - 有多个一级条目 → 一级条目本身就是章节，章节从 level>=1 起。
    """
    l1 = [name for lvl, name, _ in toc if lvl == 1]
    single_root = len(l1) == 1
    return single_root, (2 if single_root else 1)


def _resolve_title(doc: fitz.Document, toc: list, paper_id: str) -> str:
    """按优先级解析论文标题。"""
    single_root, _ = _toc_structure(toc)
    # 1) 仅当 TOC 有单一标题根节点时，用它作标题
    if single_root:
        for level, name, _page in toc:
            if level == 1 and name and len(name.strip()) > 4:
                return name.strip()
    # 2) PDF 元数据
    meta_title = (doc.metadata or {}).get("title") or ""
    if len(meta_title.strip()) > 4:
        return meta_title.strip()
    # 3) 首页最大字号文本行
    big = _largest_font_line(doc[0]) if doc.page_count else ""
    if big:
        return big
    # 4) 兜底：文件名
    return paper_id


def _largest_font_line(page: fitz.Page) -> str:
    """返回页面上最大字号的那一行文本（用于猜标题）。"""
    best_size, best_text = 0.0, ""
    data = page.get_text("dict")
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            size = max(s["size"] for s in spans)
            text = "".join(s["text"] for s in spans).strip()
            if size > best_size and len(text) > 4:
                best_size, best_text = size, text
    return best_text


def _section_map_from_toc(toc: list, n_pages: int) -> dict[int, str]:
    """由 TOC 构建「页码(1-indexed) → 当前章节标题」映射。

    取每页起始处「最近一个起始页 <= 该页」的章节。章节起始层级由 TOC 结构决定：
    有标题根节点时从 level>=2 起，否则从 level>=1 起。
    """
    _, section_min_level = _toc_structure(toc)
    # (start_page, title)，按页排序
    entries = [
        (p, name.strip()) for level, name, p in toc if level >= section_min_level and name
    ]
    entries.sort()
    page2sec: dict[int, str] = {}
    cur = ""
    idx = 0
    for page in range(1, n_pages + 1):
        while idx < len(entries) and entries[idx][0] <= page:
            cur = entries[idx][1]
            idx += 1
        page2sec[page] = cur
    return page2sec


def _detect_heading(text_line: str) -> str | None:
    """判断一行是否为章节标题，是则返回规范化标题，否则 None。"""
    line = text_line.strip()
    if not line or len(line) > 80:
        return None
    m = _NUM_HEADING_RE.match(line)
    if m:
        return f"{m.group(1)} {m.group(2).strip()}"
    if line.lower().strip(" .:") in _NAMED_HEADINGS:
        return line
    return None


def load_pdf(path: str | Path) -> ParsedPaper:
    """解析单个 PDF，返回逐页结构化结果。"""
    path = Path(path)
    paper_id = path.stem
    doc = fitz.open(path)
    try:
        toc = doc.get_toc()
        title = _resolve_title(doc, toc, paper_id)
        page2sec = _section_map_from_toc(toc, doc.page_count) if toc else {}

        pages: list[PageContent] = []
        cur_section = ""  # 无 TOC 时用正则推进
        for pno in range(doc.page_count):
            page = doc[pno]
            text = page.get_text("text").strip()

            if page2sec:  # 有 TOC：直接查表
                section = page2sec.get(pno + 1, "")
            else:         # 无 TOC：正则兜底，沿页面顺序推进当前章节
                section = cur_section
                for line in text.splitlines():
                    h = _detect_heading(line)
                    if h:
                        cur_section = h
                        if not section:  # 本页首个标题作为本页归属
                            section = h
                section = section or cur_section

            pages.append(
                PageContent(
                    paper_id=paper_id,
                    title=title,
                    section=section,
                    page=pno + 1,
                    text=text,
                )
            )
        return ParsedPaper(
            paper_id=paper_id, title=title, n_pages=doc.page_count, pages=pages
        )
    finally:
        doc.close()


def load_dir(directory: str | Path) -> list[ParsedPaper]:
    """解析目录下所有 PDF。"""
    directory = Path(directory)
    papers = [load_pdf(p) for p in sorted(directory.glob("*.pdf"))]
    return papers


if __name__ == "__main__":
    # 快速人工检查：python -m src.ingest.loader [path]
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "data/papers"
    target_path = Path(target)
    items = [load_pdf(target_path)] if target_path.suffix == ".pdf" else load_dir(target_path)
    for paper in items:
        print(f"\n=== {paper.paper_id} | {paper.n_pages}p | {paper.title[:70]} ===")
        # 打印每页归属的章节 + 文本长度
        for pc in paper.pages:
            print(f"  p{pc.page:<2} [{pc.section[:40]:<40}] chars={len(pc.text)}")

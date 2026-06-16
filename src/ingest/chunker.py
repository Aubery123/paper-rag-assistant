"""结构感知递归切分 + 元数据。

职责：把 loader 产出的结构化文本切成检索粒度的 chunk，
每个 chunk 带 {paper_id, title, section, page}，引用溯源全靠它。

TODO(P1): 按章节/段落递归切分，控制 chunk 大小与重叠，保留元数据。
"""

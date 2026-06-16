"""PDF / arXiv 解析（PyMuPDF）。

职责：把论文 PDF 解析为带页码、章节结构的文本块，供下游切分与引用溯源使用。
每个产出单元必须携带元数据：{paper_id, title, section, page}。

TODO(P0): 用 PyMuPDF(fitz) 打开 PDF，逐页提取文本与页码；尽量识别章节标题。
"""

"""CLI 入口：导入论文 / 提问 / 跑评测。

用法（规划）：
    python main.py ingest ./data/papers/
    python main.py ask "这几篇里 attention 的复杂度分别是多少？"
    python main.py eval
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="论文研读 RAG 助手 CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="导入并解析论文 PDF / arXiv")
    p_ingest.add_argument("path", help="PDF 文件或目录路径")

    p_ask = sub.add_parser("ask", help="基于知识库提问（带引用溯源）")
    p_ask.add_argument("question", help="问题文本")

    sub.add_parser("eval", help="跑评测集，输出消融对比表")

    args = parser.parse_args()

    # TODO: 各子命令在后续 Phase 接入对应模块
    raise SystemExit(f"[未实现] command={args.command!r}（P0 骨架阶段）")


if __name__ == "__main__":
    main()

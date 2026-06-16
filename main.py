"""CLI 入口：导入论文 / 提问 / 跑评测。

用法：
    python main.py ingest ./data/papers/
    python main.py ask "这几篇里 attention 的复杂度分别是多少？"
    python main.py ask "FedDrift 怎么处理概念漂移？" --papers FedDrift
    python main.py eval
"""

from __future__ import annotations

import argparse
import sys


def _cmd_ingest(args) -> None:
    from src.ingest.indexer import ingest

    stats = ingest(args.path, recreate=not args.append)
    print("导入完成：", stats)


def _cmd_ask(args) -> None:
    from src.generate.answerer import answer_stream, format_sources

    stream, hits = answer_stream(args.question, paper_ids=args.papers)
    print("\n回答：\n")
    for delta in stream:
        print(delta, end="", flush=True)
    print("\n\n来源：")
    print(format_sources(hits))


def _cmd_eval(args) -> None:
    raise SystemExit("[未实现] eval 将在 P3 接入")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows 控制台兜底

    parser = argparse.ArgumentParser(description="论文研读 RAG 助手 CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="导入并解析论文 PDF / arXiv")
    p_ingest.add_argument("path", help="PDF 文件或目录路径")
    p_ingest.add_argument(
        "--append", action="store_true", help="追加写入（默认重建 collection）"
    )
    p_ingest.set_defaults(func=_cmd_ingest)

    p_ask = sub.add_parser("ask", help="基于知识库提问（带引用溯源）")
    p_ask.add_argument("question", help="问题文本")
    p_ask.add_argument(
        "--papers", nargs="*", default=None, help="限定 paper_id（多篇对比/限定来源）"
    )
    p_ask.set_defaults(func=_cmd_ask)

    p_eval = sub.add_parser("eval", help="跑评测集，输出消融对比表")
    p_eval.set_defaults(func=_cmd_eval)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

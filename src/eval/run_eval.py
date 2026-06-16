"""消融对比实验，出对比表。

对同一评测集跑多种检索配置（纯向量 / +重排 / 混合 / 混合+重排），
计算 RAGAS 风格指标并汇总为 Markdown 对比表（进 README）。
"""

from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path

from src.config import REPORTS_DIR
from src.eval.dataset import EvalSample, load_dataset
from src.eval.metrics import (
    answer_relevancy,
    context_recall,
    faithfulness,
    page_mrr,
    page_recall_at_k,
)
from src.generate.answerer import answer, retrieve_ranked
from src.generate.llm import LLMClient

# 消融配置：检索方式 × 是否重排
CONFIGS: dict[str, dict] = {
    "纯向量 (dense)": {"mode": "dense", "use_rerank": False},
    "纯向量+重排": {"mode": "dense", "use_rerank": True},
    "混合 (hybrid)": {"mode": "hybrid", "use_rerank": False},
    "混合+重排 (full)": {"mode": "hybrid", "use_rerank": True},
}

METRICS = ["context_recall", "faithfulness", "answer_relevancy"]


def _eval_one(s: EvalSample, cfg: dict, llm: LLMClient) -> dict:
    res = answer(s.question, mode=cfg["mode"], use_rerank=cfg["use_rerank"])
    return {
        "id": s.id,
        "context_recall": context_recall(s.expected_paper_ids, res.sources),
        "faithfulness": faithfulness(res.answer, res.sources, llm),
        "answer_relevancy": answer_relevancy(s.question, res.answer, llm),
    }


def run_eval(dataset: list[EvalSample] | None = None) -> dict:
    dataset = dataset or load_dataset()
    llm = LLMClient()
    summary: dict[str, dict[str, float]] = {}
    per_sample: dict[str, list[dict]] = {}

    for name, cfg in CONFIGS.items():
        print(f"\n=== 配置：{name}  {cfg} ===")
        rows = []
        for s in dataset:
            r = _eval_one(s, cfg, llm)
            rows.append(r)
            print(f"  {s.id}: recall={r['context_recall']:.2f} "
                  f"faith={r['faithfulness']:.2f} rel={r['answer_relevancy']:.2f}")
        per_sample[name] = rows
        summary[name] = {
            m: round(sum(r[m] for r in rows) / len(rows), 3) for m in METRICS
        }
    return {"summary": summary, "per_sample": per_sample}


def run_retrieval_eval(dataset: list[EvalSample] | None = None) -> dict[str, dict[str, float]]:
    """检索层评测（确定性、无 LLM、便宜）：页级 Recall@5/10 + 页级 MRR。"""
    dataset = dataset or load_dataset()
    summary: dict[str, dict[str, float]] = {}
    for name, cfg in CONFIGS.items():
        r5 = r10 = mrr_sum = 0.0
        for s in dataset:
            gold = s.gold_page_set()
            hits = retrieve_ranked(s.question, k=20, mode=cfg["mode"], use_rerank=cfg["use_rerank"])
            r5 += page_recall_at_k(hits, gold, 5)
            r10 += page_recall_at_k(hits, gold, 10)
            mrr_sum += page_mrr(hits, gold)
        n = len(dataset)
        summary[name] = {
            "page_recall@5": round(r5 / n, 3),
            "page_recall@10": round(r10 / n, 3),
            "page_MRR": round(mrr_sum / n, 3),
        }
    return summary


def to_markdown_retrieval(summary: dict[str, dict[str, float]]) -> str:
    header = "| 配置 | 页级 Recall@5 | 页级 Recall@10 | 页级 MRR |\n|---|---|---|---|"
    lines = [header]
    for name, m in summary.items():
        lines.append(
            f"| {name} | {m['page_recall@5']:.3f} | {m['page_recall@10']:.3f} "
            f"| {m['page_MRR']:.3f} |"
        )
    return "\n".join(lines)


def to_markdown(summary: dict[str, dict[str, float]]) -> str:
    header = "| 配置 | 上下文召回 | 忠实度 | 答案相关性 |\n|---|---|---|---|"
    lines = [header]
    for name, m in summary.items():
        lines.append(
            f"| {name} | {m['context_recall']:.3f} | {m['faithfulness']:.3f} "
            f"| {m['answer_relevancy']:.3f} |"
        )
    return "\n".join(lines)


def main(gen_limit: int | None = None) -> None:
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    t0 = time.time()
    full = load_dataset()

    # 1) 检索层（确定性、便宜）：全量跑
    retrieval_summary = run_retrieval_eval(full)
    retrieval_table = to_markdown_retrieval(retrieval_summary)

    # 2) 生成层（LLM 判分、较贵）：测试阶段可用 gen_limit 只跑前 N 题
    gen_ds = full[:gen_limit] if gen_limit else full
    gen = run_eval(gen_ds)
    gen_table = to_markdown(gen["summary"])

    print("\n\n## 检索层消融（页级金标准，全量 {} 题）\n".format(len(full)))
    print(retrieval_table)
    print(f"\n## 生成层消融（RAGAS 风格，{len(gen_ds)} 题）\n")
    print(gen_table)
    print(f"\n（用时 {time.time()-t0:.0f}s）")

    REPORTS_DIR.mkdir(exist_ok=True)
    result = {"retrieval": retrieval_summary, "generation": gen}
    out = REPORTS_DIR / f"eval_{date.today().isoformat()}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORTS_DIR / "eval_table.md").write_text(
        retrieval_table + "\n\n" + gen_table, encoding="utf-8"
    )
    print(f"\n结果已存：{out}")


if __name__ == "__main__":
    main()

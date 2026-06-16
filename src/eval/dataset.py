"""评测集：问题 + 参考答案 + 应命中来源。

评测样本以 JSON 存于 src/eval/eval_set.json（随仓库提交，可离线复跑）。
- expected_paper_ids：该问题应命中的论文（上下文召回的确定性判据）；
- reference_answer：简短参考答案，供人工查看 / 答案相关性参考；
- type：single（单篇事实）| cross（跨篇对比）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_PATH = Path(__file__).parent / "eval_set.json"


@dataclass
class EvalSample:
    id: str
    question: str
    expected_paper_ids: list[str]
    reference_answer: str = ""
    # 页级金标准：答案所在的 [paper_id, page] 对（按论文章节结构标注）
    gold_pages: list[list] = field(default_factory=list)
    type: str = "single"

    def gold_page_set(self) -> set[tuple[str, int]]:
        return {(p[0], int(p[1])) for p in self.gold_pages}


def load_dataset(path: str | Path = DEFAULT_PATH) -> list[EvalSample]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [EvalSample(**item) for item in data]


if __name__ == "__main__":
    ds = load_dataset()
    print(f"评测样本数：{len(ds)}")
    for s in ds:
        print(f"  [{s.type}] {s.id}: {s.question[:40]}  → {s.expected_paper_ids}")

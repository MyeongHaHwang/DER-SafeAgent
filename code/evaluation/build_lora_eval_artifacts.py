"""Build the LoRA-eval LaTeX table and copy it into paper/tables/."""
from __future__ import annotations

import json
from pathlib import Path


RESULTS = Path("code/results/ijcip_lora_eval")
OUT_TBL_REPO = Path("code/results/ijcip_tables")
OUT_TBL_PAPER = Path("paper/tables")

ADAPTER_LABEL = {
    "qwen2_5_7b_lora":  "Qwen2.5-7B-Instruct + LoRA",
    "llama3_1_8b_lora": "Llama-3.1-8B-Instruct + LoRA",
}
ORDER = ["qwen2_5_7b_lora", "llama3_1_8b_lora"]


def _fmt(x):
    if x is None:
        return "---"
    try:
        return f"{float(x):.3f}"
    except (TypeError, ValueError):
        return str(x)


def build_table() -> None:
    rows = ["\\begin{tabular}{lrrrrrr}", "\\toprule",
              "Backbone & N & JSON-val. & schema-comp. "
              "& class-acc. & action-acc. & k1 ms / k3 ms \\\\",
              "\\midrule"]
    for tag in ORDER:
        p = RESULTS / tag / "eval_metrics.json"
        if not p.exists():
            continue
        m = json.loads(p.read_text())
        rows.append(
            f"{ADAPTER_LABEL.get(tag, tag).replace('_','-')} "
            f"& {int(m['n_samples'])} "
            f"& {_fmt(m['json_validity_rate'])} "
            f"& {_fmt(m['schema_compliance_rate'])} "
            f"& {_fmt(m['attack_class_accuracy'])} "
            f"& {_fmt(m['action_accuracy'])} "
            f"& {m['latency_k1_ms_mean']:.0f} / {m['latency_kK_ms_mean']:.0f} \\\\"
        )
    rows += ["\\bottomrule", "\\end{tabular}"]
    OUT_TBL_REPO.mkdir(parents=True, exist_ok=True)
    OUT_TBL_PAPER.mkdir(parents=True, exist_ok=True)
    text = "\n".join(rows) + "\n"
    (OUT_TBL_REPO / "table_lora_eval.tex").write_text(text)
    (OUT_TBL_PAPER / "table_lora_eval.tex").write_text(text)
    print((OUT_TBL_PAPER / "table_lora_eval.tex").read_text())


if __name__ == "__main__":
    build_table()

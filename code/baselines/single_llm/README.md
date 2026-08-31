# Single-LLM Baseline

One-shot prompt: event log → detection label + recommended action.
No tool use, no multi-agent loop, no fine-tuning.

- `prompt.txt` — locked prompt template (hash recorded in `docs/reproducibility.md`).
- `runner.py` — calls model API, returns `Detection`.
- Use the same model family as the proposed system for fair comparison.

# Baselines

Each baseline exposes a uniform interface:
```python
detect(events: list[Event]) -> list[Detection]
respond(detection: Detection) -> Action  # optional
```

## Subfolders
- `rule_ids/` — Suricata/Zeek rule set + Python adapter.
- `single_llm/` — single-prompt LLM detector (no agent loop, no tools).
- `prior_mas/` — closest published multi-agent baseline (cite in `4-method.tex`).

All baselines run through the same `code/simulation/harness.py` for parity.

#!/usr/bin/env python3
"""Record the local execution environment for provenance.

Writes ``artifacts/environment_local.json`` describing the machine on which
verification or re-execution is being performed, in the same spirit as the
archived ``code/results/ijcip_final_v3/environment.json`` from the original
experiment run.  Compare the two files when investigating any numeric
discrepancy.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/environment_local.json"

PACKAGES = ["numpy", "pandas", "scipy", "matplotlib", "yaml", "pydantic",
            "sklearn", "pytest", "torch", "transformers", "peft",
            "accelerate", "bitsandbytes", "opendssdirect", "langgraph"]

ADAPTERS = {
    "qwen2.5-7b": "code/finetuning/results/20260519-144102-lora_qwen25_7b_local/adapter/adapter_model.safetensors",
    "llama3.1-8b": "code/finetuning/results/20260519-144624-lora_llama31_8b_local/adapter/adapter_model.safetensors",
}


def _ver(name: str) -> str | None:
    try:
        m = importlib.import_module(name)
    except Exception:
        return None
    return getattr(m, "__version__", "unknown")


def _sha12(p: Path) -> str | None:
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def main() -> None:
    env: dict = {
        "recorded": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "packages": {p: _ver(p) for p in PACKAGES},
    }
    try:
        import torch
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
            env["cuda"] = torch.version.cuda
    except Exception:
        env["cuda_available"] = False
    env["adapter_sha12"] = {k: _sha12(ROOT / v) for k, v in ADAPTERS.items()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(env, indent=2) + "\n")
    print(f"[json] {OUT}")
    print(json.dumps(env, indent=2))


if __name__ == "__main__":
    main()

"""Base-model resolution for the LLM-in-the-loop experiments.

Base weights (Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct) are NOT
redistributed with this repository and must be obtained under their original
licenses (Apache-2.0 and the Llama 3.1 Community License, respectively).

By default the Hugging Face model IDs below are used, so a normal
``transformers`` cache-or-download flow applies.  To point at a local
snapshot instead (e.g. an air-gapped copy), set::

    export DER_QWEN_BASE=/path/to/qwen2_5_7b
    export DER_LLAMA_BASE=/path/to/llama3_1_8b

The QLoRA adapters in ``code/finetuning/results/…/adapter/`` were trained
against these exact base models (see ``docs/MODEL_PROVENANCE.md`` for the
adapter SHA-256 digests and training configuration).
"""

from __future__ import annotations

import os

QWEN_BASE = os.environ.get("DER_QWEN_BASE", "Qwen/Qwen2.5-7B-Instruct")
LLAMA_BASE = os.environ.get("DER_LLAMA_BASE", "meta-llama/Llama-3.1-8B-Instruct")

QWEN_ADAPTER = "code/finetuning/results/20260519-144102-lora_qwen25_7b_local/adapter"
LLAMA_ADAPTER = "code/finetuning/results/20260519-144624-lora_llama31_8b_local/adapter"

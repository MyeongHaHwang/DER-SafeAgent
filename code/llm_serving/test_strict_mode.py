"""Tests for the strict no-fallback mode and per-call provenance traces.

These tests exercise the fallback/strict logic without loading model weights,
so they run in CI. The real-model load path is covered by the Phase-3 smoke
scripts under ``code/evaluation/llm_in_loop/``.
"""
from __future__ import annotations

import os

import pytest

from .local_lora import (LLMBackendUnavailable, LocalLoRA, reset_default,
                         strict_mode_enabled)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("DER_LLM_STRICT", raising=False)
    monkeypatch.delenv("DER_LORA_ADAPTER", raising=False)
    reset_default()
    yield
    reset_default()


def test_strict_raises_when_adapter_unset(monkeypatch):
    monkeypatch.setenv("DER_LLM_STRICT", "1")
    m = LocalLoRA(adapter_path=None)
    with pytest.raises(LLMBackendUnavailable):
        m.generate_json("prompt")


def test_strict_raises_when_adapter_path_missing(monkeypatch):
    monkeypatch.setenv("DER_LLM_STRICT", "1")
    m = LocalLoRA(adapter_path="/nonexistent/adapter/dir")
    with pytest.raises(LLMBackendUnavailable):
        m.generate_json("prompt")


def test_strict_disabled_by_default():
    assert not strict_mode_enabled()


def test_fallback_trace_records_backend_and_reason():
    m = LocalLoRA(adapter_path=None)
    parsed, trace = m.generate_json_with_trace('{"kind": "command"} setpoint inv_634')
    assert trace["backend"] == "heuristic_fallback"
    assert trace["fallback_reason"]
    assert trace["adapter_sha"] is None
    assert trace["prompt_sha"]
    assert trace["latency_ms"] >= 0.0
    assert parsed["attack_class"] == "command_spoof"
    assert m.n_calls_fallback == 1 and m.n_calls_real == 0


def test_llm_json_sink_collects_role_tagged_traces():
    from ..Multi_AI_Agent.agents import _llm_json
    sink: list[dict] = []
    out = _llm_json('{"tampered": true} inv_634', role="hypothesis", sink=sink)
    assert out["attack_class"] == "fdi"
    assert len(sink) == 1
    assert sink[0]["role"] == "hypothesis"
    assert sink[0]["backend"] == "heuristic_fallback"


def test_heuristic_unchanged_reference_decisions():
    # Guard: the published heuristic decision function must stay byte-stable
    # so legacy (heuristic-backbone) results remain reproducible.
    m = LocalLoRA(adapter_path=None)
    assert m.generate_json('"kind": "command" setpoint x')["attack_class"] == "command_spoof"
    assert m.generate_json('"tampered": true')["attack_class"] == "fdi"
    assert m.generate_json("persistent_freeze")["attack_class"] == "dos"
    assert m.generate_json("quiet step")["attack_class"] == "none"

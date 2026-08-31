"""TypedDict states for the multi-agent LangGraph workflow.

Refactored from 2025/kepsoar/graph/states.py:
- Replace single-alert `soar_input` with a windowed `agent_state` carrying
  recent telemetry samples and events from the simulation harness.
- Add `expected_impact_tier` field consumed by the Caution Agent.
- Drop network 5-tuple fields that are not present in the harness output;
  keep them as optional pass-through for backward UNSW-MG24 evaluation.
"""
from __future__ import annotations

from enum import Enum, unique
from typing import Any, TypedDict


@unique
class OperationMode(Enum):
    SCRIPT_GEN = "script"
    REPORT_GEN = "report"


@unique
class PromptStrategy(Enum):
    ZERO_SHOT = "zero"
    FEW_SHOT = "few"
    COT = "cot"
    TOT = "tot"


@unique
class AttackClass(Enum):
    NONE = "none"
    FDI = "fdi"
    REPLAY = "replay"
    COMMAND_SPOOF = "command_spoof"
    DOS = "dos"
    FIRMWARE = "firmware"


@unique
class ImpactTier(Enum):
    NEGLIGIBLE = "negligible"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    SEVERE = "severe"


class AgentState(TypedDict, total=False):
    """Shared state passed between agent nodes."""
    # I/O surface from harness
    t: float
    telemetry_window: list[dict[str, Any]]   # last N TelemetrySample dicts
    event_window: list[dict[str, Any]]       # last M EventLog dicts

    # Configuration
    mode: OperationMode
    prompt_strategy: PromptStrategy

    # Script agent output
    attack_class: AttackClass
    affected_asset: str
    confidence: float
    rationale: str
    proposed_action: str                     # mitigation primitive name
    proposed_action_target: str
    expected_impact_tier: ImpactTier

    # Caution agent output
    caution: bool
    caution_reason: str

    # Report agent output (REPORT_GEN mode only)
    report: str

    # 2026 v2 additions: deterministic Telemetry Analyst output, vote-aggregated
    # Hypothesis output, per-candidate Energy Impact estimates, Coordinator decision.
    feature_view: dict[str, Any]             # output of telemetry_features.extract
    hypothesis: dict[str, Any]               # vote-aggregated hypothesis dict
    impact_estimates: list[dict[str, Any]]   # one per candidate mitigation
    final_action: str                        # Coordinator's chosen primitive
    final_action_target: str
    final_impact_tier: str
    coordinator_decision: str                # auto | hitl_required | withhold
    coordinator_reason: str
    chosen_for_hitl: str                     # primitive that was queued for HITL

    # Bookkeeping
    is_script_changed: bool                  # carried from 2025 conditional path
    decision_id: str                         # uuid4 per agent invocation

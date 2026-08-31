"""Pydantic schema for the fine-tuning corpus. Single source of truth."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class AttackClass(str, Enum):
    NONE = "none"
    FDI = "fdi"                       # false data injection
    REPLAY = "replay"
    COMMAND_SPOOF = "command_spoof"
    DOS = "dos"
    FIRMWARE = "firmware"


class ImpactTier(str, Enum):
    NEGLIGIBLE = "negligible"         # < 1% ENS proxy
    LOW = "low"                       # 1-5%
    MEDIUM = "medium"                 # 5-15%
    HIGH = "high"                     # 15-30%
    SEVERE = "severe"                 # > 30%


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class Metadata(BaseModel):
    source: Literal["synthetic", "cti", "public_log", "human"]
    license: str = Field(description="SPDX identifier")
    seed: int | None = None
    generated_at: datetime
    scenario_config_hash: str | None = None


class AssistantPayload(BaseModel):
    """Structured content carried inside assistant message (JSON string)."""
    attack_class: AttackClass
    affected_asset: str                       # e.g. "PV_inverter_BUS_634"
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    recommended_action: str
    expected_impact_tier: ImpactTier


class Example(BaseModel):
    id: str
    scenario: str
    attack_class: AttackClass
    messages: list[Message]
    metadata: Metadata

    @field_validator("messages")
    @classmethod
    def _check_roles(cls, v: list[Message]) -> list[Message]:
        if not v or v[0].role != "system":
            raise ValueError("first message must be system")
        if v[-1].role != "assistant":
            raise ValueError("last message must be assistant")
        return v

    def to_chatml(self) -> list[dict[str, Any]]:
        return [m.model_dump() for m in self.messages]

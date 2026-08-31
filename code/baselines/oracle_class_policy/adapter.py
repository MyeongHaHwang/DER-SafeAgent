"""OracleClassPolicy --- upper-bound reference, *not* a deployable detector.

Reads the ground-truth attack class from the harness's per-step record (via
a lookup against the active injectors) and applies the optimal class-
specific mitigation. Use only as an upper-bound reference in IJCIP
ablations.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ...simulation.mitigations import REGISTRY as MITIGATIONS
from ...simulation.types import Action, Detection, EventLog, TelemetrySample


_OPTIMAL_BY_CLASS: dict[str, str] = {
    "fdi":           "freeze_setpoint",
    "command_spoof": "freeze_setpoint",
    "replay":        "request_ied_revalidation",
    "dos":           "request_ied_revalidation",
    "firmware":      "request_ied_revalidation",
}


@dataclass
class OracleClassPolicy:
    name: str = "oracle_class_policy"
    _last_decision: dict | None = field(default=None, repr=False)
    _scenario_class: str | None = None
    _scenario_target: str = ""
    _attack_window: tuple[float, float] = (0.0, 0.0)

    def configure(self, attack_class: str, target: str,
                   start_s: float, end_s: float) -> None:
        """Inject the ground-truth labels at scenario load time."""
        self._scenario_class = attack_class
        self._scenario_target = target
        self._attack_window = (start_s, end_s)

    def step(self, t: float, telemetry: TelemetrySample,
             events: list[EventLog]) -> tuple[Detection, list[Action]]:
        if self._scenario_class is None:
            return Detection(asset=None, attack_class="none",
                              confidence=0.0), []
        s, e = self._attack_window
        active = (s <= t <= e)
        if not active:
            return Detection(asset=None, attack_class="none",
                              confidence=0.0), []
        action_name = _OPTIMAL_BY_CLASS.get(self._scenario_class, "no_op")
        det = Detection(asset=self._scenario_target,
                          attack_class=self._scenario_class,
                          confidence=1.0,
                          rationale="oracle ground-truth")
        if action_name == "no_op" or action_name not in MITIGATIONS:
            return det, []
        self._last_decision = {"attack_class": self._scenario_class,
                                 "final_action": action_name}
        return det, [Action(name=action_name, target=self._scenario_target,
                              params={"detector": "oracle"})]

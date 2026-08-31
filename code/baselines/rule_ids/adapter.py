"""Rule-based IDS baseline.

A small, transparent rule set that fires on:
- telemetry deviation > N stdev from rolling mean for an asset (proxy for FDI)
- duplicate event payloads within a short window (proxy for replay)
- DERMS commands originating from unexpected source field (proxy for spoof)

Intentionally simple: meant to represent what a utility would deploy with
Suricata/Zeek-style signatures, not to be competitive.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

from ...simulation.types import Action, Detection, EventLog, TelemetrySample


@dataclass
class RuleIDS:
    name: str = "rule_ids"
    window: int = 30
    z_threshold: float = 4.0
    expected_command_source: str = "DERMS"
    _hist: dict[str, deque] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=30)))
    _recent: deque = field(default_factory=lambda: deque(maxlen=120))

    def step(
        self,
        t: float,
        telemetry: TelemetrySample,
        events: list[EventLog],
    ) -> tuple[Detection, list[Action]]:
        # 1) telemetry z-score on each DER's reported P
        max_z = 0.0
        worst_asset: str | None = None
        for asset, p in telemetry.der_p_kw.items():
            h = self._hist[asset]
            if len(h) >= 5:
                mean = sum(h) / len(h)
                var = sum((x - mean) ** 2 for x in h) / len(h)
                std = var ** 0.5 or 1e-6
                z = abs(p - mean) / std
                if z > max_z:
                    max_z, worst_asset = z, asset
            h.append(p)

        # 2) duplicate-payload check (replay proxy)
        replay_hit = False
        for ev in events:
            key = (ev.source, ev.kind, str(sorted(ev.payload.items())))
            if key in self._recent:
                replay_hit = True
            self._recent.append(key)

        # 3) command-source check (spoof proxy)
        spoof_hit = any(
            ev.kind == "command" and ev.payload.get("from") != self.expected_command_source
            for ev in events
        )

        # decide — rule_ids represents today's substation practice: a hit
        # auto-fires the corresponding hard mitigation (isolate / freeze) with
        # no confidence-aware gating, no impact estimate. Strong on detection,
        # bad on FP cost — exactly the problem motivating the agent loop.
        if max_z > self.z_threshold:
            det = Detection(asset=worst_asset, attack_class="fdi",
                            confidence=min(1.0, max_z / 10.0),
                            rationale=f"z={max_z:.1f}>{self.z_threshold}")
            return det, [Action(name="isolate_inverter", target=worst_asset or "")]
        if replay_hit:
            return (Detection(asset=None, attack_class="replay", confidence=0.6,
                              rationale="duplicate event payload"),
                    [Action(name="request_ied_revalidation", target="")])
        if spoof_hit:
            spoof_target = next(
                (ev.payload.get("asset", "") for ev in events
                 if ev.kind == "command" and ev.payload.get("from") != self.expected_command_source),
                "",
            )
            return (Detection(asset=spoof_target or None, attack_class="command_spoof",
                              confidence=0.7, rationale="unexpected command source"),
                    [Action(name="freeze_setpoint", target=spoof_target)])
        return Detection(asset=None, attack_class="none", confidence=0.0), []

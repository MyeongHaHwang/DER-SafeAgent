"""Replay attack: re-emits a captured pre-attack window of telemetry/events."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace

from ..types import EventLog, TelemetrySample


@dataclass
class ReplayInjector:
    target: str
    start_s: float
    end_s: float
    capture_window_s: float = 30.0
    name: str = "replay"
    _buffer: deque = field(default_factory=lambda: deque(maxlen=4096))
    _ev_buffer: deque = field(default_factory=lambda: deque(maxlen=4096))

    def active(self, t: float) -> bool:
        return self.start_s <= t <= self.end_s

    def mutate_telemetry(self, t: float, sample: TelemetrySample) -> TelemetrySample:
        if t < self.start_s:
            self._buffer.append(sample)
            return sample
        if not self.active(t) or not self._buffer:
            return sample
        # play back the captured buffer at current time
        playback = self._buffer[(int(t - self.start_s)) % len(self._buffer)]
        return replace(playback, t=t)

    def mutate_events(self, t: float, events: list[EventLog]) -> list[EventLog]:
        # Buffer pre-attack events so we can replay real captured frames
        # (including their original sequence numbers).
        if t < self.start_s:
            for ev in events:
                self._ev_buffer.append(ev)
            return events
        if not self.active(t):
            return events
        # During the attack the adversary re-emits captured frames verbatim.
        # The values look plausible and the wire timestamp is refreshed to t,
        # but the carried protocol sequence number is the OLD captured seq ---
        # an observable regression a runtime-safe detector can catch without
        # any ground-truth flag. `tampered=True` is retained for the legacy
        # (oracle) path only.
        if not self._ev_buffer:
            return [EventLog(t=t, source=ev.source, kind=ev.kind,
                             payload=ev.payload, tampered=True, seq=ev.seq)
                    for ev in events]
        out: list[EventLog] = []
        n = len(self._ev_buffer)
        for i, ev in enumerate(events):
            cap = self._ev_buffer[(int(t - self.start_s) * len(events) + i) % n]
            out.append(EventLog(t=t, source=cap.source, kind=cap.kind,
                                payload=cap.payload, tampered=True, seq=cap.seq))
        return out

    def physical_mutate(self, t: float, feeder) -> None:
        """Replay is a message-layer attack; no physical state change."""
        return None

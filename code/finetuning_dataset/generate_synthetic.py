"""Generate a small synthetic dataset shard for pipeline validation.

Each example simulates an alert + telemetry context and a target structured
response. Replace with the real generator before v1.0.
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

ATTACK_CLASSES = ["none", "fdi", "replay", "command_spoof", "dos", "firmware"]
SCENARIOS = ["ieee13_fdi_inverter", "ieee13_replay_dnp3", "ieee34_command_spoof_derms"]
ASSETS = ["INV_634", "BESS_675", "INV_692", "RTU_650"]
ACTIONS = ["no_op", "isolate_inverter", "throttle_ramp", "freeze_setpoint", "request_ied_revalidation"]
TIERS = ["negligible", "low", "medium", "high", "severe"]

SYSTEM_PROMPT = (
    "You are the Script Agent of DER-SecAgent. Classify the recent telemetry/event "
    "window and propose a mitigation. Respond with JSON only."
)


def _telemetry_blob(rng: random.Random, asset: str, attacked: bool) -> str:
    p = rng.uniform(50, 150)
    bias = rng.uniform(40, 80) if attacked else 0.0
    return (
        f"t=180s freq=60.00Hz v_min_pu=0.98 v_max_pu=1.02 "
        f"DER P kW={{'{asset}': {p + bias:.1f}}}"
    )


def _event_blob(asset: str, attacked: bool) -> str:
    if attacked:
        return f"- t=180.0 iec61850/telemetry: {{'asset': '{asset}', 'p_kw': 'tampered'}}"
    return f"- t=180.0 iec61850/telemetry: {{'asset': '{asset}', 'p_kw': 'nominal'}}"


def make_example(idx: int, rng: random.Random) -> dict:
    attack_class = rng.choice(ATTACK_CLASSES)
    scenario = rng.choice(SCENARIOS)
    asset = rng.choice(ASSETS)
    attacked = attack_class != "none"

    user = (
        f"Recent telemetry: {_telemetry_blob(rng, asset, attacked)}\n"
        f"Recent events:\n{_event_blob(asset, attacked)}"
    )

    if attack_class == "none":
        action = "no_op"
        tier = "negligible"
        target = ""
        confidence = round(rng.uniform(0.7, 0.95), 2)
        rationale = "Telemetry within nominal bounds; no anomalous events."
    else:
        action = rng.choice([a for a in ACTIONS if a != "no_op"])
        tier = rng.choice(TIERS[1:])
        target = asset
        confidence = round(rng.uniform(0.6, 0.95), 2)
        rationale = f"Detected {attack_class} pattern affecting {asset}; recommend {action}."

    assistant_payload = {
        "attack_class": attack_class,
        "affected_asset": target or None,
        "confidence": confidence,
        "rationale": rationale,
        "recommended_action": action,
        "expected_impact_tier": tier,
    }

    return {
        "id": f"syn-{idx:05d}",
        "scenario": scenario,
        "attack_class": attack_class,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": json.dumps(assistant_payload)},
        ],
        "metadata": {
            "source": "synthetic",
            "license": "CC0-1.0",
            "seed": rng.randint(0, 1_000_000),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "scenario_config_hash": None,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="raw/synthetic/shard_dummy.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = [make_example(i, rng) for i in range(args.n)]
    out = Path(__file__).parent / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"wrote {len(rows)} examples to {out}")


if __name__ == "__main__":
    main()

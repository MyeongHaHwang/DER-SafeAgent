"""Pre-registered expanded StubFeeder scenario matrix (revision R1-C2).

Generates 24 unique scenario configurations as YAML files plus a
machine-readable manifest. Design rules:

- Factors varied causally: attack type {fdi, command_spoof, replay, dos},
  magnitude {low, med, high}, duration {short, medium, persistent}, load level
  {light, nominal, peak}, DER penetration {low, med, high}, affected asset
  {inverter, BESS, distal inverter, fleet}.
- Magnitude/duration are expressed through the injector's *physical* knobs
  (bias_kw, forged_setpoint_kw, capture_window_s, attack window length), never
  through seeds.
- ``base_load_kw`` scales the served load (light 600 / nominal 1000 /
  peak 1400); DER penetration scales every DER's max_kw (low ×0.5 / med ×1.0 /
  high ×1.5) — written out as explicit values, not runtime multipliers.
- The StubFeeder physics is untouched: ENS accrues when the DER injection
  deficit exceeds 40% of DER nominal, so configurations deliberately span
  sub-threshold (curtailment/detection only) and supra-threshold (ENS) cases.
- The harness is deterministic given a configuration (verified: 5 published
  seeds produce identical outputs), so each configuration runs with seed 0 and
  the configuration — not the seed — is the unit of statistical inference.
- firmware attack class: EXCLUDED — no physical injector exists in the
  implementation and inventing one without a firmware model would fabricate
  physics. Covered at prompt/schema level in the LoRA evaluation instead.

Run:  python3 -m code.simulation.scenarios.ijcip_revision.generate_matrix
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import yaml

HERE = Path(__file__).parent
REVISION_TAG = "ijcip_revision_r1r2_20260805"
CONFIG_OUT = Path("code/configs") / REVISION_TAG

IEEE13_FEEDER = "../../ieee13_fdi_inverter/feeder.dss"
IEEE34_FEEDER = "../../ieee34_command_spoof_derms/feeder.dss"

LOAD_KW = {"light": 600.0, "nominal": 1000.0, "peak": 1400.0}
PEN_SCALE = {"low": 0.5, "med": 1.0, "high": 1.5}

IEEE13_BUSES = ["SOURCEBUS", "634", "671", "675"]
IEEE34_BUSES = ["SOURCEBUS", "808", "824", "834", "840", "848"]


def ieee13_ders(pen: str) -> list[dict]:
    s = PEN_SCALE[pen]
    return [
        {"id": "INV_634", "type": "pv", "bus": "634", "max_kw": 200.0 * s},
        {"id": "BESS_675", "type": "bess", "bus": "675", "max_kw": 150.0 * s},
    ]


def ieee34_ders(pen: str) -> list[dict]:
    s = PEN_SCALE[pen]
    return [
        {"id": "INV_840", "type": "pv", "bus": "840", "max_kw": 240.0 * s},
        {"id": "BESS_848", "type": "bess", "bus": "848", "max_kw": 180.0 * s},
        {"id": "INV_824", "type": "pv", "bus": "824", "max_kw": 260.0 * s},
    ]


DUR_S = {"short": 60, "medium": 120, "persistent": 300}


def window(dur: str, start: int = 180) -> tuple[int, int]:
    return start, start + DUR_S[dur]


# ---------------------------------------------------------------------------
# The pre-registered matrix. Each entry: (id, feeder_family, attack list,
# magnitude label, duration label, load, penetration, affected asset+bus,
# expected_safety_issue, rationale).
# fdi bias_kw: low 20 / med 60 / high 120 (detection-difficulty axis; physical
#   harm arises only from over-mitigation, which is the class-aware avoidance
#   test).
# command_spoof forged_setpoint_kw for INV_634 (nominal 140 at pen=med):
#   low 100 (deficit 0.16, sub-threshold), med 30 (deficit 0.45, ENS),
#   high 5 (deficit 0.55, ENS). Fleet/asset variants recomputed per asset.
# replay capture_window_s: low 10 / med 30 / high 60 (staleness axis).
# dos: magnitude is the forced-zero outage itself; duration is the axis.
# ---------------------------------------------------------------------------

def _cfg(id_, family, attacks, gt, mag, dur, load, pen, asset, bus, issue, why):
    return dict(id=id_, family=family, attacks=attacks, gt=gt, mag=mag, dur=dur,
                load=load, pen=pen, asset=asset, bus=bus, issue=issue, why=why)


def build_matrix() -> list[dict]:
    M: list[dict] = []
    a, b = window("medium")

    # -- 1-4: attack-type sweep at center point (ieee13, nominal, med pen) --
    M.append(_cfg("rv13_fdi_med_medium_nom_medpen_inv",
                  "ieee13", [dict(type="fdi", target="INV_634", start_s=a, end_s=b, bias_kw=60.0)],
                  dict(attack_class="fdi", affected_asset="INV_634", start_s=a, end_s=b),
                  "med", "medium", "nominal", "med", "INV_634", "634",
                  "over-mitigation risk: isolating a healthy inverter on telemetry-layer FDI",
                  "Center point for the FDI axis; discriminates class-aware avoidance (isolate vs freeze)."))
    M.append(_cfg("rv13_spoof_med_medium_nom_medpen_inv",
                  "ieee13", [dict(type="command_spoof", target="INV_634", start_s=a, end_s=b,
                                  forged_setpoint_kw=30.0, period_s=5.0)],
                  dict(attack_class="command_spoof", affected_asset="INV_634", start_s=a, end_s=b),
                  "med", "medium", "nominal", "med", "INV_634", "634",
                  "supra-threshold generation drop -> ENS unless setpoint frozen quickly",
                  "Center point for the spoof axis; deficit 0.45 > 0.4 so mitigation speed shows in ENS."))
    M.append(_cfg("rv13_replay_med_medium_nom_medpen_inv",
                  "ieee13", [dict(type="replay", target="INV_634", start_s=a, end_s=b,
                                  capture_window_s=30.0)],
                  dict(attack_class="replay", affected_asset="INV_634", start_s=a, end_s=b),
                  "med", "medium", "nominal", "med", "INV_634", "634",
                  "stale telemetry masks true state; wrong class -> wrong primitive",
                  "Replay has no direct physical effect; discriminates classification and refusal to over-act."))
    M.append(_cfg("rv13_dos_med_medium_nom_medpen_inv",
                  "ieee13", [dict(type="dos", target="INV_634", start_s=a, end_s=b)],
                  dict(attack_class="dos", affected_asset="INV_634", start_s=a, end_s=b),
                  "med", "medium", "nominal", "med", "INV_634", "634",
                  "forced-zero output = supra-threshold deficit during the outage",
                  "DoS forces the unit to 0 kW; freeze_setpoint restores dispatch, revalidation does not — "
                  "methods differ in recovered ENS."))

    # -- 5-8: magnitude sweep (fdi low/high, spoof low/high) --
    for mag, kw in (("low", 20.0), ("high", 120.0)):
        M.append(_cfg(f"rv13_fdi_{mag}_medium_nom_medpen_inv",
                      "ieee13", [dict(type="fdi", target="INV_634", start_s=a, end_s=b, bias_kw=kw)],
                      dict(attack_class="fdi", affected_asset="INV_634", start_s=a, end_s=b),
                      mag, "medium", "nominal", "med", "INV_634", "634",
                      "detection difficulty scales inversely with bias magnitude",
                      f"FDI bias {kw} kW: {'weak signal near trigger threshold' if mag == 'low' else 'saturating signal; tests confidence calibration'}."))
    for mag, kw in (("low", 100.0), ("high", 5.0)):
        M.append(_cfg(f"rv13_spoof_{mag}_medium_nom_medpen_inv",
                      "ieee13", [dict(type="command_spoof", target="INV_634", start_s=a, end_s=b,
                                      forged_setpoint_kw=kw, period_s=5.0)],
                      dict(attack_class="command_spoof", affected_asset="INV_634", start_s=a, end_s=b),
                      mag, "medium", "nominal", "med", "INV_634", "634",
                      "sub-threshold curtailment only" if mag == "low" else "deep supra-threshold ENS",
                      f"Forged setpoint {kw} kW: {'deficit 0.16 — no ENS; tests proportionality (no over-mitigation)' if mag == 'low' else 'deficit 0.55 — worst-case drop; tests mitigation speed'}."))

    # -- 9-10: duration sweep on dos (short vs persistent) --
    for dur in ("short", "persistent"):
        s2, e2 = window(dur)
        M.append(_cfg(f"rv13_dos_med_{dur}_nom_medpen_inv",
                      "ieee13", [dict(type="dos", target="INV_634", start_s=s2, end_s=e2)],
                      dict(attack_class="dos", affected_asset="INV_634", start_s=s2, end_s=e2),
                      "med", dur, "nominal", "med", "INV_634", "634",
                      "outage ENS scales with duration; persistent case stresses fallback posture",
                      f"DoS for {DUR_S[dur]} s separates single-event from sustained-outage response."))

    # -- 11-12: duration sweep on spoof (short vs persistent) --
    for dur in ("short", "persistent"):
        s2, e2 = window(dur)
        M.append(_cfg(f"rv13_spoof_med_{dur}_nom_medpen_inv",
                      "ieee13", [dict(type="command_spoof", target="INV_634", start_s=s2, end_s=e2,
                                      forged_setpoint_kw=30.0, period_s=5.0)],
                      dict(attack_class="command_spoof", affected_asset="INV_634", start_s=s2, end_s=e2),
                      "med", dur, "nominal", "med", "INV_634", "634",
                      "ENS window scales with attack persistence",
                      f"Spoof held for {DUR_S[dur]} s; persistent case shows whether freeze holds against re-spoofing."))

    # -- 13-14: load level (light / peak) on the spoof center point --
    for load in ("light", "peak"):
        M.append(_cfg(f"rv13_spoof_med_medium_{load}_medpen_inv",
                      "ieee13", [dict(type="command_spoof", target="INV_634", start_s=a, end_s=b,
                                      forged_setpoint_kw=30.0, period_s=5.0)],
                      dict(attack_class="command_spoof", affected_asset="INV_634", start_s=a, end_s=b),
                      "med", "medium", load, "med", "INV_634", "634",
                      "absolute ENS scales with served load at fixed relative deficit",
                      f"{load} load ({LOAD_KW[load]:.0f} kW): same relative deficit, different customer impact."))

    # -- 15-16: peak-load FDI and light-load dos (load x type interaction) --
    M.append(_cfg("rv13_fdi_med_medium_peak_medpen_inv",
                  "ieee13", [dict(type="fdi", target="INV_634", start_s=a, end_s=b, bias_kw=60.0)],
                  dict(attack_class="fdi", affected_asset="INV_634", start_s=a, end_s=b),
                  "med", "medium", "peak", "med", "INV_634", "634",
                  "over-mitigation at peak load maximises the cost of a wrong isolate",
                  "Peak-load FDI: the penalty for isolating a healthy unit is largest here."))
    M.append(_cfg("rv13_dos_med_medium_light_medpen_inv",
                  "ieee13", [dict(type="dos", target="INV_634", start_s=a, end_s=b)],
                  dict(attack_class="dos", affected_asset="INV_634", start_s=a, end_s=b),
                  "med", "medium", "light", "med", "INV_634", "634",
                  "same outage, smaller absolute harm — tests proportional reporting",
                  "Light-load DoS: harm floor case for the load axis."))

    # -- 17-18: DER penetration (low / high) on the spoof center point --
    for pen in ("low", "high"):
        kw = 30.0 * PEN_SCALE[pen]  # keep the *relative* drop comparable
        M.append(_cfg(f"rv13_spoof_med_medium_nom_{pen}pen_inv",
                      "ieee13", [dict(type="command_spoof", target="INV_634", start_s=a, end_s=b,
                                      forged_setpoint_kw=kw, period_s=5.0)],
                      dict(attack_class="command_spoof", affected_asset="INV_634", start_s=a, end_s=b),
                      "med", "medium", "nominal", pen, "INV_634", "634",
                      "DER-heavy vs DER-light feeder changes the grid-support stakes",
                      f"{pen} penetration (×{PEN_SCALE[pen]}): DER share of supply changes the consequence "
                      "of both the attack and the mitigation."))

    # -- 19: penetration x fdi interaction --
    M.append(_cfg("rv13_fdi_med_medium_nom_lowpen_inv",
                  "ieee13", [dict(type="fdi", target="INV_634", start_s=a, end_s=b, bias_kw=30.0)],
                  dict(attack_class="fdi", affected_asset="INV_634", start_s=a, end_s=b),
                  "med", "medium", "nominal", "low", "INV_634", "634",
                  "low-penetration FDI: small unit, isolate is cheap — tests whether avoidance still holds",
                  "Checks that class-aware avoidance is principled, not just cost-driven."))

    # -- 20: affected asset = BESS --
    M.append(_cfg("rv13_spoof_med_medium_nom_medpen_bess",
                  "ieee13", [dict(type="command_spoof", target="BESS_675", start_s=a, end_s=b,
                                  forged_setpoint_kw=10.0, period_s=5.0)],
                  dict(attack_class="command_spoof", affected_asset="BESS_675", start_s=a, end_s=b),
                  "med", "medium", "nominal", "med", "BESS_675", "675",
                  "storage asset target; deficit 0.39 — borderline sub-threshold",
                  "Asset-type variation: BESS nominal 105 kW; drop to 10 kW gives deficit just below the "
                  "ENS threshold, separating curtailment-aware from ENS-only policies."))

    # -- 21-23: ieee34 family (distal inverter, fleet, dos) --
    s3, e3 = window("medium", start=200)
    M.append(_cfg("rv34_spoof_high_medium_nom_medpen_inv840",
                  "ieee34", [dict(type="command_spoof", target="INV_840", start_s=s3, end_s=e3,
                                  forged_setpoint_kw=10.0, period_s=5.0)],
                  dict(attack_class="command_spoof", affected_asset="INV_840", start_s=s3, end_s=e3),
                  "high", "medium", "nominal", "med", "INV_840", "840",
                  "distal-unit drop on the long radial; deficit 0.33 — curtailment signal only",
                  "Replaces the published non-discriminative ieee34 scenario with a harder drop; still "
                  "sub-threshold, so it doubles as a calibration control for ENS-driven ranking."))
    M.append(_cfg("rv34_spoof_high_medium_nom_medpen_fleet",
                  "ieee34", [dict(type="command_spoof", target="INV_840", start_s=s3, end_s=e3,
                                  forged_setpoint_kw=10.0, period_s=5.0),
                             dict(type="command_spoof", target="INV_824", start_s=s3, end_s=e3,
                                  forged_setpoint_kw=10.0, period_s=5.0)],
                  dict(attack_class="command_spoof", affected_asset="INV_840", start_s=s3, end_s=e3),
                  "high", "medium", "nominal", "med", "FLEET(INV_840+INV_824)", "840+824",
                  "coordinated fleet-level spoof -> deficit 0.69, deep ENS; DERMS-compromise proxy",
                  "Fleet-level attack (two simultaneous injectors) — the DERMS-level target class the "
                  "reviewers asked about; per-asset freeze must fire twice."))
    M.append(_cfg("rv34_dos_med_persistent_nom_medpen_bess848",
                  "ieee34", [dict(type="dos", target="BESS_848", start_s=s3, end_s=s3 + 300)],
                  dict(attack_class="dos", affected_asset="BESS_848", start_s=s3, end_s=s3 + 300),
                  "med", "persistent", "nominal", "med", "BESS_848", "848",
                  "persistent storage outage on the radial",
                  "RTU/IED-level persistent freeze on a BESS; deficit 0.26 — recovery behaviour matters "
                  "more than raw ENS."))

    # -- 24: benign calibration control (no attack) --
    M.append(_cfg("rv13_benign_none_none_nom_medpen_none",
                  "ieee13", [],
                  dict(attack_class="none", affected_asset=None, start_s=0, end_s=0),
                  "none", "none", "nominal", "med", "none", "-",
                  "no attack: any emitted mitigation is a false positive with real curtailment cost",
                  "CALIBRATION CONTROL, excluded from primary significance analysis; measures "
                  "false-positive action rate under quiet conditions."))

    # -- 25: high-staleness replay --
    M.append(_cfg("rv13_replay_high_medium_nom_medpen_inv",
                  "ieee13", [dict(type="replay", target="INV_634", start_s=a, end_s=b,
                                  capture_window_s=60.0)],
                  dict(attack_class="replay", affected_asset="INV_634", start_s=a, end_s=b),
                  "high", "medium", "nominal", "med", "INV_634", "634",
                  "long-staleness replay maximises divergence between replayed and true state",
                  "Staleness axis endpoint for the replay class."))

    return M


def emit() -> None:
    matrix = build_matrix()
    ids = [m["id"] for m in matrix]
    assert len(ids) == len(set(ids)), "duplicate scenario ids"

    CONFIG_OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for m in matrix:
        family = m["family"]
        cfg = {
            "name": m["id"],
            "description": m["why"],
            "feeder": IEEE13_FEEDER if family == "ieee13" else IEEE34_FEEDER,
            "dt_s": 1.0,
            "duration_s": 600,
            "monitored_buses": IEEE13_BUSES if family == "ieee13" else IEEE34_BUSES,
            "ders": ieee13_ders(m["pen"]) if family == "ieee13" else ieee34_ders(m["pen"]),
            "base_load_kw": LOAD_KW[m["load"]],
            "attacks": m["attacks"],
            "ground_truth": m["gt"],
            "revision_meta": {
                "revision_tag": REVISION_TAG,
                "attack_type": (m["attacks"][0]["type"] if m["attacks"] else "none"),
                "attack_magnitude": m["mag"],
                "attack_duration": m["dur"],
                "load_level": m["load"],
                "der_penetration": m["pen"],
                "affected_asset": m["asset"],
                "affected_bus": m["bus"],
                "expected_safety_issue": m["issue"],
                "rationale": m["why"],
                "is_calibration_control": m["mag"] == "none",
            },
        }
        d = HERE / m["id"]
        d.mkdir(exist_ok=True)
        (d / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        chash = hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]
        rows.append({
            "scenario_id": m["id"],
            "attack_type": cfg["revision_meta"]["attack_type"],
            "attack_magnitude": m["mag"],
            "attack_duration": m["dur"],
            "load_level": m["load"],
            "DER_penetration": m["pen"],
            "affected_asset": m["asset"],
            "affected_bus": m["bus"],
            "random_seed": 0,
            "deterministic_components": "feeder;injectors;detector-pipeline;greedy-decoding",
            "stochastic_components": "none (harness verified seed-invariant; K=3 uses distinct fixed prompt strategies, not sampling)",
            "expected_safety_issue": m["issue"],
            "is_calibration_control": cfg["revision_meta"]["is_calibration_control"],
            "configuration_hash": chash,
            "config_path": str(d / "config.yaml"),
        })

    man = CONFIG_OUT / "scenario_manifest.csv"
    with open(man, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (CONFIG_OUT / "scenario_matrix.yaml").write_text(yaml.safe_dump(
        {"revision_tag": REVISION_TAG, "n_configurations": len(rows),
         "frozen": False, "scenarios": rows}, sort_keys=False))
    print(f"wrote {len(rows)} scenario configs under {HERE}")
    print(f"manifest: {man}")


if __name__ == "__main__":
    emit()

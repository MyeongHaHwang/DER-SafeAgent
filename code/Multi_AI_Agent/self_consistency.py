"""Self-consistency wrapper — multi-strategy voting for the Hypothesis Agent.

Executes the underlying LLM under K=3 prompt-strategy variants
(zero-shot, chain-of-thought, tree-of-thought) and aggregates the K JSON
outputs by majority vote on ``attack_class``. Final confidence is set to the
vote share multiplied by the mean confidence of the winning bucket — so an
uncertain unanimous answer (e.g., three "fdi" at 0.5 each) ends up around
0.5, while a confident-but-split answer is correctly down-weighted.

This is the calibration mechanism that turns a vendored heuristic or a noisy
LoRA into something whose confidence is interpretable for downstream
threshold sweeps.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Callable


def vote(samples: list[dict]) -> dict:
    if not samples:
        return {"attack_class": "none", "confidence": 0.0,
                "vote_share": 0.0, "consensus_size": 0}
    classes = [s.get("attack_class", "none") for s in samples]
    cnt = Counter(classes)
    winner, n = cnt.most_common(1)[0]
    bucket = [s for s in samples if s.get("attack_class", "none") == winner]
    mean_conf = sum(float(s.get("confidence", 0.0)) for s in bucket) / max(len(bucket), 1)
    share = n / len(samples)
    # winner inherits the dominant bucket's modal recommendations
    actions = [s.get("proposed_action", "no_op") for s in bucket]
    proposed = Counter(actions).most_common(1)[0][0]
    targets = [s.get("proposed_action_target") for s in bucket if s.get("proposed_action_target")]
    proposed_target = Counter(targets).most_common(1)[0][0] if targets else None
    impacts = [s.get("expected_impact_tier", "negligible") for s in bucket]
    impact = Counter(impacts).most_common(1)[0][0]
    rationales = [str(s.get("rationale", "")) for s in bucket]
    rationale = max(rationales, key=len) if rationales else ""

    return {
        "attack_class": winner,
        "affected_asset": next((s.get("affected_asset") for s in bucket
                                 if s.get("affected_asset")), None),
        "confidence": share * mean_conf,
        "rationale": rationale,
        "proposed_action": proposed,
        "proposed_action_target": proposed_target,
        "expected_impact_tier": impact,
        "vote_share": share,
        "consensus_size": n,
        "n_samples": len(samples),
    }


def run_self_consistent(
    build_prompt: Callable[[str], str],
    llm_call: Callable[[str], dict],
    strategies: tuple[str, ...] = ("zero", "cot", "tot"),
) -> dict:
    """Run the LLM once per strategy, aggregate by vote.

    Parameters
    ----------
    build_prompt: takes the strategy hint string and returns the full prompt.
    llm_call: takes a prompt and returns a parsed JSON dict (heuristic or real).
    """
    samples = []
    for strat in strategies:
        try:
            samples.append(llm_call(build_prompt(strat)))
        except Exception:
            continue
    return vote(samples)

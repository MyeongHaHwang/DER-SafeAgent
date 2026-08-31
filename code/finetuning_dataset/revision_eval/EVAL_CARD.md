# Revision evaluation corpus card
Revision tag: ijcip_revision_r1r2_20260805
Built: 2026-08-05T13:59:56.686992+00:00
Total prompts: 146
Reference prompts indexed for dedup (train+val+test): 200
Candidates rejected by the leakage filter (digit-masked vs train/val/test): 0
Candidates rejected by the within-corpus diversity filter (unmasked): 80

By family: {'scenario_derived': 20, 'paraphrased_alert': 36, 'conflicting_telemetry': 15, 'malformed_alert': 15, 'prompt_injection': 20, 'memory_poisoning': 10, 'ood_attack_label': 15, 'benign_but_suspicious': 15}
By expected class: {'fdi': 21, 'command_spoof': 27, 'replay': 17, 'dos': 19, 'none': 51, 'firmware': 11}
By confidence band: {'high': 18, 'medium': 53, 'low': 75}

Label derivation: expected_action comes from the deployed class->action registry (coordinator.py override table); expected class comes from scenario ground truth (scenario_derived) or the generating rule of the family. No label was assigned or altered after observing model output.
Provenance: every record carries metadata.provenance (generator id, and for scenario_derived the exact source run directory, timestep, and frozen scenario configuration hash).
Prompt style: all prompts are programmatically generated or scenario-derived; none are human-authored; prompt_injection and memory_poisoning families are adversarially transformed.
Near-duplicate detection: normalised exact match + Jaccard >= 0.6 over 5-gram shingles against all training/val/test prompts and all accepted candidates.
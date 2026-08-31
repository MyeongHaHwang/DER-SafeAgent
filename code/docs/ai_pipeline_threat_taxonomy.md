# Derivation of the AI-pipeline threat categories (§4.7)

Revision: `ijcip_revision_r1r2_20260805`. Written in response to the reviewer
request to explain how the threat categories in §4.7 were identified. The
original manuscript asserted six categories and fourteen perturbation families
without stating a method or citing a source taxonomy; this document supplies
the method, the sources, and the mapping, and is intended to be reproducible.

**Honesty note on ordering.** The fourteen perturbation families were
implemented in the 2025–2026 development of the adversarial suite *before* this
derivation document was written. What follows is therefore a *reconstruction
and consolidation*: the categories are mapped onto published taxonomies after
the fact, and the mapping identifies both the coverage the suite has and the
published threats it does **not** exercise (§6). It is not a claim that the
suite was derived top-down from these sources.

---

## 1. Source collection

Sources were collected in two passes: (a) documents already held in the project
corpus directory `code/finetuning_dataset/training_data/` (verified by opening
the PDFs and reading their title pages and relevant sections), and (b) public
framework documents retrieved and read during this revision. Every source below
was actually opened; nothing is cited from memory.

| # | Source | Type | Verified how |
|---|---|---|---|
| S1 | Microsoft, *Taxonomy of Failure Mode in Agentic AI Systems* (2025), Bryan, Severi, de Gruyter, Jones, Bullwinkel, Minnich, Chawla, Lopez, Pouliot, Fourney, Maxwell, Pratt, Qi, Chikanov, Lutz, Dheekonda, Jagdagdorj, Kim, Song, Hines, Lundeen, Vaughan, Westerhoff, Zunger, Kawaguchi, Russinovich, Siva Kumar. 28 pp. | Agentic-AI security taxonomy | Local PDF read (`2025_Taxonomy-of-Failure-Mode-in-Agentic-AI-Systems-Whitepaper_Microsoft.pdf`); failure-mode sections extracted |
| S2 | OWASP, *Top 10 for LLM Applications 2025* (LLM01–LLM10) | LLM application security taxonomy | Retrieved from `genai.owasp.org/llm-top-10/` during this revision |
| S3 | MITRE ATT&CK for ICS — technique pages incl. T0856 *Spoof Reporting Message*, T0831 *Manipulation of Control*, T0830 *Adversary-in-the-Middle* | ICS adversary-behaviour knowledge base | Retrieved from `attack.mitre.org` during this revision |
| S4 | NIST SP 800-82 Rev. 3, *Guide to Operational Technology (OT) Security* (Sept 2023), Stouffer et al., DOI 10.6028/NIST.SP.800-82r3 | OT security guidance | Retrieved from `csrc.nist.gov` (title, authors, date, DOI confirmed) |
| S5 | NIST SP 800-61 Rev. 2, *Computer Security Incident Handling Guide*, Cichonski, Millar, Grance, Scarfone | Incident-response process guidance | Local PDF read |
| S6 | NIST, *The NIST Cybersecurity Framework (CSF) 2.0* (Feb 2024), DOI 10.6028/NIST.CSWP.29 | Risk-management framework | Local PDF read |
| S7 | Narajala & Habler, *Enterprise-Grade Security for the Model Context Protocol (MCP): Frameworks and Mitigation Strategies*, arXiv:2504.08623v2 (2025) | Agent tool-use security | Local PDF read |
| S8 | Alhelou, Hatziargyriou & Dong (eds.), *Power Systems Cybersecurity: Methods, Concepts, and Best Practices*, Springer (2023) | DER/power-system attack surface | Local PDF read |
| S9 | Li, Wang, Zeng, Wu & Yang, *A survey on LLM-based multi-agent systems*, Vicinagearth 1:9 (2024), DOI 10.1007/s44336-024-00009-2 | Multi-agent system risks | Local PDF read |
| S10 | OWASP, *Agent Name Service (ANS) for Secure AI Agent Discovery* v1.0 (May 2025) | Agent identity/discovery security | Local PDF read |

Not used, and why: MITRE ATLAS could not be read (the landing page returned no
extractable content during this revision), so no ATLAS technique ID is cited
anywhere. NERC CIP standard pages returned HTTP 403 and are logged in
`paper/reference_gaps.md` rather than cited.

## 2. Inclusion criteria

A source was included only if it (i) is a published framework, standard, or
peer-reviewed/【archived】 paper; (ii) addresses either the security of
LLM/agentic pipelines or the OT/DER attack surface the pipeline operates on;
and (iii) could be opened and read in full or in the relevant section during
this revision. A *threat* from a source was carried forward only if it can act
on **this** system's actual attack surface: the alert/telemetry ingestion path,
the prompt, the incident memory, the JSON decision object, or the operator
approval step. Threats that presuppose components DER-SecAgent does not have
(model theft, training-supply-chain compromise, autonomous tool discovery,
inter-agent network protocols) were excluded and are listed in §6 as
out-of-scope rather than silently dropped.

## 3. Consolidation procedure

1. Extract each source's threat list verbatim (S1 failure modes; S2 LLM01–LLM10;
   S3 ICS techniques relevant to telemetry/command integrity; S7 tool-use risks).
2. Discard entries excluded by the criteria in §2.
3. Merge entries that describe the same mechanism against this system's surface.
   Example: S1 *Cross domain prompt injection (XPIA)*, S2 *LLM01 Prompt
   Injection*, and S1 *Agent injection* all reduce, for DER-SecAgent, to
   "attacker-controlled text reaches the model through an ingested log/event
   field" → category C1.
4. Group the merged mechanisms by **attack channel** (where the adversary's
   bytes enter) rather than by outcome, because the containment mechanism is
   determined by the channel. This yields the six categories C1–C6.
5. Instantiate each category as concrete perturbation families in the
   executable suite, at the granularity at which the suite can actually vary
   the input (§4).

## 4. Category → perturbation-family mapping

The six categories and the fourteen implemented families. "Target component"
is the pipeline element the perturbation acts on; "containment" is the
mechanism expected to stop it; "test" is the executable evidence.

### C1 — Prompt injection through ingested logs and events
*Sources: S1 (XPIA; Agent injection), S2 (LLM01), S7 (tool-input trust).*
Attacker-controlled text in a syslog/alarm/command payload is ingested and read
by the Hypothesis Agent as if it were instruction.

| Family | Target | Containment | Test |
|---|---|---|---|
| F1 log prompt injection | Hypothesis prompt | Deterministic FeatureView grounding; fixed-enum output schema; class-aware avoidance | adversarial suite; corpus family `prompt_injection` |
| F2 role-play injection | Hypothesis prompt | as F1 | adversarial suite |
| F3 encoded-payload injection | Hypothesis prompt | as F1 | adversarial suite |
| F4 adaptive jailbreak | Hypothesis prompt | as F1 + Caution veto | adversarial suite |

### C2 — Command/parameter injection into structured fields
*Sources: S2 (LLM05 improper output handling), S7 (tool invocation), S3 (T0831).*
Shell/SQL/template payloads placed in `asset_id`, `rationale`, or protocol
fields, aiming to have attacker text cross the execution boundary.

| Family | Target | Containment | Test |
|---|---|---|---|
| F5 command injection | Action/target fields | Actions are enum values from a fixed five-primitive registry, never strings handed to a shell; target validated against known assets | `test_out_of_registry_actions_are_rejected`; `test_final_action_is_always_inside_the_registry` |
| F6 tool-call spoofing | Action selection | Registry boundary; Coordinator owns the final action, not the model | adversarial suite |

### C3 — Malformed and missing input
*Sources: S2 (LLM05), S5 (incident-data quality), S4 (OT data integrity).*
Corrupted JSON, absent required fields, wrong types, nulls.

| Family | Target | Containment | Test |
|---|---|---|---|
| F7 malformed alert | Parser/schema | Per-call JSON-validity and schema checks; `no_op` fallback below the confidence floor | corpus family `malformed_alert`; schema tests |
| F8 missing-field | Schema | Caution rule (d): missing affected asset → flag | adversarial suite |
| F9 wrong-type field | Schema | Type validation at the pipeline boundary | adversarial suite |

### C4 — Conflicting evidence across channels
*Sources: S3 (T0856 Spoof Reporting Message; T0830 AiTM), S8 (DER telemetry integrity).*
The event channel and the telemetry channel disagree — the direct consequence
of an adversary who can spoof one channel but not both.

| Family | Target | Containment | Test |
|---|---|---|---|
| F10 conflicting telemetry | Hypothesis/Coordinator | Lowest-worst-case-impact selection; Caution rationale/action consistency rule; confidence floor | corpus family `conflicting_telemetry` |
| F11 benign-but-suspicious | Hypothesis | Severity threshold; abstention; `no_op` | corpus family `benign_but_suspicious` |

### C5 — Memory poisoning
*Sources: S1 (Memory poisoning; Targeted knowledge base poisoning), S2 (LLM04), S9 (MAS state sharing).*
Adversarial entries planted in IncidentMemory so that retrieval biases future
decisions.

| Family | Target | Containment | Test |
|---|---|---|---|
| F12 single-turn memory poisoning | Retrieved memory block | Memory is prior *context* only; the Coordinator's selection never reads memory; class-aware avoidance still applies | corpus family `memory_poisoning`; `test_injected_isolation_proposal_is_contained` |
| F13 multi-turn memory poisoning | Retrieved memory block | as F12 | adversarial suite |

### C6 — Out-of-distribution attack labels
*Sources: S2 (LLM09 misinformation), S5 (classification discipline), S1 (agent misalignment).*
Upstream systems assert labels outside the six-class DER taxonomy.

| Family | Target | Containment | Test |
|---|---|---|---|
| F14 out-of-distribution attack label | Class field | Enum rejects out-of-vocabulary classes; map to `none` + abstain; HITL routing below threshold | corpus family `ood_attack_label` |

Two families that the original manuscript listed without a clear parent
category are placed explicitly here: **benign-but-suspicious** belongs to C4
(it is the null case of the conflicting-evidence channel — evidence that looks
alarming but is corroborated as benign), and **adaptive jailbreak** belongs to
C1 (it is a prompt-channel attack that adapts to observed refusals).

## 5. Human-oversight threat (identified, partially covered)

S1 lists **Human-in-the-loop (HitL) bypass** — including approval fatigue —
as a distinct failure mode. DER-SecAgent has no perturbation family for it,
but the HITL sensitivity study is directly relevant evidence: the `approve`
operator stand-in (an operator who approves everything) yields the highest
unsafe-command rate of any behaviour tested. That result is the empirical form
of the S1 concern and is reported as such rather than as a robustness success.

## 6. Published threats deliberately **not** covered

Stating these is part of the derivation, not an omission from it:

- S2 LLM03 *Supply chain* and LLM04 *Data and model poisoning* at training
  time: the adapter is trained locally on a fixed corpus; no defence is claimed.
- S2 LLM10 / model theft and LLM07 *System prompt leakage*: out of scope; the
  prompts are published in the manuscript appendix.
- S2 LLM10:2025 *Unbounded consumption*: not exercised; the trigger gate limits
  invocation but no resource-exhaustion attack was tested.
- S1 *Agent compromise* by a threat-actor-controlled model, and S10 agent
  identity/discovery attacks: the system runs a single locally-held adapter with
  no agent discovery, so these are architecturally absent rather than defended.
- S3 ICS techniques that act below the telemetry layer (e.g. firmware
  implantation): the `firmware` class exists in the schema but has no physical
  injector, so it is exercised only at prompt/schema level.

## 7. Reproducing this derivation

```bash
# sources held locally
ls code/finetuning_dataset/training_data/
pdftotext code/finetuning_dataset/training_data/2025_Taxonomy-of-Failure-Mode-in-Agentic-AI-Systems-Whitepaper_Microsoft.pdf - | less

# families as implemented
python3 -c "import pandas as pd; print(pd.read_csv('code/results/ijcip_adversarial_safety/family_breakdown_expanded.csv').perturbation.unique())"

# corpus families used in the expanded evaluation
python3 -c "import json,collections; print(collections.Counter(json.loads(l)['family'] for l in open('code/finetuning_dataset/revision_eval/eval_corpus.jsonl')))"

# containment tests
python3 -m pytest code/evaluation/trustworthy_validation/ -q
```

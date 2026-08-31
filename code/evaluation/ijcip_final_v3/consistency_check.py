"""§12 consistency checker: manuscript vs canonical results + banned-term scan.

Checks:
  1. Banned/over-claim terms in the manuscript body (proved, guaranteed,
     equivalent, real-world validated, statistically indistinguishable, ...).
  2. Key numbers present in the manuscript match canonical result files.
  3. Qwen and Llama artifacts are distinct (no duplicated adapter SHA).
  4. Bare arms report N/A (not 0) for gate coverage.

Exit code 0 if all pass, 1 otherwise.
Run: python3 -m code.evaluation.ijcip_final_v3.consistency_check
"""
from __future__ import annotations
import re, sys, json
from pathlib import Path
import pandas as pd

SEC = Path("paper/sections")
RES = Path("code/results/ijcip_final_v3")

BANNED = [
    r"\bproved\b", r"\bprove(s|n)?\b", r"\bguarantee(s|d)?\b(?!.{0,40}not)",
    r"statistically indistinguishable", r"\bequivalent\b", r"real-world validated",
    r"first-ever", r"provably safe", r"complete FDI detection",
]
# terms allowed only when negated/qualified are handled by manual review; here we
# just report occurrences for the author to confirm.
ALLOW_CONTEXT = ["not a formal proof", "no formal proof", "not a guarantee",
                 "does not guarantee", "not claim a formal proof",
                 "no statistically detectable", "not equivalence"]


def scan_terms():
    hits = []
    for f in sorted(SEC.glob("*.tex")):
        txt = f.read_text()
        for i, line in enumerate(txt.splitlines(), 1):
            if line.strip().startswith("%"):
                continue
            low = line.lower()
            if any(a in low for a in ALLOW_CONTEXT):
                continue
            for pat in BANNED:
                if re.search(pat, low):
                    hits.append((f.name, i, pat, line.strip()[:90]))
    return hits


def check_numbers():
    problems = []
    # holdout canonical
    df = pd.read_csv(RES / "holdout_e2e" / "holdout_e2e_raw.csv")
    atk = df[df.kind == "attack"]
    ens = atk.groupby("arm").ens_kwh.mean().round(3)
    body = " ".join((SEC / "5-experiments.tex").read_text().split())
    for arm, val in [("D0", ens["D0"]), ("QPROJ", ens["QPROJ"]),
                     ("LPROJ", ens["LPROJ"]), ("OPROJ", ens["OPROJ"])]:
        if f"{val:.3f}" not in body and f"{val:.2f}" not in body:
            problems.append(f"holdout ENS {arm}={val} not found verbatim in RQ3 text")
    # adapters distinct
    shas = {a: sorted(atk[atk.arm == a].adapter_sha.dropna().unique())
            for a in ["bareQ", "bareL"]}
    if shas["bareQ"] == shas["bareL"] and shas["bareQ"]:
        problems.append(f"bareQ/bareL share adapter SHA {shas['bareQ']} (duplication)")
    # bare arms N/A in holdout table
    ht = (Path("paper/tables/final_v3/table_holdout_v3.tex")).read_text()
    for arm in ["bareQ", "bareL"]:
        row = [l for l in ht.splitlines() if l.startswith(arm)]
        if row and "N/A" not in row[0]:
            problems.append(f"{arm} gate coverage not N/A in Table 8")
    # property test 0 violations
    pt = json.loads((RES / "property_tests" / "property_test_result.json").read_text())
    if not pt["ALL_PROPERTIES_HOLD"] or not pt["mutations_all_caught"]:
        problems.append("property test not clean")
    return problems


def main():
    terms = scan_terms()
    nums = check_numbers()
    print("=== banned/over-claim term occurrences (author must confirm each) ===")
    for f, i, pat, line in terms:
        print(f"  {f}:{i}  [{pat}]  {line}")
    if not terms:
        print("  (none)")
    print("\n=== number/lineage checks ===")
    for p in nums:
        print("  PROBLEM:", p)
    if not nums:
        print("  all number/lineage checks pass")
    ok = (len(nums) == 0)
    print(f"\nCONSISTENCY: {'PASS' if ok else 'FAIL'}  "
          f"({len(terms)} term occurrences to confirm, {len(nums)} hard problems)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

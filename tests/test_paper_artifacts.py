"""Paper-artifact regeneration smoke test (CPU-only).

Runs the canonical artifact builder into a temporary directory and checks
that every regenerated numeric table is byte-identical to the reference copy
of the submitted manuscript, and every regenerated figure PNG is
byte-identical to the reference. This is the CI-facing equivalent of
``make reproduce-paper`` (which additionally regenerates the statistics).
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REF_T = ROOT / "artifacts/reference/tables"
REF_F = ROOT / "artifacts/reference/figures"

TABLES = ["table_gate_v3.tex", "table_eh_v3.tex", "table_holdout_v3.tex",
          "table_adversarial_v3_qwen.tex", "table_adversarial_v3_llama.tex",
          "table_stats_primary.tex", "table_latency_v3.tex"]
PNGS = ["fig_containment_v3.png", "fig_gate_v3.png", "fig_holdout_ens_diff.png"]


@pytest.fixture(scope="module")
def built(tmp_path_factory, monkeypatch_module=None):
    from code.evaluation.final_safeagent_v3 import build_artifacts_v3 as b
    tmp = tmp_path_factory.mktemp("artifacts")
    orig = (b.TAB, b.FIG, b.PROC)
    b.TAB, b.FIG, b.PROC = tmp / "tables", tmp / "figures", tmp / "processed"
    try:
        import matplotlib
        matplotlib.rcParams["pdf.compression"] = matplotlib.rcParams["pdf.compression"]
        import os
        os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
        b.main()
    finally:
        pass
    yield tmp
    b.TAB, b.FIG, b.PROC = orig


@pytest.mark.parametrize("name", TABLES)
def test_table_matches_manuscript(built, name):
    gen = built / "tables" / name
    assert gen.exists(), f"builder did not produce {name}"
    assert gen.read_bytes() == (REF_T / name).read_bytes(), \
        f"{name} differs from the manuscript reference"


@pytest.mark.parametrize("name", PNGS)
def test_figure_png_matches_manuscript(built, name):
    gen = built / "figures" / name
    assert gen.exists(), f"builder did not produce {name}"
    assert gen.read_bytes() == (REF_F / name).read_bytes(), \
        f"{name} differs from the manuscript reference"

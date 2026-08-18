"""Repository-level test fixtures.

The research package uses repository-root-relative paths throughout
(``code/results/…``, ``code/configs/…``), so tests always run with the
repository root as the working directory regardless of where pytest was
invoked from.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

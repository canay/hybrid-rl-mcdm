"""Compatibility wrapper for the v15 artifact-driven figure generator."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.generate_figures_v15 import main


if __name__ == "__main__":
    main()

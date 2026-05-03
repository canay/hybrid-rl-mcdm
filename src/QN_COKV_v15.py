"""
QN_COKV_v15.py
==============

Thin orchestration entry point for the modular v15 experiment suite.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src import generate_figures_v15, v15_data, v15_drift, v15_extended, v15_primary, v15_robustness


STAGES = ["data", "primary", "extended", "robustness", "drift", "figures", "all"]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the modular v15 Hybrid RL-MCDM pipeline.")
    parser.add_argument(
        "--stage",
        choices=STAGES,
        default="all",
        help="Pipeline stage to run. Default runs the full stack.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    if args.stage in {"data", "all"}:
        v15_data.main([])
    if args.stage in {"primary", "all"}:
        v15_primary.main([])
    if args.stage in {"extended", "all"}:
        v15_extended.main([])
    if args.stage in {"robustness", "all"}:
        v15_robustness.main([])
    if args.stage in {"drift", "all"}:
        v15_drift.main([])
    if args.stage in {"figures", "all"}:
        generate_figures_v15.main([])


if __name__ == "__main__":
    main()

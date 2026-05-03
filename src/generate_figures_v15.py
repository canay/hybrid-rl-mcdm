"""
generate_figures_v15.py
=======================

Artifact-driven figure generation for paper_v15.tex.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.ticker as ticker
import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.v15_core import DRIFT_CHECKPOINTS, GT_SPLIT_GRID, LAMBDA_GRID, MAIN_CHECKPOINTS
from src.v15_data import REPO_ROOT


FIG_DIR = REPO_ROOT / "manuscript" / "latex" / "fig"
PRIMARY_PATH = REPO_ROOT / "results" / "v15_primary.json"
EXTENDED_PATH = REPO_ROOT / "results" / "v15_extended.json"
DRIFT_PATH = REPO_ROOT / "results" / "v15_drift.json"

C = {
    "hybrid": "#0072B2",
    "rl": "#009E73",
    "topsis": "#CC79A7",
    "popularity": "#E69F00",
    "random": "#BBBBBB",
}

ORDER = ["hybrid", "rl_only", "topsis_only", "popularity", "random"]
LABELS = {
    "hybrid": "Hybrid RL-MCDM",
    "rl_only": "RL-only",
    "topsis_only": "TOPSIS-only",
    "popularity": "Popularity",
    "random": "Random",
}
COLOR_KEY = {
    "hybrid": "hybrid",
    "rl_only": "rl",
    "topsis_only": "topsis",
    "popularity": "popularity",
    "random": "random",
}
LS = {"hybrid": "-", "rl_only": "-.", "topsis_only": ":", "popularity": "--", "random": "--"}
MARKER = {"hybrid": "o", "rl_only": "s", "topsis_only": "D", "popularity": "^", "random": "v"}
MSIZE = {"hybrid": 7, "rl_only": 6, "topsis_only": 5.5, "popularity": 5.5, "random": 5.5}
LW = {"hybrid": 2.2, "rl_only": 1.6, "topsis_only": 1.5, "popularity": 1.3, "random": 1.3}


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica Neue", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10.5,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "grid.color": "#AAAAAA",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.axisbelow": True,
}
)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save(fig: plt.Figure, name: str, pad_inches: float = 0.02) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", pad_inches=pad_inches)
    print(f"  wrote {name}.pdf and {name}.png")


def workflow_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    body: str,
    facecolor: str,
    edgecolor: str = "#4B5563",
) -> dict[str, tuple[float, float]]:
    box = mpatches.FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.1,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    box.set_path_effects(
        [
            pe.SimplePatchShadow(offset=(2.0, -2.0), alpha=0.16, rho=0.98),
            pe.Normal(),
        ]
    )
    ax.add_patch(box)

    ax.text(
        x + 0.020,
        y + height - 0.040,
        title,
        ha="left",
        va="top",
        fontsize=9.9,
        fontweight="bold",
        color="#1F2937",
    )
    ax.text(
        x + 0.020,
        y + height - 0.090,
        body,
        ha="left",
        va="top",
        fontsize=8.6,
        color="#334155",
        linespacing=1.30,
    )

    return {
        "left": (x, y + height / 2),
        "right": (x + width, y + height / 2),
        "top": (x + width / 2, y + height),
        "bottom": (x + width / 2, y),
        "center": (x + width / 2, y + height / 2),
    }


def workflow_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = "#64748B",
    connectionstyle: str = "arc3,rad=0.0",
    linestyle: str = "-",
) -> None:
    arrow = mpatches.FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=1.3,
        linestyle=linestyle,
        color=color,
        connectionstyle=connectionstyle,
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(arrow)


def workflow_polyline_arrow(
    ax: plt.Axes,
    points: list[tuple[float, float]],
    color: str = "#64748B",
    linewidth: float = 1.3,
) -> None:
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=color,
            lw=linewidth,
            solid_capstyle="round",
            zorder=2,
        )
    arrow = mpatches.FancyArrowPatch(
        points[-2],
        points[-1],
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=linewidth,
        color=color,
        shrinkA=0,
        shrinkB=2,
    )
    ax.add_patch(arrow)


def fig_architecture() -> None:
    fig, ax = plt.subplots(figsize=(11.6, 5.5))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    x1, x2, x3, x4 = 0.04, 0.285, 0.53, 0.775
    box_w = 0.185
    box_h = 0.19
    top_y = 0.55
    bottom_y = 0.22

    section_specs = [
        (x1, box_w, "CONTROLLED SETUP"),
        (x2, box_w * 2 + (x3 - x2 - box_w), "DUAL SIGNAL MODELING"),
        (x4, box_w, "OUTPUTS"),
    ]
    for x, span, label in section_specs:
        ax.text(x, 0.93, label, fontsize=8.7, fontweight="bold", color="#7C8798", va="center")
        ax.plot([x, x + span], [0.905, 0.905], color="#D4DCE6", lw=1.0)

    catalog = workflow_box(
        ax,
        x1,
        top_y,
        box_w,
        box_h,
        "Synthetic Product Catalog",
        "N = 400 items\nprice, quality,\npopularity, rating",
        facecolor="#F7F2EA",
    )
    profiles = workflow_box(
        ax,
        x1,
        bottom_y,
        box_w,
        box_h,
        "Hidden User Profiles",
        "5 segments\nbrand, price window,\ncategory, recency",
        facecolor="#F6F0FF",
    )
    topsis = workflow_box(
        ax,
        x2,
        top_y,
        box_w,
        box_h,
        "Observable Scoring",
        "Entropy-weighted TOPSIS\nprofile-agnostic score C*",
        facecolor="#EAF4FF",
        edgecolor="#3B82F6",
    )
    simulator = workflow_box(
        ax,
        x2,
        bottom_y,
        box_w,
        box_h,
        "Ground Truth + Reward",
        "observable + hidden utility\nalpha = 0.50\nbehavior-dominant reward",
        facecolor="#ECFDF5",
        edgecolor="#10B981",
    )
    qlearn = workflow_box(
        ax,
        x3,
        bottom_y,
        box_w,
        box_h,
        "Behavioral Learning",
        "Tabular Q-learning\nprofile-specific Q(u,i)",
        facecolor="#EEF2FF",
        edgecolor="#6366F1",
    )
    fusion = workflow_box(
        ax,
        x3,
        top_y,
        box_w,
        box_h,
        "Fusion + Ranking",
        "S(u,i) = 0.50 C + 0.50 Q\nsorted Top-7 list",
        facecolor="#FFF3E8",
        edgecolor="#F97316",
    )
    topk = workflow_box(
        ax,
        x4,
        top_y,
        box_w,
        box_h,
        "Recommendations",
        "Top-K items\nitem-level signal mix",
        facecolor="#F0FDF4",
        edgecolor="#22C55E",
    )
    evaluation = workflow_box(
        ax,
        x4,
        bottom_y,
        box_w,
        box_h,
        "Evaluation Suite",
        "30-run bootstrap\nF1@7, NDCG@7, ILD\ndrift, lambda, alpha, XAI",
        facecolor="#FFF7ED",
        edgecolor="#FB923C",
    )

    workflow_arrow(ax, catalog["right"], topsis["left"], color="#4B5563")
    workflow_arrow(ax, profiles["right"], simulator["left"], color="#4B5563")
    workflow_arrow(ax, simulator["right"], qlearn["left"], color="#10B981")
    workflow_arrow(ax, topsis["right"], fusion["left"], color="#3B82F6")
    workflow_arrow(ax, qlearn["top"], fusion["bottom"], color="#6366F1")
    workflow_arrow(ax, fusion["right"], topk["left"], color="#F97316")
    workflow_arrow(ax, topk["bottom"], evaluation["top"], color="#64748B")
    workflow_polyline_arrow(
        ax,
        [
            (catalog["center"][0], catalog["bottom"][1]),
            (catalog["center"][0], 0.47),
            (simulator["center"][0], 0.47),
            (simulator["center"][0], simulator["top"][1]),
        ],
        color="#4B5563",
    )

    ax.text(
        (topsis["center"][0] + fusion["center"][0]) / 2,
        0.79,
        "Observable branch",
        ha="center",
        va="center",
        fontsize=8.8,
        fontweight="bold",
        color="#3B82F6",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8},
    )
    ax.text(
        (simulator["center"][0] + qlearn["center"][0]) / 2,
        0.44,
        "Behavioral branch",
        ha="center",
        va="center",
        fontsize=8.8,
        fontweight="bold",
        color="#6366F1",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8},
    )

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.02)
    save(fig, "fig_architecture", pad_inches=0.01)
    plt.close(fig)


def stars_for_p(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def summary_point(summary: dict, method: str, checkpoint: int | str) -> dict:
    return summary[method][str(checkpoint)]


def fig_comparison(primary: dict) -> None:
    summary = primary["summary"]
    significance = primary["significance"]
    final_cp = MAIN_CHECKPOINTS[-1]

    fig, ax = plt.subplots(figsize=(7.3, 4.7))

    x = np.arange(len(ORDER))
    means = [summary_point(summary, method, final_cp)["mean"] for method in ORDER]
    lows = [means[idx] - summary_point(summary, method, final_cp)["ci_lo"] for idx, method in enumerate(ORDER)]
    highs = [summary_point(summary, method, final_cp)["ci_hi"] - means[idx] for idx, method in enumerate(ORDER)]
    colors = [C[COLOR_KEY[method]] for method in ORDER]
    labels = [LABELS[method] for method in ORDER]

    bars = ax.bar(
        x,
        means,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
        width=0.62,
        zorder=3,
        yerr=[lows, highs],
        error_kw={"elinewidth": 1.2, "capsize": 4, "capthick": 1.2, "ecolor": "#555555", "zorder": 4},
    )

    for idx, (bar, mean) in enumerate(zip(bars, means)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + highs[idx] + 0.014,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#333333",
        )

    hybrid_top = means[0] + highs[0]
    bracket_y = max(hybrid_top, max(means[idx] + highs[idx] for idx in range(len(ORDER)))) + 0.06
    for offset, baseline in enumerate(["rl_only", "topsis_only", "popularity", "random"]):
        baseline_idx = ORDER.index(baseline)
        p_value = significance[f"hybrid_vs_{baseline}"]["p_value"]
        ax.plot([x[0], x[baseline_idx]], [bracket_y, bracket_y], color="#888888", lw=0.9)
        ax.text((x[0] + x[baseline_idx]) / 2, bracket_y + 0.007, stars_for_p(p_value), ha="center", va="bottom", fontsize=8)
        bracket_y += 0.038

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("F1@7")
    ax.set_ylim(0, 1.14)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    plt.tight_layout()
    save(fig, "fig_comparison")
    plt.close(fig)


def fig_convergence(primary: dict) -> None:
    summary = primary["summary"]
    xtick_labels = ["0.5K", "1K", "2K", "5K", "10K", "20K", "30K"]
    fig, ax = plt.subplots(figsize=(7.8, 4.8))

    for method in ["hybrid", "rl_only"]:
        means = np.array([summary_point(summary, method, cp)["mean"] for cp in MAIN_CHECKPOINTS], dtype=float)
        stds = np.array([summary_point(summary, method, cp)["std"] for cp in MAIN_CHECKPOINTS], dtype=float)
        ax.fill_between(
            range(len(MAIN_CHECKPOINTS)),
            means - stds,
            means + stds,
            alpha=0.12,
            color=C[COLOR_KEY[method]],
            zorder=1,
        )

    for method in ORDER:
        means = [summary_point(summary, method, cp)["mean"] for cp in MAIN_CHECKPOINTS]
        ax.plot(
            range(len(MAIN_CHECKPOINTS)),
            means,
            color=C[COLOR_KEY[method]],
            ls=LS[method],
            lw=LW[method],
            marker=MARKER[method],
            markersize=MSIZE[method],
            label=LABELS[method],
            zorder=5 if method == "hybrid" else 4,
        )
        ax.text(len(MAIN_CHECKPOINTS) - 1 + 0.12, means[-1], f"{means[-1]:.3f}", va="center", fontsize=8, color=C[COLOR_KEY[method]])

    ax.set_xticks(range(len(MAIN_CHECKPOINTS)))
    ax.set_xticklabels(xtick_labels)
    ax.set_xlabel("Training episodes")
    ax.set_ylabel("Mean F1@7")
    ax.set_xlim(-0.2, len(MAIN_CHECKPOINTS) - 0.4)
    ax.set_ylim(-0.01, 1.00)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.legend(loc="upper left", framealpha=0.92, borderpad=0.6, labelspacing=0.4)

    plt.tight_layout()
    save(fig, "fig_convergence")
    plt.close(fig)


def fig_lambda(extended: dict, primary: dict) -> None:
    summary = extended["lambda_sensitivity"]["summary"]
    primary_summary = primary["summary"]

    lambdas = np.array(LAMBDA_GRID, dtype=float)
    f1_lam = np.array([summary[str(lam)]["mean"] for lam in lambdas], dtype=float)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(lambdas, f1_lam, color=C["hybrid"], lw=2.2, marker="o", markersize=7, zorder=4)
    ax.axvspan(0.40, 0.50, alpha=0.15, color=C["hybrid"], label="Optimal range lambda_Q in [0.40, 0.50]")
    ax.axvline(0.40, color=C["hybrid"], lw=0.9, ls=":", alpha=0.7)
    ax.axvline(0.50, color=C["hybrid"], lw=0.9, ls=":", alpha=0.7)

    rl_mean = summary_point(primary_summary, "rl_only", MAIN_CHECKPOINTS[-1])["mean"]
    topsis_mean = summary_point(primary_summary, "topsis_only", MAIN_CHECKPOINTS[-1])["mean"]
    ax.axhline(rl_mean, color=C["rl"], lw=1.2, ls="--", alpha=0.8, label=f"RL-only = {rl_mean:.3f}")
    ax.axhline(topsis_mean, color=C["topsis"], lw=1.2, ls=":", alpha=0.8, label=f"TOPSIS-only = {topsis_mean:.3f}")

    peak_idx = int(np.argmax(f1_lam))
    ax.scatter([lambdas[peak_idx]], [f1_lam[peak_idx]], color=C["hybrid"], s=85, zorder=5)
    ax.text(lambdas[peak_idx], f1_lam[peak_idx] + 0.03, f"{f1_lam[peak_idx]:.3f}", ha="center", fontsize=8.5)

    ax.set_xlabel("RL fusion weight lambda_Q")
    ax.set_ylabel("Mean F1@7")
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0.0, 1.00)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.10))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.legend(loc="lower center", framealpha=0.92, fontsize=8.5)

    plt.tight_layout()
    save(fig, "fig_lambda")
    plt.close(fig)


def fig_split_ablation(extended: dict) -> None:
    summary = extended["gt_split_sensitivity"]["summary"]
    alphas = np.array(GT_SPLIT_GRID, dtype=float)

    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    for method in ["hybrid", "rl_only", "topsis_only"]:
        ax.plot(
            alphas,
            [summary[method][str(alpha)]["mean"] for alpha in alphas],
            color=C[COLOR_KEY[method]],
            lw=2.2 if method == "hybrid" else 1.6,
            marker=MARKER[method],
            ms=7 if method == "hybrid" else 6,
            ls=LS[method],
            label=LABELS[method],
            zorder=5 if method == "hybrid" else 4,
        )

    ax.axvline(0.50, color="#888888", lw=0.9, ls="--", alpha=0.7, label="Paper design point alpha = 0.50")
    ax.set_xlabel("Observable fraction alpha in ground truth")
    ax.set_ylabel("Mean F1@7")
    ax.set_xlim(0.26, 0.84)
    ax.set_ylim(0.0, 1.00)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.10))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.legend(loc="upper left", framealpha=0.92, fontsize=8.5)

    plt.tight_layout()
    save(fig, "fig_split_ablation")
    plt.close(fig)


def fig_drift(drift: dict) -> None:
    summary = drift["summary"]
    fig, ax = plt.subplots(figsize=(7.9, 4.9))
    x = np.arange(len(DRIFT_CHECKPOINTS))

    for method in ["hybrid", "rl_only"]:
        means = np.array([summary_point(summary, method, cp)["mean"] for cp in DRIFT_CHECKPOINTS], dtype=float)
        ci_lo = np.array([summary_point(summary, method, cp)["ci_lo"] for cp in DRIFT_CHECKPOINTS], dtype=float)
        ci_hi = np.array([summary_point(summary, method, cp)["ci_hi"] for cp in DRIFT_CHECKPOINTS], dtype=float)
        ax.fill_between(x, ci_lo, ci_hi, alpha=0.12, color=C[COLOR_KEY[method]], zorder=1)
        ax.plot(
            x,
            means,
            color=C[COLOR_KEY[method]],
            lw=2.2 if method == "hybrid" else 1.8,
            marker=MARKER[method],
            markersize=7 if method == "hybrid" else 6,
            ls="-" if method == "hybrid" else "--",
            label=LABELS[method],
            zorder=4,
        )

    ax.axvspan(3.85, 4.15, alpha=0.10, color="#FF6B6B")
    ax.axvline(4.0, color="#FF4D4D", lw=1.4, ls="--")
    ax.set_xlabel("Training episodes")
    ax.set_ylabel("Mean F1@7")
    ax.set_xticks(x)
    ax.set_xticklabels(["2K", "5K", "10K", "14K", "15K", "16K", "20K", "25K", "30K"], rotation=25, ha="right")
    ax.set_ylim(0.0, 0.95)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.legend(loc="upper left", framealpha=0.92)

    plt.tight_layout()
    save(fig, "fig_drift")
    plt.close(fig)


def fig_ild(extended: dict) -> None:
    summary = extended["ild"]["summary"]
    order = ["random", "popularity", "topsis_only", "rl_only", "hybrid"]
    labels = ["Random", "Popularity", "TOPSIS", "RL-only", "Hybrid"]
    colors = [C[COLOR_KEY.get(method, method)] for method in order]
    means = [summary[method]["mean"] for method in order]
    lows = [means[idx] - summary[method]["ci_lo"] for idx, method in enumerate(order)]
    highs = [summary[method]["ci_hi"] - means[idx] for idx, method in enumerate(order)]

    fig, ax = plt.subplots(figsize=(6.7, 4.5))
    x = np.arange(len(order))
    bars = ax.bar(
        x,
        means,
        color=colors,
        edgecolor="white",
        linewidth=0.7,
        yerr=[lows, highs],
        error_kw={"elinewidth": 1.2, "capsize": 5, "capthick": 1.2, "ecolor": "#222222"},
    )

    for idx, bar in enumerate(bars):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            means[idx] + highs[idx] + 0.02,
            f"{means[idx]:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#222222",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Mean ILD (30K episodes)")
    ax.set_ylim(0.0, 1.15)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    plt.tight_layout()
    save(fig, "fig_ild")
    plt.close(fig)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate artifact-driven figures for paper_v15.")
    parser.add_argument("--primary", type=Path, default=PRIMARY_PATH, help="Primary results JSON.")
    parser.add_argument("--extended", type=Path, default=EXTENDED_PATH, help="Extended results JSON.")
    parser.add_argument("--drift", type=Path, default=DRIFT_PATH, help="Drift results JSON.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    primary = load_json(args.primary)
    extended = load_json(args.extended)
    drift = load_json(args.drift)

    print(f"Output directory: {FIG_DIR}")
    fig_architecture()
    fig_comparison(primary)
    fig_convergence(primary)
    fig_lambda(extended, primary)
    fig_split_ablation(extended)
    fig_drift(drift)
    fig_ild(extended)
    print("All figures generated.")


if __name__ == "__main__":
    main()

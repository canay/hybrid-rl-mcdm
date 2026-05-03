"""
generate_figures_v16.py
Reads v2_full_results.json and v2_supplementary.json,
produces all figures for paper_v16.tex into manuscript/v16/fig/
Uses Wong (2011) colorblind-safe palette.
"""

import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Wong colorblind-safe palette
WONG = {
    "blue":    "#0072B2",
    "orange":  "#E69F00",
    "green":   "#009E73",
    "red":     "#D55E00",
    "purple":  "#CC79A7",
    "cyan":    "#56B4E9",
    "yellow":  "#F0E442",
    "black":   "#000000",
}

ROOT = os.path.dirname(os.path.dirname(__file__))
RES_DIR = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(ROOT, "manuscript", "v16", "fig")
os.makedirs(FIG_DIR, exist_ok=True)

with open(os.path.join(RES_DIR, "v2_full_results.json")) as f:
    D = json.load(f)
with open(os.path.join(RES_DIR, "v2_supplementary.json")) as f:
    S = json.load(f)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 300,
})


def fig_comparison():
    """Bar chart: F1@7 at 30K for all 5 methods."""
    methods = ["Random", "Popularity", "TOPSIS-only", "RL-only", "Hybrid\nRL-MCDM"]
    keys = ["random", "popularity", "topsis_only", "rl_only", "hybrid"]
    means = [D["main_results"][k]["mean"] for k in keys]
    stds  = [D["main_results"][k]["std"]  for k in keys]
    colors = [WONG["purple"], WONG["yellow"], WONG["orange"], WONG["cyan"], WONG["blue"]]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=[1.96*s/np.sqrt(30) for s in stds],
                  color=colors, edgecolor="black", linewidth=0.6,
                  capsize=4, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("F1@7")
    ax.set_ylim(0, 1.05)
    # Add value labels
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f"{m:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_comparison.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "fig_comparison.png"), bbox_inches="tight")
    plt.close(fig)
    print("  fig_comparison done")


def fig_convergence():
    """Learning curves at 7 checkpoints."""
    eps = sorted(D["convergence"].keys(), key=int)
    ep_labels = [int(e) for e in eps]

    hybrid_m = [D["convergence"][e]["hybrid"]["mean"] for e in eps]
    hybrid_s = [D["convergence"][e]["hybrid"]["std"] for e in eps]
    rl_m     = [D["convergence"][e]["rl_only"]["mean"] for e in eps]
    rl_s     = [D["convergence"][e]["rl_only"]["std"] for e in eps]
    topsis_m = [D["convergence"][e]["topsis_only"]["mean"] for e in eps]
    pop_m    = [D["convergence"][e]["popularity"]["mean"] for e in eps]
    rand_m   = [D["convergence"][e]["random"]["mean"] for e in eps]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(eps))

    ax.fill_between(x, np.array(hybrid_m)-np.array(hybrid_s),
                    np.array(hybrid_m)+np.array(hybrid_s),
                    alpha=0.15, color=WONG["blue"])
    ax.fill_between(x, np.array(rl_m)-np.array(rl_s),
                    np.array(rl_m)+np.array(rl_s),
                    alpha=0.15, color=WONG["cyan"])

    ax.plot(x, hybrid_m, "o-", color=WONG["blue"], label="Hybrid RL-MCDM", linewidth=2, markersize=6)
    ax.plot(x, rl_m, "s-", color=WONG["cyan"], label="RL-only", linewidth=2, markersize=5)
    ax.plot(x, topsis_m, "^--", color=WONG["orange"], label="TOPSIS-only", linewidth=1.5, markersize=5)
    ax.plot(x, pop_m, "d--", color=WONG["yellow"], label="Popularity", linewidth=1, markersize=4)
    ax.plot(x, rand_m, "x--", color=WONG["purple"], label="Random", linewidth=1, markersize=4)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{e//1000}K" if e >= 1000 else str(e) for e in ep_labels])
    ax.set_xlabel("Training Episodes")
    ax.set_ylabel("F1@7")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="center right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_convergence.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "fig_convergence.png"), bbox_inches="tight")
    plt.close(fig)
    print("  fig_convergence done")


def fig_lambda():
    """F1@7 vs lambda_Q."""
    lams = sorted(D["lambda_ablation"].keys(), key=float)
    lam_vals = [float(l) for l in lams]
    means = [D["lambda_ablation"][l]["mean"] for l in lams]
    stds  = [D["lambda_ablation"][l]["std"] for l in lams]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.errorbar(lam_vals, means, yerr=stds, fmt="o-",
                color=WONG["blue"], linewidth=2, markersize=7,
                capsize=4, ecolor=WONG["cyan"])

    # Shade optimal range
    ax.axvspan(0.45, 0.60, alpha=0.12, color=WONG["green"],
               label="Optimal range ($\\lambda_Q \\in [0.45, 0.60]$)")

    ax.set_xlabel("RL Fusion Weight $\\lambda_Q$")
    ax.set_ylabel("F1@7")
    ax.set_ylim(0.4, 1.05)
    ax.legend(loc="lower center")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_lambda.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "fig_lambda.png"), bbox_inches="tight")
    plt.close(fig)
    print("  fig_lambda done")


def fig_split_ablation():
    """F1@7 vs GT observable fraction alpha."""
    alphas = sorted(D["split_ablation"].keys(), key=float)
    alpha_vals = [float(a) for a in alphas]
    h_means = [D["split_ablation"][a]["hybrid"]["mean"] for a in alphas]
    r_means = [D["split_ablation"][a]["rl_only"]["mean"] for a in alphas]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(alpha_vals, h_means, "o-", color=WONG["blue"], label="Hybrid RL-MCDM",
            linewidth=2, markersize=7)
    ax.plot(alpha_vals, r_means, "s-", color=WONG["cyan"], label="RL-only",
            linewidth=2, markersize=6)

    # Mark design point
    ax.axvline(0.50, color=WONG["black"], linestyle=":", alpha=0.5,
               label="Design point ($\\alpha=0.50$)")

    ax.set_xlabel("Observable GT Fraction $\\alpha$")
    ax.set_ylabel("F1@7")
    ax.set_ylim(0.4, 1.05)
    ax.legend(loc="lower left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_split_ablation.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "fig_split_ablation.png"), bbox_inches="tight")
    plt.close(fig)
    print("  fig_split_ablation done")


def fig_drift():
    """F1@7 trajectories under concept drift."""
    eps = sorted(D["concept_drift"].keys(), key=int)
    ep_vals = [int(e) for e in eps]
    h_means = [D["concept_drift"][e]["hybrid"]["mean"] for e in eps]
    r_means = [D["concept_drift"][e]["rl_only"]["mean"] for e in eps]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(eps))

    ax.plot(x, h_means, "o-", color=WONG["blue"], label="Hybrid RL-MCDM",
            linewidth=2, markersize=7)
    ax.plot(x, r_means, "s-", color=WONG["cyan"], label="RL-only",
            linewidth=2, markersize=6)

    # Vertical line at drift point (ep 15000)
    drift_idx = ep_vals.index(15000)
    ax.axvline(drift_idx, color=WONG["red"], linestyle="--", linewidth=1.5,
               label="Brand-preference flip (ep 15K)")
    ax.axvspan(drift_idx - 0.3, drift_idx + 0.3, alpha=0.08, color=WONG["red"])

    ax.set_xticks(x)
    ax.set_xticklabels([f"{e//1000}K" for e in ep_vals])
    ax.set_xlabel("Training Episodes")
    ax.set_ylabel("F1@7")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_drift.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "fig_drift.png"), bbox_inches="tight")
    plt.close(fig)
    print("  fig_drift done")


def fig_ild():
    """ILD bar chart."""
    methods = ["Random", "Popularity", "RL-only", "Hybrid\nRL-MCDM", "TOPSIS-only"]
    keys = ["random", "popularity", "rl_only", "hybrid", "topsis"]
    means = [D["ild"][k]["mean"] for k in keys]
    ci_lo = [D["ild"][k]["ci_lo"] for k in keys]
    ci_hi = [D["ild"][k]["ci_hi"] for k in keys]
    errs = [[m - lo for m, lo in zip(means, ci_lo)],
            [hi - m for m, hi in zip(means, ci_hi)]]
    colors = [WONG["purple"], WONG["yellow"], WONG["cyan"], WONG["blue"], WONG["orange"]]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=errs, color=colors, edgecolor="black",
                  linewidth=0.6, capsize=4, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Intra-List Diversity (ILD)")
    ax.set_ylim(0, 1.05)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f"{m:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_ild.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "fig_ild.png"), bbox_inches="tight")
    plt.close(fig)
    print("  fig_ild done")


def fig_architecture():
    """TikZ-based architecture diagram as standalone PDF."""
    tikz_code = r"""\documentclass[border=5pt]{standalone}
\usepackage{tikz}
\usetikzlibrary{arrows.meta, positioning, shapes.geometric, fit, calc}
\begin{document}
\begin{tikzpicture}[
    node distance=1.2cm and 1.8cm,
    block/.style={rectangle, draw, rounded corners=3pt,
                  minimum width=2.8cm, minimum height=0.9cm,
                  align=center, font=\small},
    data/.style={block, fill=blue!8},
    proc/.style={block, fill=orange!12},
    eval/.style={block, fill=green!10},
    arr/.style={-{Stealth[length=5pt]}, thick},
]
% Row 1: Inputs
\node[data] (catalog) {Product Catalog\\$N{=}400$ items};
\node[data, right=2.5cm of catalog] (profiles) {User Profiles\\$|\mathcal{U}|{=}5$};

% Row 2: Modules
\node[proc, below=1.2cm of catalog] (topsis) {Entropy-Weighted\\TOPSIS};
\node[proc, below=1.2cm of profiles] (rl) {Tabular\\Q-Learning};

% Row 3: Scores
\node[block, fill=cyan!8, below=1cm of topsis] (cscore) {$\tilde{C}^*_i$\\Observable Score};
\node[block, fill=cyan!8, below=1cm of rl] (qscore) {$\tilde{Q}(u,i)$\\Behavioral Score};

% Row 4: Fusion
\node[proc, below=1.2cm of $(cscore)!0.5!(qscore)$] (fusion)
    {Score Fusion\\$S_i = \lambda_Q \tilde{Q} + \lambda_T \tilde{C}^*$};

% Row 5: Output
\node[eval, below=1cm of fusion] (rec) {Top-$K$ Recommendations\\$R(u) = \text{Top-7}(S)$};

% Row 6: Evaluation
\node[eval, below=1cm of rec] (evaln) {Evaluation\\F1@7, NDCG@7};

% Arrows
\draw[arr] (catalog) -- (topsis);
\draw[arr] (profiles) -- (rl);
\draw[arr] (topsis) -- (cscore);
\draw[arr] (rl) -- (qscore);
\draw[arr] (cscore) -- (fusion);
\draw[arr] (qscore) -- (fusion);
\draw[arr] (fusion) -- (rec);
\draw[arr] (rec) -- (evaln);

% Reward feedback
\draw[arr, dashed, color=red!70!black]
    ($(rec.east)+(0.2,0)$) -- ++(1.2,0) |- (rl.east)
    node[pos=0.25, right, font=\scriptsize, text=red!70!black] {Reward $r(u,i)$};

% Labels
\node[font=\scriptsize\itshape, text=gray, above=0.1cm of topsis]
    {Profile-agnostic};
\node[font=\scriptsize\itshape, text=gray, above=0.1cm of rl]
    {Profile-adaptive};

\end{tikzpicture}
\end{document}
"""
    tikz_path = os.path.join(FIG_DIR, "fig_architecture.tex")
    with open(tikz_path, "w") as f:
        f.write(tikz_code)
    # Try to compile if pdflatex available
    try:
        import subprocess
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode",
             "-output-directory", FIG_DIR, tikz_path],
            capture_output=True, timeout=30
        )
        if result.returncode == 0:
            print("  fig_architecture compiled to PDF")
        else:
            print("  fig_architecture: .tex written (pdflatex failed, compile manually)")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("  fig_architecture: .tex written (pdflatex not available)")


def main():
    print("Generating v16 figures from JSON data...")
    fig_comparison()
    fig_convergence()
    fig_lambda()
    fig_split_ablation()
    fig_drift()
    fig_ild()
    fig_architecture()
    print(f"\nAll figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIG_DIR = ROOT / "manuscript" / "fig"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_json(name: str) -> dict:
    with (RESULTS / name).open("r", encoding="utf-8") as f:
        return json.load(f)


def savefig(name: str) -> None:
    for ext in ("png", "pdf"):
        plt.savefig(FIG_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close()


def style_axes(ax, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_convergence(primary: dict) -> None:
    checkpoints = [500, 1000, 2000, 5000, 10000, 20000, 30000]
    methods = [
        ("hybrid", "Hybrid RL-TOPSIS", "#1f77b4"),
        ("rl_only", "RL-only", "#ff7f0e"),
        ("topsis_only", "TOPSIS-only", "#2ca02c"),
        ("popularity", "Popularity", "#9467bd"),
        ("random", "Random", "#7f7f7f"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for key, label, color in methods:
        means = np.array([primary["summary"][key][str(c)]["mean"] for c in checkpoints])
        stds = np.array([primary["summary"][key][str(c)]["std"] for c in checkpoints])
        ax.plot(checkpoints, means, marker="o", linewidth=2, label=label, color=color)
        ax.fill_between(checkpoints, means - stds, means + stds, color=color, alpha=0.12, linewidth=0)
    ax.set_xscale("log")
    ax.set_xticks(checkpoints)
    ax.set_xticklabels(["500", "1K", "2K", "5K", "10K", "20K", "30K"])
    ax.set_ylim(0, 1.0)
    style_axes(ax, "Training episodes", "F1@7")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    savefig("fig_convergence")


def plot_lambda(extended: dict) -> None:
    grid = [str(v) for v in extended["lambda_sensitivity"]["lambda_q_grid"]]
    x = np.array([float(v) for v in grid])
    means = np.array([extended["lambda_sensitivity"]["summary"][v]["mean"] for v in grid])
    stds = np.array([extended["lambda_sensitivity"]["summary"][v]["std"] for v in grid])
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(x, means, marker="o", linewidth=2.2, color="#1f77b4")
    ax.fill_between(x, means - stds, means + stds, color="#1f77b4", alpha=0.15, linewidth=0)
    best_idx = int(np.argmax(means))
    ax.scatter([x[best_idx]], [means[best_idx]], s=60, color="#d62728", zorder=5)
    ax.axvline(0.5, linestyle="--", color="#555555", linewidth=1.1, label="Static design point")
    ax.text(x[best_idx], means[best_idx] + 0.025, f"best={x[best_idx]:.2f}", ha="center", fontsize=8)
    ax.set_ylim(0.35, 0.98)
    style_axes(ax, r"RL fusion weight $\lambda_Q$", "Mean F1@7")
    ax.legend(frameon=False, fontsize=8)
    savefig("fig_lambda")


def plot_split(extended: dict) -> None:
    grid = [str(v) for v in extended["gt_split_sensitivity"]["observable_alpha_grid"]]
    x = np.array([float(v) for v in grid])
    methods = [
        ("hybrid", "Hybrid RL-TOPSIS", "#1f77b4"),
        ("rl_only", "RL-only", "#ff7f0e"),
        ("topsis_only", "TOPSIS-only", "#2ca02c"),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for key, label, color in methods:
        means = np.array([extended["gt_split_sensitivity"]["summary"][key][v]["mean"] for v in grid])
        stds = np.array([extended["gt_split_sensitivity"]["summary"][key][v]["std"] for v in grid])
        ax.plot(x, means, marker="o", linewidth=2, label=label, color=color)
        ax.fill_between(x, means - stds, means + stds, color=color, alpha=0.12, linewidth=0)
    ax.axvline(0.5, linestyle="--", color="#555555", linewidth=1.0)
    ax.set_ylim(0, 1.0)
    style_axes(ax, r"Observable utility fraction $\alpha$", "Mean F1@7")
    ax.legend(frameon=False, fontsize=8)
    savefig("fig_split_ablation")


def plot_drift(drift: dict) -> None:
    checkpoints = [2000, 5000, 10000, 14000, 15000, 16000, 20000, 25000, 30000]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for key, label, color in [
        ("hybrid", "Hybrid RL-TOPSIS", "#1f77b4"),
        ("rl_only", "RL-only", "#ff7f0e"),
    ]:
        means = np.array([drift["summary"][key][str(c)]["mean"] for c in checkpoints])
        lows = np.array([drift["summary"][key][str(c)]["ci_lo"] for c in checkpoints])
        highs = np.array([drift["summary"][key][str(c)]["ci_hi"] for c in checkpoints])
        ax.plot(checkpoints, means, marker="o", linewidth=2, label=label, color=color)
        ax.fill_between(checkpoints, lows, highs, color=color, alpha=0.12, linewidth=0)
    ax.axvline(15000, linestyle="--", color="#333333", linewidth=1.1, label="Preference flip")
    ax.set_ylim(0.2, 0.95)
    style_axes(ax, "Training episodes", "F1@7")
    ax.legend(frameon=False, fontsize=8)
    savefig("fig_drift")


def plot_ild(primary: dict, extended: dict) -> None:
    methods = ["random", "popularity", "topsis_only", "rl_only", "hybrid"]
    labels = ["Random", "Popularity", "TOPSIS", "RL-only", "Hybrid"]
    colors = ["#7f7f7f", "#9467bd", "#2ca02c", "#ff7f0e", "#1f77b4"]
    f1 = [primary["summary"][m]["30000"]["mean"] for m in methods]
    ild = [extended["ild"]["summary"][m]["mean"] for m in methods]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for x, y, label, color in zip(ild, f1, labels, colors):
        ax.scatter(x, y, s=70, color=color)
        ax.text(x + 0.01, y + 0.01, label, fontsize=8)
    ax.set_xlim(0.35, 0.95)
    ax.set_ylim(0, 1.0)
    style_axes(ax, "Intra-list diversity (ILD)", "F1@7")
    savefig("fig_ild")


def plot_comparison(primary: dict) -> None:
    methods = ["random", "popularity", "topsis_only", "rl_only", "hybrid"]
    labels = ["Random", "Popularity", "TOPSIS", "RL-only", "Hybrid"]
    means = [primary["summary"][m]["30000"]["mean"] for m in methods]
    lows = [primary["summary"][m]["30000"]["ci_lo"] for m in methods]
    highs = [primary["summary"][m]["30000"]["ci_hi"] for m in methods]
    yerr = np.vstack([np.array(means) - np.array(lows), np.array(highs) - np.array(means)])
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    colors = ["#7f7f7f", "#9467bd", "#2ca02c", "#ff7f0e", "#1f77b4"]
    ax.bar(labels, means, yerr=yerr, capsize=4, color=colors, alpha=0.9)
    ax.set_ylim(0, 1.0)
    style_axes(ax, "", "F1@7 at 30,000 episodes")
    ax.tick_params(axis="x", rotation=20)
    savefig("fig_comparison")


def main() -> None:
    primary = load_json("amazon_primary.json")
    extended = load_json("amazon_extended.json")
    drift = load_json("amazon_drift.json")
    plot_convergence(primary)
    plot_lambda(extended)
    plot_split(extended)
    plot_drift(drift)
    plot_ild(primary, extended)
    plot_comparison(primary)


if __name__ == "__main__":
    main()


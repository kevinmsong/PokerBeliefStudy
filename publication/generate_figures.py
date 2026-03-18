"""
generate_figures.py — Produce all 300 dpi publication figures for main.tex.

Run from repo root:
    python publication/generate_figures.py
"""

import json
import glob
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from collections import Counter, defaultdict
from scipy import stats
from scipy.stats import gaussian_kde

matplotlib.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "font.size":          11,
    "axes.titlesize":     12,
    "axes.labelsize":     11,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    10,
    "legend.title_fontsize": 10,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.08,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     1.0,
    "xtick.major.width":  1.0,
    "ytick.major.width":  1.0,
    "xtick.major.size":   4,
    "ytick.major.size":   4,
    "lines.linewidth":    1.6,
    "patch.linewidth":    0.8,
    "text.usetex":        False,
    "axes.titlepad":      8,
})

OUT = "publication/figures"
os.makedirs(OUT, exist_ok=True)

# ── Colour palette (Nature-style, colourblind-safe) ──────────────────────────
C_HEURISTIC = "#878787"   # grey
C_STATIC    = "#2166AC"   # blue
C_BELIEF    = "#D6604D"   # orange-red
C_HELD1     = "#4DAC26"   # green
C_HELD2     = "#762A83"   # purple

AGENT_COLORS = {
    "HeuristicAgent": C_HEURISTIC,
    "StaticEVAgent":  C_STATIC,
    "BeliefEVAgent":  C_BELIEF,
}
AGENT_LABELS = {
    "HeuristicAgent": "Heuristic",
    "StaticEVAgent":  "Static-EV",
    "BeliefEVAgent":  "Belief-EV",
}


def bootstrap_ci(x, n=5000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(n)])
    return np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])


# ─────────────────────────────────────────────────────────────────────────────
# FIG 1 — Main comparison: chips/hand with 95 % CI, grouped by agent
# ─────────────────────────────────────────────────────────────────────────────
def fig_main_comparison():
    mc = pd.read_csv("outputs/processed/main_comparison_summary.csv")

    # Three agents × three opponent contexts; heuristic only has balanced
    groups = [
        # (matchup_label,            agent_type,      opponent_label)
        ("heuristic_vs_balanced",    "HeuristicAgent", "Balanced"),
        ("ev_static_vs_balanced",    "StaticEVAgent",  "Balanced"),
        ("ev_static_vs_aggressive",  "StaticEVAgent",  "Aggressive"),
        ("ev_belief_vs_balanced",    "BeliefEVAgent",  "Balanced"),
        ("ev_belief_vs_aggressive",  "BeliefEVAgent",  "Aggressive"),
        ("ev_belief_vs_passive",     "BeliefEVAgent",  "Passive"),
    ]

    rows = []
    for ml, at, opp in groups:
        r = mc[(mc.matchup_label == ml) & (mc.agent0_type == at)]["terminal_reward_0"].values
        if len(r) == 0:
            continue
        lo, hi = bootstrap_ci(r)
        rows.append(dict(label=f"{AGENT_LABELS[at]}\n({opp})", agent=at,
                         mean=r.mean(), lo=lo, hi=hi))
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    xs = np.arange(len(df))
    bar_colors = [AGENT_COLORS[a] for a in df["agent"]]

    bars = ax.bar(xs, df["mean"], color=bar_colors, alpha=0.88, width=0.60,
                  zorder=3, edgecolor="white", linewidth=0.5)
    ax.errorbar(xs, df["mean"],
                yerr=[df["mean"] - df["lo"], df["hi"] - df["mean"]],
                fmt="none", color="#222222", capsize=4, capthick=1.2,
                elinewidth=1.2, zorder=4)

    for i, row in df.iterrows():
        ax.text(i, row["hi"] + 4, f'{row["mean"]:.0f}',
                ha="center", va="bottom", fontsize=9, color="#222222")

    ax.set_xticks(xs)
    ax.set_xticklabels(df["label"], fontsize=9.5)
    ax.set_ylabel("Mean chips won per hand", labelpad=6)
    ax.set_ylim(0, 250)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(25))
    ax.grid(axis="y", which="major", linestyle=":", linewidth=0.6,
            alpha=0.7, zorder=0)
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.35)

    legend_patches = [
        mpatches.Patch(color=C_HEURISTIC, alpha=0.88, label="Heuristic"),
        mpatches.Patch(color=C_STATIC,    alpha=0.88, label="Static-EV"),
        mpatches.Patch(color=C_BELIEF,    alpha=0.88, label="Belief-EV"),
    ]
    ax.legend(handles=legend_patches, loc="upper left",
              frameon=True, framealpha=0.9, edgecolor="0.8",
              fontsize=10, borderpad=0.6)

    ax.set_title("Experiment 1 — Agent performance across opponent families\n"
                 "(n = 5,000 hands per condition; error bars = 95% bootstrap CI)",
                 fontsize=11)

    fig.tight_layout(pad=1.2)
    fig.savefig(f"{OUT}/fig1_main_comparison.pdf")
    fig.savefig(f"{OUT}/fig1_main_comparison.png")
    plt.close(fig)
    print("Fig 1 saved.")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 2 — Belief ablation: head-to-head chip differential distributions
# ─────────────────────────────────────────────────────────────────────────────
def fig_belief_ablation():
    ba = pd.read_csv("outputs/processed/belief_ablation_summary.csv")

    conditions = [
        ("belief_vs_static_balanced",    "Balanced"),
        ("belief_vs_static_aggressive",  "Aggressive"),
        ("belief_vs_static_trappy",      "Trappy"),
    ]

    # Extra top margin for suptitle; extra bottom margin for sub-axis annotations
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 4.0))
    fig.subplots_adjust(wspace=0.42, top=0.78, bottom=0.28)

    for ax, (ml, lbl) in zip(axes, conditions):
        grp      = ba[ba.matchup_label == ml]
        belief_r = grp["terminal_reward_0"].values
        static_r = grp["terminal_reward_1"].values
        diff     = belief_r - static_r

        bins = np.linspace(-700, 700, 45)
        ax.hist(diff, bins=bins, color=C_BELIEF, alpha=0.72,
                edgecolor="none", zorder=3)
        ax.axvline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.6, zorder=4)
        # Mean line — no legend label (annotation goes below x-axis instead)
        ax.axvline(diff.mean(), color=C_BELIEF, linewidth=1.8, linestyle="-", zorder=5)

        t, p = stats.ttest_rel(belief_r, static_r)
        p_str = f"p = {p:.3f}" if p >= 0.001 else "p < 0.001"

        # ── Sub-axis annotation row, below x-axis label ──────────────────────
        # Left: mean value (coloured to match the mean line)
        ax.text(0.0, -0.30, f"Mean = {diff.mean():.1f}",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=9.5, color=C_BELIEF)
        # Right: t and p stats
        ax.text(1.0, -0.30, f"$t$ = {t:.2f},  {p_str}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9.5, color="#333333")

        # Panel title with extra padding
        ax.set_title(lbl, fontsize=12, pad=10, fontweight="bold")
        ax.set_xlabel("Δ chips/hand  (Belief-EV − Static-EV)", fontsize=10, labelpad=5)
        # Shift x-axis label 2 pts left on rightmost subplot only
        if ax is axes[2]:
            ax.xaxis.set_label_coords(0.47, -0.13)
        if ax is axes[0]:
            ax.set_ylabel("Frequency", fontsize=10)
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6, zorder=0)
        ax.tick_params(labelsize=9.5)

    # suptitle well above the panel titles
    fig.suptitle(
        "Experiment 2 — Belief ablation: per-hand chip differential, "
        "Belief-EV vs. Static-EV (head-to-head)\n"
        "n = 5,000 hands per condition; dashed vertical line = zero",
        fontsize=11, y=1.01,
    )
    fig.savefig(f"{OUT}/fig2_belief_ablation.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/fig2_belief_ablation.png", bbox_inches="tight")
    plt.close(fig)
    print("Fig 2 saved.")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3 — Robustness: dev vs held-out with seed-level scatter
# ─────────────────────────────────────────────────────────────────────────────
def fig_robustness():
    mc  = pd.read_csv("outputs/processed/main_comparison_summary.csv")
    rob = pd.read_csv("outputs/processed/robustness_summary.csv")

    def seed_means(df, matchup):
        return df[df.matchup_label == matchup].groupby("seed")["terminal_reward_0"].mean().values

    entries = [
        # (label,                 values,                                 color,       hatch)
        ("Heuristic\nDev",         seed_means(mc,  "heuristic_vs_balanced"),       C_HEURISTIC, ""),
        ("Heuristic\nHeld-out 1",  seed_means(rob, "heuristic_vs_held_out_1"),     C_HEURISTIC, "//"),
        ("Heuristic\nHeld-out 2",  seed_means(rob, "heuristic_vs_held_out_2"),     C_HEURISTIC, "xx"),
        ("Static-EV\nDev",         seed_means(mc,  "ev_static_vs_balanced"),       C_STATIC,    ""),
        ("Static-EV\nHeld-out 1",  seed_means(rob, "ev_static_vs_held_out_1"),     C_STATIC,    "//"),
        ("Static-EV\nHeld-out 2",  seed_means(rob, "ev_static_vs_held_out_2"),     C_STATIC,    "xx"),
        ("Belief-EV\nDev",         seed_means(mc,  "ev_belief_vs_balanced"),       C_BELIEF,    ""),
        ("Belief-EV\nHeld-out 1",  seed_means(rob, "ev_belief_vs_held_out_1"),     C_BELIEF,    "//"),
        ("Belief-EV\nHeld-out 2",  seed_means(rob, "ev_belief_vs_held_out_2"),     C_BELIEF,    "xx"),
    ]

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    xs = np.arange(len(entries))

    for i, (lbl, vals, col, hatch) in enumerate(entries):
        m = vals.mean()
        lo, hi = bootstrap_ci(vals)
        ax.bar(xs[i], m, color=col, alpha=0.82, width=0.65,
               hatch=hatch, edgecolor="white" if not hatch else col,
               linewidth=0.6, zorder=3)
        ax.errorbar(xs[i], m,
                    yerr=[[m - lo], [hi - m]],
                    fmt="none", color="#222222", capsize=4,
                    capthick=1.1, elinewidth=1.1, zorder=4)
        rng_j = np.random.default_rng(i + 42)
        jitter = rng_j.uniform(-0.20, 0.20, len(vals))
        ax.scatter(xs[i] + jitter, vals, color="#222222",
                   s=10, zorder=5, alpha=0.55)

    for xd in [2.5, 5.5]:
        ax.axvline(xd, color="0.65", linewidth=0.8, linestyle="--")

    ax.set_xticks(xs)
    ax.set_xticklabels([e[0] for e in entries], fontsize=9)
    ax.set_ylabel("Mean chips won per hand", labelpad=6)
    ax.set_ylim(0, 230)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6, zorder=0)

    legend_patches = [
        mpatches.Patch(facecolor="0.5", label="Dev (no hatch)"),
        mpatches.Patch(facecolor="0.5", hatch="//", edgecolor="0.5", label="Held-out 1  (//)"),
        mpatches.Patch(facecolor="0.5", hatch="xx", edgecolor="0.5", label="Held-out 2  (×)"),
    ]
    ax.legend(handles=legend_patches, loc="lower right",
              frameon=True, framealpha=0.92, edgecolor="0.75",
              fontsize=9.5, borderpad=0.7)

    ax.set_title(
        "Experiment 4 — Robustness: development vs. held-out opponent families\n"
        "(dots = individual seeds; error bars = 95% bootstrap CI)",
        fontsize=11,
    )
    fig.tight_layout(pad=1.3)
    fig.savefig(f"{OUT}/fig4_robustness.pdf")
    fig.savefig(f"{OUT}/fig4_robustness.png")
    plt.close(fig)
    print("Fig 3 saved.")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 4 — Calibration: posterior entropy KDE  +  action frequency profile
# ─────────────────────────────────────────────────────────────────────────────
def fig_calibration():
    # ── Posterior entropy ────────────────────────────────────────────────────
    entropy_by_family = defaultdict(list)
    for fpath in sorted(glob.glob(
            "outputs/raw_runs/calibration_belief_calibration_*_seed*.json")):
        family = fpath.split("belief_calibration_")[1].split("_seed")[0]
        with open(fpath) as f:
            hands = json.load(f)
        for hand in hands:
            for act in hand.get("action_history", []):
                if isinstance(act, dict) and act.get("player") == 0:
                    ent = act.get("posterior_entropy")
                    if ent is not None:
                        entropy_by_family[family].append(ent)

    # ── Action frequencies ───────────────────────────────────────────────────
    agent_action_data = {}
    for ag_lbl, matchup in [
        ("Heuristic", "main_comparison_heuristic_vs_balanced"),
        ("Static-EV",  "main_comparison_ev_static_vs_balanced"),
        ("Belief-EV",  "main_comparison_ev_belief_vs_balanced"),
    ]:
        acts = Counter()
        for fpath in sorted(glob.glob(f"outputs/raw_runs/{matchup}_seed*.json")):
            with open(fpath) as f:
                hands = json.load(f)
            for hand in hands:
                for a in hand.get("action_history", []):
                    if isinstance(a, dict) and a.get("player") == 0:
                        acts[a.get("action", "")] += 1
        total = sum(acts.values())
        agent_action_data[ag_lbl] = {k: v / total for k, v in acts.items()}

    # ── Layout: 1 row × 2 panels ─────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.6))
    fig.subplots_adjust(wspace=0.40)

    # — Left: entropy KDE —
    fam_map = {
        "balanced":   ("Balanced",   "#2166AC"),
        "aggressive": ("Aggressive", "#D6604D"),
        "maniac":     ("Maniac",     "#4DAC26"),
    }
    max_ent = np.log2(7)

    for fam, (lbl, col) in fam_map.items():
        vals = entropy_by_family.get(fam, [])
        if not vals:
            continue
        kde  = gaussian_kde(vals, bw_method=0.25)
        xs   = np.linspace(1.75, 2.00, 400)
        ys   = kde(xs)
        ax1.plot(xs, ys, color=col, linewidth=1.8, label=lbl)
        ax1.fill_between(xs, ys, alpha=0.12, color=col)
        ax1.axvline(np.mean(vals), color=col, linewidth=1.0,
                    linestyle="--", alpha=0.75)

    ax1.axvline(max_ent, color="black", linewidth=1.0, linestyle=":",
                label=f"Max entropy\n(H = {max_ent:.2f} bits)")
    ax1.set_xlabel("Posterior entropy (bits)", labelpad=5)
    ax1.set_ylabel("Density", labelpad=5)
    ax1.set_title("Posterior entropy by opponent family", fontsize=11, pad=6)
    # Place legend slightly left of upper-right corner
    ax1.legend(frameon=True, framealpha=0.9, edgecolor="0.8",
               fontsize=9.0, loc="upper right",
               bbox_to_anchor=(0.93, 1.0), borderpad=0.6)
    ax1.grid(linestyle=":", linewidth=0.5, alpha=0.55)
    ax1.tick_params(labelsize=10)

    # — Right: action frequency grouped bars —
    action_order  = ["fold", "call", "check", "bet_half_pot", "bet_pot", "jam"]
    action_labels = ["Fold", "Call", "Check", "Bet ½", "Bet 1×", "Jam"]
    agent_list    = ["Heuristic", "Static-EV", "Belief-EV"]
    agent_cols_r  = [C_HEURISTIC, C_STATIC, C_BELIEF]
    bar_w = 0.24
    xs2   = np.arange(len(action_order))

    for j, (ag, col) in enumerate(zip(agent_list, agent_cols_r)):
        freqs  = [agent_action_data[ag].get(a, 0) for a in action_order]
        offset = (j - 1) * bar_w
        ax2.bar(xs2 + offset, freqs, width=bar_w, color=col,
                alpha=0.88, label=ag, zorder=3,
                edgecolor="white", linewidth=0.4)

    ax2.set_xticks(xs2)
    ax2.set_xticklabels(action_labels, fontsize=10)
    ax2.set_ylabel("Proportion of decisions", labelpad=5)
    ax2.set_ylim(0, 0.70)
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax2.set_title("Action frequency profile by agent type", fontsize=11, pad=6)
    ax2.legend(frameon=True, framealpha=0.9, edgecolor="0.8",
               fontsize=9.5, loc="upper right", borderpad=0.7)
    ax2.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6, zorder=0)
    ax2.tick_params(labelsize=10)

    fig.suptitle(
        "Experiment 3 — Belief calibration and behavioural profiles",
        fontsize=12, y=1.03,
    )
    fig.savefig(f"{OUT}/fig3_calibration.pdf")
    fig.savefig(f"{OUT}/fig3_calibration.png")
    plt.close(fig)
    print("Fig 4 saved.")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 5 — Seed-level stability across 5 independent runs
# ─────────────────────────────────────────────────────────────────────────────
def fig_seed_variance():
    mc = pd.read_csv("outputs/processed/main_comparison_summary.csv")

    matchups = [
        ("heuristic_vs_balanced",   "HeuristicAgent", "Heuristic (Balanced)",   C_HEURISTIC, "o", "-"),
        ("ev_static_vs_balanced",   "StaticEVAgent",  "Static-EV (Balanced)",   C_STATIC,    "s", "-"),
        ("ev_belief_vs_balanced",   "BeliefEVAgent",  "Belief-EV (Balanced)",   C_BELIEF,    "^", "-"),
        ("ev_belief_vs_aggressive", "BeliefEVAgent",  "Belief-EV (Aggressive)", C_BELIEF,    "^", "--"),
        ("ev_belief_vs_passive",    "BeliefEVAgent",  "Belief-EV (Passive)",    C_BELIEF,    "^", ":"),
    ]

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    seeds = sorted(mc["seed"].unique())

    for ml, at, lbl, col, marker, ls in matchups:
        grp = mc[(mc.matchup_label == ml) & (mc.agent0_type == at)]
        if grp.empty:
            continue
        means = grp.groupby("seed")["terminal_reward_0"].mean().reindex(seeds).values
        ax.plot(seeds, means, marker=marker, markersize=5.5, color=col,
                linestyle=ls, label=lbl, linewidth=1.5, markeredgecolor="white",
                markeredgewidth=0.5)

    ax.set_xlabel("Random seed", labelpad=5)
    ax.set_ylabel("Mean chips won per hand", labelpad=5)
    ax.set_xticks(seeds)
    ax.set_ylim(60, 230)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(40))
    ax.grid(linestyle=":", linewidth=0.5, alpha=0.6)
    ax.tick_params(labelsize=10)
    ax.legend(frameon=True, framealpha=0.92, edgecolor="0.8",
              fontsize=6.0, loc="upper right", borderpad=0.5,
              labelspacing=0.22, handlelength=1.4, handletextpad=0.4)
    ax.set_title(
        "Seed-to-seed stability across 5 independent replications\n"
        "(n = 1,000 hands per seed)",
        fontsize=11, pad=6,
    )

    fig.tight_layout(pad=1.3)
    fig.savefig(f"{OUT}/fig5_seed_variance.pdf")
    fig.savefig(f"{OUT}/fig5_seed_variance.png")
    plt.close(fig)
    print("Fig 5 saved.")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fig_main_comparison()
    fig_belief_ablation()
    fig_robustness()
    fig_calibration()
    fig_seed_variance()
    print(f"\nAll figures saved to {OUT}/")

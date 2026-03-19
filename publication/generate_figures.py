"""
generate_figures.py — Produce publication figures for the refreshed study.

Run from repo root:
    python publication/generate_figures.py
"""

from __future__ import annotations

import glob
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis import compute_switch_case_metrics
from src.utils import bootstrap_ci


matplotlib.use("Agg")

OUT = ROOT / "publication" / "figures"
PROCESSED = ROOT / "outputs" / "processed"
RAW = ROOT / "outputs" / "raw_runs"

OUT.mkdir(parents=True, exist_ok=True)

matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "lines.linewidth": 1.8,
        "patch.linewidth": 0.8,
        "text.usetex": False,
    }
)


C_HEURISTIC = "#7A7A7A"
C_STATIC = "#2B6CB0"
C_BELIEF = "#D76B41"
C_CONTROL = "#5F6CAF"
C_ADAPTIVE = "#2F855A"
C_SWITCH = "#AA3A38"
C_BALANCED = "#4C78A8"
C_AGGRESSIVE = "#E45756"
C_PASSIVE = "#72B7B2"
C_TRAPPY = "#B279A2"

AGENT_COLORS = {
    "HeuristicAgent": C_HEURISTIC,
    "StaticEVAgent": C_STATIC,
    "BeliefEVAgent": C_BELIEF,
}
AGENT_LABELS = {
    "HeuristicAgent": "Heuristic",
    "StaticEVAgent": "Static-EV",
    "BeliefEVAgent": "Belief-EV",
}
FAMILY_LABELS = {
    "balanced": "Balanced",
    "aggressive": "Aggressive",
    "passive": "Passive",
    "maniac": "Maniac",
    "trappy": "Trappy",
    "held_out_1": "Held-out 1",
    "held_out_2": "Held-out 2",
}
FAMILY_COLORS = {
    "balanced": C_BALANCED,
    "aggressive": C_AGGRESSIVE,
    "passive": C_PASSIVE,
    "maniac": "#54A24B",
    "trappy": C_TRAPPY,
    "held_out_1": "#3C8D2F",
    "held_out_2": "#8E5EA2",
}


def _read_processed(experiment_id: str) -> pd.DataFrame:
    path = PROCESSED / f"{experiment_id}_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing processed file: {path}")
    return pd.read_csv(path)


def _save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png")
    fig.savefig(OUT / f"{stem}.pdf")
    plt.close(fig)


def _ci(values: pd.Series | np.ndarray | list[float]) -> tuple[float, float, float]:
    arr = pd.Series(values).dropna().to_numpy()
    stats_ci = bootstrap_ci(arr.tolist(), n_boot=2000, ci=0.95)
    return stats_ci["mean"], stats_ci["lower"], stats_ci["upper"]


def _entropy_bits(raw_entropy_nats: float) -> float:
    return raw_entropy_nats / math.log(2.0)


def _load_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def fig_main_comparison() -> None:
    df = _read_processed("main_comparison")
    family_order = ["balanced", "aggressive", "passive"]
    agent_order = ["HeuristicAgent", "StaticEVAgent", "BeliefEVAgent"]

    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    x = np.arange(len(family_order))
    width = 0.24

    for idx, agent in enumerate(agent_order):
        means = []
        lowers = []
        uppers = []
        for family in family_order:
            subset = df[(df["agent0_type"] == agent) & (df["behavior_family_1"] == family)]
            mean, lo, hi = _ci(subset["terminal_reward_0"])
            means.append(mean)
            lowers.append(mean - lo)
            uppers.append(hi - mean)

        offset = (idx - 1) * width
        ax.bar(
            x + offset,
            means,
            width=width,
            color=AGENT_COLORS[agent],
            label=AGENT_LABELS[agent],
            alpha=0.9,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        ax.errorbar(
            x + offset,
            means,
            yerr=[lowers, uppers],
            fmt="none",
            color="#222222",
            capsize=4,
            elinewidth=1.1,
            zorder=4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([FAMILY_LABELS[f] for f in family_order])
    ax.set_ylabel("Net chips won per hand")
    ax.set_title(
        "Experiment 1 — Main comparison across family-policy opponents\n"
        "(45,000 hands total; 5,000 hands per cell; error bars = 95% bootstrap CI)"
    )
    ax.yaxis.set_major_locator(mticker.MultipleLocator(25))
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7, zorder=0)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", frameon=True, framealpha=0.92, edgecolor="0.8",
              bbox_to_anchor=(-0.0016, 1.0), bbox_transform=ax.transAxes)

    _save(fig, "fig1_main_comparison")


def fig_belief_ablation() -> None:
    df = _read_processed("belief_ablation")
    conditions = [
        ("belief_vs_static_balanced", "Balanced"),
        ("belief_vs_static_aggressive", "Aggressive"),
        ("belief_vs_static_trappy", "Trappy"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 4.0))
    fig.subplots_adjust(wspace=0.38, top=0.80, bottom=0.28)

    for ax, (matchup, label) in zip(axes, conditions):
        subset = df[df["matchup_label"] == matchup]
        diff = subset["terminal_reward_0"] - subset["terminal_reward_1"]
        bound = max(200, int(math.ceil(np.nanpercentile(np.abs(diff), 99) / 50.0) * 50))
        bins = np.linspace(-bound, bound, 36)

        ax.hist(diff, bins=bins, color=C_BELIEF, alpha=0.78, edgecolor="none", zorder=3)
        ax.axvline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.55, zorder=4)
        ax.axvline(diff.mean(), color=C_SWITCH, linewidth=1.8, zorder=5)

        t_stat, p_val = stats.ttest_rel(subset["terminal_reward_0"], subset["terminal_reward_1"])
        mean, lo, hi = _ci(diff)
        p_text = f"p = {p_val:.3f}" if p_val >= 0.001 else "p < 0.001"

        ax.text(0.0, -0.30, f"Mean = {mean:.1f} [{lo:.1f}, {hi:.1f}]",
                transform=ax.transAxes, ha="left", va="top", fontsize=9, color=C_SWITCH)
        ax.text(1.0, -0.30, f"t = {t_stat:.2f}, {p_text}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9, color="#333333")

        ax.set_title(label, fontweight="bold", pad=10)
        ax.set_xlabel("Per-hand differential\n(Belief-EV − Static-EV)")
        if ax is axes[0]:
            ax.set_ylabel("Frequency")
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6, zorder=0)

    fig.suptitle(
        "Experiment 2 — Direct ablation of belief updating\n"
        "(15,000 hands total; 5,000 hands per condition)"
    )
    _save(fig, "fig2_belief_ablation")


def fig_calibration() -> None:
    entropy_by_family: dict[str, list[float]] = defaultdict(list)
    for path_str in glob.glob(str(RAW / "calibration_*_seed*.json")):
        path = Path(path_str)
        family = path.stem.split("belief_calibration_")[1].split("_seed")[0]
        for hand in _load_json(path):
            for step in hand.get("action_history", []):
                if step.get("player") == 0 and step.get("posterior_entropy") is not None:
                    entropy_by_family[family].append(_entropy_bits(float(step["posterior_entropy"])))

    action_counts = {
        "Heuristic": Counter(),
        "Static-EV": Counter(),
        "Belief-EV": Counter(),
    }
    action_map = {
        "heuristic_vs_": "Heuristic",
        "ev_static_vs_": "Static-EV",
        "ev_belief_vs_": "Belief-EV",
    }
    for path_str in glob.glob(str(RAW / "main_comparison_*_seed*.json")):
        path = Path(path_str)
        stem = path.stem.replace("main_comparison_", "", 1)
        agent_label = None
        for prefix, name in action_map.items():
            if stem.startswith(prefix):
                agent_label = name
                break
        if agent_label is None:
            continue
        for hand in _load_json(path):
            for step in hand.get("action_history", []):
                if step.get("player") == 0:
                    action_counts[agent_label][step.get("action", "")] += 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.8))
    fig.subplots_adjust(wspace=0.38)

    for family in ["balanced", "aggressive", "maniac"]:
        values = entropy_by_family.get(family, [])
        if len(values) < 2:
            continue
        xs = np.linspace(min(values), max(values), 300)
        kde = gaussian_kde(values, bw_method=0.25)
        ys = kde(xs)
        ax1.plot(xs, ys, color=FAMILY_COLORS[family], linewidth=1.8, label=FAMILY_LABELS[family])
        ax1.fill_between(xs, ys, alpha=0.15, color=FAMILY_COLORS[family])
        ax1.axvline(np.mean(values), color=FAMILY_COLORS[family], linewidth=1.0, linestyle="--", alpha=0.8)

    max_entropy = math.log2(7)
    ax1.axvline(max_entropy, color="black", linewidth=1.0, linestyle=":", label=f"Max = {max_entropy:.3f} bits")
    ax1.set_xlabel("Posterior entropy (bits)")
    ax1.set_ylabel("Density")
    ax1.set_title("Calibration entropy by opponent family")
    ax1.grid(linestyle=":", linewidth=0.5, alpha=0.6)
    ax1.legend(frameon=True, framealpha=0.92, edgecolor="0.8", loc="upper left")

    action_order = ["fold", "call", "check", "bet_half_pot", "bet_pot", "jam"]
    action_labels = ["Fold", "Call", "Check", "Bet 1/2", "Bet pot", "Jam"]
    x = np.arange(len(action_order))
    width = 0.24
    action_colors = [C_HEURISTIC, C_STATIC, C_BELIEF]
    for idx, (agent_label, color) in enumerate(zip(action_counts.keys(), action_colors)):
        counts = action_counts[agent_label]
        total = max(1, sum(counts.values()))
        freqs = [counts.get(action, 0) / total for action in action_order]
        offset = (idx - 1) * width
        ax2.bar(
            x + offset,
            freqs,
            width=width,
            color=color,
            alpha=0.88,
            label=agent_label,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels(action_labels, rotation=15)
    ax2.set_ylabel("Share of player-0 decisions")
    ax2.set_ylim(0, 0.75)
    ax2.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax2.set_title("Action profile across development opponents")
    ax2.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6, zorder=0)
    ax2.legend(frameon=True, framealpha=0.92, edgecolor="0.8", loc="upper right")

    fig.suptitle("Experiment 3 — Calibration and behavioral profiles (10,000 calibration hands)", y=1.02)
    _save(fig, "fig3_calibration")


def fig_robustness() -> None:
    main_df = _read_processed("main_comparison")
    rob_df = _read_processed("robustness")
    agent_order = ["HeuristicAgent", "StaticEVAgent", "BeliefEVAgent"]

    def seed_values(df: pd.DataFrame, agent: str, family: str | None = None) -> np.ndarray:
        subset = df[df["agent0_type"] == agent]
        if family is not None:
            subset = subset[subset["behavior_family_1"] == family]
        return subset.groupby("seed")["terminal_reward_0"].mean().to_numpy()

    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    labels = ["Development", "Held-out 1", "Held-out 2"]
    x = np.arange(len(labels))
    width = 0.24

    for idx, agent in enumerate(agent_order):
        dev_values = seed_values(main_df, agent)
        h1_values = seed_values(rob_df, agent, "held_out_1")
        h2_values = seed_values(rob_df, agent, "held_out_2")
        grouped = [dev_values, h1_values, h2_values]

        means = []
        lowers = []
        uppers = []
        for values in grouped:
            mean, lo, hi = _ci(values)
            means.append(mean)
            lowers.append(mean - lo)
            uppers.append(hi - mean)

        offset = (idx - 1) * width
        ax.bar(
            x + offset,
            means,
            width=width,
            color=AGENT_COLORS[agent],
            alpha=0.9,
            label=AGENT_LABELS[agent],
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        ax.errorbar(
            x + offset,
            means,
            yerr=[lowers, uppers],
            fmt="none",
            color="#222222",
            capsize=4,
            elinewidth=1.0,
            zorder=4,
        )
        rng = np.random.default_rng(idx + 17)
        for jdx, values in enumerate(grouped):
            jitter = rng.uniform(-0.05, 0.05, size=len(values))
            ax.scatter(np.full(len(values), x[jdx] + offset) + jitter, values, s=16, color="#222222", alpha=0.55, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Seed-level mean net chips won per hand")
    ax.set_title(
        "Experiment 4 — Robustness to held-out opponent families\n"
        "(Development baseline = all three development families; dots = seeds)"
    )
    ax.yaxis.set_major_locator(mticker.MultipleLocator(25))
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7, zorder=0)
    ax.legend(frameon=True, framealpha=0.92, edgecolor="0.8", loc="upper left")

    _save(fig, "fig4_robustness")


def fig_main_study_diagnostics() -> None:
    main_ids = ["main_comparison", "belief_ablation", "calibration", "robustness"]
    frames = [_read_processed(exp_id) for exp_id in main_ids]
    combined = pd.concat(frames, ignore_index=True)
    main_df = frames[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.8))
    fig.subplots_adjust(wspace=0.34)

    for agent in ["HeuristicAgent", "StaticEVAgent", "BeliefEVAgent"]:
        subset = main_df[main_df["agent0_type"] == agent]
        if subset.empty:
            continue
        seed_means = subset.groupby("seed")["terminal_reward_0"].mean()
        ax1.plot(
            seed_means.index,
            seed_means.values,
            marker="o",
            color=AGENT_COLORS[agent],
            linewidth=1.8,
            label=AGENT_LABELS[agent],
        )

    ax1.set_xlabel("Random seed")
    ax1.set_ylabel("Mean net chips won per hand")
    ax1.set_title("Seed-level stability in the main comparison")
    ax1.grid(linestyle=":", linewidth=0.5, alpha=0.6)
    ax1.legend(frameon=True, framealpha=0.92, edgecolor="0.8", loc="best")

    counts = combined["first_to_act"].value_counts().reindex([0, 1], fill_value=0)
    bars = ax2.bar(
        ["Player 0 first", "Player 1 first"],
        counts.values,
        color=[C_STATIC, C_SWITCH],
        alpha=0.88,
        edgecolor="white",
        linewidth=0.6,
    )
    for bar, value in zip(bars, counts.values):
        ax2.text(bar.get_x() + bar.get_width() / 2, value + 800, f"{int(value):,}", ha="center", va="bottom", fontsize=10)
    ax2.set_ylabel("Hands in the 100,000-hand main study")
    ax2.set_title("Alternating first-to-act assignment")
    ax2.yaxis.set_major_locator(mticker.MultipleLocator(10000))
    ax2.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)

    fig.suptitle("Figure 5 — Main-study diagnostics and design checks", y=1.02)
    _save(fig, "fig5_main_study_diagnostics")


def fig_switch_event_study() -> None:
    df = _read_processed("switch_case_study")
    adaptive = df[df["matchup_label"] == "adaptive_counter_vs_hidden_switch"].copy()
    control = df[df["matchup_label"] == "balanced_control_vs_hidden_switch"].copy()

    for frame in (adaptive, control):
        frame["offset"] = frame["hand_num"] - frame["switch_hand"]
        frame["offset_bin"] = (np.floor(frame["offset"] / 50) * 50).astype(int)

    window = 1000
    adaptive_plot = adaptive[adaptive["offset"].between(-window, window)]
    control_plot = control[control["offset"].between(-window, window)]

    event_adaptive = adaptive_plot.groupby("offset_bin")["terminal_reward_0"].mean().reset_index()
    event_control = control_plot.groupby("offset_bin")["terminal_reward_0"].mean().reset_index()

    case_per_seed, _ = compute_switch_case_metrics(df)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.0))
    fig.subplots_adjust(wspace=0.34)

    ax1.plot(event_adaptive["offset_bin"], event_adaptive["terminal_reward_0"], color=C_ADAPTIVE, label="Adaptive responder")
    ax1.plot(event_control["offset_bin"], event_control["terminal_reward_0"], color=C_CONTROL, label="Balanced control")
    ax1.axvline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.65)
    ax1.set_xlabel("Hands relative to hidden switch")
    ax1.set_ylabel("Average net chips won per hand")
    ax1.set_title("Event-study reward trace around the hidden switch")
    ax1.grid(linestyle=":", linewidth=0.5, alpha=0.6)
    ax1.legend(frameon=True, framealpha=0.92, edgecolor="0.8", loc="best")

    grouped = {
        "Adaptive\npre": case_per_seed["pre_adaptive_reward"],
        "Adaptive\npost": case_per_seed["post_adaptive_reward"],
        "Control\npre": case_per_seed["pre_control_reward"],
        "Control\npost": case_per_seed["post_control_reward"],
    }
    colors = [C_ADAPTIVE, C_ADAPTIVE, C_CONTROL, C_CONTROL]
    means, lowers, uppers = [], [], []
    for values in grouped.values():
        mean, lo, hi = _ci(values)
        means.append(mean)
        lowers.append(mean - lo)
        uppers.append(hi - mean)
    xpos = np.arange(len(grouped))
    ax2.bar(xpos, means, color=colors, alpha=0.88, edgecolor="white", linewidth=0.6, zorder=3)
    ax2.errorbar(xpos, means, yerr=[lowers, uppers], fmt="none", color="#222222", capsize=4, elinewidth=1.0, zorder=4)
    ax2.set_xticks(xpos)
    ax2.set_xticklabels(list(grouped.keys()))
    ax2.set_ylabel("Seed-level mean net chips won per hand")
    ax2.set_title("Pre- and post-switch reward levels")
    ax2.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6, zorder=0)

    fig.suptitle("Figure 6 — Hidden-switch case study: event trace and pre/post outcomes", y=1.02)
    _save(fig, "fig6_switch_event_study")


def fig_switch_detection() -> None:
    df = _read_processed("switch_case_study")
    case_per_seed, case_summary = compute_switch_case_metrics(df)
    if case_per_seed.empty:
        raise ValueError("Switch case-study metrics are empty.")

    case_per_seed = case_per_seed.sort_values("seed").copy()
    family_colors = [FAMILY_COLORS.get(family, C_SWITCH) for family in case_per_seed["true_family_post"]]
    edge_colors = ["black" if ok else "#9B2C2C" for ok in case_per_seed["family_identification_correct"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.0))
    fig.subplots_adjust(wspace=0.34)

    detected = case_per_seed["detection_hand"].fillna(case_per_seed["switch_hand"])
    ax1.scatter(
        case_per_seed["switch_hand"],
        detected,
        s=80,
        c=family_colors,
        edgecolors=edge_colors,
        linewidths=1.2,
        alpha=0.92,
    )
    low = min(case_per_seed["switch_hand"].min(), detected.min()) - 100
    high = max(case_per_seed["switch_hand"].max(), detected.max()) + 100
    ax1.plot([low, high], [low, high], linestyle="--", color="black", linewidth=1.0, alpha=0.6)
    for _, row in case_per_seed.iterrows():
        ax1.text(row["switch_hand"] + 25, (row["detection_hand"] if pd.notna(row["detection_hand"]) else row["switch_hand"]) + 25, str(int(row["seed"])), fontsize=8)
    ax1.set_xlabel("True hidden-switch hand")
    ax1.set_ylabel("Detected switch hand")
    ax1.set_title("Detection timing by seed")
    ax1.grid(linestyle=":", linewidth=0.5, alpha=0.6)

    y = np.arange(len(case_per_seed))
    bars = ax2.barh(
        y,
        case_per_seed["post_switch_lift_vs_control"],
        color=family_colors,
        alpha=0.88,
        edgecolor=edge_colors,
        linewidth=1.1,
    )
    ax2.axvline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax2.set_yticks(y)
    ax2.set_yticklabels([f"Seed {int(seed)}" for seed in case_per_seed["seed"]])
    ax2.set_xlabel("Post-switch lift vs. balanced control")
    ax2.set_title(
        "Adaptive benefit by seed\n"
        f"(mean lift = {case_summary['mean_post_switch_lift_vs_control'].iloc[0]:.2f} chips/hand)"
    )
    ax2.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.6)

    legend_handles = [
        mpatches.Patch(color=FAMILY_COLORS["aggressive"], label="Switch to aggressive"),
        mpatches.Patch(color=FAMILY_COLORS["passive"], label="Switch to passive"),
        mpatches.Patch(color=FAMILY_COLORS["trappy"], label="Switch to trappy"),
    ]
    ax2.legend(handles=legend_handles, frameon=True, framealpha=0.92, edgecolor="0.8", loc="upper left")

    fig.suptitle("Figure 7 — Hidden-switch case study: detection timing and adaptive lift", y=1.02)
    _save(fig, "fig7_switch_detection")


def main() -> None:
    fig_main_comparison()
    fig_belief_ablation()
    fig_calibration()
    fig_robustness()
    fig_main_study_diagnostics()
    fig_switch_event_study()
    fig_switch_detection()
    print(f"All figures saved to {OUT}")


if __name__ == "__main__":
    main()

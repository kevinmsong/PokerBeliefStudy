"""
analysis.py — Compute metrics and generate manuscript-ready figures at 300 DPI.
"""
import os
import json
import math
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from src.utils import bootstrap_ci, load_json


# Style defaults
FIGURE_DPI = 300
FIGURE_STYLE = "seaborn-v0_8-whitegrid"
PALETTE = "colorblind"


def _setup_style():
    """Set up matplotlib style."""
    try:
        plt.style.use(FIGURE_STYLE)
    except OSError:
        plt.style.use("seaborn-whitegrid")


def compute_performance_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute average chips per hand with bootstrap confidence intervals.

    Parameters
    ----------
    df : pd.DataFrame
        Hand-level results DataFrame.

    Returns
    -------
    pd.DataFrame
        Summary DataFrame with columns:
        agent, matchup_label, mean_reward, ci_lower, ci_upper, n_hands.
    """
    rows = []

    groups = df.groupby(["agent0_type", "matchup_label"]) if "agent0_type" in df.columns else df.groupby(["matchup_id"])

    for (agent_type, matchup_label), grp in df.groupby(["agent0_type", "matchup_label"]):
        rewards_0 = grp["terminal_reward_0"].tolist()
        rewards_1 = grp["terminal_reward_1"].tolist()

        ci_0 = bootstrap_ci(rewards_0)
        ci_1 = bootstrap_ci(rewards_1)

        rows.append({
            "agent": agent_type,
            "player": 0,
            "matchup_label": matchup_label,
            "mean_reward": ci_0["mean"],
            "ci_lower": ci_0["lower"],
            "ci_upper": ci_0["upper"],
            "n_hands": len(grp),
        })

        agent1_type = grp["agent1_type"].iloc[0] if "agent1_type" in grp.columns else "agent1"
        rows.append({
            "agent": agent1_type,
            "player": 1,
            "matchup_label": matchup_label,
            "mean_reward": ci_1["mean"],
            "ci_lower": ci_1["lower"],
            "ci_upper": ci_1["upper"],
            "n_hands": len(grp),
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def compute_calibration_metrics(hand_records: List[dict]) -> Dict:
    """Compute Brier score, ECE, and reliability diagram data.

    Parameters
    ----------
    hand_records : List[dict]
        List of raw hand record dicts.

    Returns
    -------
    dict
        Contains: brier_score, ece, reliability_data (bins, mean_conf, mean_acc, counts).
    """
    # Extract predictions and outcomes from belief states
    predictions = []  # (predicted_prob, outcome)

    for record in hand_records:
        action_history = record.get("action_history", [])
        winner = record.get("showdown_winner")

        for step in action_history:
            posterior = step.get("posterior")
            if posterior is None:
                continue
            player = step.get("player", 0)

            # Predicted probability of winning for this player
            # Approximate using posterior entropy as proxy for confidence
            entropy_val = step.get("posterior_entropy")
            if entropy_val is None:
                continue

            max_entropy = math.log(7)  # 7 hand classes
            confidence = 1.0 - (entropy_val / max_entropy)

            # Outcome: did this player win?
            if winner == player:
                outcome = 1.0
            elif winner == -1:
                outcome = 0.5  # Tie
            else:
                outcome = 0.0

            predictions.append((confidence, outcome))

    if not predictions:
        return {
            "brier_score": None,
            "ece": None,
            "reliability_data": None,
            "n_predictions": 0,
        }

    preds = np.array(predictions)
    confidences = preds[:, 0]
    outcomes = preds[:, 1]

    # Brier score
    brier_score = float(np.mean((confidences - outcomes) ** 2))

    # Reliability diagram data (10 bins)
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    mean_conf = []
    mean_acc = []
    counts = []

    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i+1])
        if i == n_bins - 1:
            mask = (confidences >= bins[i]) & (confidences <= bins[i+1])
        n = mask.sum()
        if n > 0:
            mean_conf.append(float(np.mean(confidences[mask])))
            mean_acc.append(float(np.mean(outcomes[mask])))
            counts.append(int(n))
        else:
            mean_conf.append(float(bin_centers[i]))
            mean_acc.append(0.0)
            counts.append(0)

    # Expected Calibration Error
    ece = float(sum(
        c * abs(mc - ma) for c, mc, ma in zip(counts, mean_conf, mean_acc)
    ) / max(1, len(predictions)))

    return {
        "brier_score": brier_score,
        "ece": ece,
        "reliability_data": {
            "bins": bin_centers.tolist(),
            "mean_confidence": mean_conf,
            "mean_accuracy": mean_acc,
            "counts": counts,
        },
        "n_predictions": len(predictions),
    }


def compute_robustness_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute performance drop from development to held-out opponents.

    Parameters
    ----------
    df : pd.DataFrame
        Hand-level results DataFrame.

    Returns
    -------
    pd.DataFrame
        Robustness metrics with performance on dev vs held-out opponents.
    """
    held_out_families = ["held_out_1", "held_out_2"]

    rows = []
    for agent_type in df["agent0_type"].unique():
        agent_df = df[df["agent0_type"] == agent_type]

        # Dev performance (not held-out)
        dev_df = agent_df[~agent_df["opponent_family_1"].isin(held_out_families)]
        held_df = agent_df[agent_df["opponent_family_1"].isin(held_out_families)]

        dev_mean = float(dev_df["terminal_reward_0"].mean()) if len(dev_df) > 0 else 0.0
        held_mean = float(held_df["terminal_reward_0"].mean()) if len(held_df) > 0 else 0.0

        rows.append({
            "agent": agent_type,
            "dev_mean_reward": dev_mean,
            "held_out_mean_reward": held_mean,
            "performance_drop": dev_mean - held_mean,
            "n_dev": len(dev_df),
            "n_held_out": len(held_df),
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def plot_performance_comparison(summary: pd.DataFrame, output_path: str):
    """Bar chart with CI error bars at 300 DPI.

    Parameters
    ----------
    summary : pd.DataFrame
        Output of compute_performance_summary().
    output_path : str
        Path to save the figure.
    """
    _setup_style()
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    if summary.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    agents = summary["agent"].unique()
    x = np.arange(len(agents))
    width = 0.6

    colors = sns.color_palette(PALETTE, len(agents))

    means = []
    lowers = []
    uppers = []

    for agent in agents:
        ag_df = summary[summary["agent"] == agent]
        mean_val = ag_df["mean_reward"].mean()
        lower_val = ag_df["ci_lower"].mean()
        upper_val = ag_df["ci_upper"].mean()
        means.append(mean_val)
        lowers.append(mean_val - lower_val)
        uppers.append(upper_val - mean_val)

    bars = ax.bar(x, means, width, color=colors, alpha=0.85, edgecolor="black", linewidth=0.8)
    ax.errorbar(
        x, means,
        yerr=[lowers, uppers],
        fmt="none", color="black", capsize=5, capthick=1.5, linewidth=1.5
    )

    ax.set_xticks(x)
    ax.set_xticklabels(agents, fontsize=11)
    ax.set_ylabel("Mean Chips per Hand", fontsize=12)
    ax.set_title("Agent Performance Comparison", fontsize=14, fontweight="bold")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.legend(bars, agents, loc="upper right", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_reliability_diagram(cal_data: dict, output_path: str):
    """Calibration reliability diagram at 300 DPI.

    Parameters
    ----------
    cal_data : dict
        Output of compute_calibration_metrics().
    output_path : str
        Path to save the figure.
    """
    _setup_style()
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if cal_data.get("reliability_data") is None:
        for ax in axes:
            ax.text(0.5, 0.5, "No calibration data", ha="center", va="center")
        fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        return

    rel_data = cal_data["reliability_data"]
    bins = rel_data["bins"]
    mean_conf = rel_data["mean_confidence"]
    mean_acc = rel_data["mean_accuracy"]
    counts = rel_data["counts"]

    # Reliability diagram
    ax1 = axes[0]
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration")
    ax1.plot(mean_conf, mean_acc, "o-", color="steelblue", linewidth=2, markersize=6, label="Model")
    ax1.fill_between(mean_conf, mean_acc, mean_conf,
                     alpha=0.2, color="red", label="Calibration gap")
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.set_xlabel("Mean Confidence", fontsize=12)
    ax1.set_ylabel("Mean Accuracy", fontsize=12)
    ax1.set_title(f"Reliability Diagram\n(ECE={cal_data.get('ece', 0):.4f}, "
                  f"Brier={cal_data.get('brier_score', 0):.4f})", fontsize=12)
    ax1.legend(fontsize=10)

    # Histogram of confidence distribution
    ax2 = axes[1]
    ax2.bar(bins, counts, width=0.08, color="steelblue", alpha=0.7, edgecolor="black")
    ax2.set_xlabel("Confidence", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Confidence Distribution", fontsize=12)

    brier = cal_data.get("brier_score")
    ece = cal_data.get("ece")
    info_text = f"Brier Score: {brier:.4f}\nECE: {ece:.4f}" if brier is not None else "No data"
    ax2.text(0.05, 0.95, info_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_robustness_heatmap(robustness: pd.DataFrame, output_path: str):
    """Opponent shift robustness heatmap at 300 DPI.

    Parameters
    ----------
    robustness : pd.DataFrame
        Output of compute_robustness_metrics().
    output_path : str
        Path to save the figure.
    """
    _setup_style()
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    if robustness.empty:
        ax.text(0.5, 0.5, "No robustness data", ha="center", va="center")
        fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        return

    # Create matrix for heatmap
    metrics = ["dev_mean_reward", "held_out_mean_reward", "performance_drop"]
    labels = ["Dev Reward", "Held-Out Reward", "Performance Drop"]

    plot_data = robustness[["agent"] + metrics].set_index("agent")
    plot_data.columns = labels

    sns.heatmap(
        plot_data,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": "Chips per Hand"},
    )
    ax.set_title("Agent Robustness: Dev vs Held-Out Opponents", fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Agent", fontsize=11)

    plt.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_belief_trace(hand_record: dict, output_path: str):
    """Posterior belief evolution over a single hand at 300 DPI.

    Parameters
    ----------
    hand_record : dict
        A single hand record dict.
    output_path : str
        Path to save the figure.
    """
    from src.hand_classes import HAND_CLASSES
    _setup_style()
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    action_history = hand_record.get("action_history", [])

    # Extract belief states from belief agent's steps
    belief_steps = []
    for step in action_history:
        posterior = step.get("posterior")
        if posterior:
            belief_steps.append({
                "decision_index": step["decision_index"],
                "action": step["action"],
                "player": step["player"],
                "posterior": posterior,
            })

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    if not belief_steps:
        ax1.text(0.5, 0.5, "No belief data in this hand", ha="center", va="center")
        ax2.text(0.5, 0.5, "No belief data in this hand", ha="center", va="center")
        fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        return

    # Build belief trace
    steps = [s["decision_index"] for s in belief_steps]
    colors = sns.color_palette(PALETTE, len(HAND_CLASSES))

    for i, hc in enumerate(HAND_CLASSES):
        probs = [s["posterior"].get(hc, 0.0) for s in belief_steps]
        ax1.plot(steps, probs, "o-", color=colors[i], label=hc, linewidth=2, markersize=4)

    ax1.set_ylim([0, 1])
    ax1.set_xlabel("Decision Index", fontsize=11)
    ax1.set_ylabel("Posterior Probability", fontsize=11)
    ax1.set_title("Belief State Evolution", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=8, loc="upper right", ncol=2)

    # Action annotations
    for step in action_history:
        di = step["decision_index"]
        action = step["action"]
        ax1.axvline(x=di, color="gray", alpha=0.3, linestyle="--", linewidth=0.8)

    # Entropy trace
    entropies = []
    for step in belief_steps:
        post = step["posterior"]
        h = -sum(p * math.log(p) for p in post.values() if p > 0)
        entropies.append(h)

    ax2.plot(steps, entropies, "s-", color="purple", linewidth=2, markersize=5)
    ax2.set_xlabel("Decision Index", fontsize=11)
    ax2.set_ylabel("Posterior Entropy (nats)", fontsize=11)
    ax2.set_title("Belief Entropy Over Hand", fontsize=13)
    ax2.axhline(y=math.log(7), color="gray", linestyle="--", linewidth=0.8, label="Max entropy")
    ax2.legend(fontsize=10)

    board = hand_record.get("board", [])
    plt.suptitle(f"Belief Trace | Board: {' '.join(board)}", fontsize=12, y=1.01)

    plt.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def generate_all_tables(results_dir: str, output_dir: str):
    """Generate LaTeX/CSV tables for manuscript.

    Parameters
    ----------
    results_dir : str
        Directory containing processed results.
    output_dir : str
        Directory to save tables.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Look for summary CSV
    summary_files = [f for f in os.listdir(results_dir) if f.endswith("_summary.csv")]

    for fname in summary_files:
        fpath = os.path.join(results_dir, fname)
        df = pd.read_csv(fpath)

        experiment_id = fname.replace("_summary.csv", "")

        # Performance table
        if "agent0_type" in df.columns:
            perf = compute_performance_summary(df)
            if not perf.empty:
                perf_csv = os.path.join(output_dir, f"{experiment_id}_performance.csv")
                perf.to_csv(perf_csv, index=False)

                # LaTeX table
                perf_tex = os.path.join(output_dir, f"{experiment_id}_performance.tex")
                with open(perf_tex, "w") as f:
                    f.write(_df_to_latex(perf, caption=f"Agent Performance — {experiment_id}"))

        # Robustness table
        if "agent0_type" in df.columns and "opponent_family_1" in df.columns:
            rob = compute_robustness_metrics(df)
            if not rob.empty:
                rob_csv = os.path.join(output_dir, f"{experiment_id}_robustness.csv")
                rob.to_csv(rob_csv, index=False)

                rob_tex = os.path.join(output_dir, f"{experiment_id}_robustness.tex")
                with open(rob_tex, "w") as f:
                    f.write(_df_to_latex(rob, caption=f"Robustness Metrics — {experiment_id}"))

    print(f"Tables generated in {output_dir}")


def _df_to_latex(df: pd.DataFrame, caption: str = "") -> str:
    """Convert DataFrame to LaTeX table string."""
    latex = "\\begin{table}[h]\n"
    latex += "\\centering\n"
    if caption:
        latex += f"\\caption{{{caption}}}\n"
    latex += df.to_latex(index=False, float_format="%.3f")
    latex += "\\end{table}\n"
    return latex


def generate_all_figures(results_dir: str, raw_dir: str, figures_dir: str):
    """Generate all manuscript figures from results.

    Parameters
    ----------
    results_dir : str
        Directory with processed CSV results.
    raw_dir : str
        Directory with raw JSON hand records.
    figures_dir : str
        Output directory for figures.
    """
    os.makedirs(figures_dir, exist_ok=True)

    # Find summary files
    summary_files = [f for f in os.listdir(results_dir) if f.endswith("_summary.csv")]

    for fname in summary_files:
        experiment_id = fname.replace("_summary.csv", "")
        df = pd.read_csv(os.path.join(results_dir, fname))

        # Performance comparison
        if "agent0_type" in df.columns:
            summary = compute_performance_summary(df)
            plot_performance_comparison(
                summary,
                os.path.join(figures_dir, f"{experiment_id}_performance.png")
            )

            # Robustness
            robustness = compute_robustness_metrics(df)
            plot_robustness_heatmap(
                robustness,
                os.path.join(figures_dir, f"{experiment_id}_robustness.png")
            )

    # Load raw records for calibration and belief traces
    raw_files = [f for f in os.listdir(raw_dir) if f.endswith(".json")]

    all_records = []
    for fname in raw_files[:10]:  # Limit to avoid memory issues
        try:
            records = load_json(os.path.join(raw_dir, fname))
            all_records.extend(records)
        except Exception:
            pass

    if all_records:
        # Calibration
        cal_data = compute_calibration_metrics(all_records)
        plot_reliability_diagram(
            cal_data,
            os.path.join(figures_dir, "calibration_reliability.png")
        )

        # Belief trace for first hand with belief data
        for record in all_records:
            has_belief = any(
                step.get("posterior") is not None
                for step in record.get("action_history", [])
            )
            if has_belief:
                plot_belief_trace(
                    record,
                    os.path.join(figures_dir, "belief_trace_example.png")
                )
                break

    print(f"Figures generated in {figures_dir}")

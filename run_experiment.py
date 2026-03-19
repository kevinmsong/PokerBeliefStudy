"""
run_experiment.py — CLI entry point for running PokerBeliefStudy experiments.

Usage:
    python run_experiment.py --experiment main_comparison --seeds 42,43,44,45,46 --hands 1000 --output outputs/
    python run_experiment.py --experiment calibration --seeds 42 --hands 500 --output outputs/
    python run_experiment.py --experiment robustness --seeds 42,43,44 --hands 1000 --output outputs/
    python run_experiment.py --experiment belief_ablation --seeds 42,43 --hands 1000 --output outputs/
"""
import argparse
import os
import sys
import time
import json
from typing import Optional

import yaml


def parse_seeds(seeds_str: str):
    """Parse comma-separated seeds string into list of ints."""
    return [int(s.strip()) for s in seeds_str.split(",")]


def load_config(config_dir: str) -> dict:
    """Load all configuration files."""
    env_path = os.path.join(config_dir, "env.yaml")
    experiments_path = os.path.join(config_dir, "experiments.yaml")
    agents_path = os.path.join(config_dir, "agents.yaml")

    with open(env_path) as f:
        env_config = yaml.safe_load(f)

    with open(experiments_path) as f:
        experiments_config = yaml.safe_load(f)

    with open(agents_path) as f:
        agents_config = yaml.safe_load(f)

    return {
        "env": env_config,
        "experiments": experiments_config,
        "agents": agents_config,
    }


def run_experiment(
    experiment_name: str,
    seeds: list,
    hands: Optional[int],
    output_dir: str,
    config_dir: str = "config",
    verbose: bool = True,
    dry_run: bool = False,
):
    """Run a named experiment.

    Parameters
    ----------
    experiment_name : str
        Name of experiment defined in config/experiments.yaml.
    seeds : list
        List of integer seeds.
    hands : int | None
        Number of hands per seed. If None, use the experiment default from
        config/experiments.yaml.
    output_dir : str
        Output directory for results.
    config_dir : str
        Directory containing config YAML files.
    verbose : bool
        Whether to print progress.
    dry_run : bool
        If True, print what would be run without running.
    """
    start_time = time.time()

    # Load configuration
    config = load_config(config_dir)
    env_config = config["env"]
    experiments_config = config["experiments"]

    if experiment_name not in experiments_config:
        print(f"ERROR: Experiment '{experiment_name}' not found in {config_dir}/experiments.yaml")
        print(f"Available experiments: {list(experiments_config.keys())}")
        sys.exit(1)

    exp_config = experiments_config[experiment_name]
    matchups = exp_config["matchups"]
    configured_hands = exp_config.get("hands_per_seed", 100)
    hands_from_config = hands is None
    resolved_hands = configured_hands if hands_from_config else hands
    total_hands = sum(len(seeds) * matchup.get("hands_per_seed", resolved_hands) for matchup in matchups)
    has_overrides = any("hands_per_seed" in matchup for matchup in matchups)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Experiment: {experiment_name}")
        print(f"  Description: {exp_config.get('description', 'N/A')}")
        print(f"  Seeds: {seeds}")
        hands_line = f"  Hands/seed: {resolved_hands}"
        if hands_from_config:
            hands_line += " (from config)"
        if has_overrides:
            hands_line += "; matchup overrides active"
        print(hands_line)
        print(f"  Matchups: {len(matchups)}")
        print(f"  Total hands: {total_hands:,}")
        print(f"  Output: {output_dir}")
        print(f"{'='*60}\n")

    if dry_run:
        print("[DRY RUN] Would run the above experiment. Exiting.")
        return None

    # Import here to avoid slow startup when just checking args
    from src.tournament import run_tournament
    from src.analysis import (
        compute_performance_summary, compute_robustness_metrics,
        plot_performance_comparison, plot_robustness_heatmap,
        generate_all_tables
    )

    # Run the tournament
    df = run_tournament(
        matchups=matchups,
        seeds=seeds,
        hands_per_seed=resolved_hands,
        config=env_config,
        output_dir=output_dir,
        experiment_id=experiment_name,
        verbose=verbose,
    )

    elapsed = time.time() - start_time

    if verbose:
        print(f"\nTournament complete in {elapsed:.1f}s")
        if df is not None and not df.empty:
            print(f"Total hands played: {len(df):,}")
            print(f"\nPerformance by agent type:")
            if "agent0_type" in df.columns:
                for agent_type, grp in df.groupby("agent0_type"):
                    mean_r = grp["terminal_reward_0"].mean()
                    print(f"  {agent_type}: {mean_r:.2f} chips/hand (player 0)")

    # Generate analysis
    if df is not None and not df.empty:
        try:
            figures_dir = os.path.join(output_dir, "figures")
            tables_dir = os.path.join(output_dir, "tables")
            processed_dir = os.path.join(output_dir, "processed")

            os.makedirs(figures_dir, exist_ok=True)
            os.makedirs(tables_dir, exist_ok=True)

            # Performance summary
            if "agent0_type" in df.columns and "matchup_label" in df.columns:
                summary = compute_performance_summary(df)
                plot_performance_comparison(
                    summary,
                    os.path.join(figures_dir, f"{experiment_name}_performance.png")
                )
                if verbose:
                    print(f"  Performance plot saved.")

                robustness = compute_robustness_metrics(df)
                plot_robustness_heatmap(
                    robustness,
                    os.path.join(figures_dir, f"{experiment_name}_robustness.png")
                )
                if verbose:
                    print(f"  Robustness heatmap saved.")

            generate_all_tables(processed_dir, tables_dir)
            if verbose:
                print(f"  Tables saved.")

        except Exception as e:
            if verbose:
                print(f"  Warning: analysis failed: {e}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Run PokerBeliefStudy experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiment.py --experiment main_comparison --seeds 42,43,44,45,46 --hands 1000 --output outputs/
  python run_experiment.py --experiment calibration --seeds 42 --hands 500 --output outputs/
  python run_experiment.py --experiment robustness --seeds 42,43,44 --hands 1000 --output outputs/
  python run_experiment.py --experiment belief_ablation --seeds 42,43 --hands 1000 --output outputs/
        """,
    )
    parser.add_argument(
        "--experiment", "-e",
        type=str,
        required=True,
        help="Experiment name (defined in config/experiments.yaml)",
    )
    parser.add_argument(
        "--seeds", "-s",
        type=str,
        default="42",
        help="Comma-separated list of random seeds (default: 42)",
    )
    parser.add_argument(
        "--hands", "-n",
        type=int,
        default=None,
        help="Default hands per seed. If omitted, use the experiment config.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="outputs/",
        help="Output directory (default: outputs/)",
    )
    parser.add_argument(
        "--config-dir", "-c",
        type=str,
        default="config",
        help="Configuration directory (default: config)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be run without running",
    )

    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    verbose = not args.quiet

    df = run_experiment(
        experiment_name=args.experiment,
        seeds=seeds,
        hands=args.hands,
        output_dir=args.output,
        config_dir=args.config_dir,
        verbose=verbose,
        dry_run=args.dry_run,
    )

    if verbose and df is not None and not df.empty:
        print(f"\nDone! Results saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

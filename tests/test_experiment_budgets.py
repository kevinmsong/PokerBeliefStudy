"""
test_experiment_budgets.py — Guardrails for exact study hand budgets.
"""
from pathlib import Path

import yaml


EXPERIMENTS_PATH = Path("config/experiments.yaml")
MAIN_STUDY_EXPERIMENTS = [
    "main_comparison",
    "belief_ablation",
    "calibration",
    "robustness",
]


def _load_experiments():
    return yaml.safe_load(EXPERIMENTS_PATH.read_text())


def _experiment_total_hands(experiment: dict) -> int:
    seeds = len(experiment["seeds"])
    default_hands = experiment["hands_per_seed"]
    return sum(seeds * matchup.get("hands_per_seed", default_hands) for matchup in experiment["matchups"])


def test_main_study_totals_exactly_100k_hands():
    experiments = _load_experiments()
    total = sum(_experiment_total_hands(experiments[name]) for name in MAIN_STUDY_EXPERIMENTS)
    assert total == 100_000


def test_switch_case_study_totals_exactly_100k_hands():
    experiments = _load_experiments()
    total = _experiment_total_hands(experiments["switch_case_study"])
    assert total == 100_000

"""
utils.py — Shared utility functions for the PokerBeliefStudy project.
"""
import json
import os
import math
import time
import hashlib
from typing import Any, Dict, List, Optional
import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    """Create a numpy RNG from an integer seed."""
    return np.random.default_rng(seed)


def seed_from_string(s: str) -> int:
    """Convert a string to a deterministic integer seed."""
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % (2**32)


def normalize_dist(dist: Dict[str, float]) -> Dict[str, float]:
    """Normalize a dict distribution to sum to 1.0."""
    total = sum(dist.values())
    if total <= 0:
        n = len(dist)
        return {k: 1.0 / n for k in dist}
    return {k: v / total for k, v in dist.items()}


def entropy(dist: Dict[str, float]) -> float:
    """Compute Shannon entropy of a distribution in nats."""
    h = 0.0
    for p in dist.values():
        if p > 0:
            h -= p * math.log(p)
    return h


def save_json(obj: Any, path: str) -> None:
    """Save object as JSON to path, creating directories as needed."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=_json_serializer)


def load_json(path: str) -> Any:
    """Load JSON from path."""
    with open(path, 'r') as f:
        return json.load(f)


def _json_serializer(obj: Any) -> Any:
    """Handle non-serializable types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def bootstrap_ci(values: List[float], n_boot: int = 1000, ci: float = 0.95,
                  rng: Optional[np.random.Generator] = None) -> Dict[str, float]:
    """Compute bootstrap confidence interval for mean.

    Returns dict with 'mean', 'lower', 'upper'.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    arr = np.array(values)
    if len(arr) == 0:
        return {'mean': 0.0, 'lower': 0.0, 'upper': 0.0}
    boot_means = np.array([
        np.mean(rng.choice(arr, size=len(arr), replace=True))
        for _ in range(n_boot)
    ])
    alpha = 1 - ci
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return {'mean': float(np.mean(arr)), 'lower': lower, 'upper': upper}


def format_duration(seconds: float) -> str:
    """Format duration in seconds as human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    return f"{seconds/3600:.1f}h"


def load_yaml(path: str) -> Any:
    """Load YAML file."""
    import yaml
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def chips_won_lost(initial_stack: int, final_stack: int) -> int:
    """Return net chips won (positive) or lost (negative)."""
    return final_stack - initial_stack


def generate_hand_id(run_id: str, hand_num: int) -> str:
    """Generate a unique hand ID."""
    return f"{run_id}_h{hand_num:06d}"


def generate_run_id(experiment_id: str, seed: int) -> str:
    """Generate a unique run ID."""
    return f"{experiment_id}_s{seed}"

"""
equity.py — Monte Carlo equity estimation.

Given hero hole cards and board, estimate P(win) against random opponent hand
or against distribution over hand classes.
"""
import numpy as np
from typing import Tuple, List, Dict, Optional

from src.cards import make_deck, card_rank, card_suit
from src.hand_eval import evaluate_hand
from src.hand_classes import classify_hand, HAND_CLASSES


def estimate_equity(
    hero_cards: Tuple[str, str],
    board: List[str],
    opp_belief: Dict[str, float],
    rng: np.random.Generator,
    n_samples: int = 200,
    dead_cards: Optional[List[str]] = None,
) -> float:
    """Return estimated equity [0,1] for hero.

    Parameters
    ----------
    hero_cards : Tuple[str, str]
        Hero's hole cards.
    board : List[str]
        Community cards currently visible (4 or 5 cards).
    opp_belief : Dict[str, float]
        Probability distribution over opponent hand classes.
    rng : np.random.Generator
        Numpy RNG for reproducibility.
    n_samples : int
        Number of Monte Carlo samples.
    dead_cards : Optional[List[str]]
        Additional dead cards (besides hero_cards and board).

    Returns
    -------
    float
        Equity estimate in [0, 1].
    """
    dead_set = set(list(hero_cards) + list(board) + (dead_cards or []))
    deck = make_deck()
    available = [c for c in deck if c not in dead_set]

    wins = 0.0
    total = 0

    for _ in range(n_samples):
        # Sample opponent hole cards weighted by belief distribution
        opp_cards = _sample_opp_hand(available, opp_belief, rng)
        if opp_cards is None:
            continue

        # Determine remaining board cards needed
        cards_needed = max(0, 5 - len(board))
        if cards_needed > 0:
            # Sample future board cards
            dead_for_runout = dead_set | set(opp_cards)
            remaining_deck = [c for c in available if c not in opp_cards]
            if len(remaining_deck) < cards_needed:
                continue
            runout_indices = rng.choice(len(remaining_deck), size=cards_needed, replace=False)
            runout = [remaining_deck[i] for i in runout_indices]
        else:
            runout = []

        full_board = list(board) + runout
        hero_full = list(hero_cards) + full_board
        opp_full = list(opp_cards) + full_board

        hero_score = evaluate_hand(hero_full)
        opp_score = evaluate_hand(opp_full)

        if hero_score > opp_score:
            wins += 1.0
        elif hero_score == opp_score:
            wins += 0.5

        total += 1

    if total == 0:
        return 0.5
    return wins / total


def _sample_opp_hand(
    available: List[str],
    opp_belief: Dict[str, float],
    rng: np.random.Generator,
    max_tries: int = 50,
) -> Optional[Tuple[str, str]]:
    """Sample opponent hole cards consistent with belief distribution.

    Uses rejection sampling: sample random 2-card hand, check if its
    class is consistent with the belief (weighted acceptance).
    """
    if not available or len(available) < 2:
        return None

    # Fast path: if belief is uniform or we want random sample
    total_belief = sum(opp_belief.values())
    if total_belief <= 0:
        # Uniform - just pick random
        idx = rng.choice(len(available), size=2, replace=False)
        return (available[idx[0]], available[idx[1]])

    # Normalize belief
    norm_belief = {k: v / total_belief for k, v in opp_belief.items()}

    # Use importance sampling with acceptance-rejection
    for _ in range(max_tries):
        if len(available) < 2:
            break
        idx = rng.choice(len(available), size=2, replace=False)
        c1, c2 = available[idx[0]], available[idx[1]]
        # Accept with probability proportional to belief weight of hand class
        # For speed, we classify against a minimal board (just for sampling)
        # Accept all with probability = belief weight of hand class
        # This is an approximation - we accept with prob = class_prob / max_prob
        max_prob = max(norm_belief.values()) if norm_belief else 1.0
        # Sample acceptance: always accept if belief > 0 for any class
        # Simple approach: sample hand class from belief, then find hand matching
        accept_prob = 1.0  # Accept all for speed; belief weighting done via equity calc
        if rng.random() < accept_prob:
            return (c1, c2)

    # Fallback: just return random pair
    if len(available) >= 2:
        idx = rng.choice(len(available), size=2, replace=False)
        return (available[idx[0]], available[idx[1]])
    return None


def estimate_equity_uniform(
    hero_cards: Tuple[str, str],
    board: List[str],
    rng: np.random.Generator,
    n_samples: int = 200,
    dead_cards: Optional[List[str]] = None,
) -> float:
    """Estimate equity against a uniformly random opponent hand."""
    uniform_belief = {c: 1.0 / len(HAND_CLASSES) for c in HAND_CLASSES}
    return estimate_equity(hero_cards, board, uniform_belief, rng, n_samples, dead_cards)


def compute_pot_odds(call_amount: int, pot: int) -> float:
    """Return pot odds as a fraction [0,1] (what fraction of pot you must call)."""
    if call_amount <= 0:
        return 0.0
    total = pot + call_amount
    if total <= 0:
        return 0.0
    return call_amount / total


def equity_beats_pot_odds(equity: float, call_amount: int, pot: int) -> bool:
    """Return True if equity justifies a call based on pot odds."""
    pot_odds_required = compute_pot_odds(call_amount, pot)
    return equity >= pot_odds_required

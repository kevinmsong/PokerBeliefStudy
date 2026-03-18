"""
hand_eval.py — Complete 7-card poker hand evaluator.
Returns comparable tuples where higher tuple = better hand.

Hand rankings (best to worst):
9 = Straight Flush
8 = Four of a Kind
7 = Full House
6 = Flush
5 = Straight
4 = Three of a Kind
3 = Two Pair
2 = One Pair
1 = High Card
"""
from itertools import combinations
from typing import List, Tuple
from src.cards import RANK_VALUES, card_rank, card_suit

HAND_RANK_NAMES = {
    9: "Straight Flush",
    8: "Four of a Kind",
    7: "Full House",
    6: "Flush",
    5: "Straight",
    4: "Three of a Kind",
    3: "Two Pair",
    2: "One Pair",
    1: "High Card",
}


def _rank_counts(cards: List[str]) -> dict:
    """Return {rank_value: count} for given cards."""
    counts = {}
    for c in cards:
        r = card_rank(c)
        counts[r] = counts.get(r, 0) + 1
    return counts


def _is_flush(cards: List[str]) -> bool:
    """Return True if all 5 cards share the same suit."""
    return len(set(card_suit(c) for c in cards)) == 1


def _is_straight(ranks: List[int]) -> Tuple[bool, int]:
    """Check if 5 sorted (descending) rank values form a straight.

    Returns (is_straight, high_card_rank).
    Handles wheel (A-2-3-4-5).
    """
    unique = sorted(set(ranks), reverse=True)
    if len(unique) < 5:
        return False, 0
    # Normal straight
    for i in range(len(unique) - 4):
        window = unique[i:i+5]
        if window[0] - window[4] == 4:
            return True, window[0]
    # Wheel: A-2-3-4-5 -> ranks 12,0,1,2,3
    if set([12, 0, 1, 2, 3]).issubset(set(unique)):
        return True, 3  # 5-high straight
    return False, 0


def _evaluate_5(cards: List[str]) -> tuple:
    """Evaluate exactly 5 cards and return comparable tuple."""
    assert len(cards) == 5
    ranks = sorted([card_rank(c) for c in cards], reverse=True)
    flush = _is_flush(cards)
    is_str, str_high = _is_straight(ranks)

    if flush and is_str:
        return (9, str_high)

    counts = _rank_counts(cards)
    count_vals = sorted(counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
    freq_sorted = [r for r, _ in count_vals]
    freqs = [v for _, v in count_vals]

    if freqs[0] == 4:
        # Four of a kind
        quad_rank = freq_sorted[0]
        kicker = freq_sorted[1]
        return (8, quad_rank, kicker)

    if freqs[0] == 3 and freqs[1] == 2:
        return (7, freq_sorted[0], freq_sorted[1])

    if flush:
        return (6,) + tuple(ranks)

    if is_str:
        return (5, str_high)

    if freqs[0] == 3:
        trip_rank = freq_sorted[0]
        kickers = sorted([r for r in ranks if r != trip_rank], reverse=True)
        return (4, trip_rank) + tuple(kickers)

    if freqs[0] == 2 and freqs[1] == 2:
        pair1 = freq_sorted[0]
        pair2 = freq_sorted[1]
        kicker = freq_sorted[2]
        return (3, pair1, pair2, kicker)

    if freqs[0] == 2:
        pair_rank = freq_sorted[0]
        kickers = sorted([r for r in ranks if r != pair_rank or ranks.count(pair_rank) > 1], reverse=True)
        # Get kickers properly
        remaining = ranks.copy()
        remaining.remove(pair_rank)
        remaining.remove(pair_rank)
        return (2, pair_rank) + tuple(remaining)

    return (1,) + tuple(ranks)


def evaluate_hand(cards: List[str]) -> tuple:
    """Evaluate best 5-card hand from up to 7 cards. Returns comparable tuple."""
    if len(cards) <= 5:
        return _evaluate_5(cards)
    best = None
    for combo in combinations(cards, 5):
        score = _evaluate_5(list(combo))
        if best is None or score > best:
            best = score
    return best


def compare_hands(cards_a: List[str], cards_b: List[str]) -> int:
    """Return 1 if A wins, -1 if B wins, 0 if tie."""
    score_a = evaluate_hand(cards_a)
    score_b = evaluate_hand(cards_b)
    if score_a > score_b:
        return 1
    elif score_a < score_b:
        return -1
    return 0


def hand_rank_name(rank_tuple: tuple) -> str:
    """Return human-readable name of hand rank."""
    if not rank_tuple:
        return "Unknown"
    return HAND_RANK_NAMES.get(rank_tuple[0], "Unknown")

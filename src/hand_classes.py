"""
hand_classes.py — Map (hole_cards, board) to one of 7 hand classes.

Classification is deterministic and uses:
1. Actual hand strength (via hand_eval)
2. Draw detection
3. Board-relative strength percentile

Hand Classes (ordered best to worst):
- nuts_or_near_nuts
- strong_made
- medium_made
- weak_showdown
- strong_draw
- weak_draw
- air
"""
from itertools import combinations
from typing import Tuple, List, Dict
import numpy as np

from src.cards import make_deck, RANK_VALUES, card_rank, card_suit
from src.hand_eval import evaluate_hand, compare_hands

HAND_CLASSES = [
    'nuts_or_near_nuts',
    'strong_made',
    'medium_made',
    'weak_showdown',
    'strong_draw',
    'weak_draw',
    'air',
]


def _has_pair(hole_cards: Tuple[str, str], board: List[str]) -> bool:
    all_cards = list(hole_cards) + board
    ranks = [card_rank(c) for c in all_cards]
    return len(ranks) != len(set(ranks))


def _get_hand_strength_category(hole_cards: Tuple[str, str], board: List[str]) -> int:
    """Return hand strength category from evaluate_hand tuple."""
    all_cards = list(hole_cards) + board
    score = evaluate_hand(all_cards)
    return score[0]  # 1=High Card, 2=Pair, ..., 9=Straight Flush


def _estimate_percentile(hole_cards: Tuple[str, str], board: List[str],
                          n_samples: int = 100) -> float:
    """Estimate hand strength percentile vs random opponent hands [0,1]."""
    deck = make_deck()
    dead = set(list(hole_cards) + board)
    available = [c for c in deck if c not in dead]

    hero_cards = list(hole_cards) + board
    hero_score = evaluate_hand(hero_cards)

    wins = 0
    ties = 0
    total = 0
    # Deterministic sampling using fixed seed for classification
    rng = np.random.default_rng(hash(tuple(sorted(hole_cards) + sorted(board))) & 0xFFFFFFFF)
    indices = rng.choice(len(available), size=min(n_samples * 2, len(available)), replace=False)
    sampled = [available[i] for i in indices]

    i = 0
    while total < n_samples and i + 1 < len(sampled):
        opp_hole = (sampled[i], sampled[i+1])
        i += 2
        # Avoid using hole cards
        if opp_hole[0] in dead or opp_hole[1] in dead:
            continue
        opp_cards = list(opp_hole) + board
        opp_score = evaluate_hand(opp_cards)
        if hero_score > opp_score:
            wins += 1
        elif hero_score == opp_score:
            ties += 1
        total += 1

    if total == 0:
        return 0.5
    return (wins + 0.5 * ties) / total


def _detect_flush_draw(hole_cards: Tuple[str, str], board: List[str]) -> str:
    """Detect flush draw type: 'nut_flush_draw', 'flush_draw', 'backdoor_flush_draw', or None."""
    for suit in ['c', 'd', 'h', 's']:
        suited_hole = [c for c in hole_cards if card_suit(c) == suit]
        suited_board = [c for c in board if card_suit(c) == suit]
        total_suited = len(suited_hole) + len(suited_board)
        if total_suited >= 4 and len(suited_hole) >= 1:
            # Check if it's nut flush draw (highest card of suit)
            all_suited = suited_hole + suited_board
            deck_suited = [c for c in make_deck() if card_suit(c) == suit]
            max_rank = max(card_rank(c) for c in deck_suited)
            hero_max = max(card_rank(c) for c in suited_hole)
            if hero_max == max_rank or card_rank(sorted(all_suited, key=card_rank)[-1]) == hero_max:
                return 'nut_flush_draw'
            return 'flush_draw'
        elif total_suited == 3 and len(suited_hole) >= 1:
            return 'backdoor_flush_draw'
    return None


def _detect_straight_draw(hole_cards: Tuple[str, str], board: List[str]) -> str:
    """Detect straight draw type: 'oesd', 'gutshot', 'backdoor_straight', or None."""
    all_cards = list(hole_cards) + board
    ranks = sorted(set(card_rank(c) for c in all_cards))
    # Include ace as low
    if 12 in ranks:
        ranks = [0] + ranks if 0 not in ranks else ranks

    # Check for open-ended straight draw (4 consecutive cards)
    for i in range(len(ranks) - 3):
        window = ranks[i:i+4]
        if window[-1] - window[0] == 3 and len(window) == 4:
            # Check both ends extend
            low_ext = window[0] - 1
            high_ext = window[-1] + 1
            if low_ext >= 0 and high_ext <= 12:
                return 'oesd'

    # Gutshot: 4 cards with exactly one gap
    for i in range(len(ranks) - 3):
        window = ranks[i:i+4]
        if window[-1] - window[0] == 4 and len(window) == 4:
            return 'gutshot'

    # Backdoor straight draw: 3 consecutive ranks using hole cards
    hole_ranks = set(card_rank(c) for c in hole_cards)
    board_ranks = set(card_rank(c) for c in board)
    for r in hole_ranks:
        # Check if we have 2 of 3 consecutive ranks using at least one hole card
        for span in [(r-2, r-1, r), (r-1, r, r+1), (r, r+1, r+2)]:
            if all(0 <= x <= 12 for x in span):
                count = sum(1 for x in span if x in hole_ranks or x in board_ranks)
                if count >= 2 and any(x in hole_ranks for x in span):
                    return 'backdoor_straight'
    return None


def _board_is_paired(board: List[str]) -> bool:
    """Check if board contains a pair."""
    ranks = [card_rank(c) for c in board]
    return len(ranks) != len(set(ranks))


def _has_set_or_better(hole_cards: Tuple[str, str], board: List[str]) -> bool:
    """Check if hero has three of a kind or better using hole cards."""
    all_cards = list(hole_cards) + board
    score = evaluate_hand(all_cards)
    return score[0] >= 4  # Three of a Kind or better


def _has_two_pair_or_better(hole_cards: Tuple[str, str], board: List[str]) -> bool:
    all_cards = list(hole_cards) + board
    score = evaluate_hand(all_cards)
    return score[0] >= 3


def _has_top_pair(hole_cards: Tuple[str, str], board: List[str]) -> bool:
    """Check if hero has top pair (using a hole card to pair highest board card)."""
    if not board:
        return False
    max_board_rank = max(card_rank(c) for c in board)
    hole_ranks = [card_rank(c) for c in hole_cards]
    return max_board_rank in hole_ranks


def _top_pair_kicker_strength(hole_cards: Tuple[str, str], board: List[str]) -> str:
    """Return 'strong', 'medium', or 'weak' kicker for top pair."""
    max_board_rank = max(card_rank(c) for c in board)
    kickers = [card_rank(c) for c in hole_cards if card_rank(c) != max_board_rank]
    if not kickers:
        # Both hole cards pair the top board card (unlikely but handle it)
        return 'strong'
    kicker = max(kickers)
    if kicker >= RANK_VALUES['J']:  # J or better
        return 'strong'
    elif kicker >= RANK_VALUES['7']:
        return 'medium'
    return 'weak'


def _has_overpair(hole_cards: Tuple[str, str], board: List[str]) -> Tuple[bool, bool]:
    """Return (has_overpair, is_small_overpair)."""
    if card_rank(hole_cards[0]) != card_rank(hole_cards[1]):
        return False, False
    hole_rank = card_rank(hole_cards[0])
    if not board:
        return True, hole_rank <= RANK_VALUES['8']
    max_board_rank = max(card_rank(c) for c in board)
    if hole_rank > max_board_rank:
        return True, hole_rank <= RANK_VALUES['8']
    return False, False


def _has_second_pair(hole_cards: Tuple[str, str], board: List[str]) -> Tuple[bool, str]:
    """Return (has_second_pair, kicker_strength)."""
    if len(board) < 2:
        return False, 'none'
    board_ranks = sorted(set(card_rank(c) for c in board), reverse=True)
    if len(board_ranks) < 2:
        return False, 'none'
    second_board_rank = board_ranks[1]
    hole_ranks = [card_rank(c) for c in hole_cards]
    if second_board_rank in hole_ranks:
        kicker_ranks = [r for r in hole_ranks if r != second_board_rank]
        kicker = max(kicker_ranks) if kicker_ranks else 0
        if kicker >= RANK_VALUES['J']:
            return True, 'strong'
        elif kicker >= RANK_VALUES['7']:
            return True, 'medium'
        return True, 'weak'
    return False, 'none'


def _has_bottom_pair(hole_cards: Tuple[str, str], board: List[str]) -> bool:
    """Return True if hero has bottom pair (pairs the lowest board card)."""
    if not board:
        return False
    board_ranks = sorted(set(card_rank(c) for c in board), reverse=True)
    if len(board_ranks) < 2:
        return False
    bottom_board_rank = board_ranks[-1]
    hole_ranks = [card_rank(c) for c in hole_cards]
    return bottom_board_rank in hole_ranks


def _has_third_pair(hole_cards: Tuple[str, str], board: List[str]) -> bool:
    """Return True if hero has third pair (pairs the third-highest board card)."""
    if len(board) < 3:
        return False
    board_ranks = sorted(set(card_rank(c) for c in board), reverse=True)
    if len(board_ranks) < 3:
        return False
    third_board_rank = board_ranks[2]
    hole_ranks = [card_rank(c) for c in hole_cards]
    return third_board_rank in hole_ranks


def _has_ace_high(hole_cards: Tuple[str, str], board: List[str]) -> bool:
    """Return True if hero has ace high (ace in hole, no pair)."""
    hole_ranks = [card_rank(c) for c in hole_cards]
    board_ranks = [card_rank(c) for c in board]
    has_ace = RANK_VALUES['A'] in hole_ranks
    has_pair = any(r in board_ranks for r in hole_ranks)
    return has_ace and not has_pair


def classify_hand(hole_cards: Tuple[str, str], board: List[str]) -> str:
    """Return the hand class for these hole cards given this board.

    Classification is deterministic, mutually exclusive, and exhaustive.
    """
    all_cards = list(hole_cards) + board
    score = evaluate_hand(all_cards)
    hand_cat = score[0]

    # Category 1-9: 1=High Card, 2=Pair, 3=Two Pair, 4=Trips, 5=Straight,
    #               6=Flush, 7=Full House, 8=Quads, 9=Straight Flush

    # Straight Flush, Quads -> nuts_or_near_nuts
    if hand_cat >= 8:
        return 'nuts_or_near_nuts'

    # Full House -> nuts_or_near_nuts
    if hand_cat == 7:
        return 'nuts_or_near_nuts'

    # Flush
    if hand_cat == 6:
        # Nut flush or near-nut flush -> nuts_or_near_nuts
        # Determine if it's nut flush
        flush_suit = None
        for suit in ['c', 'd', 'h', 's']:
            suited_all = [c for c in all_cards if card_suit(c) == suit]
            if len(suited_all) >= 5:
                flush_suit = suit
                break
        if flush_suit:
            suited_hole = [c for c in hole_cards if card_suit(c) == flush_suit]
            suited_all_suit = [c for c in all_cards if card_suit(c) == flush_suit]
            max_hero_flush_rank = max(card_rank(c) for c in suited_hole) if suited_hole else -1
            max_all_flush_rank = max(card_rank(c) for c in suited_all_suit)
            if max_hero_flush_rank == max_all_flush_rank:
                return 'nuts_or_near_nuts'
        return 'strong_made'

    # Straight
    if hand_cat == 5:
        # Check if it's near-nut straight (using board-relative context)
        percentile = _estimate_percentile(hole_cards, board, n_samples=50)
        if percentile >= 0.92:
            return 'nuts_or_near_nuts'
        return 'strong_made'

    # Three of a Kind
    if hand_cat == 4:
        if _has_set_or_better(hole_cards, board):
            # Set (using both hole cards to make trips) -> nuts_or_near_nuts
            hole_ranks = [card_rank(c) for c in hole_cards]
            if hole_ranks[0] == hole_ranks[1]:
                # Pocket pair hit the board
                trip_rank = score[1]
                if trip_rank == hole_ranks[0]:
                    return 'nuts_or_near_nuts'
            return 'strong_made'
        return 'strong_made'

    # Two Pair
    if hand_cat == 3:
        # Top two pair using both hole cards -> nuts_or_near_nuts
        board_ranks = sorted(set(card_rank(c) for c in board), reverse=True)
        hole_ranks = sorted([card_rank(c) for c in hole_cards], reverse=True)
        # Check if both hole cards pair board cards
        pairs_formed = [r for r in hole_ranks if r in [card_rank(c) for c in board]]
        if len(pairs_formed) == 2:
            # Two pair using two different hole cards
            if board_ranks and hole_ranks[0] == board_ranks[0]:
                return 'nuts_or_near_nuts'
            return 'strong_made'
        # Pocket pair + board pair
        if hole_ranks[0] == hole_ranks[1]:
            return 'strong_made'
        return 'strong_made'

    # One Pair
    if hand_cat == 2:
        pair_rank = score[1]
        board_ranks = sorted(set(card_rank(c) for c in board), reverse=True)
        hole_ranks = [card_rank(c) for c in hole_cards]
        max_board_rank = board_ranks[0] if board_ranks else -1

        # Overpair check
        has_overpair, is_small_overpair = _has_overpair(hole_cards, board)
        if has_overpair:
            if is_small_overpair:
                return 'medium_made'
            return 'strong_made'

        # Top pair check
        if _has_top_pair(hole_cards, board):
            kicker_str = _top_pair_kicker_strength(hole_cards, board)
            if kicker_str == 'strong':
                return 'strong_made'
            elif kicker_str == 'medium':
                return 'medium_made'
            else:
                return 'medium_made'

        # Second pair
        has_second, kicker_str = _has_second_pair(hole_cards, board)
        if has_second:
            if kicker_str == 'strong':
                return 'medium_made'
            return 'weak_showdown'

        # Bottom or third pair
        if _has_bottom_pair(hole_cards, board) or _has_third_pair(hole_cards, board):
            return 'weak_showdown'

        # Any other pair
        return 'weak_showdown'

    # High Card (hand_cat == 1) - check for draws and ace high
    flush_draw = _detect_flush_draw(hole_cards, board)
    straight_draw = _detect_straight_draw(hole_cards, board)

    # Combo draw (flush draw + straight draw) -> strong_draw
    if flush_draw in ('nut_flush_draw', 'flush_draw') and straight_draw in ('oesd', 'gutshot'):
        return 'strong_draw'

    # Nut flush draw or OESD -> strong_draw
    if flush_draw == 'nut_flush_draw' or straight_draw == 'oesd':
        return 'strong_draw'

    # Regular flush draw or gutshot -> weak_draw (unless nut flush draw handled above)
    if flush_draw == 'flush_draw':
        return 'weak_draw'

    if straight_draw == 'gutshot':
        return 'weak_draw'

    # Ace high with decent kicker -> weak_showdown
    if _has_ace_high(hole_cards, board):
        hole_ranks = [card_rank(c) for c in hole_cards]
        non_ace_rank = [r for r in hole_ranks if r != RANK_VALUES['A']]
        if non_ace_rank and max(non_ace_rank) >= RANK_VALUES['9']:
            return 'weak_showdown'

    return 'air'


def uniform_prior() -> Dict[str, float]:
    """Return uniform prior over hand classes."""
    n = len(HAND_CLASSES)
    return {c: 1.0 / n for c in HAND_CLASSES}


def get_hand_class_index(hand_class: str) -> int:
    """Return index of hand class in HAND_CLASSES list."""
    return HAND_CLASSES.index(hand_class)

"""
cards.py — Card representation, deck creation, shuffling, dealing.
All randomness goes through an explicit numpy RNG for reproducibility.
"""
import numpy as np
from typing import List, Tuple, Optional

RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
SUITS = ['c', 'd', 'h', 's']
RANK_VALUES = {r: i for i, r in enumerate(RANKS)}


def make_deck() -> List[str]:
    """Return a standard 52-card deck as list of strings like 'Ah', 'Td'."""
    return [r + s for r in RANKS for s in SUITS]


def shuffle_deck(deck: List[str], rng: np.random.Generator) -> List[str]:
    """Return a shuffled copy of the deck using the provided RNG."""
    deck = deck.copy()
    rng.shuffle(deck)
    return deck


def deal_cards(deck: List[str], n: int, exclude: Optional[List[str]] = None) -> Tuple[List[str], List[str]]:
    """Deal n cards from deck, excluding any in exclude list.

    Returns (dealt, remaining).
    """
    exclude_set = set(exclude or [])
    available = [c for c in deck if c not in exclude_set]
    return available[:n], available[n:]


def card_rank(card: str) -> int:
    """Return integer rank of card (0=2, 12=Ace)."""
    return RANK_VALUES[card[0]]


def card_suit(card: str) -> str:
    """Return suit character of card."""
    return card[1]


def parse_card(s: str) -> str:
    """Validate and normalize card string."""
    s = s.strip()
    assert len(s) == 2, f"Invalid card: {s}"
    assert s[0] in RANK_VALUES, f"Invalid rank: {s[0]}"
    assert s[1] in SUITS, f"Invalid suit: {s[1]}"
    return s


def cards_to_str(cards: List[str]) -> str:
    """Return human-readable card list string."""
    return ' '.join(cards)

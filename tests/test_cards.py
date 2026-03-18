"""
test_cards.py — Tests for src/cards.py
"""
import pytest
import numpy as np
from src.cards import make_deck, shuffle_deck, deal_cards, card_rank, card_suit, parse_card, RANKS, SUITS


class TestMakeDeck:
    def test_deck_has_52_cards(self):
        deck = make_deck()
        assert len(deck) == 52

    def test_deck_all_unique(self):
        deck = make_deck()
        assert len(set(deck)) == 52

    def test_deck_all_valid(self):
        deck = make_deck()
        for card in deck:
            assert len(card) == 2
            assert card[0] in RANKS
            assert card[1] in SUITS

    def test_deck_contains_known_cards(self):
        deck = make_deck()
        assert "Ah" in deck
        assert "2c" in deck
        assert "Ks" in deck
        assert "Td" in deck


class TestShuffleDeck:
    def test_shuffle_is_deterministic_under_seed(self):
        deck = make_deck()
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        shuffled1 = shuffle_deck(deck, rng1)
        shuffled2 = shuffle_deck(deck, rng2)
        assert shuffled1 == shuffled2

    def test_shuffle_different_seeds_different_order(self):
        deck = make_deck()
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(99)
        shuffled1 = shuffle_deck(deck, rng1)
        shuffled2 = shuffle_deck(deck, rng2)
        assert shuffled1 != shuffled2

    def test_shuffle_preserves_cards(self):
        deck = make_deck()
        rng = np.random.default_rng(42)
        shuffled = shuffle_deck(deck, rng)
        assert sorted(shuffled) == sorted(deck)

    def test_shuffle_does_not_modify_original(self):
        deck = make_deck()
        original = deck.copy()
        rng = np.random.default_rng(42)
        _ = shuffle_deck(deck, rng)
        assert deck == original


class TestDealCards:
    def test_deal_correct_count(self):
        deck = make_deck()
        dealt, remaining = deal_cards(deck, 5)
        assert len(dealt) == 5
        assert len(remaining) == 47

    def test_deal_no_duplicates(self):
        deck = make_deck()
        dealt, remaining = deal_cards(deck, 10)
        all_cards = dealt + remaining
        assert len(set(all_cards)) == len(all_cards)

    def test_deal_excludes_specified_cards(self):
        deck = make_deck()
        exclude = ["Ah", "Kd", "Qc"]
        dealt, remaining = deal_cards(deck, 5, exclude=exclude)
        assert "Ah" not in dealt
        assert "Kd" not in dealt
        assert "Qc" not in dealt

    def test_deal_excludes_not_in_remaining(self):
        deck = make_deck()
        exclude = ["Ah", "Kd"]
        dealt, remaining = deal_cards(deck, 5, exclude=exclude)
        assert "Ah" not in remaining
        assert "Kd" not in remaining

    def test_deal_empty_exclude(self):
        deck = make_deck()
        dealt, remaining = deal_cards(deck, 2, exclude=[])
        assert len(dealt) == 2

    def test_deal_none_exclude(self):
        deck = make_deck()
        dealt, remaining = deal_cards(deck, 2, exclude=None)
        assert len(dealt) == 2


class TestCardRank:
    def test_ace_is_highest(self):
        assert card_rank("Ah") == 12

    def test_two_is_lowest(self):
        assert card_rank("2c") == 0

    def test_ten_rank(self):
        assert card_rank("Td") == 8

    def test_rank_ordering(self):
        assert card_rank("Kh") > card_rank("Qh")
        assert card_rank("Qh") > card_rank("Jh")
        assert card_rank("3s") > card_rank("2s")


class TestCardSuit:
    def test_heart_suit(self):
        assert card_suit("Ah") == "h"

    def test_spade_suit(self):
        assert card_suit("Ks") == "s"

    def test_diamond_suit(self):
        assert card_suit("Td") == "d"

    def test_club_suit(self):
        assert card_suit("2c") == "c"


class TestParseCard:
    def test_valid_card(self):
        assert parse_card("Ah") == "Ah"

    def test_valid_card_strips_whitespace(self):
        assert parse_card("  Ah  ") == "Ah"

    def test_invalid_rank_raises(self):
        with pytest.raises(AssertionError):
            parse_card("Xh")

    def test_invalid_suit_raises(self):
        with pytest.raises(AssertionError):
            parse_card("Ax")

    def test_too_long_raises(self):
        with pytest.raises(AssertionError):
            parse_card("Ahd")

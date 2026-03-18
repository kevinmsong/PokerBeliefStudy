"""
test_hand_eval.py — Tests for src/hand_eval.py
"""
import pytest
from src.hand_eval import evaluate_hand, compare_hands, hand_rank_name


class TestHandRankings:
    def test_straight_flush(self):
        cards = ["Ah", "Kh", "Qh", "Jh", "Th", "2c", "3d"]
        score = evaluate_hand(cards)
        assert score[0] == 9, f"Expected Straight Flush (9), got {score}"

    def test_four_of_a_kind(self):
        cards = ["Ah", "As", "Ad", "Ac", "Kh", "2c", "3d"]
        score = evaluate_hand(cards)
        assert score[0] == 8, f"Expected Four of a Kind (8), got {score}"

    def test_full_house(self):
        cards = ["Ah", "As", "Ad", "Kh", "Ks", "2c", "3d"]
        score = evaluate_hand(cards)
        assert score[0] == 7, f"Expected Full House (7), got {score}"

    def test_flush(self):
        cards = ["Ah", "9h", "7h", "5h", "2h", "Kc", "3d"]
        score = evaluate_hand(cards)
        assert score[0] == 6, f"Expected Flush (6), got {score}"

    def test_straight(self):
        cards = ["Ah", "Ks", "Qd", "Jc", "Th", "2c", "3d"]
        score = evaluate_hand(cards)
        assert score[0] == 5, f"Expected Straight (5), got {score}"

    def test_three_of_a_kind(self):
        cards = ["Ah", "As", "Ad", "Kh", "Qc", "2d", "3s"]
        score = evaluate_hand(cards)
        assert score[0] == 4, f"Expected Three of a Kind (4), got {score}"

    def test_two_pair(self):
        cards = ["Ah", "As", "Kd", "Ks", "Qh", "2c", "3d"]
        score = evaluate_hand(cards)
        assert score[0] == 3, f"Expected Two Pair (3), got {score}"

    def test_one_pair(self):
        cards = ["Ah", "As", "Kd", "Qh", "Jc", "2c", "3d"]
        score = evaluate_hand(cards)
        assert score[0] == 2, f"Expected One Pair (2), got {score}"

    def test_high_card(self):
        cards = ["Ah", "Ks", "Qd", "Jc", "9h", "7c", "2d"]
        score = evaluate_hand(cards)
        assert score[0] == 1, f"Expected High Card (1), got {score}"

    def test_wheel_straight(self):
        """A-2-3-4-5 is a valid straight (five-high)."""
        cards = ["Ah", "2s", "3d", "4c", "5h", "Kd", "Qc"]
        score = evaluate_hand(cards)
        assert score[0] == 5, f"Expected Straight (5), got {score}"
        assert score[1] == 3, f"Expected 5-high straight (rank 3), got {score[1]}"

    def test_rankings_ordered(self):
        """Higher hand types must have higher scores."""
        sf = evaluate_hand(["Ah", "Kh", "Qh", "Jh", "Th", "2c", "3d"])
        quads = evaluate_hand(["Ah", "As", "Ad", "Ac", "Kh", "2c", "3d"])
        fh = evaluate_hand(["Ah", "As", "Ad", "Kh", "Ks", "2c", "3d"])
        flush = evaluate_hand(["Ah", "9h", "7h", "5h", "2h", "Kc", "3d"])
        straight = evaluate_hand(["Ah", "Ks", "Qd", "Jc", "Th", "2c", "3d"])
        trips = evaluate_hand(["Ah", "As", "Ad", "Kh", "Qc", "2d", "3s"])
        two_pair = evaluate_hand(["Ah", "As", "Kd", "Ks", "Qh", "2c", "3d"])
        one_pair = evaluate_hand(["Ah", "As", "Kd", "Qh", "Jc", "2c", "3d"])
        high_card = evaluate_hand(["Ah", "Ks", "Qd", "Jc", "9h", "7c", "2d"])

        assert sf > quads > fh > flush > straight > trips > two_pair > one_pair > high_card


class TestCompareHands:
    def test_better_hand_wins(self):
        hand_a = ["Ah", "As", "Ad", "Ac", "Kh", "2c", "3d"]  # Quads
        hand_b = ["Ah", "As", "Ad", "Kh", "Ks", "2c", "3d"]  # Full House
        assert compare_hands(hand_a, hand_b) == 1

    def test_worse_hand_loses(self):
        hand_a = ["Ah", "As", "Ad", "Kh", "Ks", "2c", "3d"]  # Full House
        hand_b = ["Ah", "As", "Ad", "Ac", "Kh", "2c", "3d"]  # Quads
        assert compare_hands(hand_a, hand_b) == -1

    def test_tie_detection(self):
        # Same exact 7 cards => tie
        cards = ["Ah", "Ks", "Qd", "Jc", "9h", "7c", "2d"]
        assert compare_hands(cards, cards) == 0

    def test_kicker_matters(self):
        hand_a = ["Ah", "As", "Kd", "Qh", "Jc", "2c", "3d"]  # AA with K kicker
        hand_b = ["Ah", "As", "Qd", "Jh", "9c", "2c", "3d"]  # AA with Q kicker
        assert compare_hands(hand_a, hand_b) == 1

    def test_5_card_hands(self):
        hand_a = ["Ah", "As", "Ad", "Ac", "Kh"]  # Quads
        hand_b = ["Ah", "As", "Ad", "Kh", "Ks"]  # Full House
        assert compare_hands(hand_a, hand_b) == 1


class TestHandRankName:
    def test_straight_flush_name(self):
        cards = ["Ah", "Kh", "Qh", "Jh", "Th", "2c", "3d"]
        score = evaluate_hand(cards)
        assert hand_rank_name(score) == "Straight Flush"

    def test_four_of_a_kind_name(self):
        cards = ["Ah", "As", "Ad", "Ac", "Kh", "2c", "3d"]
        score = evaluate_hand(cards)
        assert hand_rank_name(score) == "Four of a Kind"

    def test_high_card_name(self):
        cards = ["Ah", "Ks", "Qd", "Jc", "9h", "7c", "2d"]
        score = evaluate_hand(cards)
        assert hand_rank_name(score) == "High Card"


class TestSevenCardSelection:
    def test_finds_best_5_from_7(self):
        # With flush on board, player has one card to improve
        # Best hand from 7 cards should be correctly found
        cards = ["Ah", "Kh", "Qh", "Jh", "9h", "2c", "5d"]
        score = evaluate_hand(cards)
        # Best hand is Ace-high flush
        assert score[0] == 6

    def test_7_cards_vs_5_card_subset(self):
        """7-card evaluation should find at least as good a hand as any 5-card subset."""
        from itertools import combinations
        cards = ["Ah", "As", "Ad", "Kh", "Ks", "Qd", "Qc"]
        best_7 = evaluate_hand(cards)
        best_5 = max(evaluate_hand(list(combo)) for combo in combinations(cards, 5))
        assert best_7 == best_5

"""
test_state.py — Tests for src/state.py
"""
import pytest
from src.state import (
    make_initial_state, apply_action, get_legal_actions, is_terminal,
    PublicState, PlayerState
)


def make_test_state(pot=100, stack=200, street="turn", last_bet=0):
    return make_initial_state(
        board=["Ah", "Kd", "Qc", "2s"],
        hole_cards_0=("Jh", "Ts"),
        hole_cards_1=("9d", "8c"),
        effective_stack=stack + (pot // 2),
        starting_commitment_0=pot // 2,
        starting_commitment_1=pot // 2,
        first_to_act=0,
        street=street,
    )


class TestLegalActions:
    def test_no_bet_faced_actions(self):
        state = make_test_state(pot=100, stack=200)
        player = state.players[0]
        pub = state.public
        pub.last_bet = 0
        legal = get_legal_actions(pub, player)
        action_names = [a for a, _ in legal]
        assert "check" in action_names
        assert "bet_half_pot" in action_names
        assert "bet_pot" in action_names
        assert "jam" in action_names
        assert "fold" not in action_names
        assert "call" not in action_names

    def test_facing_bet_actions(self):
        state = make_test_state(pot=100, stack=200)
        player = state.players[0]
        pub = state.public
        pub.last_bet = 50
        legal = get_legal_actions(pub, player)
        action_names = [a for a, _ in legal]
        assert "fold" in action_names
        assert "call" in action_names
        assert "check" not in action_names

    def test_call_amount_capped_at_stack(self):
        state = make_test_state(pot=100, stack=30)
        player = state.players[0]
        pub = state.public
        pub.last_bet = 100  # More than stack
        legal = get_legal_actions(pub, player)
        call_actions = [(a, amt) for a, amt in legal if a == "call"]
        assert len(call_actions) == 1
        assert call_actions[0][1] == 30  # Capped at stack

    def test_all_in_player_no_actions(self):
        state = make_test_state(pot=100, stack=0)
        player = state.players[0]
        player.stack = 0
        player.all_in = True
        pub = state.public
        legal = get_legal_actions(pub, player)
        assert legal == []

    def test_bet_amounts_correct(self):
        state = make_test_state(pot=100, stack=200)
        player = state.players[0]
        pub = state.public
        pub.last_bet = 0
        legal = get_legal_actions(pub, player)
        legal_dict = dict(legal)
        assert legal_dict["bet_half_pot"] == 50  # pot // 2 = 50
        assert legal_dict["bet_pot"] == 100       # pot = 100


class TestPotConservation:
    def test_pot_increases_on_bet(self):
        state = make_test_state(pot=100, stack=200)
        initial_pot = state.public.pot
        new_state = apply_action(state, "bet_half_pot", 50)
        assert new_state.public.pot == initial_pot + 50

    def test_stack_decreases_on_bet(self):
        state = make_test_state(pot=100, stack=200)
        initial_stack = state.players[0].stack
        new_state = apply_action(state, "bet_half_pot", 50)
        assert new_state.players[0].stack == initial_stack - 50

    def test_total_chips_conserved_on_bet(self):
        state = make_test_state(pot=100, stack=200)
        total_before = state.public.pot + sum(p.stack for p in state.players)
        new_state = apply_action(state, "bet_half_pot", 50)
        total_after = new_state.public.pot + sum(p.stack for p in new_state.players)
        assert total_before == total_after

    def test_total_chips_conserved_on_call(self):
        state = make_test_state(pot=100, stack=200)
        state.public.last_bet = 50
        state.public.to_act = 1  # Player 1 calls
        total_before = state.public.pot + sum(p.stack for p in state.players)
        new_state = apply_action(state, "call", 50)
        total_after = new_state.public.pot + sum(p.stack for p in new_state.players)
        assert total_before == total_after

    def test_total_chips_conserved_on_jam(self):
        state = make_test_state(pot=100, stack=200)
        total_before = state.public.pot + sum(p.stack for p in state.players)
        new_state = apply_action(state, "jam", 200)
        total_after = new_state.public.pot + sum(p.stack for p in new_state.players)
        assert total_before == total_after


class TestStackConservation:
    def test_fold_does_not_change_stack(self):
        state = make_test_state(pot=100, stack=200)
        state.public.last_bet = 50
        initial_stack = state.players[0].stack
        new_state = apply_action(state, "fold", 0)
        # Stack unchanged when folding (already committed)
        assert new_state.players[0].stack == initial_stack

    def test_check_does_not_change_stack(self):
        state = make_test_state(pot=100, stack=200)
        initial_stack = state.players[0].stack
        new_state = apply_action(state, "check", 0)
        assert new_state.players[0].stack == initial_stack


class TestTerminalDetection:
    def test_fold_is_terminal(self):
        state = make_test_state(pot=100, stack=200)
        state.public.last_bet = 50
        new_state = apply_action(state, "fold", 0)
        assert is_terminal(new_state)

    def test_check_check_advances_street(self):
        state = make_test_state(pot=100, stack=200, street="turn")
        state.public.last_bet = 0
        # Player 0 checks
        state = apply_action(state, "check", 0)
        assert not is_terminal(state) or state.public.street == "river"

    def test_initial_state_not_terminal(self):
        state = make_test_state()
        assert not is_terminal(state)

    def test_fold_winner_correct(self):
        state = make_test_state(pot=100, stack=200)
        state.public.last_bet = 50
        state.public.to_act = 0  # Player 0 folds
        new_state = apply_action(state, "fold", 0)
        assert is_terminal(new_state)
        assert new_state.winner == 1  # Opponent wins

    def test_river_call_is_terminal(self):
        state = make_test_state(pot=100, stack=200, street="river")
        state.public.last_bet = 50
        new_state = apply_action(state, "call", 50)
        assert is_terminal(new_state)

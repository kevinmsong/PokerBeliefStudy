"""
state.py — Full game state implementation.

Handles:
- pot conservation (total chips constant)
- stack conservation
- legal action generation (context-sensitive)
- terminal detection
- street transitions
- action application
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any


Card = str  # e.g. "Ah", "2s", "Td"


@dataclass
class PlayerState:
    stack: int
    committed: int = 0
    folded: bool = False
    all_in: bool = False


@dataclass
class PublicState:
    board: List[Card]
    pot: int
    street: str              # "turn" or "river"
    to_act: int
    last_bet: int
    history: List[Tuple[int, str, int]] = field(default_factory=list)


@dataclass
class PrivateState:
    hole_cards: Tuple[Card, Card]


@dataclass
class FullState:
    public: PublicState
    players: List[PlayerState]
    private_states: List[PrivateState]
    terminal: bool = False
    winner: Optional[int] = None


def get_legal_actions(public_state: PublicState, player_state: PlayerState) -> List[Tuple[str, int]]:
    """Return list of legal (action_name, amount) tuples for the current player.

    When NO bet faced: check, bet_half_pot, bet_pot, jam
    When FACING bet: fold, call, jam
    """
    pot = public_state.pot
    last_bet = public_state.last_bet
    stack = player_state.stack

    if stack == 0:
        # Player is all-in, no actions available
        return []

    if last_bet > 0:
        # Facing a bet
        call_amount = min(last_bet, stack)
        actions = [
            ("fold", 0),
            ("call", call_amount),
        ]
        if stack > call_amount:
            actions.append(("jam", stack))
        return actions
    else:
        # No bet faced
        half_pot = max(1, pot // 2)
        full_pot = max(1, pot)
        actions = [("check", 0)]
        if stack >= half_pot:
            actions.append(("bet_half_pot", half_pot))
        if stack >= full_pot and full_pot != half_pot:
            actions.append(("bet_pot", full_pot))
        if stack > 0:
            # Jam if different from bet_pot
            jam_amount = stack
            if jam_amount not in [half_pot, full_pot]:
                actions.append(("jam", jam_amount))
            elif jam_amount == full_pot:
                # Replace bet_pot with jam (they're the same)
                pass
            elif jam_amount == half_pot:
                pass
        return actions


def apply_action(state: FullState, action: str, amount: int) -> FullState:
    """Apply an action to the state and return new state.

    This performs a deep copy-like operation to maintain immutability.
    """
    import copy
    new_state = copy.deepcopy(state)
    pub = new_state.public
    actor = pub.to_act
    player = new_state.players[actor]

    if action == "fold":
        player.folded = True
        # Other player wins
        winner = 1 - actor
        new_state.terminal = True
        new_state.winner = winner
        pub.history.append((actor, action, amount))
        return new_state

    if action == "check":
        pub.history.append((actor, action, amount))
        # Check if both players have acted (or street over)
        if _street_done_after_check(pub, actor):
            return _advance_or_end(new_state)
        pub.to_act = 1 - actor
        return new_state

    if action in ("bet_half_pot", "bet_pot", "jam"):
        # Bet/raise
        bet_size = amount
        bet_size = min(bet_size, player.stack)
        player.stack -= bet_size
        player.committed += bet_size
        pub.pot += bet_size
        pub.last_bet = bet_size
        if player.stack == 0:
            player.all_in = True
        pub.history.append((actor, action, bet_size))
        pub.to_act = 1 - actor
        return new_state

    if action == "call":
        call_amount = min(amount, player.stack)
        player.stack -= call_amount
        player.committed += call_amount
        pub.pot += call_amount
        if player.stack == 0:
            player.all_in = True
        pub.history.append((actor, action, call_amount))
        # After call, street is over (in our 2-player model)
        return _advance_or_end(new_state)

    raise ValueError(f"Unknown action: {action}")


def _street_done_after_check(pub: PublicState, checker: int) -> bool:
    """Determine if the street is done after a check.

    The street is done if the other player also checked (or hasn't acted yet
    and we're in a heads-up check-check situation).
    In our model: street is done if the checker was not the first to act
    (i.e., both players have had a chance to act).
    """
    # Count checks in current street
    checks_this_street = sum(
        1 for pid, act, amt in pub.history
        if act == "check"
    )
    # If we just checked and there was already a check (from opponent), street over
    # Or if checker is the second to act (to_act = 1 when player 0 opened)
    # Simple rule: if the last action in history from the OTHER player was check, done
    other = 1 - checker
    other_actions = [(pid, act, amt) for pid, act, amt in pub.history if pid == other]
    if other_actions:
        last_other_action = other_actions[-1][1]
        if last_other_action == "check":
            return True
    return False


def _advance_or_end(state: FullState) -> FullState:
    """Advance to next street or end the hand."""
    pub = state.public

    if pub.street == "turn":
        # Advance to river
        pub.street = "river"
        pub.last_bet = 0
        pub.to_act = 0  # Player 0 acts first post-flop
        # River card will be dealt by simulate.py
        return state
    else:
        # River is done -> showdown
        state.terminal = True
        state.winner = None  # Showdown, determined externally
        return state


def is_terminal(state: FullState) -> bool:
    """Check if the state is terminal."""
    return state.terminal


def get_reward(state: FullState, player_idx: int) -> int:
    """Return the reward for player_idx in a terminal state.

    This returns the net chip gain (positive = won, negative = lost).
    Actual payoff is computed in simulate.py based on committed chips.
    """
    if not state.terminal:
        return 0
    if state.winner is not None:
        if state.winner == player_idx:
            return state.public.pot
        else:
            return 0
    # Showdown - computed externally
    return 0


def make_initial_state(
    board: List[Card],
    hole_cards_0: Tuple[Card, Card],
    hole_cards_1: Tuple[Card, Card],
    starting_pot: int,
    effective_stack: int,
    street: str = "turn",
) -> FullState:
    """Create the initial FullState for a hand."""
    pub = PublicState(
        board=list(board),
        pot=starting_pot,
        street=street,
        to_act=0,
        last_bet=0,
        history=[],
    )
    players = [
        PlayerState(stack=effective_stack, committed=0),
        PlayerState(stack=effective_stack, committed=0),
    ]
    private_states = [
        PrivateState(hole_cards=hole_cards_0),
        PrivateState(hole_cards=hole_cards_1),
    ]
    return FullState(
        public=pub,
        players=players,
        private_states=private_states,
        terminal=False,
        winner=None,
    )

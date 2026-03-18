"""
simulate.py — Run a single hand of poker.

Manages:
- card dealing
- action loop
- notifying agents of opponent actions
- terminal detection
- showdown
- hand record emission
"""
from typing import Dict, Any, Optional, Tuple, List
import copy
import math

import numpy as np

from src.cards import make_deck, shuffle_deck
from src.state import (
    FullState, PublicState, PlayerState, PrivateState,
    make_initial_state, apply_action, is_terminal, get_legal_actions
)
from src.infoset import extract_infostate
from src.hand_classes import classify_hand
from src.hand_eval import compare_hands, hand_rank_name, evaluate_hand
from src.beliefs import BeliefState
from src.agents.base import BaseAgent


def run_hand(
    agent0: BaseAgent,
    agent1: BaseAgent,
    rng: np.random.Generator,
    config: dict,
    opp_family_0: str = "balanced",
    opp_family_1: str = "balanced",
    hand_id: str = "",
    experiment_id: str = "",
    seed: int = 0,
    run_id: str = "",
) -> dict:
    """Run one complete hand of heads-up poker.

    Parameters
    ----------
    agent0, agent1 : BaseAgent
        The two agents. agent0 is player 0, agent1 is player 1.
    rng : np.random.Generator
        Numpy RNG for card dealing.
    config : dict
        Configuration dict (starting_pot, effective_stack, street_start).
    opp_family_0 : str
        Opponent family that agent1 belongs to (from agent0's perspective).
    opp_family_1 : str
        Opponent family that agent0 belongs to (from agent1's perspective).
    hand_id : str
        Unique identifier for this hand.
    experiment_id : str
        Experiment identifier.
    seed : int
        Seed used for this hand (for logging).
    run_id : str
        Run identifier.

    Returns
    -------
    dict
        Hand record with all logging fields.
    """
    agents = [agent0, agent1]
    starting_pot = config.get("starting_pot", 100)
    effective_stack = config.get("effective_stack", 200)
    street_start = config.get("street_start", "turn")

    # Reset agent state for new hand
    for agent in agents:
        agent.new_hand()

    # Deal cards
    deck = make_deck()
    deck = shuffle_deck(deck, rng)

    # Deal hole cards
    hole_cards_0 = (deck[0], deck[1])
    hole_cards_1 = (deck[2], deck[3])
    remaining = deck[4:]

    # Deal board cards
    if street_start == "turn":
        board_initial = remaining[:4]
        river_card = remaining[4]
        remaining = remaining[5:]
    else:
        # River - 5 cards already on board
        board_initial = remaining[:5]
        river_card = None
        remaining = remaining[5:]

    # Create initial state
    state = make_initial_state(
        board=board_initial,
        hole_cards_0=hole_cards_0,
        hole_cards_1=hole_cards_1,
        starting_pot=starting_pot,
        effective_stack=effective_stack,
        street=street_start,
    )

    action_history_log = []
    decision_index = 0

    # --- Turn action loop ---
    state = _run_street(
        state, agents, opp_family_0, opp_family_1,
        action_history_log, decision_index, "turn"
    )
    decision_index = len(action_history_log)

    # Check if hand ended on turn
    if not is_terminal(state):
        # Advance to river
        if river_card is not None:
            state.public.board.append(river_card)
        state.public.street = "river"
        state.public.last_bet = 0
        state.public.to_act = 0

        # --- River action loop ---
        state = _run_street(
            state, agents, opp_family_0, opp_family_1,
            action_history_log, decision_index, "river"
        )

    # --- Resolve terminal state ---
    terminal_reward_0, terminal_reward_1, showdown_winner = _resolve_terminal(
        state, hole_cards_0, hole_cards_1
    )

    realized_class_0 = classify_hand(hole_cards_0, state.public.board)
    realized_class_1 = classify_hand(hole_cards_1, state.public.board)

    hand_record = {
        "run_id": run_id,
        "seed": seed,
        "experiment_id": experiment_id,
        "matchup_id": f"{agent0.name}_vs_{agent1.name}",
        "opponent_family_0": opp_family_0,
        "opponent_family_1": opp_family_1,
        "hand_id": hand_id,
        "board": list(state.public.board),
        "hole_cards_0": list(hole_cards_0),
        "hole_cards_1": list(hole_cards_1),
        "action_history": action_history_log,
        "terminal_reward_0": terminal_reward_0,
        "terminal_reward_1": terminal_reward_1,
        "showdown_winner": showdown_winner,
        "realized_hand_class_0": realized_class_0,
        "realized_hand_class_1": realized_class_1,
        "final_pot": state.public.pot,
    }

    return hand_record


def _run_street(
    state: FullState,
    agents: List[BaseAgent],
    opp_family_0: str,
    opp_family_1: str,
    action_history_log: List[dict],
    decision_index: int,
    street: str,
) -> FullState:
    """Run the action loop for a single street.

    Returns the state after the street is complete.
    """
    opp_families = [opp_family_0, opp_family_1]
    max_actions = 20  # Safety limit to prevent infinite loops

    actions_this_street = 0
    while not is_terminal(state) and state.public.street == street:
        if actions_this_street >= max_actions:
            # Force check/fold to end infinite loop
            state.terminal = True
            break

        actor = state.public.to_act
        agent = agents[actor]

        infostate = extract_infostate(state, actor)

        if not infostate.legal_actions:
            # No legal actions (e.g., all-in) - advance
            break

        # Agent decides
        action, amount = agent.act(infostate)

        # Get EV table and belief for logging (before applying action)
        ev_table = agent.get_ev_table()
        belief_state = agent.get_belief_state()
        entropy_val = None
        prior_log = None
        posterior_log = None

        # Get prior/posterior from belief agent
        if hasattr(agent, 'get_prior'):
            prior_log = agent.get_prior()
        if hasattr(agent, 'get_posterior'):
            posterior_log = agent.get_posterior()
        if hasattr(agent, 'get_entropy'):
            entropy_val = agent.get_entropy()

        # Log the decision
        action_log_entry = {
            "street": street,
            "decision_index": decision_index + actions_this_street,
            "player": actor,
            "action": action,
            "size": amount,
            "legal_actions": list(infostate.legal_actions),
            "ev_table": ev_table,
            "prior": prior_log or belief_state,
            "posterior": posterior_log,
            "posterior_entropy": entropy_val,
        }
        action_history_log.append(action_log_entry)

        # Notify opponent of action
        opponent_idx = 1 - actor
        opp_agent = agents[opponent_idx]
        opp_agent.observe_opponent_action(action, amount, state.public)

        # Apply action
        state = apply_action(state, action, amount)
        actions_this_street += 1

    return state


def _resolve_terminal(
    state: FullState,
    hole_cards_0: Tuple[str, str],
    hole_cards_1: Tuple[str, str],
) -> Tuple[int, int, Optional[int]]:
    """Resolve the terminal state and return (reward_0, reward_1, showdown_winner).

    Rewards are relative chip gains/losses.
    """
    pot = state.public.pot
    board = state.public.board

    if state.winner is not None:
        # One player folded
        winner = state.winner
        loser = 1 - winner
        if winner == 0:
            return pot, 0, winner
        else:
            return 0, pot, winner

    # Showdown
    result = compare_hands(
        list(hole_cards_0) + board,
        list(hole_cards_1) + board,
    )
    if result == 1:
        return pot, 0, 0
    elif result == -1:
        return 0, pot, 1
    else:
        # Tie - split pot
        half = pot // 2
        return half, pot - half, -1  # -1 = tie

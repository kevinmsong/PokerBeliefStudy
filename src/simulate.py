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
from typing import List, Optional, Tuple

import numpy as np

from src.agents.base import BaseAgent
from src.cards import make_deck, shuffle_deck
from src.hand_classes import classify_hand
from src.hand_eval import compare_hands
from src.infoset import extract_infostate
from src.state import FullState, apply_action, is_terminal, make_initial_state
from src.utils import chips_won_lost


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
    hand_num: int = 0,
) -> dict:
    """Run one complete hand of heads-up poker."""
    agents = [agent0, agent1]
    effective_stack = config.get("effective_stack", 200)
    street_start = config.get("street_start", "turn")
    starting_commitment_0 = config.get("starting_commitment_0", config.get("starting_pot", 100) // 2)
    starting_commitment_1 = config.get("starting_commitment_1", config.get("starting_pot", 100) // 2)
    first_to_act = hand_num % 2
    initial_stack_0 = max(0, effective_stack - starting_commitment_0)
    initial_stack_1 = max(0, effective_stack - starting_commitment_1)

    for agent in agents:
        agent.new_hand()

    deck = shuffle_deck(make_deck(), rng)

    hole_cards_0 = (deck[0], deck[1])
    hole_cards_1 = (deck[2], deck[3])
    remaining = deck[4:]

    if street_start == "turn":
        board_initial = remaining[:4]
        river_card = remaining[4]
    else:
        board_initial = remaining[:5]
        river_card = None

    state = make_initial_state(
        board=board_initial,
        hole_cards_0=hole_cards_0,
        hole_cards_1=hole_cards_1,
        effective_stack=effective_stack,
        starting_commitment_0=starting_commitment_0,
        starting_commitment_1=starting_commitment_1,
        first_to_act=first_to_act,
        street=street_start,
    )

    action_history_log = []
    decision_index = 0

    state = _run_street(
        state=state,
        agents=agents,
        action_history_log=action_history_log,
        decision_index=decision_index,
        street="turn",
    )
    decision_index = len(action_history_log)

    if not is_terminal(state):
        if river_card is not None and len(state.public.board) == 4:
            state.public.board.append(river_card)
        state.public.street = "river"
        state.public.last_bet = 0
        state.public.to_act = state.public.first_to_act

        state = _run_street(
            state=state,
            agents=agents,
            action_history_log=action_history_log,
            decision_index=decision_index,
            street="river",
        )

    final_stack_0, final_stack_1, showdown_winner = _resolve_terminal(
        state,
        hole_cards_0,
        hole_cards_1,
    )
    terminal_reward_0 = chips_won_lost(initial_stack_0, final_stack_0)
    terminal_reward_1 = chips_won_lost(initial_stack_1, final_stack_1)

    hand_record = {
        "run_id": run_id,
        "seed": seed,
        "experiment_id": experiment_id,
        "matchup_id": f"{agent0.name}_vs_{agent1.name}",
        "opponent_family_0": opp_family_0,
        "opponent_family_1": opp_family_1,
        "hand_id": hand_id,
        "hand_num": hand_num,
        "first_to_act": first_to_act,
        "starting_commitment_0": starting_commitment_0,
        "starting_commitment_1": starting_commitment_1,
        "board": list(state.public.board),
        "hole_cards_0": list(hole_cards_0),
        "hole_cards_1": list(hole_cards_1),
        "action_history": action_history_log,
        "terminal_reward_0": terminal_reward_0,
        "terminal_reward_1": terminal_reward_1,
        "final_stack_0": final_stack_0,
        "final_stack_1": final_stack_1,
        "showdown_winner": showdown_winner,
        "realized_hand_class_0": classify_hand(hole_cards_0, state.public.board),
        "realized_hand_class_1": classify_hand(hole_cards_1, state.public.board),
        "final_pot": state.public.pot,
    }
    return hand_record


def _run_street(
    state: FullState,
    agents: List[BaseAgent],
    action_history_log: List[dict],
    decision_index: int,
    street: str,
) -> FullState:
    """Run the action loop for a single street."""
    max_actions = 20
    actions_this_street = 0

    while not is_terminal(state) and state.public.street == street:
        if actions_this_street >= max_actions:
            state.terminal = True
            break

        actor = state.public.to_act
        agent = agents[actor]
        infostate = extract_infostate(state, actor)

        if not infostate.legal_actions:
            break

        action, amount = agent.act(infostate)
        ev_table = agent.get_ev_table()
        belief_state = agent.get_belief_state()
        prior_log = getattr(agent, "get_prior", lambda: None)()
        posterior_log = getattr(agent, "get_posterior", lambda: None)()
        entropy_val = getattr(agent, "get_entropy", lambda: None)()

        action_history_log.append(
            {
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
        )

        agents[1 - actor].observe_opponent_action(action, amount, state.public)
        state = apply_action(state, action, amount)
        actions_this_street += 1

    return state


def _resolve_terminal(
    state: FullState,
    hole_cards_0: Tuple[str, str],
    hole_cards_1: Tuple[str, str],
) -> Tuple[int, int, Optional[int]]:
    """Resolve the terminal state and return final stacks plus showdown winner."""
    pot = state.public.pot
    board = state.public.board

    if state.winner is not None:
        if state.winner == 0:
            return state.players[0].stack + pot, state.players[1].stack, 0
        return state.players[0].stack, state.players[1].stack + pot, 1

    result = compare_hands(
        list(hole_cards_0) + board,
        list(hole_cards_1) + board,
    )
    if result == 1:
        return state.players[0].stack + pot, state.players[1].stack, 0
    if result == -1:
        return state.players[0].stack, state.players[1].stack + pot, 1

    half = pot // 2
    return state.players[0].stack + half, state.players[1].stack + (pot - half), -1

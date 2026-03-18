"""
infoset.py — Convert FullState + player_index -> InfoState.

Ensures NO hidden state leakage (opponent hole cards not included).
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any

from src.state import FullState, PublicState, get_legal_actions

Card = str


@dataclass
class InfoState:
    hole_cards: Tuple[Card, Card]
    board: List[Card]
    pot: int
    street: str
    stack_self: int
    stack_opp: int
    to_act: int
    last_bet: int
    history: List[Tuple[int, str, int]]
    legal_actions: List[Tuple[str, int]]


def extract_infostate(state: FullState, player_idx: int) -> InfoState:
    """Extract InfoState for player_idx from FullState.

    No hidden state (opponent hole cards) is included.
    """
    pub = state.public
    player = state.players[player_idx]
    opp_idx = 1 - player_idx
    opp_player = state.players[opp_idx]

    legal_actions = get_legal_actions(pub, player)

    return InfoState(
        hole_cards=state.private_states[player_idx].hole_cards,
        board=list(pub.board),
        pot=pub.pot,
        street=pub.street,
        stack_self=player.stack,
        stack_opp=opp_player.stack,
        to_act=pub.to_act,
        last_bet=pub.last_bet,
        history=list(pub.history),
        legal_actions=legal_actions,
    )

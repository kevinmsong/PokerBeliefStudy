"""
heuristic.py — Fixed-rule heuristic poker agent.

Uses hand strength buckets from hand_classes.py to make decisions.

Rules:
- nuts_or_near_nuts: bet_pot / jam when aggressor, call/raise when facing bet
- strong_made: bet_half_pot / call
- medium_made: check-call pot-odds based
- weak_showdown: check-fold unless excellent odds
- strong_draw: semibluff based on pot odds
- weak_draw: call only with direct pot odds
- air: check-fold, occasional bluff (20% freq)
"""
from typing import Tuple, Optional, Dict, List

import numpy as np

from src.agents.base import BaseAgent
from src.infoset import InfoState
from src.hand_classes import classify_hand
from src.state import PublicState
from src.equity import compute_pot_odds


class HeuristicAgent(BaseAgent):
    """Fixed-rule heuristic agent based on hand class."""

    BLUFF_FREQ = 0.20  # 20% bluff frequency with air

    def __init__(self, player_idx: int, rng: np.random.Generator, name: str = "HeuristicAgent"):
        super().__init__(name=name, player_idx=player_idx, rng=rng)

    def act(self, infostate: InfoState) -> Tuple[str, int]:
        """Choose action based on hand class heuristic rules."""
        hand_class = classify_hand(infostate.hole_cards, infostate.board)
        legal = infostate.legal_actions
        facing_bet = infostate.last_bet > 0

        action = self._select_action(hand_class, legal, facing_bet, infostate)
        return action

    def _select_action(
        self,
        hand_class: str,
        legal: List[Tuple[str, int]],
        facing_bet: bool,
        infostate: InfoState,
    ) -> Tuple[str, int]:
        """Select action based on hand class and situation."""
        legal_dict = {a: amt for a, amt in legal}

        if facing_bet:
            return self._act_facing_bet(hand_class, legal_dict, infostate)
        else:
            return self._act_no_bet(hand_class, legal_dict, infostate)

    def _act_facing_bet(
        self,
        hand_class: str,
        legal_dict: Dict[str, int],
        infostate: InfoState,
    ) -> Tuple[str, int]:
        """Choose action when facing a bet."""
        call_amt = legal_dict.get("call", 0)
        pot = infostate.pot
        pot_odds = compute_pot_odds(call_amt, pot)

        if hand_class == 'nuts_or_near_nuts':
            # Raise or call (trap occasionally)
            if "jam" in legal_dict:
                return ("jam", legal_dict["jam"])
            return ("call", call_amt)

        elif hand_class == 'strong_made':
            # Call or raise
            if "jam" in legal_dict and self.rng.random() < 0.3:
                return ("jam", legal_dict["jam"])
            return ("call", call_amt)

        elif hand_class == 'medium_made':
            # Call if pot odds justify
            if pot_odds < 0.45:  # Need <45% equity from pot odds = call if good price
                return ("call", call_amt)
            elif "call" in legal_dict:
                return ("call", call_amt)
            return ("fold", 0)

        elif hand_class == 'weak_showdown':
            # Fold unless great odds
            if pot_odds < 0.25:
                return ("call", call_amt)
            return ("fold", 0)

        elif hand_class == 'strong_draw':
            # Semibluff or call
            equity_approx = 0.40  # Approximate equity for strong draw
            if equity_approx >= pot_odds:
                if "jam" in legal_dict and self.rng.random() < 0.35:
                    return ("jam", legal_dict["jam"])
                return ("call", call_amt)
            return ("fold", 0)

        elif hand_class == 'weak_draw':
            # Call only with direct pot odds
            equity_approx = 0.20  # Approximate equity for weak draw
            if equity_approx >= pot_odds:
                return ("call", call_amt)
            return ("fold", 0)

        elif hand_class == 'air':
            # Fold usually, occasional bluff-raise
            if self.rng.random() < self.BLUFF_FREQ * 0.3 and "jam" in legal_dict:
                return ("jam", legal_dict["jam"])
            return ("fold", 0)

        # Fallback
        return ("fold", 0)

    def _act_no_bet(
        self,
        hand_class: str,
        legal_dict: Dict[str, int],
        infostate: InfoState,
    ) -> Tuple[str, int]:
        """Choose action when no bet is facing."""
        if hand_class == 'nuts_or_near_nuts':
            # Bet pot or jam for value
            if "bet_pot" in legal_dict:
                return ("bet_pot", legal_dict["bet_pot"])
            if "jam" in legal_dict:
                return ("jam", legal_dict["jam"])
            return ("check", 0)

        elif hand_class == 'strong_made':
            # Bet half pot for value
            if "bet_half_pot" in legal_dict:
                return ("bet_half_pot", legal_dict["bet_half_pot"])
            return ("check", 0)

        elif hand_class == 'medium_made':
            # Check or bet small
            if self.rng.random() < 0.4 and "bet_half_pot" in legal_dict:
                return ("bet_half_pot", legal_dict["bet_half_pot"])
            return ("check", 0)

        elif hand_class == 'weak_showdown':
            # Mostly check
            return ("check", 0)

        elif hand_class == 'strong_draw':
            # Semibluff occasionally
            if self.rng.random() < 0.45:
                if "bet_half_pot" in legal_dict:
                    return ("bet_half_pot", legal_dict["bet_half_pot"])
            return ("check", 0)

        elif hand_class == 'weak_draw':
            # Check usually
            if self.rng.random() < 0.15 and "bet_half_pot" in legal_dict:
                return ("bet_half_pot", legal_dict["bet_half_pot"])
            return ("check", 0)

        elif hand_class == 'air':
            # Check-fold strategy, bluff occasionally
            if self.rng.random() < self.BLUFF_FREQ:
                if "bet_half_pot" in legal_dict:
                    return ("bet_half_pot", legal_dict["bet_half_pot"])
            return ("check", 0)

        # Fallback
        return ("check", 0)

    def new_hand(self) -> None:
        """Reset per-hand state (nothing to reset for heuristic agent)."""
        pass

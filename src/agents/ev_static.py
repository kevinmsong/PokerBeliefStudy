"""
ev_static.py — EV-based agent with FIXED prior (no belief updating).

Uses equity.py + response_model.py to compute expected value of actions
under a static (non-updated) prior belief about opponent hand class.
"""
from typing import Tuple, Dict, List, Optional

import numpy as np

from src.agents.base import BaseAgent
from src.infoset import InfoState
from src.hand_classes import classify_hand, HAND_CLASSES
from src.state import PublicState
from src.equity import estimate_equity, compute_pot_odds
from src.response_model import ResponseModel, make_default_response_model
from src.beliefs import BeliefState, make_uniform_belief


class StaticEVAgent(BaseAgent):
    """EV-maximizing agent using a static (non-updated) prior.

    Computes EV(action | state, prior) = Σ P(outcome | action, state, prior) * U(outcome)
    using Monte Carlo rollouts.
    """

    def __init__(
        self,
        player_idx: int,
        rng: np.random.Generator,
        opp_family: str = "balanced",
        n_rollouts: int = 200,
        prior: Optional[Dict[str, float]] = None,
        response_model: Optional[ResponseModel] = None,
        name: str = "StaticEVAgent",
    ):
        super().__init__(name=name, player_idx=player_idx, rng=rng)
        self.opp_family = opp_family
        self.n_rollouts = n_rollouts
        self.response_model = response_model or make_default_response_model()

        if prior is None:
            n = len(HAND_CLASSES)
            self.prior = {c: 1.0 / n for c in HAND_CLASSES}
        else:
            total = sum(prior.values())
            self.prior = {k: v / total for k, v in prior.items()}

        self._last_ev_table = {}

    def act(self, infostate: InfoState) -> Tuple[str, int]:
        """Choose action that maximizes EV under static prior."""
        legal = infostate.legal_actions
        if not legal:
            return ("check", 0)

        ev_table = self._compute_ev_table(infostate)
        self._last_ev_table = ev_table

        # Pick action with highest EV
        best_action = max(ev_table, key=ev_table.get)
        best_amount = {a: amt for a, amt in legal}[best_action]
        return (best_action, best_amount)

    def _compute_ev_table(self, infostate: InfoState) -> Dict[str, float]:
        """Compute EV for each legal action."""
        ev_table = {}
        for action, amount in infostate.legal_actions:
            ev = self._compute_action_ev(action, amount, infostate)
            ev_table[action] = ev
        return ev_table

    def _compute_action_ev(self, action: str, amount: int, infostate: InfoState) -> float:
        """Compute EV of a single action via Monte Carlo rollout."""
        pot = infostate.pot
        stack_self = infostate.stack_self

        if action == "fold":
            return 0.0  # Forfeit pot, relative EV = 0 (we gain nothing)

        if action == "check":
            # Equity in pot without investing more chips
            equity = estimate_equity(
                hero_cards=infostate.hole_cards,
                board=infostate.board,
                opp_belief=self.prior,
                rng=self.rng,
                n_samples=max(50, self.n_rollouts // 4),
            )
            return equity * pot

        if action in ("call",):
            # Pay amount, then equity in new pot
            equity = estimate_equity(
                hero_cards=infostate.hole_cards,
                board=infostate.board,
                opp_belief=self.prior,
                rng=self.rng,
                n_samples=max(50, self.n_rollouts // 4),
            )
            new_pot = pot + amount
            ev = equity * new_pot - amount
            return ev

        if action in ("bet_half_pot", "bet_pot", "jam"):
            # We bet `amount`; opponent responds based on prior + response model
            equity = estimate_equity(
                hero_cards=infostate.hole_cards,
                board=infostate.board,
                opp_belief=self.prior,
                rng=self.rng,
                n_samples=max(50, self.n_rollouts // 4),
            )

            # Simulate opponent response
            ev = self._compute_bet_ev(action, amount, equity, infostate)
            return ev

        return 0.0

    def _compute_bet_ev(
        self,
        action: str,
        bet_amount: int,
        hero_equity: float,
        infostate: InfoState,
    ) -> float:
        """Compute EV of a bet considering opponent's likely response.

        EV(bet) = P(fold) * (pot) + P(call) * equity * (pot + 2*bet) - bet
                + P(jam) * (complicated)
        """
        pot = infostate.pot
        stack_opp = infostate.stack_opp

        # Get average opponent response using prior-weighted response probs
        # Construct a synthetic PublicState for the response model
        synth_state = PublicState(
            board=infostate.board,
            pot=pot,
            street=infostate.street,
            to_act=1 - self.player_idx,
            last_bet=bet_amount,
            history=list(infostate.history),
        )

        # Average response probs over prior
        avg_fold = 0.0
        avg_call = 0.0
        avg_jam = 0.0

        facing_bet_legal = [("fold", 0), ("call", bet_amount), ("jam", stack_opp)]

        for hand_class, prob in self.prior.items():
            if prob <= 0:
                continue
            resp = self.response_model.action_probs(
                hand_class=hand_class,
                public_state=synth_state,
                family_name=self.opp_family,
                legal_actions=facing_bet_legal,
            )
            avg_fold += prob * resp.get("fold", 0)
            avg_call += prob * resp.get("call", 0)
            avg_jam += prob * resp.get("jam", 0)

        # EV calculation
        # If fold: we win the pot
        ev_if_fold = pot

        # If call: we go to showdown (simplified - use equity)
        new_pot_call = pot + 2 * bet_amount
        ev_if_call = hero_equity * new_pot_call - bet_amount

        # If jam: opponent shoves, we call or fold
        # Simplified: assume we always call the jam (we've already bet)
        jam_amount = stack_opp
        new_pot_jam = pot + bet_amount + jam_amount
        ev_if_jam = hero_equity * new_pot_jam - bet_amount

        # Total EV
        total_ev = avg_fold * ev_if_fold + avg_call * ev_if_call + avg_jam * ev_if_jam
        return total_ev

    def observe_opponent_action(self, action: str, amount: int, public_state: PublicState) -> None:
        """Static agent does not update beliefs."""
        pass

    def new_hand(self) -> None:
        """Reset per-hand state."""
        self._last_ev_table = {}

    def get_ev_table(self) -> Optional[Dict[str, float]]:
        """Return last computed EV table for logging."""
        return self._last_ev_table.copy() if self._last_ev_table else None

    def get_belief_state(self) -> Optional[Dict[str, float]]:
        """Return static prior as belief state."""
        return self.prior.copy()

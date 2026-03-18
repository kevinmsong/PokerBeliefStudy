"""
ev_belief.py — EV-based agent with BAYESIAN belief updating.

Uses equity.py + response_model.py + beliefs.py.
Updates posterior after each observed opponent action.
"""
from typing import Tuple, Dict, List, Optional
import math

import numpy as np

from src.agents.base import BaseAgent
from src.infoset import InfoState
from src.hand_classes import classify_hand, HAND_CLASSES
from src.state import PublicState
from src.equity import estimate_equity, compute_pot_odds
from src.response_model import ResponseModel, make_default_response_model
from src.beliefs import BeliefState, make_uniform_belief


class BeliefEVAgent(BaseAgent):
    """EV-maximizing agent using Bayesian posterior belief updating.

    Maintains a posterior over opponent hand classes and updates it
    after each observed opponent action.
    """

    def __init__(
        self,
        player_idx: int,
        rng: np.random.Generator,
        opp_family: str = "balanced",
        n_rollouts: int = 200,
        prior: Optional[Dict[str, float]] = None,
        response_model: Optional[ResponseModel] = None,
        smoothing: float = 0.05,
        name: str = "BeliefEVAgent",
    ):
        super().__init__(name=name, player_idx=player_idx, rng=rng)
        self.opp_family = opp_family
        self.n_rollouts = n_rollouts
        self.smoothing = smoothing
        self.response_model = response_model or make_default_response_model()

        if prior is None:
            n = len(HAND_CLASSES)
            raw_prior = {c: 1.0 / n for c in HAND_CLASSES}
        else:
            total = sum(prior.values())
            raw_prior = {k: v / total for k, v in prior.items()}

        self.belief = BeliefState(raw_prior)
        self._last_ev_table = {}
        self._prior_at_decision = {}
        self._posterior_at_decision = {}

    def act(self, infostate: InfoState) -> Tuple[str, int]:
        """Choose action that maximizes EV under current posterior belief."""
        legal = infostate.legal_actions
        if not legal:
            return ("check", 0)

        # Save current belief for logging
        self._prior_at_decision = self.belief.get_prior()
        self._posterior_at_decision = self.belief.get_posterior()

        ev_table = self._compute_ev_table(infostate)
        self._last_ev_table = ev_table

        # Pick action with highest EV
        best_action = max(ev_table, key=ev_table.get)
        best_amount = {a: amt for a, amt in legal}[best_action]
        return (best_action, best_amount)

    def _compute_ev_table(self, infostate: InfoState) -> Dict[str, float]:
        """Compute EV for each legal action, sharing a single equity estimate."""
        posterior = self.belief.get_posterior()
        smoothed = self._apply_smoothing(posterior)

        # Compute equity once for this decision point
        equity = estimate_equity(
            hero_cards=infostate.hole_cards,
            board=infostate.board,
            opp_belief=smoothed,
            rng=self.rng,
            n_samples=self.n_rollouts,
        )
        ev_table = {}
        for action, amount in infostate.legal_actions:
            ev = self._compute_action_ev(action, amount, infostate, equity, smoothed)
            ev_table[action] = ev
        return ev_table

    def _compute_action_ev(self, action: str, amount: int, infostate: InfoState,
                           equity: float, smoothed: Dict[str, float]) -> float:
        """Compute EV of a single action using pre-computed equity."""
        pot = infostate.pot

        if action == "fold":
            return 0.0

        if action == "check":
            return equity * pot

        if action == "call":
            new_pot = pot + amount
            return equity * new_pot - amount

        if action in ("bet_half_pot", "bet_pot", "jam"):
            return self._compute_bet_ev(action, amount, equity, infostate, smoothed)

        return 0.0

    def _apply_smoothing(self, posterior: Dict[str, float]) -> Dict[str, float]:
        """Apply Laplace smoothing to avoid zero probabilities."""
        n = len(HAND_CLASSES)
        smoothed = {}
        total = sum(posterior.values())
        for c in HAND_CLASSES:
            raw = posterior.get(c, 0.0)
            smoothed[c] = (raw + self.smoothing / n) / (total + self.smoothing)
        return smoothed

    def _compute_bet_ev(
        self,
        action: str,
        bet_amount: int,
        hero_equity: float,
        infostate: InfoState,
        smoothed_posterior: Dict[str, float],
    ) -> float:
        """Compute EV of a bet considering opponent's likely response."""
        pot = infostate.pot
        stack_opp = infostate.stack_opp

        # Construct a synthetic PublicState
        synth_state = PublicState(
            board=infostate.board,
            pot=pot,
            street=infostate.street,
            to_act=1 - self.player_idx,
            last_bet=bet_amount,
            history=list(infostate.history),
        )

        # Average response probs over posterior
        avg_fold = 0.0
        avg_call = 0.0
        avg_jam = 0.0

        facing_bet_legal = [("fold", 0), ("call", bet_amount), ("jam", stack_opp)]

        for hand_class in HAND_CLASSES:
            prob = smoothed_posterior.get(hand_class, 0.0)
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
        ev_if_fold = pot
        new_pot_call = pot + 2 * bet_amount
        ev_if_call = hero_equity * new_pot_call - bet_amount
        jam_amount = stack_opp
        new_pot_jam = pot + bet_amount + jam_amount
        ev_if_jam = hero_equity * new_pot_jam - bet_amount

        total_ev = avg_fold * ev_if_fold + avg_call * ev_if_call + avg_jam * ev_if_jam
        return total_ev

    def observe_opponent_action(self, action: str, amount: int, public_state: PublicState) -> None:
        """Update belief posterior after observing opponent action."""
        self.belief.update(
            action=action,
            public_state=public_state,
            response_model=self.response_model,
            opp_family=self.opp_family,
        )

    def new_hand(self) -> None:
        """Reset per-hand state: reset beliefs to prior."""
        self.belief.reset()
        self._last_ev_table = {}
        self._prior_at_decision = {}
        self._posterior_at_decision = {}

    def get_ev_table(self) -> Optional[Dict[str, float]]:
        """Return last computed EV table for logging."""
        return self._last_ev_table.copy() if self._last_ev_table else None

    def get_belief_state(self) -> Optional[Dict[str, float]]:
        """Return current posterior for logging."""
        return self.belief.get_posterior()

    def get_prior(self) -> Dict[str, float]:
        """Return prior at last decision for logging."""
        return self._prior_at_decision.copy()

    def get_posterior(self) -> Dict[str, float]:
        """Return posterior at last decision for logging."""
        return self._posterior_at_decision.copy()

    def get_entropy(self) -> float:
        """Return entropy of current posterior."""
        return self.belief.entropy()

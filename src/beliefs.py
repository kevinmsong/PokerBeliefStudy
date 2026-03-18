"""
beliefs.py — Bayesian belief state over opponent hand classes.

Update rule: P(c | a, s) ∝ P(a | c, s) * P(c)
"""
import math
from typing import Dict, Optional

from src.hand_classes import HAND_CLASSES
from src.state import PublicState


class BeliefState:
    """Maintains prior and posterior over opponent hand classes."""

    def __init__(self, prior: Dict[str, float]):
        """Initialize with a prior over hand classes.

        prior: dict mapping hand_class -> probability (must sum to ~1.0)
        """
        self.prior = self._normalize(prior.copy())
        self.posterior = self.prior.copy()

    @staticmethod
    def _normalize(dist: Dict[str, float]) -> Dict[str, float]:
        """Normalize a distribution to sum to 1.0."""
        total = sum(dist.values())
        if total <= 0:
            # Uniform fallback
            n = len(HAND_CLASSES)
            return {c: 1.0 / n for c in HAND_CLASSES}
        return {k: v / total for k, v in dist.items()}

    def update(self, action: str, public_state: PublicState,
               response_model, opp_family: str):
        """Bayesian update: P(c|a,s) ∝ P(a|c,s) * P(c)

        Parameters
        ----------
        action : str
            The action observed from the opponent.
        public_state : PublicState
            The current public game state.
        response_model : ResponseModel
            Model providing P(action | hand_class, state, family).
        opp_family : str
            The opponent's family name.
        """
        new_posterior = {}
        for hand_class in HAND_CLASSES:
            # Get action distribution for this hand class
            # We need to infer what legal actions were available when opponent acted
            # Use the action_probs method with a synthetic legal actions set
            from src.state import get_legal_actions, PlayerState
            # Approximate legal actions based on the action taken
            if action in ("fold", "call", "jam") and public_state.last_bet > 0:
                legal = [("fold", 0), ("call", public_state.last_bet), ("jam", 999)]
            elif action == "jam" and public_state.last_bet == 0:
                legal = [("check", 0), ("bet_half_pot", public_state.pot // 2),
                         ("bet_pot", public_state.pot), ("jam", 999)]
            elif action in ("check", "bet_half_pot", "bet_pot"):
                legal = [("check", 0), ("bet_half_pot", public_state.pot // 2),
                         ("bet_pot", public_state.pot), ("jam", 999)]
            else:
                legal = [("check", 0), ("bet_half_pot", public_state.pot // 2),
                         ("bet_pot", public_state.pot), ("jam", 999)]

            probs = response_model.action_probs(
                hand_class=hand_class,
                public_state=public_state,
                family_name=opp_family,
                legal_actions=legal,
            )
            likelihood = probs.get(action, 1e-6)
            prior_prob = self.posterior.get(hand_class, 0.0)
            new_posterior[hand_class] = likelihood * prior_prob

        self.posterior = self._normalize(new_posterior)

    def entropy(self) -> float:
        """Compute Shannon entropy of current posterior in nats."""
        h = 0.0
        for p in self.posterior.values():
            if p > 0:
                h -= p * math.log(p)
        return h

    def reset(self):
        """Reset posterior to prior."""
        self.posterior = self.prior.copy()

    def get_posterior(self) -> Dict[str, float]:
        """Return copy of current posterior."""
        return self.posterior.copy()

    def get_prior(self) -> Dict[str, float]:
        """Return copy of prior."""
        return self.prior.copy()

    def __repr__(self) -> str:
        sorted_post = sorted(self.posterior.items(), key=lambda x: -x[1])
        top = sorted_post[:3]
        parts = [f"{k}:{v:.3f}" for k, v in top]
        return f"BeliefState(top3=[{', '.join(parts)}], H={self.entropy():.3f})"


def make_uniform_belief() -> BeliefState:
    """Create a BeliefState with uniform prior over all hand classes."""
    n = len(HAND_CLASSES)
    prior = {c: 1.0 / n for c in HAND_CLASSES}
    return BeliefState(prior)


def make_heuristic_prior() -> Dict[str, float]:
    """Return a heuristic prior based on approximate hand class frequencies."""
    # Rough empirical frequencies for heads-up poker on turn/river
    return {
        'nuts_or_near_nuts': 0.05,
        'strong_made': 0.12,
        'medium_made': 0.18,
        'weak_showdown': 0.15,
        'strong_draw': 0.10,
        'weak_draw': 0.12,
        'air': 0.28,
    }

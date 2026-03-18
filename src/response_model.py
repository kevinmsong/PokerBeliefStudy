"""
response_model.py — Parameterized response model P(action | hand_class, public_state, opponent_family).

Each opponent family is defined by a dict of behavioral parameters.
The model converts params + hand_class + context -> probability distribution over legal actions.
"""
from typing import Dict, List, Tuple, Optional
import numpy as np

from src.hand_classes import HAND_CLASSES
from src.state import PublicState

# Default opponent family parameters
# Keys:
#   aggression: float [0,1] - tendency to bet/raise when no bet faced
#   bluff_rate: float [0,1] - tendency to bluff with weak hands
#   call_looseness: float [0,1] - tendency to call when facing bets
#   jam_aggression: float [0,1] - tendency to jam vs normal-sized bets
#   draw_semibluff_freq: float [0,1] - tendency for draws to semi-bluff
#   trap_freq: float [0,1] - tendency to slow-play strong hands

DEFAULT_FAMILIES = {
    "balanced": {
        "aggression": 0.5,
        "bluff_rate": 0.25,
        "call_looseness": 0.5,
        "jam_aggression": 0.2,
        "draw_semibluff_freq": 0.4,
        "trap_freq": 0.15,
    },
    "aggressive": {
        "aggression": 0.8,
        "bluff_rate": 0.55,
        "call_looseness": 0.45,
        "jam_aggression": 0.5,
        "draw_semibluff_freq": 0.7,
        "trap_freq": 0.05,
    },
    "passive": {
        "aggression": 0.2,
        "bluff_rate": 0.10,
        "call_looseness": 0.75,
        "jam_aggression": 0.05,
        "draw_semibluff_freq": 0.15,
        "trap_freq": 0.10,
    },
    "tight": {
        "aggression": 0.45,
        "bluff_rate": 0.05,
        "call_looseness": 0.25,
        "jam_aggression": 0.15,
        "draw_semibluff_freq": 0.20,
        "trap_freq": 0.20,
    },
    "loose": {
        "aggression": 0.55,
        "bluff_rate": 0.35,
        "call_looseness": 0.80,
        "jam_aggression": 0.25,
        "draw_semibluff_freq": 0.50,
        "trap_freq": 0.10,
    },
    "maniac": {
        "aggression": 0.95,
        "bluff_rate": 0.80,
        "call_looseness": 0.60,
        "jam_aggression": 0.75,
        "draw_semibluff_freq": 0.90,
        "trap_freq": 0.02,
    },
    "trappy": {
        "aggression": 0.30,
        "bluff_rate": 0.15,
        "call_looseness": 0.55,
        "jam_aggression": 0.10,
        "draw_semibluff_freq": 0.25,
        "trap_freq": 0.70,
    },
    "overbluffer": {
        "aggression": 0.75,
        "bluff_rate": 0.85,
        "call_looseness": 0.40,
        "jam_aggression": 0.45,
        "draw_semibluff_freq": 0.80,
        "trap_freq": 0.05,
    },
    "underbluffer": {
        "aggression": 0.40,
        "bluff_rate": 0.02,
        "call_looseness": 0.45,
        "jam_aggression": 0.10,
        "draw_semibluff_freq": 0.10,
        "trap_freq": 0.25,
    },
    "held_out_1": {
        "aggression": 0.65,
        "bluff_rate": 0.40,
        "call_looseness": 0.35,
        "jam_aggression": 0.35,
        "draw_semibluff_freq": 0.60,
        "trap_freq": 0.30,
    },
    "held_out_2": {
        "aggression": 0.35,
        "bluff_rate": 0.20,
        "call_looseness": 0.65,
        "jam_aggression": 0.08,
        "draw_semibluff_freq": 0.30,
        "trap_freq": 0.45,
    },
}


def _softmax(values: Dict[str, float], temperature: float = 1.0) -> Dict[str, float]:
    """Convert logit-like values to probabilities via softmax."""
    keys = list(values.keys())
    vals = np.array([values[k] for k in keys])
    vals = vals / temperature
    vals = vals - vals.max()  # numerical stability
    exp_vals = np.exp(vals)
    total = exp_vals.sum()
    return {k: float(exp_vals[i] / total) for i, k in enumerate(keys)}


class ResponseModel:
    """Parameterized response model mapping (hand_class, state, family) -> action distribution."""

    def __init__(self, families: Optional[Dict[str, Dict]] = None):
        """Initialize with opponent family parameter dicts.

        If None, uses DEFAULT_FAMILIES.
        """
        self.families = families if families is not None else DEFAULT_FAMILIES.copy()

    def get_family_names(self) -> List[str]:
        return list(self.families.keys())

    def action_probs(
        self,
        hand_class: str,
        public_state: PublicState,
        family_name: str,
        legal_actions: List[Tuple[str, int]],
    ) -> Dict[str, float]:
        """Return probability distribution over legal actions.

        Parameters
        ----------
        hand_class : str
            One of HAND_CLASSES.
        public_state : PublicState
            Current public game state for context.
        family_name : str
            Name of opponent family.
        legal_actions : List[Tuple[str, int]]
            List of legal (action, amount) tuples.

        Returns
        -------
        Dict[str, float]
            Probability for each action name in legal_actions.
        """
        if family_name not in self.families:
            family_name = "balanced"
        params = self.families[family_name]
        action_names = [a for a, _ in legal_actions]

        # Compute unnormalized logits for each action based on hand class + params
        logits = self._compute_logits(hand_class, public_state, params, action_names)

        # Normalize to probabilities
        total = sum(logits.values())
        if total <= 0:
            # Uniform fallback
            n = len(action_names)
            return {a: 1.0 / n for a in action_names}

        probs = {a: logits[a] / total for a in action_names}
        return probs

    def _compute_logits(
        self,
        hand_class: str,
        public_state: PublicState,
        params: Dict[str, float],
        action_names: List[str],
    ) -> Dict[str, float]:
        """Compute unnormalized action weights based on hand class and params."""
        agg = params["aggression"]
        bluff = params["bluff_rate"]
        loose = params["call_looseness"]
        jam_agg = params["jam_aggression"]
        draw_semi = params["draw_semibluff_freq"]
        trap = params["trap_freq"]

        facing_bet = public_state.last_bet > 0

        # Base logits per hand class
        logits = {}

        if facing_bet:
            # Actions: fold, call, jam
            if hand_class == 'nuts_or_near_nuts':
                fold_w = max(0.01, trap * 0.1)  # rarely fold, sometimes trap by calling
                call_w = 0.3 + trap * 0.5       # trap by calling
                jam_w = 0.6 + jam_agg * 0.3 - trap * 0.4
                logits = {"fold": fold_w, "call": call_w, "jam": jam_w}

            elif hand_class == 'strong_made':
                fold_w = 0.02
                call_w = 0.5 - agg * 0.2
                jam_w = 0.4 + jam_agg * 0.3
                logits = {"fold": fold_w, "call": call_w, "jam": jam_w}

            elif hand_class == 'medium_made':
                fold_w = 0.15 - loose * 0.1
                call_w = 0.6 + loose * 0.2
                jam_w = 0.1 + jam_agg * 0.15
                logits = {"fold": fold_w, "call": call_w, "jam": jam_w}

            elif hand_class == 'weak_showdown':
                fold_w = 0.4 - loose * 0.3
                call_w = 0.5 + loose * 0.3
                jam_w = 0.05 + jam_agg * 0.05
                logits = {"fold": fold_w, "call": call_w, "jam": jam_w}

            elif hand_class == 'strong_draw':
                fold_w = 0.10 - loose * 0.05
                call_w = 0.50 + loose * 0.15
                jam_w = 0.30 + draw_semi * 0.2 + jam_agg * 0.1
                logits = {"fold": fold_w, "call": call_w, "jam": jam_w}

            elif hand_class == 'weak_draw':
                fold_w = 0.50 - loose * 0.25
                call_w = 0.40 + loose * 0.25
                jam_w = 0.05 + draw_semi * 0.05
                logits = {"fold": fold_w, "call": call_w, "jam": jam_w}

            elif hand_class == 'air':
                fold_w = 0.65 - bluff * 0.3
                call_w = 0.25 + loose * 0.1
                jam_w = 0.05 + bluff * 0.3
                logits = {"fold": fold_w, "call": call_w, "jam": jam_w}

            else:
                logits = {"fold": 0.3, "call": 0.5, "jam": 0.2}

        else:
            # Actions: check, bet_half_pot, bet_pot, jam
            if hand_class == 'nuts_or_near_nuts':
                check_w = 0.2 + trap * 0.6     # trap by checking
                half_w = 0.2 - trap * 0.1
                pot_w = 0.4 + agg * 0.2 - trap * 0.3
                jam_w = 0.15 + jam_agg * 0.2 - trap * 0.1
                logits = {"check": check_w, "bet_half_pot": half_w, "bet_pot": pot_w, "jam": jam_w}

            elif hand_class == 'strong_made':
                check_w = 0.20
                half_w = 0.35 + agg * 0.1
                pot_w = 0.30 + agg * 0.1
                jam_w = 0.10 + jam_agg * 0.15
                logits = {"check": check_w, "bet_half_pot": half_w, "bet_pot": pot_w, "jam": jam_w}

            elif hand_class == 'medium_made':
                check_w = 0.45 - agg * 0.1
                half_w = 0.35 + agg * 0.1
                pot_w = 0.15 + agg * 0.05
                jam_w = 0.05
                logits = {"check": check_w, "bet_half_pot": half_w, "bet_pot": pot_w, "jam": jam_w}

            elif hand_class == 'weak_showdown':
                check_w = 0.65 - agg * 0.05
                half_w = 0.20 + agg * 0.05
                pot_w = 0.10
                jam_w = 0.03
                logits = {"check": check_w, "bet_half_pot": half_w, "bet_pot": pot_w, "jam": jam_w}

            elif hand_class == 'strong_draw':
                check_w = 0.40 - draw_semi * 0.2
                half_w = 0.25 + draw_semi * 0.1
                pot_w = 0.20 + draw_semi * 0.1
                jam_w = 0.10 + draw_semi * 0.1
                logits = {"check": check_w, "bet_half_pot": half_w, "bet_pot": pot_w, "jam": jam_w}

            elif hand_class == 'weak_draw':
                check_w = 0.70 - draw_semi * 0.15
                half_w = 0.15 + draw_semi * 0.1
                pot_w = 0.10 + draw_semi * 0.05
                jam_w = 0.02
                logits = {"check": check_w, "bet_half_pot": half_w, "bet_pot": pot_w, "jam": jam_w}

            elif hand_class == 'air':
                check_w = 0.65 - bluff * 0.35
                half_w = 0.15 + bluff * 0.2
                pot_w = 0.10 + bluff * 0.1
                jam_w = 0.05 + bluff * 0.1
                logits = {"check": check_w, "bet_half_pot": half_w, "bet_pot": pot_w, "jam": jam_w}

            else:
                logits = {"check": 0.4, "bet_half_pot": 0.3, "bet_pot": 0.2, "jam": 0.1}

        # Filter to only legal actions and clamp negatives
        result = {}
        for a in action_names:
            result[a] = max(1e-6, logits.get(a, 1e-6))

        return result

    def get_family_params(self, family_name: str) -> Dict[str, float]:
        """Return parameter dict for a family."""
        return self.families.get(family_name, self.families["balanced"]).copy()


def make_default_response_model() -> ResponseModel:
    """Create ResponseModel with all default opponent families."""
    return ResponseModel(DEFAULT_FAMILIES)

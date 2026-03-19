"""
family_policy.py — Stochastic family-conditioned poker policy.

This agent uses the response model directly as its actual behavioral policy,
making opponent-family labels correspond to observable action tendencies.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.agents.base import BaseAgent
from src.hand_classes import classify_hand
from src.infoset import InfoState
from src.response_model import ResponseModel, make_default_response_model
from src.state import PublicState


class FamilyPolicyAgent(BaseAgent):
    """Sample actions from a parameterized response family."""

    def __init__(
        self,
        player_idx: int,
        rng: np.random.Generator,
        behavior_family: str = "balanced",
        response_model: Optional[ResponseModel] = None,
        name: str = "FamilyPolicyAgent",
    ):
        super().__init__(name=name, player_idx=player_idx, rng=rng)
        self.behavior_family = behavior_family
        self.response_model = response_model or make_default_response_model()
        self._last_action_probs: Dict[str, float] = {}

    def act(self, infostate: InfoState) -> Tuple[str, int]:
        """Sample an action from the active behavior family."""
        if not infostate.legal_actions:
            return ("check", 0)

        hand_class = classify_hand(infostate.hole_cards, infostate.board)
        public_state = PublicState(
            board=list(infostate.board),
            pot=infostate.pot,
            street=infostate.street,
            to_act=infostate.to_act,
            last_bet=infostate.last_bet,
            history=list(infostate.history),
            first_to_act=infostate.to_act,
        )
        probs = self.response_model.action_probs(
            hand_class=hand_class,
            public_state=public_state,
            family_name=self.behavior_family,
            legal_actions=infostate.legal_actions,
        )
        self._last_action_probs = probs

        actions = list(probs.keys())
        weights = np.array([probs[a] for a in actions], dtype=float)
        weights = weights / weights.sum()
        action_name = str(self.rng.choice(actions, p=weights))
        legal_amounts = {a: amt for a, amt in infostate.legal_actions}
        return (action_name, legal_amounts[action_name])

    def new_hand(self) -> None:
        self._last_action_probs = {}

    def set_behavior_family(self, family_name: str) -> None:
        self.behavior_family = family_name

    def get_behavior_family(self) -> Optional[str]:
        return self.behavior_family


"""
adaptive_counter.py — Adaptive responder that detects hidden family shifts.

The agent begins with a balanced family policy, scores rolling log-likelihoods
for candidate opponent families from observed actions, and switches once to a
predefined counter-style when the evidence gap clears a threshold.
"""
from collections import deque
import math
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from src.agents.base import BaseAgent
from src.beliefs import make_heuristic_prior
from src.hand_classes import HAND_CLASSES, classify_hand
from src.infoset import InfoState
from src.response_model import ResponseModel, make_default_response_model
from src.state import PublicState


class AdaptiveCounterFamilyAgent(BaseAgent):
    """Adaptive family-policy agent with one-time hidden-switch response."""

    def __init__(
        self,
        player_idx: int,
        rng: np.random.Generator,
        initial_behavior_family: str = "balanced",
        candidate_families: Optional[List[str]] = None,
        counter_map: Optional[Dict[str, str]] = None,
        evidence_window: int = 80,
        evidence_threshold: float = 6.0,
        min_observations: int = 30,
        min_hand_before_switch: int = 0,
        response_model: Optional[ResponseModel] = None,
        prior: Optional[Dict[str, float]] = None,
        name: str = "AdaptiveCounterFamilyAgent",
    ):
        super().__init__(name=name, player_idx=player_idx, rng=rng)
        self.response_model = response_model or make_default_response_model()
        self.behavior_family = initial_behavior_family
        self.reference_family = initial_behavior_family
        self.candidate_families = candidate_families or ["aggressive", "passive", "trappy"]
        self.counter_map = counter_map or {
            "aggressive": "tight",
            "passive": "aggressive",
            "trappy": "balanced",
        }
        self.evidence_window = evidence_window
        self.evidence_threshold = evidence_threshold
        self.min_observations = min_observations
        self.min_hand_before_switch = min_hand_before_switch
        self.prior = prior or make_heuristic_prior()
        self._evidence: Deque[Dict[str, float]] = deque(maxlen=evidence_window)
        self._cusum_scores: Dict[str, float] = {family: 0.0 for family in self.candidate_families}
        self._current_hand_num = -1
        self._observations_seen = 0
        self._detected_family: Optional[str] = None
        self._detection_hand: Optional[int] = None
        self._switched = False

    def act(self, infostate: InfoState) -> Tuple[str, int]:
        """Act according to the current behavior family."""
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
        actions = list(probs.keys())
        weights = np.array([probs[a] for a in actions], dtype=float)
        weights = weights / weights.sum()
        action_name = str(self.rng.choice(actions, p=weights))
        legal_amounts = {a: amt for a, amt in infostate.legal_actions}
        return (action_name, legal_amounts[action_name])

    def observe_opponent_action(
        self,
        action: str,
        amount: int,
        public_state: PublicState,
    ) -> None:
        """Update rolling family evidence from an observed opponent action."""
        del amount
        legal_actions = self._infer_legal_actions(public_state)
        families = [self.reference_family] + [f for f in self.candidate_families if f != self.reference_family]
        log_likelihoods: Dict[str, float] = {}

        for family in families:
            marginal = 0.0
            for hand_class in HAND_CLASSES:
                class_prob = self.prior.get(hand_class, 0.0)
                probs = self.response_model.action_probs(
                    hand_class=hand_class,
                    public_state=public_state,
                    family_name=family,
                    legal_actions=legal_actions,
                )
                marginal += class_prob * probs.get(action, 1e-9)
            log_likelihoods[family] = math.log(max(marginal, 1e-9))

        self._evidence.append(log_likelihoods)
        self._observations_seen += 1

        baseline_log = log_likelihoods[self.reference_family]
        for family in self.candidate_families:
            delta = log_likelihoods[family] - baseline_log
            self._cusum_scores[family] = max(0.0, self._cusum_scores.get(family, 0.0) + delta)

        if self._current_hand_num < self.min_hand_before_switch:
            for family in self.candidate_families:
                self._cusum_scores[family] = 0.0
            return

        if self._switched or self._observations_seen < self.min_observations:
            return

        best_family = max(self.candidate_families, key=lambda family: self._cusum_scores.get(family, 0.0))
        best_score = self._cusum_scores.get(best_family, 0.0)
        if best_score >= self.evidence_threshold:
            self._detected_family = best_family
            self._detection_hand = self._current_hand_num
            self.behavior_family = self.counter_map.get(best_family, self.reference_family)
            self._switched = True

    def new_hand(self) -> None:
        """Preserve cross-hand evidence; no per-hand reset required."""
        pass

    def set_hand_context(self, hand_num: int, seed: int, run_id: str) -> None:
        del seed, run_id
        self._current_hand_num = hand_num

    def set_behavior_family(self, family_name: str) -> None:
        self.behavior_family = family_name

    def get_behavior_family(self) -> Optional[str]:
        return self.behavior_family

    def get_model_family(self) -> Optional[str]:
        return self._detected_family or self.reference_family

    def get_detection_state(self) -> Optional[Dict[str, object]]:
        return {
            "detected_family": self._detected_family,
            "detection_hand": self._detection_hand,
            "switched": self._switched,
            "observations_seen": self._observations_seen,
            "current_behavior_family": self.behavior_family,
            "cusum_scores": dict(self._cusum_scores),
        }

    def _infer_legal_actions(self, public_state: PublicState) -> List[Tuple[str, int]]:
        """Approximate the legal actions the opponent faced when acting."""
        if public_state.last_bet > 0:
            jam_amount = max(public_state.last_bet * 2, public_state.pot)
            return [
                ("fold", 0),
                ("call", public_state.last_bet),
                ("jam", jam_amount),
            ]

        half_pot = max(1, public_state.pot // 2)
        full_pot = max(1, public_state.pot)
        return [
            ("check", 0),
            ("bet_half_pot", half_pot),
            ("bet_pot", full_pot),
            ("jam", max(full_pot * 2, 1)),
        ]

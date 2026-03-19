"""
base.py — Abstract base class for all poker agents.
"""
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Optional, List

import numpy as np

from src.infoset import InfoState
from src.state import PublicState


class BaseAgent(ABC):
    """Abstract base class for all poker agents."""

    def __init__(self, name: str, player_idx: int, rng: np.random.Generator):
        """Initialize agent.

        Parameters
        ----------
        name : str
            Agent name for logging.
        player_idx : int
            Which player position this agent occupies (0 or 1).
        rng : np.random.Generator
            Numpy RNG for any stochastic decisions.
        """
        self.name = name
        self.player_idx = player_idx
        self.rng = rng

    @abstractmethod
    def act(self, infostate: InfoState) -> Tuple[str, int]:
        """Choose an action given the current information state.

        Parameters
        ----------
        infostate : InfoState
            Current information state for this player.

        Returns
        -------
        Tuple[str, int]
            (action_name, amount) chosen by the agent.
        """
        ...

    def observe_opponent_action(
        self,
        action: str,
        amount: int,
        public_state: PublicState,
    ) -> None:
        """Called after opponent takes an action.

        Default implementation does nothing.
        Subclasses can override to update beliefs.

        Parameters
        ----------
        action : str
            Action taken by opponent.
        amount : int
            Amount of the action.
        public_state : PublicState
            Public state at the time of the action.
        """
        pass

    def new_hand(self) -> None:
        """Called at the start of each new hand.

        Default implementation does nothing.
        Subclasses can override to reset per-hand state.
        """
        pass

    def set_hand_context(
        self,
        hand_num: int,
        seed: int,
        run_id: str,
    ) -> None:
        """Provide per-hand context before a new hand starts.

        Default implementation does nothing. Agents that need cross-hand
        adaptation metadata can override this hook.
        """
        pass

    def set_behavior_family(self, family_name: str) -> None:
        """Update the agent's active behavior family.

        Default implementation does nothing.
        """
        pass

    def set_model_family(self, family_name: str) -> None:
        """Update the agent's opponent-model family.

        Default implementation does nothing.
        """
        pass

    def get_ev_table(self) -> Optional[Dict[str, float]]:
        """Return EV table for logging purposes.

        Default returns None. EV agents override this.
        """
        return None

    def get_belief_state(self) -> Optional[Dict[str, float]]:
        """Return current belief posterior for logging.

        Default returns None. Belief agents override this.
        """
        return None

    def get_behavior_family(self) -> Optional[str]:
        """Return the agent's active behavior family, if applicable."""
        return None

    def get_model_family(self) -> Optional[str]:
        """Return the agent's current opponent-model family, if applicable."""
        return None

    def get_detection_state(self) -> Optional[Dict[str, object]]:
        """Return optional adaptation metadata for logging."""
        return None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, player={self.player_idx})"

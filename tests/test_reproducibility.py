"""
test_reproducibility.py — Tests for reproducibility of simulation.

Same seed => same hand trace.
Different seed => different hand trace.
"""
import pytest
import numpy as np
from src.simulate import run_hand
from src.agents.heuristic import HeuristicAgent
from src.agents.ev_belief import BeliefEVAgent
from src.response_model import make_default_response_model


DEFAULT_CONFIG = {
    "starting_pot": 100,
    "effective_stack": 200,
    "street_start": "turn",
}


def make_agents(seed: int):
    rng0 = np.random.default_rng(seed * 2)
    rng1 = np.random.default_rng(seed * 2 + 1)
    agent0 = HeuristicAgent(player_idx=0, rng=rng0, name="heuristic_0")
    agent1 = HeuristicAgent(player_idx=1, rng=rng1, name="heuristic_1")
    return agent0, agent1


def make_belief_agents(seed: int):
    rng0 = np.random.default_rng(seed * 2)
    rng1 = np.random.default_rng(seed * 2 + 1)
    rm = make_default_response_model()
    agent0 = BeliefEVAgent(
        player_idx=0, rng=rng0, opp_family="balanced",
        n_rollouts=10, response_model=rm, name="belief_0"
    )
    agent1 = HeuristicAgent(player_idx=1, rng=rng1, name="heuristic_1")
    return agent0, agent1


class TestSameSeedSameTrace:
    def test_same_seed_same_board(self):
        """Same seed should produce same board cards."""
        seed = 42
        agent0a, agent1a = make_agents(seed)
        agent0b, agent1b = make_agents(seed)

        rng_a = np.random.default_rng(seed * 3)
        rng_b = np.random.default_rng(seed * 3)

        record_a = run_hand(agent0a, agent1a, rng_a, DEFAULT_CONFIG, seed=seed)
        record_b = run_hand(agent0b, agent1b, rng_b, DEFAULT_CONFIG, seed=seed)

        assert record_a["board"] == record_b["board"]

    def test_same_seed_same_hole_cards(self):
        """Same seed should produce same hole cards."""
        seed = 42
        agent0a, agent1a = make_agents(seed)
        agent0b, agent1b = make_agents(seed)

        rng_a = np.random.default_rng(seed * 3)
        rng_b = np.random.default_rng(seed * 3)

        record_a = run_hand(agent0a, agent1a, rng_a, DEFAULT_CONFIG, seed=seed)
        record_b = run_hand(agent0b, agent1b, rng_b, DEFAULT_CONFIG, seed=seed)

        assert record_a["hole_cards_0"] == record_b["hole_cards_0"]
        assert record_a["hole_cards_1"] == record_b["hole_cards_1"]

    def test_same_seed_same_action_history(self):
        """Same seed and same agents should produce same action history."""
        seed = 42
        agent0a, agent1a = make_agents(seed)
        agent0b, agent1b = make_agents(seed)

        rng_a = np.random.default_rng(seed * 3)
        rng_b = np.random.default_rng(seed * 3)

        record_a = run_hand(agent0a, agent1a, rng_a, DEFAULT_CONFIG, seed=seed)
        record_b = run_hand(agent0b, agent1b, rng_b, DEFAULT_CONFIG, seed=seed)

        actions_a = [(s["action"], s["player"]) for s in record_a["action_history"]]
        actions_b = [(s["action"], s["player"]) for s in record_b["action_history"]]
        assert actions_a == actions_b

    def test_same_seed_same_rewards(self):
        """Same seed should produce same terminal rewards."""
        seed = 42
        agent0a, agent1a = make_agents(seed)
        agent0b, agent1b = make_agents(seed)

        rng_a = np.random.default_rng(seed * 3)
        rng_b = np.random.default_rng(seed * 3)

        record_a = run_hand(agent0a, agent1a, rng_a, DEFAULT_CONFIG, seed=seed)
        record_b = run_hand(agent0b, agent1b, rng_b, DEFAULT_CONFIG, seed=seed)

        assert record_a["terminal_reward_0"] == record_b["terminal_reward_0"]
        assert record_a["terminal_reward_1"] == record_b["terminal_reward_1"]

    def test_same_seed_belief_agent_reproducible(self):
        """Belief agent with same seed should produce same results."""
        seed = 123
        agent0a, agent1a = make_belief_agents(seed)
        agent0b, agent1b = make_belief_agents(seed)

        rng_a = np.random.default_rng(seed * 3)
        rng_b = np.random.default_rng(seed * 3)

        record_a = run_hand(agent0a, agent1a, rng_a, DEFAULT_CONFIG, seed=seed)
        record_b = run_hand(agent0b, agent1b, rng_b, DEFAULT_CONFIG, seed=seed)

        assert record_a["board"] == record_b["board"]
        assert record_a["terminal_reward_0"] == record_b["terminal_reward_0"]


class TestDifferentSeedDifferentTrace:
    def test_different_seeds_different_boards(self):
        """Different seeds should (almost certainly) produce different boards."""
        seed_a, seed_b = 42, 99

        agent0a, agent1a = make_agents(seed_a)
        agent0b, agent1b = make_agents(seed_b)

        rng_a = np.random.default_rng(seed_a * 3)
        rng_b = np.random.default_rng(seed_b * 3)

        record_a = run_hand(agent0a, agent1a, rng_a, DEFAULT_CONFIG, seed=seed_a)
        record_b = run_hand(agent0b, agent1b, rng_b, DEFAULT_CONFIG, seed=seed_b)

        # Boards should differ (probability of collision is negligible)
        assert record_a["board"] != record_b["board"] or \
               record_a["hole_cards_0"] != record_b["hole_cards_0"]

    def test_many_seeds_variety(self):
        """Running many seeds should produce a variety of hand strengths."""
        boards = []
        for seed in range(10):
            agent0, agent1 = make_agents(seed)
            rng = np.random.default_rng(seed * 3)
            record = run_hand(agent0, agent1, rng, DEFAULT_CONFIG, seed=seed)
            boards.append(tuple(record["board"]))

        # Should have multiple distinct boards
        assert len(set(boards)) > 1

    def test_multiple_hands_in_sequence(self):
        """Running multiple hands should use RNG state correctly."""
        seed = 42
        records = []
        for hand_num in range(5):
            agent0, agent1 = make_agents(seed)
            hand_rng = np.random.default_rng(seed * 100000 + hand_num)
            record = run_hand(agent0, agent1, hand_rng, DEFAULT_CONFIG, seed=seed)
            records.append(record)

        boards = [tuple(r["board"]) for r in records]
        # Multiple hands should have different boards
        assert len(set(boards)) > 1

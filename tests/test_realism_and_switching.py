"""
test_realism_and_switching.py — Coverage for refreshed realism and case-study logic.
"""
import numpy as np

from src.agents.adaptive_counter import AdaptiveCounterFamilyAgent
from src.agents.base import BaseAgent
from src.agents.family_policy import FamilyPolicyAgent
from src.infoset import InfoState
from src.simulate import run_hand
from src.state import PublicState
from src.tournament import _make_switch_schedule


DEFAULT_CONFIG = {
    "effective_stack": 200,
    "starting_commitment_0": 50,
    "starting_commitment_1": 50,
    "street_start": "turn",
}


class BetPotThenCallAgent(BaseAgent):
    def act(self, infostate):
        legal = dict(infostate.legal_actions)
        if "bet_pot" in legal:
            return ("bet_pot", legal["bet_pot"])
        if "call" in legal:
            return ("call", legal["call"])
        return ("check", 0)


class FoldFacingBetAgent(BaseAgent):
    def act(self, infostate):
        legal = dict(infostate.legal_actions)
        if "fold" in legal:
            return ("fold", 0)
        return ("check", 0)


def make_no_bet_public_state():
    return PublicState(
        board=["Ah", "Kd", "7c", "2s"],
        pot=100,
        street="turn",
        to_act=0,
        last_bet=0,
        first_to_act=0,
        history=[],
    )


def make_air_infostate():
    return InfoState(
        hole_cards=("3d", "9s"),
        board=["Ah", "Kd", "7c", "2s"],
        pot=100,
        street="turn",
        stack_self=150,
        stack_opp=150,
        to_act=0,
        last_bet=0,
        history=[],
        legal_actions=[("check", 0), ("bet_half_pot", 50), ("bet_pot", 100), ("jam", 150)],
    )


class TestNetRewardAccounting:
    def test_rewards_are_net_of_remaining_stack(self):
        agent0 = BetPotThenCallAgent("bettor", 0, np.random.default_rng(1))
        agent1 = FoldFacingBetAgent("folder", 1, np.random.default_rng(2))

        record = run_hand(
            agent0=agent0,
            agent1=agent1,
            rng=np.random.default_rng(7),
            config=DEFAULT_CONFIG,
            hand_num=0,
            seed=7,
        )

        assert record["starting_commitment_0"] == 50
        assert record["starting_commitment_1"] == 50
        assert record["terminal_reward_0"] == 100
        assert record["terminal_reward_1"] == 0

    def test_first_to_act_alternates_by_hand(self):
        agent0 = FoldFacingBetAgent("p0", 0, np.random.default_rng(3))
        agent1 = FoldFacingBetAgent("p1", 1, np.random.default_rng(4))

        record_0 = run_hand(agent0, agent1, np.random.default_rng(10), DEFAULT_CONFIG, hand_num=0, seed=10)
        record_1 = run_hand(agent0, agent1, np.random.default_rng(11), DEFAULT_CONFIG, hand_num=1, seed=11)

        assert record_0["first_to_act"] == 0
        assert record_1["first_to_act"] == 1


class TestFamilyPolicyAgent:
    def test_aggressive_family_bets_more_than_passive(self):
        infostate = make_air_infostate()
        aggressive = FamilyPolicyAgent(0, np.random.default_rng(42), behavior_family="aggressive")
        passive = FamilyPolicyAgent(0, np.random.default_rng(42), behavior_family="passive")

        aggressive_bets = 0
        passive_bets = 0
        n_samples = 400
        for _ in range(n_samples):
            aggressive_bets += aggressive.act(infostate)[0] != "check"
            passive_bets += passive.act(infostate)[0] != "check"

        assert aggressive_bets > passive_bets


class TestAdaptiveCounterFamilyAgent:
    def test_no_switch_before_threshold(self):
        agent = AdaptiveCounterFamilyAgent(
            player_idx=0,
            rng=np.random.default_rng(5),
            evidence_window=20,
            evidence_threshold=20.0,
            min_observations=5,
        )
        pub = make_no_bet_public_state()
        agent.set_hand_context(hand_num=3, seed=1, run_id="r")
        for _ in range(8):
            agent.observe_opponent_action("bet_pot", 100, pub)

        detection = agent.get_detection_state()
        assert detection["switched"] is False
        assert detection["detected_family"] is None

    def test_switches_to_counter_family_after_evidence(self):
        agent = AdaptiveCounterFamilyAgent(
            player_idx=0,
            rng=np.random.default_rng(6),
            evidence_window=20,
            evidence_threshold=0.5,
            min_observations=5,
        )
        pub = make_no_bet_public_state()
        agent.set_hand_context(hand_num=12, seed=1, run_id="r")
        for _ in range(10):
            agent.observe_opponent_action("bet_pot", 100, pub)

        detection = agent.get_detection_state()
        assert detection["switched"] is True
        assert detection["detected_family"] in {"aggressive", "passive", "trappy"}
        assert detection["current_behavior_family"] in {"tight", "aggressive", "balanced"}
        assert detection["detection_hand"] == 12


class TestSwitchSchedule:
    def test_schedule_is_deterministic_for_seed_and_group(self):
        schedule_a = _make_switch_schedule(42, 7500, "hidden_pair", "balanced", ["aggressive", "passive", "trappy"])
        schedule_b = _make_switch_schedule(42, 7500, "hidden_pair", "balanced", ["aggressive", "passive", "trappy"])
        assert schedule_a == schedule_b

    def test_schedule_switch_occurs_in_middle_half(self):
        schedule = _make_switch_schedule(42, 7500, "hidden_pair", "balanced", ["aggressive", "passive", "trappy"])
        assert 7500 // 4 <= schedule["switch_hand"] < (3 * 7500) // 4

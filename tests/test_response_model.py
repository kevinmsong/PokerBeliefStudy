"""
test_response_model.py — Tests for src/response_model.py
"""
import pytest
from src.response_model import ResponseModel, make_default_response_model, DEFAULT_FAMILIES
from src.hand_classes import HAND_CLASSES
from src.state import PublicState


def make_pub(last_bet=0, pot=100, street="turn"):
    return PublicState(
        board=["Ah", "Kd", "Qc", "2s"],
        pot=pot,
        street=street,
        to_act=0,
        last_bet=last_bet,
        history=[],
    )


def no_bet_legal():
    return [("check", 0), ("bet_half_pot", 50), ("bet_pot", 100), ("jam", 200)]


def facing_bet_legal(bet=50):
    return [("fold", 0), ("call", bet), ("jam", 200)]


class TestActionProbsSumToOne:
    def test_no_bet_probs_sum_to_one(self):
        rm = make_default_response_model()
        for hc in HAND_CLASSES:
            for fam in DEFAULT_FAMILIES:
                probs = rm.action_probs(hc, make_pub(last_bet=0), fam, no_bet_legal())
                total = sum(probs.values())
                assert abs(total - 1.0) < 1e-6, \
                    f"Probs don't sum to 1 for {hc}/{fam}: {total}"

    def test_facing_bet_probs_sum_to_one(self):
        rm = make_default_response_model()
        for hc in HAND_CLASSES:
            for fam in DEFAULT_FAMILIES:
                probs = rm.action_probs(hc, make_pub(last_bet=50), fam, facing_bet_legal())
                total = sum(probs.values())
                assert abs(total - 1.0) < 1e-6, \
                    f"Probs don't sum to 1 for {hc}/{fam}: {total}"

    def test_all_probs_nonnegative(self):
        rm = make_default_response_model()
        for hc in HAND_CLASSES:
            for fam in DEFAULT_FAMILIES:
                probs = rm.action_probs(hc, make_pub(last_bet=0), fam, no_bet_legal())
                for action, prob in probs.items():
                    assert prob >= 0, f"Negative prob for {hc}/{fam}/{action}: {prob}"


class TestAllFamiliesAccessible:
    def test_all_default_families_present(self):
        rm = make_default_response_model()
        for fam in DEFAULT_FAMILIES:
            assert fam in rm.families

    def test_all_families_listed(self):
        rm = make_default_response_model()
        families = rm.get_family_names()
        expected = list(DEFAULT_FAMILIES.keys())
        assert set(families) == set(expected)

    def test_unknown_family_falls_back_to_balanced(self):
        rm = make_default_response_model()
        # Should not raise
        probs = rm.action_probs("air", make_pub(), "nonexistent_family", no_bet_legal())
        assert sum(probs.values()) > 0.99


class TestParameterEffects:
    def test_aggressive_bets_more_than_passive(self):
        """Aggressive family should bet more often than passive family."""
        rm = make_default_response_model()
        pub = make_pub(last_bet=0)
        legal = no_bet_legal()

        # For medium hand, aggressive should bet more than passive
        agg_probs = rm.action_probs("medium_made", pub, "aggressive", legal)
        pas_probs = rm.action_probs("medium_made", pub, "passive", legal)

        # Aggressive betting probability = bet_half_pot + bet_pot + jam
        agg_bet = agg_probs.get("bet_half_pot", 0) + agg_probs.get("bet_pot", 0) + agg_probs.get("jam", 0)
        pas_bet = pas_probs.get("bet_half_pot", 0) + pas_probs.get("bet_pot", 0) + pas_probs.get("jam", 0)
        assert agg_bet > pas_bet, f"Aggressive bet {agg_bet} not > passive bet {pas_bet}"

    def test_trappy_checks_more_with_strong_hands(self):
        """Trappy family should check more with nuts than aggressive family."""
        rm = make_default_response_model()
        pub = make_pub(last_bet=0)
        legal = no_bet_legal()

        trap_probs = rm.action_probs("nuts_or_near_nuts", pub, "trappy", legal)
        agg_probs = rm.action_probs("nuts_or_near_nuts", pub, "aggressive", legal)

        trap_check = trap_probs.get("check", 0)
        agg_check = agg_probs.get("check", 0)
        assert trap_check > agg_check, f"Trappy check {trap_check} not > aggressive check {agg_check}"

    def test_loose_calls_more_than_tight(self):
        """Loose family should call more often than tight family."""
        rm = make_default_response_model()
        pub = make_pub(last_bet=50)
        legal = facing_bet_legal()

        loose_probs = rm.action_probs("medium_made", pub, "loose", legal)
        tight_probs = rm.action_probs("medium_made", pub, "tight", legal)

        loose_call = loose_probs.get("call", 0)
        tight_call = tight_probs.get("call", 0)
        assert loose_call >= tight_call, f"Loose call {loose_call} not >= tight call {tight_call}"

    def test_overbluffer_bluffs_more_than_underbluffer(self):
        """Overbluffer should bet more with air than underbluffer."""
        rm = make_default_response_model()
        pub = make_pub(last_bet=0)
        legal = no_bet_legal()

        over_probs = rm.action_probs("air", pub, "overbluffer", legal)
        under_probs = rm.action_probs("air", pub, "underbluffer", legal)

        over_bet = over_probs.get("bet_half_pot", 0) + over_probs.get("bet_pot", 0) + over_probs.get("jam", 0)
        under_bet = under_probs.get("bet_half_pot", 0) + under_probs.get("bet_pot", 0) + under_probs.get("jam", 0)
        assert over_bet > under_bet, f"Overbluffer bet {over_bet} not > underbluffer bet {under_bet}"

    def test_nuts_folds_less_than_air(self):
        """Nuts should fold less often than air when facing a bet."""
        rm = make_default_response_model()
        pub = make_pub(last_bet=50)
        legal = facing_bet_legal()

        nuts_probs = rm.action_probs("nuts_or_near_nuts", pub, "balanced", legal)
        air_probs = rm.action_probs("air", pub, "balanced", legal)

        nuts_fold = nuts_probs.get("fold", 0)
        air_fold = air_probs.get("fold", 0)
        assert nuts_fold < air_fold, f"Nuts fold {nuts_fold} not < air fold {air_fold}"


class TestEdgeCases:
    def test_custom_legal_actions_subset(self):
        """Should work with just fold/call (no jam available)."""
        rm = make_default_response_model()
        pub = make_pub(last_bet=50)
        legal = [("fold", 0), ("call", 50)]  # No jam
        probs = rm.action_probs("medium_made", pub, "balanced", legal)
        total = sum(probs.values())
        assert abs(total - 1.0) < 1e-6

    def test_single_legal_action(self):
        """With only one legal action, it should have probability 1."""
        rm = make_default_response_model()
        pub = make_pub(last_bet=0)
        legal = [("check", 0)]
        probs = rm.action_probs("air", pub, "balanced", legal)
        assert abs(probs.get("check", 0) - 1.0) < 1e-6

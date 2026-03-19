"""
tournament.py — Run many hands across seeds and matchups.
"""
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.agents.adaptive_counter import AdaptiveCounterFamilyAgent
from src.agents.base import BaseAgent
from src.agents.ev_belief import BeliefEVAgent
from src.agents.ev_static import StaticEVAgent
from src.agents.family_policy import FamilyPolicyAgent
from src.agents.heuristic import HeuristicAgent
from src.analysis import (
    compute_performance_summary,
    compute_robustness_metrics,
    generate_all_tables,
    plot_performance_comparison,
    plot_robustness_heatmap,
)
from src.response_model import make_default_response_model
from src.simulate import run_hand
from src.utils import generate_hand_id, generate_run_id, save_json, seed_from_string


def make_agent(
    agent_config: dict,
    player_idx: int,
    rng: np.random.Generator,
    behavior_family: Optional[str] = None,
    model_family: Optional[str] = None,
) -> BaseAgent:
    """Instantiate an agent from config dict."""
    agent_type = agent_config.get("name", "HeuristicAgent")
    response_model = make_default_response_model()

    if agent_type == "HeuristicAgent":
        return HeuristicAgent(
            player_idx=player_idx,
            rng=rng,
            name=agent_config.get("label", "heuristic"),
        )

    if agent_type == "StaticEVAgent":
        return StaticEVAgent(
            player_idx=player_idx,
            rng=rng,
            opp_family=model_family or "balanced",
            n_rollouts=agent_config.get("n_rollouts", 200),
            prior=None,
            response_model=response_model,
            name=agent_config.get("label", "ev_static"),
        )

    if agent_type == "BeliefEVAgent":
        return BeliefEVAgent(
            player_idx=player_idx,
            rng=rng,
            opp_family=model_family or "balanced",
            n_rollouts=agent_config.get("n_rollouts", 200),
            prior=None,
            smoothing=agent_config.get("smoothing", 0.05),
            response_model=response_model,
            name=agent_config.get("label", "ev_belief"),
        )

    if agent_type == "FamilyPolicyAgent":
        return FamilyPolicyAgent(
            player_idx=player_idx,
            rng=rng,
            behavior_family=behavior_family or "balanced",
            response_model=response_model,
            name=agent_config.get("label", "family_policy"),
        )

    if agent_type == "AdaptiveCounterFamilyAgent":
        return AdaptiveCounterFamilyAgent(
            player_idx=player_idx,
            rng=rng,
            initial_behavior_family=behavior_family or "balanced",
            candidate_families=agent_config.get("candidate_families"),
            counter_map=agent_config.get("counter_map"),
            evidence_window=agent_config.get("evidence_window", 80),
            evidence_threshold=agent_config.get("evidence_threshold", 6.0),
            min_observations=agent_config.get("min_observations", 30),
            min_hand_before_switch=agent_config.get("min_hand_before_switch", 0),
            response_model=response_model,
            name=agent_config.get("label", "adaptive_counter"),
        )

    raise ValueError(f"Unknown agent type: {agent_type}")


def run_tournament(
    matchups: List[Dict],
    seeds: List[int],
    hands_per_seed: int,
    config: dict,
    output_dir: str,
    experiment_id: str,
    agent_configs: Optional[Dict] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run many hands across seeds and matchups."""
    del agent_configs
    os.makedirs(os.path.join(output_dir, "raw_runs"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "processed"), exist_ok=True)

    all_records: List[Dict[str, Any]] = []
    total_hands = sum(
        len(seeds) * matchup.get("hands_per_seed", hands_per_seed)
        for matchup in matchups
    )
    schedule_cache: Dict[Tuple[int, str], Dict[str, Any]] = {}
    pbar = tqdm(total=total_hands, desc=f"Tournament [{experiment_id}]", disable=not verbose)

    for matchup in matchups:
        matchup_label = matchup.get("label", "matchup")
        agent0_cfg = matchup.get("agent0", {"name": "HeuristicAgent"})
        agent1_cfg = matchup.get("agent1", {"name": "HeuristicAgent"})
        behavior_family_0 = matchup.get("behavior_family_0")
        behavior_family_1 = matchup.get("behavior_family_1")
        model_family_0 = matchup.get("model_family_0", matchup.get("opp_family_0", "balanced"))
        model_family_1 = matchup.get("model_family_1", matchup.get("opp_family_1", "balanced"))
        switching_player = matchup.get("switching_player")
        schedule_group = matchup.get("switch_schedule_group")
        switch_candidates = matchup.get("switch_candidates", ["aggressive", "passive", "trappy"])
        pre_switch_family = matchup.get("pre_switch_family", "balanced")
        hands_this_matchup = matchup.get("hands_per_seed", hands_per_seed)

        for seed in seeds:
            run_id = generate_run_id(f"{experiment_id}_{matchup_label}", seed)
            rng0 = np.random.default_rng(seed * 2)
            rng1 = np.random.default_rng(seed * 2 + 1)

            agent0 = make_agent(
                agent0_cfg,
                player_idx=0,
                rng=rng0,
                behavior_family=behavior_family_0,
                model_family=model_family_0,
            )
            agent1 = make_agent(
                agent1_cfg,
                player_idx=1,
                rng=rng1,
                behavior_family=behavior_family_1,
                model_family=model_family_1,
            )

            switch_schedule = None
            if switching_player is not None and schedule_group:
                schedule_key = (seed, schedule_group)
                if schedule_key not in schedule_cache:
                    schedule_cache[schedule_key] = _make_switch_schedule(
                        seed=seed,
                        hands_per_seed=hands_this_matchup,
                        schedule_group=schedule_group,
                        pre_family=pre_switch_family,
                        switch_candidates=switch_candidates,
                    )
                switch_schedule = schedule_cache[schedule_key]

            seed_records = []
            for hand_num in range(hands_this_matchup):
                for agent in (agent0, agent1):
                    agent.set_hand_context(hand_num=hand_num, seed=seed, run_id=run_id)

                if switch_schedule is not None and switching_player in (0, 1):
                    active_family = (
                        switch_schedule["true_family_pre"]
                        if hand_num < switch_schedule["switch_hand"]
                        else switch_schedule["true_family_post"]
                    )
                    if switching_player == 0:
                        agent0.set_behavior_family(active_family)
                    else:
                        agent1.set_behavior_family(active_family)

                hand_id = generate_hand_id(run_id, hand_num)
                hand_rng = np.random.default_rng(seed * 100000 + hand_num)

                try:
                    hand_record = run_hand(
                        agent0=agent0,
                        agent1=agent1,
                        rng=hand_rng,
                        config=config,
                        opp_family_0=model_family_0 or "balanced",
                        opp_family_1=model_family_1 or "balanced",
                        hand_id=hand_id,
                        experiment_id=experiment_id,
                        seed=seed,
                        run_id=run_id,
                        hand_num=hand_num,
                    )
                    hand_record["matchup_label"] = matchup_label
                    hand_record["agent0_type"] = agent0_cfg.get("name", "unknown")
                    hand_record["agent1_type"] = agent1_cfg.get("name", "unknown")
                    hand_record["behavior_family_0"] = agent0.get_behavior_family()
                    hand_record["behavior_family_1"] = agent1.get_behavior_family()
                    hand_record["model_family_0"] = agent0.get_model_family() or model_family_0
                    hand_record["model_family_1"] = agent1.get_model_family() or model_family_1

                    if switch_schedule is not None:
                        hand_record["true_family_pre"] = switch_schedule["true_family_pre"]
                        hand_record["true_family_post"] = switch_schedule["true_family_post"]
                        hand_record["switch_hand"] = switch_schedule["switch_hand"]
                        hand_record["phase"] = (
                            "pre_switch"
                            if hand_num < switch_schedule["switch_hand"]
                            else "post_switch"
                        )

                    detection_state = agent0.get_detection_state() or agent1.get_detection_state()
                    if detection_state:
                        hand_record["detected_family"] = detection_state.get("detected_family")
                        hand_record["detection_hand"] = detection_state.get("detection_hand")
                        hand_record["responder_family_post"] = detection_state.get("current_behavior_family")
                        hand_record["responder_switched"] = detection_state.get("switched")
                        hand_record["observations_seen"] = detection_state.get("observations_seen")

                    seed_records.append(hand_record)
                    all_records.append(_flatten_record(hand_record))
                except Exception as exc:
                    if verbose:
                        print(f"  Warning: hand {hand_id} failed: {exc}")

                pbar.update(1)

            raw_path = os.path.join(
                output_dir,
                "raw_runs",
                f"{experiment_id}_{matchup_label}_seed{seed}.json",
            )
            try:
                save_json(seed_records, raw_path)
            except Exception as exc:
                if verbose:
                    print(f"  Warning: could not save raw records: {exc}")

    pbar.close()

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    processed_path = os.path.join(output_dir, "processed", f"{experiment_id}_summary.csv")
    try:
        df.to_csv(processed_path, index=False)
    except Exception as exc:
        if verbose:
            print(f"  Warning: could not save processed summary: {exc}")

    return df


def _make_switch_schedule(
    seed: int,
    hands_per_seed: int,
    schedule_group: str,
    pre_family: str,
    switch_candidates: List[str],
) -> Dict[str, Any]:
    """Derive a deterministic hidden-switch schedule shared across paired matchups."""
    schedule_seed = seed_from_string(f"{schedule_group}:{seed}:{hands_per_seed}")
    rng = np.random.default_rng(schedule_seed)
    low = max(1, hands_per_seed // 4)
    high = max(low + 1, (3 * hands_per_seed) // 4)
    switch_hand = int(rng.integers(low, high))
    true_family_post = str(rng.choice(switch_candidates))
    return {
        "true_family_pre": pre_family,
        "true_family_post": true_family_post,
        "switch_hand": switch_hand,
    }


def _flatten_record(record: dict) -> dict:
    """Flatten a hand record for DataFrame storage."""
    flat = {
        "run_id": record.get("run_id", ""),
        "seed": record.get("seed", 0),
        "experiment_id": record.get("experiment_id", ""),
        "matchup_id": record.get("matchup_id", ""),
        "matchup_label": record.get("matchup_label", ""),
        "hand_id": record.get("hand_id", ""),
        "hand_num": record.get("hand_num", 0),
        "first_to_act": record.get("first_to_act", 0),
        "starting_commitment_0": record.get("starting_commitment_0", 0),
        "starting_commitment_1": record.get("starting_commitment_1", 0),
        "opponent_family_0": record.get("opponent_family_0", ""),
        "opponent_family_1": record.get("opponent_family_1", ""),
        "behavior_family_0": record.get("behavior_family_0"),
        "behavior_family_1": record.get("behavior_family_1"),
        "model_family_0": record.get("model_family_0"),
        "model_family_1": record.get("model_family_1"),
        "true_family_pre": record.get("true_family_pre"),
        "true_family_post": record.get("true_family_post"),
        "switch_hand": record.get("switch_hand"),
        "phase": record.get("phase"),
        "detected_family": record.get("detected_family"),
        "detection_hand": record.get("detection_hand"),
        "responder_family_post": record.get("responder_family_post"),
        "responder_switched": record.get("responder_switched"),
        "observations_seen": record.get("observations_seen"),
        "agent0_type": record.get("agent0_type", ""),
        "agent1_type": record.get("agent1_type", ""),
        "board": " ".join(record.get("board", [])),
        "hole_cards_0": " ".join(record.get("hole_cards_0", [])),
        "hole_cards_1": " ".join(record.get("hole_cards_1", [])),
        "terminal_reward_0": record.get("terminal_reward_0", 0),
        "terminal_reward_1": record.get("terminal_reward_1", 0),
        "final_stack_0": record.get("final_stack_0", 0),
        "final_stack_1": record.get("final_stack_1", 0),
        "showdown_winner": record.get("showdown_winner"),
        "realized_hand_class_0": record.get("realized_hand_class_0", ""),
        "realized_hand_class_1": record.get("realized_hand_class_1", ""),
        "final_pot": record.get("final_pot", 0),
        "n_actions": len(record.get("action_history", [])),
    }
    return flat

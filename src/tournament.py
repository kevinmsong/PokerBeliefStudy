"""
tournament.py — Run many hands across seeds and matchups.
"""
import os
import json
import time
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.simulate import run_hand
from src.agents.base import BaseAgent
from src.agents.heuristic import HeuristicAgent
from src.agents.ev_static import StaticEVAgent
from src.agents.ev_belief import BeliefEVAgent
from src.response_model import make_default_response_model
from src.utils import save_json, generate_run_id, generate_hand_id


def make_agent(
    agent_config: dict,
    player_idx: int,
    rng: np.random.Generator,
    opp_family: str = "balanced",
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

    elif agent_type == "StaticEVAgent":
        n_rollouts = agent_config.get("n_rollouts", 200)
        prior_type = agent_config.get("prior", "uniform")
        if prior_type == "uniform":
            prior = None  # Will use uniform default
        else:
            prior = None
        return StaticEVAgent(
            player_idx=player_idx,
            rng=rng,
            opp_family=opp_family,
            n_rollouts=n_rollouts,
            prior=prior,
            response_model=response_model,
            name=agent_config.get("label", "ev_static"),
        )

    elif agent_type == "BeliefEVAgent":
        n_rollouts = agent_config.get("n_rollouts", 200)
        smoothing = agent_config.get("smoothing", 0.05)
        prior_type = agent_config.get("prior", "uniform")
        if prior_type == "uniform":
            prior = None
        else:
            prior = None
        return BeliefEVAgent(
            player_idx=player_idx,
            rng=rng,
            opp_family=opp_family,
            n_rollouts=n_rollouts,
            prior=prior,
            smoothing=smoothing,
            response_model=response_model,
            name=agent_config.get("label", "ev_belief"),
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
    """Run many hands across seeds and matchups.

    Parameters
    ----------
    matchups : List[Dict]
        List of matchup configurations.
        Each matchup has: agent0, agent1, opp_family_0, opp_family_1, label.
    seeds : List[int]
        Seeds to run.
    hands_per_seed : int
        Number of hands per seed.
    config : dict
        Game configuration.
    output_dir : str
        Directory to save results.
    experiment_id : str
        Experiment identifier.
    agent_configs : dict, optional
        Agent type configurations.
    verbose : bool
        Whether to show progress bars.

    Returns
    -------
    pd.DataFrame
        Summary DataFrame with one row per hand.
    """
    os.makedirs(os.path.join(output_dir, "raw_runs"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "processed"), exist_ok=True)

    all_records = []
    response_model = make_default_response_model()

    total_hands = len(matchups) * len(seeds) * hands_per_seed
    pbar = tqdm(total=total_hands, desc=f"Tournament [{experiment_id}]", disable=not verbose)

    for matchup in matchups:
        matchup_label = matchup.get("label", "matchup")
        agent0_cfg = matchup.get("agent0", {"name": "HeuristicAgent"})
        agent1_cfg = matchup.get("agent1", {"name": "HeuristicAgent"})
        opp_family_0 = matchup.get("opp_family_0", "balanced")
        opp_family_1 = matchup.get("opp_family_1", "balanced")

        for seed in seeds:
            run_id = generate_run_id(f"{experiment_id}_{matchup_label}", seed)
            rng = np.random.default_rng(seed)

            # Create agents with this seed's RNG
            # Split RNG for two agents to avoid correlation
            rng0 = np.random.default_rng(seed * 2)
            rng1 = np.random.default_rng(seed * 2 + 1)
            rng_deal = np.random.default_rng(seed * 3)

            agent0 = make_agent(agent0_cfg, player_idx=0, rng=rng0, opp_family=opp_family_0)
            agent1 = make_agent(agent1_cfg, player_idx=1, rng=rng1, opp_family=opp_family_1)

            seed_records = []

            for hand_num in range(hands_per_seed):
                hand_id = generate_hand_id(run_id, hand_num)
                hand_rng = np.random.default_rng(seed * 100000 + hand_num)

                try:
                    hand_record = run_hand(
                        agent0=agent0,
                        agent1=agent1,
                        rng=hand_rng,
                        config=config,
                        opp_family_0=opp_family_0,
                        opp_family_1=opp_family_1,
                        hand_id=hand_id,
                        experiment_id=experiment_id,
                        seed=seed,
                        run_id=run_id,
                    )

                    # Add matchup info
                    hand_record["matchup_label"] = matchup_label
                    hand_record["agent0_type"] = agent0_cfg.get("name", "unknown")
                    hand_record["agent1_type"] = agent1_cfg.get("name", "unknown")
                    hand_record["hand_num"] = hand_num

                    seed_records.append(hand_record)
                    all_records.append(_flatten_record(hand_record))

                except Exception as e:
                    if verbose:
                        print(f"  Warning: hand {hand_id} failed: {e}")

                pbar.update(1)

            # Save raw records for this seed/matchup
            raw_path = os.path.join(
                output_dir, "raw_runs",
                f"{experiment_id}_{matchup_label}_seed{seed}.json"
            )
            try:
                save_json(seed_records, raw_path)
            except Exception as e:
                if verbose:
                    print(f"  Warning: could not save raw records: {e}")

    pbar.close()

    # Create summary DataFrame
    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)

    # Save processed summary
    processed_path = os.path.join(output_dir, "processed", f"{experiment_id}_summary.csv")
    try:
        df.to_csv(processed_path, index=False)
    except Exception as e:
        if verbose:
            print(f"  Warning: could not save processed summary: {e}")

    return df


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
        "opponent_family_0": record.get("opponent_family_0", ""),
        "opponent_family_1": record.get("opponent_family_1", ""),
        "agent0_type": record.get("agent0_type", ""),
        "agent1_type": record.get("agent1_type", ""),
        "board": " ".join(record.get("board", [])),
        "hole_cards_0": " ".join(record.get("hole_cards_0", [])),
        "hole_cards_1": " ".join(record.get("hole_cards_1", [])),
        "terminal_reward_0": record.get("terminal_reward_0", 0),
        "terminal_reward_1": record.get("terminal_reward_1", 0),
        "showdown_winner": record.get("showdown_winner"),
        "realized_hand_class_0": record.get("realized_hand_class_0", ""),
        "realized_hand_class_1": record.get("realized_hand_class_1", ""),
        "final_pot": record.get("final_pot", 0),
        "n_actions": len(record.get("action_history", [])),
    }
    return flat

# PokerBeliefStudy

A research-grade Python repository studying Bayesian belief updating in heads-up poker, with a paired hidden-switch case study for adaptive response.

## Project Overview

PokerBeliefStudy investigates whether an agent that maintains and updates a Bayesian belief distribution over opponent hand classes outperforms agents that use fixed strategies or static priors. The main study now covers exactly 100,000 simulated hands across four experiments, plus a separate 100,000-hand hidden-switch case study. The core comparison includes three study agents:

1. **HeuristicAgent** — Fixed rule-based agent using hand class buckets
2. **StaticEVAgent** — Expected value maximizer with a fixed (non-updating) prior over opponent hands
3. **BeliefEVAgent** — Expected value maximizer with Bayesian posterior that updates after each observed opponent action

## Setup Instructions

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
# Clone or download the repository
cd Poker_Belief_Study

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode (optional)
pip install -e .
```

### Verify Installation

```bash
# Run tests
python -m pytest tests/ -v

# Quick smoke test (10 hands)
python run_experiment.py --experiment calibration --seeds 42 --hands 10 --output outputs/
```

## Usage Examples

### Running Experiments

```bash
# Main comparison experiment (all three agents vs family-conditioned opponents)
python run_experiment.py --experiment main_comparison --seeds 42,43,44,45,46 --hands 1000 --output outputs/

# Calibration analysis (belief agent only)
python run_experiment.py --experiment calibration --seeds 42,43,44,45,46 --hands 650 --output outputs/

# Robustness to held-out opponents
python run_experiment.py --experiment robustness --seeds 42,43,44,45,46 --hands 1000 --output outputs/

# Belief updating ablation study
python run_experiment.py --experiment belief_ablation --seeds 42,43,44,45,46 --hands 1000 --output outputs/

# Separate 100,000-hand hidden-switch case study
python run_experiment.py --experiment switch_case_study --seeds 42,43,44,45,46,47,48,49,50,51 --hands 5000 --output outputs/
```

### CLI Options

```
python run_experiment.py --help

Options:
  --experiment, -e   Experiment name (see config/experiments.yaml)
  --seeds, -s        Comma-separated random seeds (default: 42)
  --hands, -n        Hands per seed (default: 100)
  --output, -o       Output directory (default: outputs/)
  --config-dir, -c   Config directory (default: config)
  --quiet, -q        Suppress progress output
  --dry-run          Print plan without running
```

### Using the Python API

```python
import numpy as np
from src.agents.ev_belief import BeliefEVAgent
from src.agents.family_policy import FamilyPolicyAgent
from src.simulate import run_hand

# Create agents
rng0 = np.random.default_rng(42)
rng1 = np.random.default_rng(43)
agent0 = BeliefEVAgent(player_idx=0, rng=rng0, opp_family="balanced")
agent1 = FamilyPolicyAgent(player_idx=1, rng=rng1, behavior_family="balanced")

# Run a single hand
config = {
    "starting_commitment_0": 50,
    "starting_commitment_1": 50,
    "effective_stack": 200,
    "street_start": "turn",
}
hand_rng = np.random.default_rng(99)
record = run_hand(agent0, agent1, hand_rng, config, opp_family_0="balanced", opp_family_1="balanced")

print(f"First to act: {record['first_to_act']}")
print(f"Board: {record['board']}")
print(f"Net reward (agent0): {record['terminal_reward_0']}")
print(f"Hand class (agent0): {record['realized_hand_class_0']}")
```

## Experiment Descriptions

### `main_comparison`
Compares all three study agents (heuristic, ev_static, ev_belief) against balanced, aggressive, and passive family-conditioned opponents. This is the primary comparison for the paper.

**Seeds:** 42, 43, 44, 45, 46
**Hands/seed:** 1000
**Matchups:** 9 (3 agent types × 3 opponent families)
**Total hands:** 45,000

### `calibration`
Analyzes the calibration of the belief agent's posterior beliefs. Computes Brier score and Expected Calibration Error.

**Seeds:** 42, 43, 44, 45, 46
**Hands/seed:** 650 by default, with one 700-hand balanced matchup override
**Matchups:** 3 (belief agent vs balanced, aggressive, maniac opponents)
**Total hands:** 10,000

### `robustness`
Tests all three agents against held-out opponent families (held_out_1, held_out_2) not used during development. Measures generalization.

**Seeds:** 42, 43, 44, 45, 46
**Hands/seed:** 1000
**Matchups:** 6 (3 agents × 2 held-out families)
**Total hands:** 30,000

### `belief_ablation`
Direct comparison of BeliefEVAgent vs StaticEVAgent to isolate the value of belief updating. Tests on balanced, aggressive, and trappy opponents.

**Seeds:** 42, 43, 44, 45, 46
**Hands/seed:** 1000
**Matchups:** 3 (belief vs static × 3 opponent types)
**Total hands:** 15,000

### `switch_case_study`
Runs a separate hidden-switch case study in which one initially balanced family-policy opponent switches once to a sampled family, while an adaptive responder is compared against a balanced control.

**Seeds:** 42, 43, 44, 45, 46, 47, 48, 49, 50, 51
**Hands/seed:** 5000
**Matchups:** 2 (adaptive counter vs hidden switch, balanced control vs hidden switch)
**Total hands:** 100,000

The four main experiments total exactly 100,000 hands. The separate hidden-switch study also totals exactly 100,000 hands.

## Output Structure

```
outputs/
├── raw_runs/          # Raw JSON hand records per seed/matchup
├── processed/         # Aggregated CSV summaries
├── figures/           # 300 DPI publication-ready figures
│   ├── *_performance.png      # Bar chart with CI error bars
│   ├── *_robustness.png       # Robustness heatmap
│   ├── calibration_reliability.png  # Reliability diagram
│   └── belief_trace_example.png     # Belief evolution trace
└── tables/            # LaTeX and CSV tables for manuscript
    ├── *_performance.csv/.tex
    └── *_robustness.csv/.tex
```

## Hand Classes

The system classifies each hand into one of 7 mutually exclusive, exhaustive classes:

| Class | Description |
|-------|-------------|
| `nuts_or_near_nuts` | Top ~5% of board-relative hands (sets+, nut flush, straights) |
| `strong_made` | Top pair strong kicker, two pair, overpair, non-nut flush |
| `medium_made` | Top pair weak kicker, second pair strong kicker, small overpair |
| `weak_showdown` | Bottom pair, third pair, ace high |
| `strong_draw` | Nut flush draw, OESD, combo draw |
| `weak_draw` | Gutshot, weak flush draw |
| `air` | No pair, no draw, no showdown value |

## Opponent Families

Nine development families + two held-out families:

| Family | Key Characteristics |
|--------|---------------------|
| `balanced` | Moderate all parameters |
| `aggressive` | High aggression, high bluff rate |
| `passive` | Low aggression, high call looseness |
| `tight` | Low bluff rate, low call looseness |
| `loose` | High call looseness, moderate bluff |
| `maniac` | Very high aggression, very high bluff |
| `trappy` | High trap frequency, low initiative betting |
| `overbluffer` | Very high bluff rate |
| `underbluffer` | Very low bluff rate |
| `held_out_1` | Novel combination (held-out testing) |
| `held_out_2` | Different novel combination (held-out testing) |

## Architecture

### Core Modules

- `src/cards.py` — Card representation and deck operations
- `src/hand_eval.py` — 7-card hand evaluator returning comparable tuples
- `src/hand_classes.py` — Classify hands into 7 classes
- `src/state.py` — Game state, legal actions, action application
- `src/infoset.py` — Extract information set (no hidden state leakage)
- `src/beliefs.py` — Bayesian belief state with Bayes' rule updating
- `src/response_model.py` — P(action | hand_class, state, opponent_family)
- `src/equity.py` — Monte Carlo equity estimation
- `src/simulate.py` — Single hand simulation
- `src/tournament.py` — Multi-hand tournament runner
- `src/analysis.py` — Metrics and 300 DPI figure generation

### Agent Types

- `src/agents/heuristic.py` — Fixed-rule policy
- `src/agents/ev_static.py` — EV maximizer with static prior
- `src/agents/ev_belief.py` — EV maximizer with Bayesian updating
- `src/agents/family_policy.py` — Stochastic family-conditioned policy agent
- `src/agents/adaptive_counter.py` — Adaptive responder for the hidden-switch case study

## Reproducibility

All randomness is seeded through explicit `numpy.random.Generator` objects. The same seed always produces the same hand sequence and agent decisions.

```python
# Verify reproducibility
from src.simulate import run_hand
from src.agents.heuristic import HeuristicAgent
import numpy as np

def run_with_seed(seed):
    a0 = HeuristicAgent(0, np.random.default_rng(seed*2))
    a1 = HeuristicAgent(1, np.random.default_rng(seed*2+1))
    return run_hand(a0, a1, np.random.default_rng(seed*3),
                    {"starting_pot":100,"effective_stack":200,"street_start":"turn"})

r1 = run_with_seed(42)
r2 = run_with_seed(42)
assert r1["board"] == r2["board"]  # Always True
```

## Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Specific test file
python -m pytest tests/test_hand_eval.py -v

# Quick test
python -m pytest tests/ -x --tb=short
```

## License

MIT License — see LICENSE file for details.

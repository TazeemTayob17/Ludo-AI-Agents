# Stage 1 scoring script: runs the Expert or rollout agent (or a reference) as one seat against three identical opponents.

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from agents.expert_agent import ExpertAgent
from agents.heuristic_agent import HeuristicAgent
from agents.random_agent import RandomAgent
from agents.rollout_agent import RolloutAgent
from training.run_full_evaluation import TOURNAMENT_SEED_OFFSET
from training.statistics import wilson_confidence_interval
from training.tournament import TournamentResult, run_tournament, wrap_choose_action_fn

AGENT_NAMES = ("expert", "rollout", "heuristic", "random")
OPPONENT_NAMES = ("heuristic", "random", "expert")


# Builds a policy by name; rollout takes its rollout count and dice seed.
def _make_policy(name: str, num_rollouts: int, seed: int):
    if name == "expert":
        return ExpertAgent()
    if name == "rollout":
        return RolloutAgent(num_rollouts=num_rollouts, seed=seed)
    if name == "heuristic":
        return HeuristicAgent()
    return RandomAgent(rng=np.random.default_rng(seed))


# Plays one chunk of games on its own dice seed; top-level so worker processes can import it.
def _score_chunk(agent: str, opponent: str, num_games: int, num_rollouts: int, seed: int) -> TournamentResult:
    return run_tournament(
        agent,
        wrap_choose_action_fn(_make_policy(agent, num_rollouts, seed)),
        _make_policy(opponent, num_rollouts, seed),
        num_games,
        seed=seed,
    )


# Scores the agent over num_games; one worker reproduces the paired tournament dice seed exactly, more workers split the games into chunks.
def score(agent: str, opponent: str, num_games: int, num_rollouts: int, workers: int) -> TournamentResult:
    if workers <= 1:
        return _score_chunk(agent, opponent, num_games, num_rollouts, TOURNAMENT_SEED_OFFSET)
    sizes = [num_games // workers + (1 if i < num_games % workers else 0) for i in range(workers)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_score_chunk, agent, opponent, size, num_rollouts, TOURNAMENT_SEED_OFFSET + i)
            for i, size in enumerate(sizes)
            if size > 0
        ]
        parts = [f.result() for f in futures]
    return TournamentResult(
        agent_label=agent,
        num_episodes=sum(p.num_episodes for p in parts),
        wins=sum(p.wins for p in parts),
        total_episode_length=sum(p.total_episode_length for p in parts),
        total_captures_made=sum(p.total_captures_made for p in parts),
        total_times_captured=sum(p.total_times_captured for p in parts),
    )


# Parses arguments, scores the agent, prints the win rate with a 95% interval, and appends a row to the CSV.
def _main() -> None:
    parser = argparse.ArgumentParser(description="Score one baseline agent against three identical opponents.")
    parser.add_argument("--agent", choices=AGENT_NAMES, required=True)
    parser.add_argument("--opponent", choices=OPPONENT_NAMES, default="heuristic")
    parser.add_argument("--games", type=int, default=2000)
    parser.add_argument("--rollouts", type=int, default=16, help="rollouts per candidate move (rollout agent only)")
    parser.add_argument("--workers", type=int, default=1, help="processes; 1 keeps the paired dice seed")
    parser.add_argument("--out", type=Path, default=Path("runs/stage1/baseline_scores.csv"))
    args = parser.parse_args()

    result = score(args.agent, args.opponent, args.games, args.rollouts, args.workers)
    interval = wilson_confidence_interval(result.wins, result.num_episodes)
    low, high = interval.lower, interval.upper
    print(
        f"{args.agent} vs 3 {args.opponent}: {result.wins}/{result.num_episodes} = {result.win_rate:.2%} "
        f"(95% CI [{low:.2%}, {high:.2%}]), avg length {result.avg_episode_length:.1f}"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    is_new = not args.out.exists()
    with args.out.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if is_new:
            writer.writerow(["agent", "opponent", "games", "rollouts", "workers", "wins", "win_rate", "ci_low", "ci_high"])
        writer.writerow(
            [args.agent, args.opponent, result.num_episodes, args.rollouts, args.workers, result.wins,
             f"{result.win_rate:.4f}", f"{low:.4f}", f"{high:.4f}"]
        )


if __name__ == "__main__":
    _main()

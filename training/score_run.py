# One-command scoring of trained runs: paired 2,000-game tournament, Wilson interval, difference from a baseline and the +3pp promotion rule.

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from agents.expert_agent import ExpertAgent
from agents.heuristic_agent import HeuristicAgent
from agents.random_agent import RandomAgent
from agents.third_party.alhajry_expert import AlhajryExpert
from training.run_full_evaluation import TOURNAMENT_SEED_OFFSET, load_trained_policy, run_observation_spec
from training.statistics import t_confidence_interval_across_seeds, wilson_confidence_interval
from training.tournament import TournamentResult, run_tournament


BASELINE_WIN_RATE = 0.3185
PROMOTION_MARGIN = 0.03
NOISE_BAND = 0.03


# Builds the opponent policy for the three other seats.
def _opponent(name: str):
    if name == "random":
        return RandomAgent(rng=np.random.default_rng(TOURNAMENT_SEED_OFFSET + 1))
    if name == "alhajry":
        return AlhajryExpert()
    return ExpertAgent() if name == "expert" else HeuristicAgent()


# Returns the verdict wording for a difference from the baseline under the project's decision rules.
def verdict(difference: float) -> str:
    if difference >= PROMOTION_MARGIN:
        return "PROMOTE: confirm on seeds 1 and 2"
    if difference <= -PROMOTION_MARGIN:
        return "REJECT: clearly below the baseline"
    return "no detectable change (within +/-3pp)"


# Plays the games in one seat, or spread evenly over all four seats (each on its own dice stream) when rotate_seats is set.
def _score_run(run: Path, opponent: str, games: int, rotate_seats: bool) -> TournamentResult:
    policy, spec = load_trained_policy(run), run_observation_spec(run)
    if not rotate_seats:
        return run_tournament(run.name, policy, _opponent(opponent), games, seed=TOURNAMENT_SEED_OFFSET, observation_spec=spec)
    parts = [
        run_tournament(run.name, policy, _opponent(opponent), games // 4, seed=TOURNAMENT_SEED_OFFSET + seat,
                       observation_spec=spec, agent_player_id=seat)
        for seat in range(4)
    ]
    return TournamentResult(
        agent_label=run.name,
        num_episodes=sum(p.num_episodes for p in parts),
        wins=sum(p.wins for p in parts),
        total_episode_length=sum(p.total_episode_length for p in parts),
        total_captures_made=sum(p.total_captures_made for p in parts),
        total_times_captured=sum(p.total_times_captured for p in parts),
    )


# Scores each run directory's final checkpoint, prints one line per run and appends the rows to the CSV.
def _main() -> None:
    parser = argparse.ArgumentParser(description="Score trained runs against three identical opponents on the paired dice seed.")
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--opponent", choices=["heuristic", "random", "expert", "alhajry"], default="heuristic")
    parser.add_argument("--games", type=int, default=2000)
    parser.add_argument("--rotate-seats", action="store_true", help="split the games evenly over the four seats")
    parser.add_argument("--baseline", type=float, default=BASELINE_WIN_RATE, help="win rate to compare against")
    parser.add_argument("--out", type=Path, default=Path("runs/pilot_scores.csv"))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    is_new = not args.out.exists()
    with args.out.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if is_new:
            writer.writerow(["run", "opponent", "games", "wins", "win_rate", "ci_low", "ci_high", "diff_vs_baseline", "verdict"])
        rates = []
        for run in args.runs:
            result = _score_run(run, args.opponent, args.games, args.rotate_seats)
            rates.append(result.win_rate)
            interval = wilson_confidence_interval(result.wins, result.num_episodes)
            difference = result.win_rate - args.baseline
            text = verdict(difference)
            print(
                f"{run.name} vs 3 {args.opponent}: {result.wins}/{result.num_episodes} = {result.win_rate:.2%} "
                f"(95% CI [{interval.lower:.2%}, {interval.upper:.2%}]); {difference * 100:+.2f}pp vs {args.baseline:.2%}: {text}"
            )
            writer.writerow([run.name, args.opponent, result.num_episodes, result.wins, f"{result.win_rate:.4f}",
                             f"{interval.lower:.4f}", f"{interval.upper:.4f}", f"{difference:.4f}", text])
            handle.flush()
    if len(rates) > 1:
        interval = t_confidence_interval_across_seeds(rates)
        print(f"across {len(rates)} runs: mean {np.mean(rates):.2%}, SD {np.std(rates, ddof=1) * 100:.2f}pp, "
              f"95% t-interval [{interval.lower:.2%}, {interval.upper:.2%}]")


if __name__ == "__main__":
    _main()

# Error analysis for one agent seat vs 3 Heuristic: agreement with the Heuristic and a ranked report of the agent's mistakes over many games.

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from agents.expert_agent import ExpertAgent
from agents.heuristic_agent import HeuristicAgent, _classify_token
from env.board import is_on_main_track, relative_to_global
from env.legal_moves import get_legal_tokens
from env.ludo_env import LudoEnv
from env.moves import apply_move
from env.rewards import progress_potential
from env.state_encoding import ObservationSpec
from env.threats import capturing_rolls
from training.episode_stats import agent_won
from training.run_full_evaluation import TOURNAMENT_SEED_OFFSET, load_trained_policy, run_observation_spec
from training.tournament import wrap_choose_action_fn

LEAD_THRESHOLD = 0.10
BEHIND_THRESHOLD = -0.10


# Returns how many distinct rolls would let an opponent capture the token after the given move (0 if it is off the shared track).
def risk_after(board, player_id: int, token_id: int, roll: int) -> int:
    after = board.copy()
    apply_move(after, player_id, token_id, roll)
    position = after.get(player_id, token_id)
    if not is_on_main_track(position):
        return 0
    return len(capturing_rolls(after, player_id, relative_to_global(player_id, position)))


# Plays num_games of the agent as seat 0 against 3 Heuristic and tallies every decision and game-level pattern.
def play_diagnostic_games(select_action, spec: ObservationSpec, num_games: int, seed: int = TOURNAMENT_SEED_OFFSET) -> dict:
    heuristic = HeuristicAgent()
    env = LudoEnv(opponent_policy=heuristic, rng=np.random.default_rng(seed), observation_spec=spec)
    counts: Counter = Counter()
    swaps: Counter = Counter()
    wins = lost_while_leading = lost_games = 0
    first_behind_fractions: list[float] = []
    captures = captured = 0

    for _ in range(num_games):
        obs, info = env.reset()
        terminated, truncated = env.game.terminated, env.game.truncated
        potentials: list[float] = []
        while not (terminated or truncated):
            board, roll = env.game.board, info["dice_roll"]
            legal = get_legal_tokens(board, 0, roll)
            action = select_action(env, obs, info)
            chosen = env.token_for_action(action)
            potentials.append(progress_potential(board, 0))
            counts["decisions"] += 1
            if len(legal) > 1:
                counts["multi"] += 1
                classes = {t: _classify_token(board, 0, t, roll) for t in legal}
                reference = heuristic(board, 0, roll, legal)
                if chosen == reference:
                    counts["agree"] += 1
                else:
                    swaps[(classes[reference], classes[chosen])] += 1
                risks = {t: risk_after(board, 0, t, roll) for t in legal}
                if risks[chosen] > min(risks.values()):
                    counts["riskier_than_available"] += 1
                if "capture" in classes.values():
                    counts["capture_chances"] += 1
                    counts["skipped_capture"] += classes[chosen] != "capture"
                if "exit_base" in classes.values() and "capture" not in classes.values():
                    counts["exit_chances"] += 1
                    counts["skipped_exit"] += classes[chosen] != "exit_base"
                if "enter_home" in classes.values() and not {"capture", "exit_base"} & set(classes.values()):
                    counts["home_chances"] += 1
                    counts["skipped_home"] += classes[chosen] != "enter_home"
            obs, reward, terminated, truncated, info = env.step(action)
        won = agent_won(terminated, info, 0)
        wins += won
        captures += info["episode_captures_made"]
        captured += info["episode_times_captured"]
        if not won:
            lost_games += 1
            lost_while_leading += bool(potentials) and max(potentials) > LEAD_THRESHOLD
            behind = [i for i, p in enumerate(potentials) if p < BEHIND_THRESHOLD]
            if behind:
                first_behind_fractions.append(behind[0] / len(potentials))
    return {
        "games": num_games, "wins": wins, "counts": dict(counts), "swaps": {f"{a}->{b}": n for (a, b), n in swaps.items()},
        "lost_games": lost_games, "lost_while_leading": lost_while_leading,
        "mean_first_behind_fraction": float(np.mean(first_behind_fractions)) if first_behind_fractions else None,
        "captures_made": captures, "times_captured": captured,
    }


# Turns the tallies into per-game rates and a table of mistake categories ranked by mistakes per game.
def build_report(tallies: dict) -> dict:
    counts, games = tallies["counts"], tallies["games"]
    multi = max(counts.get("multi", 0), 1)
    categories = []
    for name, mistakes_key, chances_key in (
        ("skipped a capture", "skipped_capture", "capture_chances"),
        ("skipped entering home", "skipped_home", "home_chances"),
        ("skipped leaving base on a 6", "skipped_exit", "exit_chances"),
        ("chose a riskier move than available", "riskier_than_available", "multi"),
    ):
        mistakes, chances = counts.get(mistakes_key, 0), max(counts.get(chances_key, 0), 1)
        categories.append({"mistake": name, "per_game": mistakes / games, "rate_when_available": mistakes / chances})
    categories.append({"mistake": "own token captured", "per_game": tallies["times_captured"] / games, "rate_when_available": None})
    categories.append(
        {"mistake": "lost a game after leading the race", "per_game": tallies["lost_while_leading"] / games,
         "rate_when_available": tallies["lost_while_leading"] / max(tallies["lost_games"], 1)}
    )
    categories.sort(key=lambda row: row["per_game"], reverse=True)
    return {
        "win_rate": tallies["wins"] / games,
        "decisions_per_game": counts.get("decisions", 0) / games,
        "agreement_with_heuristic": counts.get("agree", 0) / multi,
        "mean_first_behind_fraction_in_lost_games": tallies["mean_first_behind_fraction"],
        "capture_to_death_ratio": tallies["captures_made"] / max(tallies["times_captured"], 1),
        "ranked_mistakes": categories,
        "top_swaps_from_heuristic": dict(sorted(tallies["swaps"].items(), key=lambda kv: -kv[1])[:10]),
    }


# Prints the report as a short readable table.
def print_report(label: str, report: dict) -> None:
    print(f"== {label}: win rate {report['win_rate']:.2%}, agreement with Heuristic {report['agreement_with_heuristic']:.1%}, "
          f"capture/death {report['capture_to_death_ratio']:.2f}, {report['decisions_per_game']:.0f} decisions per game")
    for row in report["ranked_mistakes"]:
        rate = "" if row["rate_when_available"] is None else f"  ({row['rate_when_available']:.1%} of chances)"
        print(f"   {row['per_game']:7.3f} per game  {row['mistake']}{rate}")
    behind = report["mean_first_behind_fraction_in_lost_games"]
    if behind is not None:
        print(f"   lost games: the agent first fell clearly behind {behind:.0%} of the way through the game")
    print("   biggest departures from the Heuristic:", report["top_swaps_from_heuristic"])


# Parses --run or --baseline, plays the games, prints the report and optionally saves it as JSON.
def _main() -> None:
    parser = argparse.ArgumentParser(description="Mistake report for a trained run or a hand-made baseline.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", type=Path, help="a training run directory with config.json and checkpoints")
    group.add_argument("--baseline", choices=["heuristic", "expert"])
    parser.add_argument("--games", type=int, default=300)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.run:
        select, spec, label = load_trained_policy(args.run), run_observation_spec(args.run), args.run.name
    else:
        policy = HeuristicAgent() if args.baseline == "heuristic" else ExpertAgent()
        select, spec, label = wrap_choose_action_fn(policy), ObservationSpec(), args.baseline
    report = build_report(play_diagnostic_games(select, spec, args.games))
    print_report(label, report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    _main()

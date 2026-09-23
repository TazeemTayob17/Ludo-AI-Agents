# Step 11's main script: the large-scale tournament (11.2/11.3) for every Step 10 checkpoint
# plus Random/Heuristic reference policies, and the reward-shaping ablation (11.4). Reads
# already-trained checkpoints from runs/full/final/ - does not retrain Step 10's agents.
# Everything is written to CSV (11.5), never left only as terminal output.

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from agents.dqn_agent import DQNAgent
from agents.heuristic_agent import HeuristicAgent
from agents.random_agent import RandomAgent
from agents.tabular_q_agent import TabularQAgent
from training.checkpoint import load_dqn_checkpoint, load_tabular_checkpoint
from training.config import TrainingConfig, load_config
from training.run_training import run_training
from training.statistics import (
    holm_adjusted_p_values,
    mean_and_std,
    t_confidence_interval_across_seeds,
    two_proportion_z_test,
    wilson_confidence_interval,
)
from training.tournament import GreedyPolicy, TournamentResult, run_tournament, wrap_choose_action_fn

DEFAULT_TOURNAMENT_EPISODES = 2000
DEFAULT_ABLATION_EPISODES = 10000
DEFAULT_ABLATION_SEEDS = 5
DEFAULT_ABLATION_EVAL_EPISODES = 50
# Every tournament uses this one dice seed, so all agents face the identical sequence of games
# and their win-rate differences are policy differences rather than differences in dice luck.
TOURNAMENT_SEED_OFFSET = 100000

AGENT_TYPES = ("dqn", "double_dqn", "tabular_q")
NUM_TRAINED_SEEDS = 5

# Where _log mirrors its output, set once run_full_evaluation knows its output directory.
_progress_log_path: Path | None = None

# Prints a timestamped progress line and mirrors it to progress.log. This run takes hours with
# no other feedback, so silence must be distinguishable from a hang, both live and afterwards.
def _log(message: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
    print(line, flush=True)
    if _progress_log_path is not None:
        with open(_progress_log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

# Rebuilds a trained agent from its run directory's saved best checkpoint and wraps it as
# a greedy (epsilon=0) policy, matching the GreedyPolicy interface run_tournament expects.
def load_trained_policy(run_dir: Path) -> GreedyPolicy:
    config = load_config(run_dir / "config.json")
    extension = "pt" if config.agent_type in ("dqn", "double_dqn") else "pkl"
    checkpoint_path = run_dir / "checkpoints" / f"best.{extension}"

    if config.agent_type in ("dqn", "double_dqn"):
        agent = DQNAgent(
            double_dqn=(config.agent_type == "double_dqn"),
            hidden_size=config.hidden_size,
            include_dice_roll=config.include_dice_roll,
        )
        load_dqn_checkpoint(checkpoint_path, agent)
        agent.epsilon = 0.0

        def select(env, obs, info):
            return agent.select_action(obs, info["action_mask"])

        return select

    agent = TabularQAgent()
    load_tabular_checkpoint(checkpoint_path, agent)
    agent.epsilon = 0.0
    return wrap_choose_action_fn(agent)

# Reads back whether a run's agent expects the dice roll appended to its observation, so the
# tournament env is built to match the checkpoint it is about to evaluate.
def run_uses_dice_roll(run_dir: Path) -> bool:
    return load_config(run_dir / "config.json").include_dice_roll

# Runs the large-scale tournament (11.2) for every trained final run plus the Random and
# Heuristic reference policies (one seat vs. 3 Heuristic opponents throughout, matching
# Step 10's evaluation setup), and returns one TournamentResult per label.
def run_all_tournaments(final_root: Path, num_episodes: int) -> dict[str, TournamentResult]:
    results: dict[str, TournamentResult] = {}
    total = len(AGENT_TYPES) * NUM_TRAINED_SEEDS + 2
    _log(f"tournaments: {total} agents x {num_episodes} games")

    for agent_type in AGENT_TYPES:
        for seed in range(NUM_TRAINED_SEEDS):
            run_dir = final_root / f"{agent_type}_seed{seed}"
            label = f"{agent_type}_seed{seed}"
            _log(f"  [{len(results) + 1}/{total}] {label}")
            policy = load_trained_policy(run_dir)
            results[label] = run_tournament(
                label,
                policy,
                HeuristicAgent(),
                num_episodes,
                seed=TOURNAMENT_SEED_OFFSET,
                include_dice_roll=run_uses_dice_roll(run_dir),
            )
            _log(f"      win rate {results[label].win_rate:.2%}")

    _log(f"  [{total - 1}/{total}] random_baseline")
    results["random_baseline"] = run_tournament(
        "random_baseline",
        wrap_choose_action_fn(RandomAgent(rng=np.random.default_rng(TOURNAMENT_SEED_OFFSET))),
        HeuristicAgent(),
        num_episodes,
        seed=TOURNAMENT_SEED_OFFSET,
    )
    # A Heuristic agent evaluated the same way (one seat vs. 3 Heuristic opponents) gives a
    # reference point for "what does a fairly-matched competitor score in this same setup" -
    # useful context since the flat 25% no-skill baseline doesn't apply against non-random opponents.
    _log(f"      win rate {results['random_baseline'].win_rate:.2%}")
    _log(f"  [{total}/{total}] heuristic_mirror")
    results["heuristic_mirror"] = run_tournament(
        "heuristic_mirror",
        wrap_choose_action_fn(HeuristicAgent()),
        HeuristicAgent(),
        num_episodes,
        seed=TOURNAMENT_SEED_OFFSET,
    )
    _log(f"      win rate {results['heuristic_mirror'].win_rate:.2%}")
    return results

# Trains num_seeds fresh DQN runs with SPARSE_REWARD_CONFIG instead of the dense default,
# reusing the sweep's winning hyperparameters, for Step 11.4's reward-shaping ablation.
# Scoped to DQN only (not Double DQN/Tabular Q too) for the same hardware/time-budget
# reasoning already used to scope Step 10's sweep - see docs/decisions_log.md.
def run_reward_ablation_training(
    output_root: Path,
    winning_hyperparameters: dict,
    num_episodes: int,
    num_seeds: int,
    eval_episodes: int,
    gamma: float,
    include_dice_roll: bool,
) -> None:
    _log(f"reward ablation: retraining {num_seeds} sparse-reward DQN seeds x {num_episodes} episodes")
    for seed in range(num_seeds):
        _log(f"  [{seed + 1}/{num_seeds}] training dqn_sparse_seed{seed} (this is the slow part)")
        config = TrainingConfig(
            agent_type="dqn",
            seed=seed,
            num_episodes=num_episodes,
            eval_episodes=eval_episodes,
            run_name="dqn_sparse_final",
            reward_mode="sparse",
            gamma=gamma,
            include_dice_roll=include_dice_roll,
            learning_rate=winning_hyperparameters["learning_rate"],
            target_sync_every_steps=winning_hyperparameters["target_sync_every_steps"],
            epsilon_decay_episodes=winning_hyperparameters["epsilon_decay_episodes"],
        )
        run_training(config, output_root / f"dqn_sparse_seed{seed}")

# Runs the tournament for each of the sparse-reward ablation seeds just trained.
def run_ablation_tournaments(ablation_root: Path, num_seeds: int, num_episodes: int) -> dict[str, TournamentResult]:
    results = {}
    _log(f"ablation tournaments: {num_seeds} agents x {num_episodes} games")
    for seed in range(num_seeds):
        run_dir = ablation_root / f"dqn_sparse_seed{seed}"
        label = f"dqn_sparse_seed{seed}"
        _log(f"  [{seed + 1}/{num_seeds}] {label}")
        policy = load_trained_policy(run_dir)
        results[label] = run_tournament(
            label,
            policy,
            HeuristicAgent(),
            num_episodes,
            seed=TOURNAMENT_SEED_OFFSET,
            include_dice_roll=run_uses_dice_roll(run_dir),
        )
    return results

# Builds one agent-type's summary: mean +/- std win rate across seeds, a pooled-sample
# Wilson confidence interval, and a significance test against the Random baseline (11.3).
def summarize_by_agent_type(
    seed_results: list[TournamentResult], agent_type: str, baseline: TournamentResult
) -> dict:
    win_rates = [r.win_rate for r in seed_results]
    pooled_wins = sum(r.wins for r in seed_results)
    pooled_episodes = sum(r.num_episodes for r in seed_results)

    stats = mean_and_std(win_rates)
    seed_ci = t_confidence_interval_across_seeds(win_rates)
    pooled_ci = wilson_confidence_interval(pooled_wins, pooled_episodes)
    z_test = two_proportion_z_test(pooled_wins, pooled_episodes, baseline.wins, baseline.num_episodes)
    length_stats = mean_and_std([r.avg_episode_length for r in seed_results])
    ratios = [r for r in (res.capture_to_death_ratio for res in seed_results) if r is not None]
    ratio_stats = mean_and_std(ratios) if ratios else None

    return {
        "agent_type": agent_type,
        "num_seeds": len(seed_results),
        "mean_win_rate": stats.mean,
        "std_win_rate": stats.std,
        # Headline interval: seeds are the unit of analysis, since each one is a different policy.
        "seed_win_rate_95ci_lower": seed_ci.lower,
        "seed_win_rate_95ci_upper": seed_ci.upper,
        # Secondary: all seeds' games pooled as one binomial - describes this set of agents'
        # aggregate play strength, NOT the uncertainty in the method's win rate.
        "pooled_win_rate": pooled_wins / pooled_episodes,
        "pooled_win_rate_95ci_lower": pooled_ci.lower,
        "pooled_win_rate_95ci_upper": pooled_ci.upper,
        "z_vs_random_baseline": z_test.z,
        "p_value_vs_random_baseline": z_test.p_value,
        "mean_episode_length": length_stats.mean,
        "mean_capture_to_death_ratio": ratio_stats.mean if ratio_stats else "",
        "std_capture_to_death_ratio": ratio_stats.std if ratio_stats else "",
    }

# Writes one row per tournament result (agent/seed-level detail) to a CSV file - 11.5.
def save_tournament_results_csv(results: dict[str, TournamentResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "agent_label", "num_episodes", "wins", "win_rate", "avg_episode_length",
        "total_captures_made", "total_times_captured", "capture_to_death_ratio",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results.values():
            writer.writerow(
                {
                    "agent_label": result.agent_label,
                    "num_episodes": result.num_episodes,
                    "wins": result.wins,
                    "win_rate": result.win_rate,
                    "avg_episode_length": result.avg_episode_length,
                    "total_captures_made": result.total_captures_made,
                    "total_times_captured": result.total_times_captured,
                    "capture_to_death_ratio": result.capture_to_death_ratio,
                }
            )

# Writes one row per agent-type summary (mean +/- std, CI, significance vs. baseline) to CSV.
def save_summary_csv(summaries: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "agent_type", "num_seeds", "mean_win_rate", "std_win_rate",
        "seed_win_rate_95ci_lower", "seed_win_rate_95ci_upper",
        "pooled_win_rate", "pooled_win_rate_95ci_lower", "pooled_win_rate_95ci_upper",
        "z_vs_random_baseline", "p_value_vs_random_baseline", "holm_adjusted_p_value",
        "mean_episode_length", "mean_capture_to_death_ratio", "std_capture_to_death_ratio",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(summary)

# Runs Step 11 end to end: the large-scale tournament for every Step 10 checkpoint plus
# baselines, the reward-shaping ablation, and writes every result to CSV.
def run_full_evaluation(
    final_root: Path,
    output_dir: Path,
    tournament_episodes: int = DEFAULT_TOURNAMENT_EPISODES,
    run_reward_ablation: bool = True,
    ablation_episodes: int = DEFAULT_ABLATION_EPISODES,
    ablation_seeds: int = DEFAULT_ABLATION_SEEDS,
    ablation_eval_episodes: int = DEFAULT_ABLATION_EVAL_EPISODES,
) -> dict:
    global _progress_log_path
    output_dir.mkdir(parents=True, exist_ok=True)
    _progress_log_path = output_dir / "progress.log"
    started = datetime.now()
    _log(f"Step 11 started - reward ablation {'ON' if run_reward_ablation else 'OFF'}")

    results = run_all_tournaments(final_root, tournament_episodes)
    baseline = results["random_baseline"]

    summaries = [
        summarize_by_agent_type(
            [results[f"{agent_type}_seed{seed}"] for seed in range(NUM_TRAINED_SEEDS)], agent_type, baseline
        )
        for agent_type in AGENT_TYPES
    ]

    if run_reward_ablation:
        summary_json = json.loads((final_root.parent / "summary.json").read_text())
        winning_hyperparameters = summary_json["winning_hyperparameters"]
        ablation_root = output_dir / "reward_ablation_runs"
        # The sparse arm must match the dense arm it is compared against on everything except
        # the reward dict, so gamma and the observation format are read from the dense runs
        # rather than defaulted - otherwise the ablation measures more than reward shaping.
        dense_config = load_config(final_root / "dqn_seed0" / "config.json")
        run_reward_ablation_training(
            ablation_root,
            winning_hyperparameters,
            ablation_episodes,
            ablation_seeds,
            ablation_eval_episodes,
            gamma=dense_config.gamma,
            include_dice_roll=dense_config.include_dice_roll,
        )
        ablation_results = run_ablation_tournaments(ablation_root, ablation_seeds, tournament_episodes)
        results.update(ablation_results)
        summaries.append(
            summarize_by_agent_type(
                [ablation_results[f"dqn_sparse_seed{seed}"] for seed in range(ablation_seeds)],
                "dqn_sparse",
                baseline,
            )
        )

    # Every agent type is tested against the same Random baseline, so the p-values are a family
    # and need a family-wise correction before any of them is read as "significant".
    adjusted = holm_adjusted_p_values([s["p_value_vs_random_baseline"] for s in summaries])
    for summary, adjusted_p in zip(summaries, adjusted):
        summary["holm_adjusted_p_value"] = adjusted_p

    save_tournament_results_csv(results, output_dir / "tournament_results.csv")
    save_summary_csv(summaries, output_dir / "summary_stats.csv")

    elapsed = datetime.now() - started
    _log(f"DONE in {elapsed} - wrote tournament_results.csv and summary_stats.csv to {output_dir}")
    for summary in summaries:
        _log(f"  {summary['agent_type']}: {summary['mean_win_rate']:.2%} +/- {summary['std_win_rate']:.2%}")

    return {"tournament_results": results, "summaries": summaries}

# Parses flags and runs the full Step 11 scope from the command line.
def _main() -> None:
    parser = argparse.ArgumentParser(description="Run Step 11: large-scale evaluation and the reward-shaping ablation.")
    parser.add_argument("--final-root", default="runs/full/final", help="directory holding Step 10's trained final runs")
    parser.add_argument("--output-dir", default="runs/full/evaluation", help="where to write CSV results and ablation runs")
    parser.add_argument("--tournament-episodes", type=int, default=DEFAULT_TOURNAMENT_EPISODES)
    parser.add_argument("--skip-reward-ablation", action="store_true", help="skip Step 11.4's sparse-reward retraining")
    parser.add_argument("--ablation-episodes", type=int, default=DEFAULT_ABLATION_EPISODES)
    parser.add_argument("--ablation-seeds", type=int, default=DEFAULT_ABLATION_SEEDS)
    parser.add_argument("--ablation-eval-episodes", type=int, default=DEFAULT_ABLATION_EVAL_EPISODES)
    args = parser.parse_args()
    run_full_evaluation(
        Path(args.final_root),
        Path(args.output_dir),
        tournament_episodes=args.tournament_episodes,
        run_reward_ablation=not args.skip_reward_ablation,
        ablation_episodes=args.ablation_episodes,
        ablation_seeds=args.ablation_seeds,
        ablation_eval_episodes=args.ablation_eval_episodes,
    )

if __name__ == "__main__":
    _main()

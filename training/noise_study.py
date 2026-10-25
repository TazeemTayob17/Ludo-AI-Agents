# Noise study: per-rollout outcomes for every candidate move at sampled real decisions, and an offline analysis of how well n rollouts pick the best move.
import sys
import time
from multiprocessing import Pool

import numpy as np

from agents.heuristic_agent import HeuristicAgent
from env.board import NUM_PLAYERS
from env.game_state import GameState
from env.ludo_env import LudoEnv
from env.moves import apply_move
from env.rewards import progress_potential
from env.turns import has_player_won

NUM_ROLLOUTS = 600
HORIZONS = (40, 100)
HEUR = HeuristicAgent()


# Plays one rollout after the move; returns (win, phi at each horizon).
def one_rollout(after, player_id, next_player, seed):
    game = GameState(board=after.copy(), current_player=next_player)
    rng = np.random.default_rng(seed)
    phis = {}
    turns = 0
    while not (game.terminated or game.truncated):
        game.step_turn(HEUR, rng)
        turns += 1
        for h in HORIZONS:
            if turns == h:
                phis[h] = progress_potential(game.board, player_id)
    win = float(game.winner == player_id)
    for h in HORIZONS:
        if h not in phis:
            phis[h] = 1.0 if win else -1.0
    return win, [phis[h] for h in HORIZONS]


# Evaluates all candidates of one decision with common dice seeds; returns an array [cand, rollout, 1 + horizons].
def evaluate_decision(args):
    board, roll, legal, base_seed = args
    out = np.zeros((len(legal), NUM_ROLLOUTS, 1 + len(HORIZONS)))
    for ci, token in enumerate(legal):
        after = board.copy()
        outcome = apply_move(after, 0, token, roll)
        if has_player_won(after, 0):
            out[ci, :, 0] = 1.0
            out[ci, :, 1:] = 1.0
            continue
        extra = roll == 6 or outcome.captured_player is not None or outcome.reached_home
        nxt = 0 if extra else 1
        for i in range(NUM_ROLLOUTS):
            win, phis = one_rollout(after, 0, nxt, base_seed + i)
            out[ci, i, 0] = win
            out[ci, i, 1:] = phis
    return legal, out


# Collects multi-move decisions from Heuristic-vs-Heuristic games and samples a spread of them.
def collect_decisions(num_games=8, per_game=6):
    decisions = []
    for g in range(num_games):
        env = LudoEnv(opponent_policy=HEUR, rng=np.random.default_rng(900 + g))
        obs, info = env.reset()
        done = False
        found = []
        while not done:
            legal = tuple(int(t) for t in info["action_mask"].nonzero()[0])
            roll = info["dice_roll"]
            board = env.game.board
            if len(legal) > 1:
                found.append((board.copy(), roll, legal))
            obs, r, te, tr, info = env.step(HEUR(board, 0, roll, legal))
            done = te or tr
        idx = np.linspace(0, len(found) - 1, per_game).astype(int)
        decisions += [found[i] for i in idx]
    return decisions


# Runs the rollout collection and saves an object array of [candidate, rollout, 1 + horizons] arrays, one per decision.
def collect_main(out_path):
    t0 = time.time()
    decisions = collect_decisions()
    jobs = [(b, r, l, 10_000 + 1000 * i) for i, (b, r, l) in enumerate(decisions)]
    print("decisions", len(jobs), flush=True)
    results = []
    with Pool(4) as pool:
        for k, res in enumerate(pool.imap(evaluate_decision, jobs), 1):
            results.append(res)
            print(f"done {k}/{len(jobs)} elapsed {time.time() - t0:.0f}s", flush=True)
    np.save(out_path, np.array([r[1] for r in results], dtype=object), allow_pickle=True)
    print("saved", flush=True)


# Prints best-worst spread and regret-versus-budget tables for each estimator, judged on held-out rollouts.
def analyse(path):
    data = np.load(path, allow_pickle=True)
    EVAL = slice(300, 600)
    names = {0: "win", 1: "phi@40", 2: "phi@100"}

    gaps_best_second, spreads = [], []
    for d in data:
        win = d[:, :, 0].mean(axis=1)
        s = np.sort(win)[::-1]
        gaps_best_second.append(s[0] - s[1])
        spreads.append(s[0] - s[-1])
    print(f"decisions {len(data)}; full-600 win-fraction: mean best-worst spread {np.mean(spreads)*100:.1f}pp, "
          f"mean best-vs-second gap {np.mean(gaps_best_second)*100:.1f}pp, median gap {np.median(gaps_best_second)*100:.1f}pp")

    print("\nestimator  n    regret(pp)  picks-held-out-best   (random-choice regret shown last)")
    for k in (0, 1, 2):
        for n in (16, 64, 150):
            regrets, hits = [], []
            for d in data:
                true = d[:, EVAL, 0].mean(axis=1)
                for start in range(0, 300 - n + 1, n):
                    est = d[:, start:start + n, k].mean(axis=1)
                    pick = int(np.argmax(est))
                    regrets.append(true.max() - true[pick])
                    hits.append(pick == int(np.argmax(true)))
            print(f"{names[k]:8s} {n:4d}  {np.mean(regrets)*100:8.2f}   {np.mean(hits):.2f}")
    rand = np.mean([d[:, EVAL, 0].mean(axis=1).max() - d[:, EVAL, 0].mean(axis=1).mean() for d in data]) * 100
    print(f"random choice regret {rand:.2f}pp")
    # Heuristic's own choice regret on the same decisions is reported by the caller if available.


if __name__ == "__main__":
    if sys.argv[1] == "collect":
        collect_main(sys.argv[2])
    else:
        analyse(sys.argv[2])

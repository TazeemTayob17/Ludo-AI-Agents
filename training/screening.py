# Value-prediction screening: collects labelled positions from self-play and compares encodings and network sizes by how well they predict who wins.

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from agents.dqn_network import QNetwork
from agents.heuristic_agent import HeuristicAgent
from env.board import BoardState
from env.game_state import GameState
from env.rewards import progress_potential
from env.state_encoding import ObservationSpec, encode_observation

NUM_SPLIT_BUCKETS = 20
TRAIN_BUCKETS = 14
VAL_BUCKETS = 17
_SPLIT_NAMES = ("train", "val", "test")

ENCODINGS = {
    "token": ObservationSpec(),
    "token_canon": ObservationSpec(canonical_token_order=True),
    "track": ObservationSpec(board_encoding="track"),
    "track_canon": ObservationSpec(board_encoding="track", canonical_token_order=True),
    "token_dice": ObservationSpec(include_dice_roll=True),
    "track_dice": ObservationSpec(board_encoding="track", include_dice_roll=True),
    "features": ObservationSpec(include_move_features=True, include_threat_features=True),
    "features_canon": ObservationSpec(include_move_features=True, include_threat_features=True, canonical_token_order=True),
    "features_canon_risk": ObservationSpec(
        include_move_features=True, include_threat_features=True, include_risk_features=True, canonical_token_order=True
    ),
    "features_canon_race": ObservationSpec(
        include_move_features=True, include_threat_features=True, include_race_features=True, canonical_token_order=True
    ),
    "features_canon_risk_race": ObservationSpec(
        include_move_features=True, include_threat_features=True, include_risk_features=True,
        include_race_features=True, canonical_token_order=True,
    ),
    "track_features": ObservationSpec(board_encoding="track", include_move_features=True, include_threat_features=True),
    "track_features_canon": ObservationSpec(
        board_encoding="track", include_move_features=True, include_threat_features=True, canonical_token_order=True
    ),
}


# Plays the Heuristic (with some random moves for variety) and records the board, seat and roll at every decision.
class _SelfPlayRecorder:

    # Stores the move-randomisation probability, its random generator, and the empty row list.
    def __init__(self, rng: np.random.Generator, random_move_prob: float) -> None:
        self.rng = rng
        self.random_move_prob = random_move_prob
        self.heuristic = HeuristicAgent()
        self.rows: list[tuple[np.ndarray, int, int]] = []

    # Records the decision, then returns a random legal token with probability random_move_prob, else the Heuristic's choice.
    def __call__(self, board: BoardState, player_id: int, roll: int, legal_tokens: tuple[int, ...]) -> int:
        self.rows.append((board.positions.copy(), player_id, roll))
        if self.rng.random() < self.random_move_prob:
            return legal_tokens[int(self.rng.integers(0, len(legal_tokens)))]
        return self.heuristic(board, player_id, roll, legal_tokens)


# Plays num_games finished games and returns every decision as raw boards, seats, rolls, win labels and game ids (truncated games are dropped).
def collect_positions(num_games: int, seed: int = 0, random_move_prob: float = 0.1, max_turns: int = 1000) -> dict[str, np.ndarray]:
    boards, players, rolls, labels, games = [], [], [], [], []
    for game_id in range(num_games):
        rng = np.random.default_rng([seed, game_id])
        recorder = _SelfPlayRecorder(np.random.default_rng([seed, game_id, 1]), random_move_prob)
        game = GameState(max_turns=max_turns)
        while not (game.terminated or game.truncated):
            game.step_turn(recorder, rng)
        if not game.terminated or not recorder.rows:
            continue
        for board_positions, player_id, roll in recorder.rows:
            boards.append(board_positions)
            players.append(player_id)
            rolls.append(roll)
            labels.append(int(player_id == game.winner))
            games.append(game_id)
    return {
        "boards": np.array(boards, dtype=np.int8),
        "players": np.array(players, dtype=np.int8),
        "rolls": np.array(rolls, dtype=np.int8),
        "labels": np.array(labels, dtype=np.int8),
        "games": np.array(games, dtype=np.int32),
    }


# Assigns each game to train, val or test by game id, so no game ever appears in two splits.
def split_of_games(game_ids: np.ndarray) -> np.ndarray:
    bucket = game_ids % NUM_SPLIT_BUCKETS
    return np.where(bucket < TRAIN_BUCKETS, 0, np.where(bucket < VAL_BUCKETS, 1, 2))


# Picks up to the capped number of rows for each split with a fixed seed, returning the row indices per split name.
def select_rows(data: dict[str, np.ndarray], caps: dict[str, int], seed: int = 0) -> dict[str, np.ndarray]:
    split = split_of_games(data["games"])
    rng = np.random.default_rng(seed)
    chosen = {}
    for index, name in enumerate(_SPLIT_NAMES):
        rows = np.flatnonzero(split == index)
        if len(rows) > caps[name]:
            rows = np.sort(rng.choice(rows, size=caps[name], replace=False))
        chosen[name] = rows
    return chosen


# Builds a BoardState holding the given 4x4 position array.
def _board_from(positions: np.ndarray) -> BoardState:
    board = BoardState()
    board.positions = positions.astype(board.positions.dtype)
    return board


# Encodes the chosen rows with one named encoding into a float32 matrix.
def encode_rows(data: dict[str, np.ndarray], rows: np.ndarray, spec: ObservationSpec) -> np.ndarray:
    return np.stack(
        [
            encode_observation(_board_from(data["boards"][i]), int(data["players"][i]), int(data["rolls"][i]), spec)
            for i in rows
        ]
    )


# Returns each chosen row's race standing (own progress minus the opponents' mean), the one number the potential reward uses.
def race_standing_rows(data: dict[str, np.ndarray], rows: np.ndarray) -> np.ndarray:
    return np.array(
        [progress_potential(_board_from(data["boards"][i]), int(data["players"][i])) for i in rows], dtype=np.float32
    )


# Returns the mean binary log-loss and Brier score of predicted win probabilities.
def loss_metrics(probabilities: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    p = np.clip(probabilities.astype(np.float64), 1e-7, 1 - 1e-7)
    y = labels.astype(np.float64)
    logloss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return logloss, float(np.mean((p - y) ** 2))


# Returns the log-loss of always predicting the constant probability p against labels with positive rate rate.
def constant_logloss(p: float, rate: float) -> float:
    return -(rate * math.log(p) + (1 - rate) * math.log(1 - p))


# Fits a value network (QNetwork with one output) with early stopping on the validation loss and returns test metrics at the best epoch.
def fit_value_network(
    train: tuple[np.ndarray, np.ndarray],
    val: tuple[np.ndarray, np.ndarray],
    test: tuple[np.ndarray, np.ndarray],
    hidden_size: int,
    seed: int,
    epochs: int = 12,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
) -> dict[str, float]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x_train, y_train = torch.as_tensor(train[0]), torch.as_tensor(train[1], dtype=torch.float32)
    network = QNetwork(input_size=x_train.shape[1], hidden_size=hidden_size, output_size=1)
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

    # Returns the network's win probabilities for a feature matrix.
    def predict(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return torch.sigmoid(network(torch.as_tensor(x)).squeeze(1)).numpy()

    best = {"val_loss": float("inf")}
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(x_train))
        for start in range(0, len(order), batch_size):
            index = torch.as_tensor(order[start:start + batch_size])
            loss = F.binary_cross_entropy_with_logits(network(x_train[index]).squeeze(1), y_train[index])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val_loss, _ = loss_metrics(predict(val[0]), val[1])
        if val_loss < best["val_loss"]:
            test_loss, test_brier = loss_metrics(predict(test[0]), test[1])
            train_loss, _ = loss_metrics(predict(train[0]), train[1])
            best = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "test_logloss": test_loss,
                "test_brier": test_brier,
                "best_epoch": float(epoch),
            }
    return best


# Fits a logistic regression of the label on race standing alone and returns its test log-loss and Brier score.
def race_standing_baseline(
    train: tuple[np.ndarray, np.ndarray], test: tuple[np.ndarray, np.ndarray]
) -> tuple[float, float]:
    x = torch.as_tensor(train[0]).unsqueeze(1)
    y = torch.as_tensor(train[1], dtype=torch.float32)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.LBFGS(model.parameters(), max_iter=100)

    # Computes the loss and gradients for one LBFGS step.
    def closure():
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(x).squeeze(1), y)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        probabilities = torch.sigmoid(model(torch.as_tensor(test[0]).unsqueeze(1)).squeeze(1)).numpy()
    return loss_metrics(probabilities, test[1])


# Runs the fit for each encoding, hidden size and seed, writes one CSV row per fit and prints a mean-and-SD summary with reference rows.
def run_fits(
    positions_path: Path,
    encodings: list[str],
    hidden_sizes: list[int],
    num_seeds: int,
    caps: dict[str, int],
    out_path: Path,
    epochs: int,
) -> list[dict]:
    data = {key: value for key, value in np.load(positions_path).items()}
    rows = select_rows(data, caps)
    labels = {name: data["labels"][rows[name]].astype(np.float32) for name in _SPLIT_NAMES}
    rate = float(labels["train"].mean())
    results: list[dict] = []

    prior_loss, prior_brier = loss_metrics(np.full(len(labels["test"]), rate), labels["test"])
    standing = {name: race_standing_rows(data, rows[name]) for name in ("train", "test")}
    standing_loss, standing_brier = race_standing_baseline(
        (standing["train"], labels["train"]), (standing["test"], labels["test"])
    )
    print(f"win rate in train {rate:.4f}; constant prior log-loss {prior_loss:.4f} (formula {constant_logloss(rate, rate):.4f})")
    print(f"race-standing logistic regression: log-loss {standing_loss:.4f}, Brier {standing_brier:.4f}")

    for encoding in encodings:
        spec = ENCODINGS[encoding]
        start = time.time()
        features = {name: encode_rows(data, rows[name], spec) for name in _SPLIT_NAMES}
        print(f"[{encoding}] encoded {sum(len(v) for v in features.values())} rows, size {features['train'].shape[1]}, {time.time() - start:.0f}s", flush=True)
        for hidden in hidden_sizes:
            for seed in range(num_seeds):
                metrics = fit_value_network(
                    (features["train"], labels["train"]),
                    (features["val"], labels["val"]),
                    (features["test"], labels["test"]),
                    hidden, seed, epochs=epochs,
                )
                results.append({"encoding": encoding, "hidden": hidden, "seed": seed, "input_size": features["train"].shape[1], **metrics})
            subset = [r["test_logloss"] for r in results if r["encoding"] == encoding and r["hidden"] == hidden]
            gap = np.mean([r["val_loss"] - r["train_loss"] for r in results if r["encoding"] == encoding and r["hidden"] == hidden])
            print(
                f"[{encoding}] hidden {hidden}: test log-loss {np.mean(subset):.4f} +/- {np.std(subset, ddof=1) if len(subset) > 1 else 0:.4f}"
                f" (val-train gap {gap:+.4f})", flush=True,
            )
        del features
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    return results


# Parses the collect and fit subcommands.
def _main() -> None:
    parser = argparse.ArgumentParser(description="Value-prediction screening of board encodings and network sizes.")
    sub = parser.add_subparsers(dest="command", required=True)
    collect = sub.add_parser("collect")
    collect.add_argument("--games", type=int, default=3000)
    collect.add_argument("--seed", type=int, default=0)
    collect.add_argument("--random-move-prob", type=float, default=0.1)
    collect.add_argument("--out", type=Path, default=Path("runs/screening/positions.npz"))
    fit = sub.add_parser("fit")
    fit.add_argument("--positions", type=Path, default=Path("runs/screening/positions.npz"))
    fit.add_argument("--encodings", nargs="+", default=["token", "track"], choices=list(ENCODINGS))
    fit.add_argument("--hidden", nargs="+", type=int, default=[128])
    fit.add_argument("--seeds", type=int, default=3)
    fit.add_argument("--epochs", type=int, default=12)
    fit.add_argument("--train-rows", type=int, default=120000)
    fit.add_argument("--eval-rows", type=int, default=30000)
    fit.add_argument("--out", type=Path, default=Path("runs/screening/results.csv"))
    args = parser.parse_args()

    if args.command == "collect":
        start = time.time()
        data = collect_positions(args.games, args.seed, args.random_move_prob)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out, **data)
        print(f"{len(data['labels'])} decisions from {len(np.unique(data['games']))} games, win-label rate {data['labels'].mean():.4f}, {time.time() - start:.0f}s")
    else:
        caps = {"train": args.train_rows, "val": args.eval_rows, "test": args.eval_rows}
        run_fits(args.positions, args.encodings, args.hidden, args.seeds, caps, args.out, args.epochs)


if __name__ == "__main__":
    _main()

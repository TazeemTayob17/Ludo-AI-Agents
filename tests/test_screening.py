# Tests for the value-prediction screening tool: collection labels, game-level splits, metrics and a tiny end-to-end fit.

import math

import numpy as np

from training.screening import (
    constant_logloss,
    collect_positions,
    encode_rows,
    ENCODINGS,
    loss_metrics,
    run_fits,
    select_rows,
    split_of_games,
)


# Checks every game contributes exactly one winning seat's rows as label 1, and the arrays line up.
def test_collect_positions_labels_and_shapes():
    data = collect_positions(6, seed=1)
    n = len(data["labels"])
    assert n > 0
    assert data["boards"].shape == (n, 4, 4)
    assert set(np.unique(data["labels"])) <= {0, 1}
    for game in np.unique(data["games"]):
        rows = data["games"] == game
        winners = np.unique(data["players"][rows & (data["labels"] == 1)])
        assert len(winners) == 1


# Checks collection is reproducible from its seed.
def test_collect_positions_is_deterministic():
    a, b = collect_positions(3, seed=2), collect_positions(3, seed=2)
    assert np.array_equal(a["boards"], b["boards"]) and np.array_equal(a["labels"], b["labels"])


# Checks no game id appears in more than one split, and all three splits are used.
def test_splits_never_share_a_game():
    games = np.repeat(np.arange(200), 5)
    split = split_of_games(games)
    for game in range(200):
        assert len(set(split[games == game])) == 1
    assert set(split) == {0, 1, 2}
    assert abs((split == 0).mean() - 0.7) < 0.05


# Checks the constant-prior log-loss matches the closed form at p = 0.25 and the metric function.
def test_constant_prior_matches_formula():
    labels = np.array([1] * 25 + [0] * 75)
    logloss, brier = loss_metrics(np.full(100, 0.25), labels)
    assert abs(logloss - constant_logloss(0.25, 0.25)) < 1e-9
    assert abs(logloss - (-(0.25 * math.log(0.25) + 0.75 * math.log(0.75)))) < 1e-9
    assert abs(brier - 0.1875) < 1e-9


# Checks every named encoding produces the size its spec promises.
def test_encodings_have_expected_sizes():
    data = collect_positions(2, seed=3)
    rows = np.arange(3)
    sizes = {name: encode_rows(data, rows, spec).shape[1] for name, spec in ENCODINGS.items()}
    assert sizes["token"] == 64 and sizes["track"] == 241 and sizes["features"] == 108
    assert sizes["track_features_canon"] == 241 + 44


# Checks the whole fit pipeline runs on a tiny dataset and writes a CSV with finite metrics.
def test_run_fits_end_to_end(tmp_path):
    data = collect_positions(40, seed=4)
    np.savez_compressed(tmp_path / "positions.npz", **data)
    caps = {"train": 400, "val": 100, "test": 100}
    results = run_fits(tmp_path / "positions.npz", ["token", "track"], [16], 1, caps, tmp_path / "out.csv", epochs=2)
    assert len(results) == 2 and (tmp_path / "out.csv").exists()
    assert all(np.isfinite(r["test_logloss"]) for r in results)
    rows = select_rows(data, caps)
    assert not set(data["games"][rows["train"]]) & set(data["games"][rows["test"]])

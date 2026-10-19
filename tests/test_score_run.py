# Tests for the scoring script: promotion wording and a tiny end-to-end score of a freshly trained run.

import sys

from training.config import TrainingConfig
from training.run_training import run_training
from training.score_run import _main, verdict


# Checks the verdict follows the +3pp promotion rule and the +/-3pp no-change band.
def test_verdict_thresholds():
    assert verdict(0.031).startswith("PROMOTE")
    assert verdict(0.029).startswith("no detectable change")
    assert verdict(-0.029).startswith("no detectable change")
    assert verdict(-0.031).startswith("REJECT")


# Trains a 3-episode run with the track encoding and scores it for a few games, expecting a CSV row.
def test_score_run_end_to_end(tmp_path, monkeypatch, capsys):
    config = TrainingConfig(
        agent_type="dqn", seed=0, num_episodes=3, eval_episodes=2, checkpoint_every_episodes=3, min_buffer_size=10,
        include_move_features=True, board_encoding="track", canonical_token_order=True, hidden_size=16,
    )
    run_training(config, tmp_path / "run")
    out = tmp_path / "scores.csv"
    monkeypatch.setattr(sys, "argv", ["score_run", str(tmp_path / "run"), "--games", "3", "--out", str(out)])
    _main()
    assert "vs 3 heuristic" in capsys.readouterr().out
    assert len(out.read_text().strip().splitlines()) == 2


# Scores the same tiny run with seat rotation and a second copy, expecting the cross-run summary line.
def test_score_run_rotating_seats_and_summary(tmp_path, monkeypatch, capsys):
    config = TrainingConfig(
        agent_type="dqn", seed=0, num_episodes=2, eval_episodes=2, checkpoint_every_episodes=2, min_buffer_size=10,
        include_move_features=True, canonical_token_order=True, hidden_size=16,
    )
    run_training(config, tmp_path / "a")
    run_training(config, tmp_path / "b")
    monkeypatch.setattr(sys, "argv", ["score_run", str(tmp_path / "a"), str(tmp_path / "b"), "--games", "8",
                                      "--rotate-seats", "--out", str(tmp_path / "s.csv")])
    _main()
    assert "across 2 runs" in capsys.readouterr().out

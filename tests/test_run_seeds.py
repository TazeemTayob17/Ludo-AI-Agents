# Tests for the multi-seed runner: it builds one quick_run command per seed and skips seeds that are already trained.

import subprocess

from training import run_seeds


# Checks each seed gets its own output directory and the shared flags are passed through, skipping a finished seed.
def test_train_seeds_builds_commands_and_skips_finished(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda command, check: calls.append(command))
    done = tmp_path / "x_seed1" / "checkpoints"
    done.mkdir(parents=True)
    (done / "episode_5.pt").write_text("")
    dirs = run_seeds.train_seeds("x", [0, 1, 2], tmp_path, 5, ["--canonical-token-order"])
    assert [d.name for d in dirs] == ["x_seed0", "x_seed1", "x_seed2"]
    assert len(calls) == 2
    assert "--canonical-token-order" in calls[0] and calls[0][calls[0].index("--seed") + 1] == "0"
    assert calls[1][calls[1].index("--seed") + 1] == "2"

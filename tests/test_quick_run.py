# Tests for training/quick_run.py's CLI wrapper. Tiny episode count - proves the CLI
# wiring works, not a real training run.

import sys

from training.quick_run import _main


# Checks the CLI builds a config from flags and runs it to the expected output directory.
def test_quick_run_cli_produces_expected_output(tmp_path, monkeypatch):
    output_dir = tmp_path / "cli_run"
    argv = [
        "quick_run.py",
        "--agent-type", "tabular_q",
        "--seed", "0",
        "--num-episodes", "2",
        "--run-name", "smoke",
        "--opponent-type", "random",
        "--output-dir", str(output_dir),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    _main()

    assert (output_dir / "config.json").exists()
    assert (output_dir / "log.csv").exists()
    assert (output_dir / "checkpoints" / "best.pkl").exists()

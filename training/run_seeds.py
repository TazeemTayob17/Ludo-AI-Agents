# Trains one configuration on several seeds one after another (each as a separate quick_run process), then scores them together.

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


# Runs quick_run once per seed with the shared training arguments, skipping seeds whose final checkpoint already exists.
def train_seeds(name: str, seeds: list[int], root: Path, num_episodes: int, train_args: list[str]) -> list[Path]:
    run_dirs = []
    for seed in seeds:
        run_dir = root / f"{name}_seed{seed}"
        run_dirs.append(run_dir)
        if (run_dir / "checkpoints" / f"episode_{num_episodes}.pt").exists():
            print(f"seed {seed}: already trained, skipping", flush=True)
            continue
        command = [
            sys.executable, "-m", "training.quick_run", "--agent-type", "dqn", "--seed", str(seed),
            "--num-episodes", str(num_episodes), "--run-name", name, "--output-dir", str(run_dir), *train_args,
        ]
        print("running:", " ".join(command), flush=True)
        subprocess.run(command, check=True)
    return run_dirs


# Parses the seed list and passes every unrecognised flag straight through to quick_run (the training settings).
def _main() -> None:
    parser = argparse.ArgumentParser(description="Train a configuration on several seeds, then score them together.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--root", type=Path, default=Path("runs/final"))
    parser.add_argument("--num-episodes", type=int, default=10000)
    parser.add_argument("--no-score", action="store_true")
    args, train_args = parser.parse_known_args()

    run_dirs = train_seeds(args.name, args.seeds, args.root, args.num_episodes, train_args)
    if not args.no_score:
        subprocess.run([sys.executable, "-m", "training.score_run", *map(str, run_dirs)], check=True)


if __name__ == "__main__":
    _main()

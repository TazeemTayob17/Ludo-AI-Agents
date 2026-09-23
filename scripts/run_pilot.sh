#!/usr/bin/env bash
# Relaunches the Step 11.5 pilot: one DQN run with the dice roll in the observation and
# gamma=0.99, and a matched control on the old representation, so the difference between
# them isolates those two changes. Both use the sweep's winning hyperparameters, seed 0.
# Deletes any previous pilot output first - these runs start from episode 1, there is no resume.
# Run sequentially, not in parallel: a first attempt at running both at once was killed under
# memory pressure on this machine, and Step 10's full scope ran sequentially for the same reason.
set -euo pipefail

cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
COMMON="--agent-type dqn --seed 0 --num-episodes 10000 --learning-rate 0.0005 \
  --target-sync-every-steps 1000 --epsilon-decay-episodes 6000 \
  --checkpoint-every-episodes 1000 --eval-episodes 100"

rm -rf runs/pilot
mkdir -p runs/pilot

echo "[1/2] treatment: dice roll in observation, gamma=0.99"
$PY -m training.quick_run $COMMON \
  --run-name pilot_treatment --include-dice-roll --gamma 0.99 \
  --output-dir runs/pilot/treatment_roll_gamma99_seed0 > runs/pilot/treatment.log 2>&1

echo "[2/2] control: original observation, gamma=0.95"
$PY -m training.quick_run $COMMON \
  --run-name pilot_control --gamma 0.95 \
  --output-dir runs/pilot/control_baseline_seed0 > runs/pilot/control.log 2>&1

echo "pilot complete - compare runs/pilot/*/log.csv"

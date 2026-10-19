# Pilot and final-run commands (branch supervisor-improvements)

Run everything from the project root, in a terminal that has the project's virtual environment available (the commands use `.venv/Scripts/python.exe`). Hypotheses and pass marks are in `docs/decisions_log.md` (Step 19). Each pilot takes about 2 hours alone; the machine has 4 cores, so two at a time is reasonable (each will be slower). Every command writes its own log file, so a closed terminal loses nothing. Create the log folder first: `mkdir runs/lecturer`.

All pilots use the baseline settings: seed 0, 10,000 episodes, learning rate 0.0005, target sync 1,000, epsilon decay 6,000, 100-game in-training evaluations. The commands are written out in full (no shell variables) so they work in Git Bash or PowerShell.

## Pilots (seed 0), in order of priority

P1 - canonical token order on the feature DQN (compare with 31.85%):

```
.venv/Scripts/python.exe -m training.quick_run --agent-type dqn --seed 0 --num-episodes 10000 --learning-rate 0.0005 --target-sync-every-steps 1000 --epsilon-decay-episodes 6000 --checkpoint-every-episodes 1000 --eval-episodes 100 --include-move-features --include-threat-features --canonical-token-order --run-name p1_canon --output-dir runs/lecturer/p1_canon_seed0 > runs/lecturer/p1.log 2>&1
```

R2 - canonical token order on the raw DQN (compare with the raw control, 7.50%):

```
.venv/Scripts/python.exe -m training.quick_run --agent-type dqn --seed 0 --num-episodes 10000 --learning-rate 0.0005 --target-sync-every-steps 1000 --epsilon-decay-episodes 6000 --checkpoint-every-episodes 1000 --eval-episodes 100 --canonical-token-order --run-name r2_raw_canon --output-dir runs/lecturer/r2_raw_canon_seed0 > runs/lecturer/r2.log 2>&1
```

R1 - track board on the raw DQN (compare with 7.50%):

```
.venv/Scripts/python.exe -m training.quick_run --agent-type dqn --seed 0 --num-episodes 10000 --learning-rate 0.0005 --target-sync-every-steps 1000 --epsilon-decay-episodes 6000 --checkpoint-every-episodes 1000 --eval-episodes 100 --board-encoding track --run-name r1_raw_track --output-dir runs/lecturer/r1_raw_track_seed0 > runs/lecturer/r1.log 2>&1
```

P2 - track board on top of canonical order, feature DQN (compare with P1's score):

```
.venv/Scripts/python.exe -m training.quick_run --agent-type dqn --seed 0 --num-episodes 10000 --learning-rate 0.0005 --target-sync-every-steps 1000 --epsilon-decay-episodes 6000 --checkpoint-every-episodes 1000 --eval-episodes 100 --include-move-features --include-threat-features --canonical-token-order --board-encoding track --run-name p2_track_canon --output-dir runs/lecturer/p2_track_canon_seed0 > runs/lecturer/p2.log 2>&1
```

N1 - 256 hidden units on the current feature model (compare with 31.85%):

```
.venv/Scripts/python.exe -m training.quick_run --agent-type dqn --seed 0 --num-episodes 10000 --learning-rate 0.0005 --target-sync-every-steps 1000 --epsilon-decay-episodes 6000 --checkpoint-every-episodes 1000 --eval-episodes 100 --include-move-features --include-threat-features --hidden-size 256 --run-name n1_hidden256 --output-dir runs/lecturer/n1_hidden256_seed0 > runs/lecturer/n1.log 2>&1
```

## Conditional pilots (only if wanted, or if an earlier pilot is promoted)

F1 - risk block (add `--canonical-token-order` if P1 was promoted); F2 - race block; N2 - a bigger network on the best representation (add `--hidden-size 256` or `512` to the best pilot's command):

```
.venv/Scripts/python.exe -m training.quick_run --agent-type dqn --seed 0 --num-episodes 10000 --learning-rate 0.0005 --target-sync-every-steps 1000 --epsilon-decay-episodes 6000 --checkpoint-every-episodes 1000 --eval-episodes 100 --include-move-features --include-threat-features --include-risk-features --run-name f1_risk --output-dir runs/lecturer/f1_risk_seed0 > runs/lecturer/f1.log 2>&1

.venv/Scripts/python.exe -m training.quick_run --agent-type dqn --seed 0 --num-episodes 10000 --learning-rate 0.0005 --target-sync-every-steps 1000 --epsilon-decay-episodes 6000 --checkpoint-every-episodes 1000 --eval-episodes 100 --include-move-features --include-threat-features --include-race-features --run-name f2_race --output-dir runs/lecturer/f2_race_seed0 > runs/lecturer/f2.log 2>&1
```

## Scoring a finished pilot (about 3 minutes each)

Feature-agent pilots are compared with 31.85% (the default); raw-agent pilots with the raw control, 7.50%:

```
.venv/Scripts/python.exe -m training.score_run runs/lecturer/p1_canon_seed0
.venv/Scripts/python.exe -m training.score_run runs/lecturer/r2_raw_canon_seed0 --baseline 0.075
```

Each line prints the win rate, the 95% interval, the difference from the baseline and the verdict (PROMOTE at +3pp, no detectable change within +/-3pp, REJECT at -3pp). Rows are appended to `runs/pilot_scores.csv`.

## Error analysis of a finished run (about 5 minutes)

```
.venv/Scripts/python.exe -m training.diagnostics --run runs/lecturer/p1_canon_seed0 --games 400 --out runs/diagnostics/p1.json
```

## Confirming a promoted pilot on seeds 1 and 2, then the final five seeds

`training.run_seeds` trains the seeds one after another, skips seeds that are already finished, and then scores them together with a cross-seed mean, SD and t-interval. Pass the promoted configuration's training flags after the runner's own options (not `--seed`, `--num-episodes`, `--run-name` or `--output-dir`; the runner adds them). Example for P1:

```
.venv/Scripts/python.exe -m training.run_seeds --name p1_canon --seeds 1 2 --root runs/lecturer --learning-rate 0.0005 --target-sync-every-steps 1000 --epsilon-decay-episodes 6000 --checkpoint-every-episodes 1000 --eval-episodes 100 --include-move-features --include-threat-features --canonical-token-order
```

Final: five seeds (0 to 4) of the best feature agent and of the best raw agent, using the promoted flags:

```
.venv/Scripts/python.exe -m training.run_seeds --name final_feature --seeds 0 1 2 3 4 --root runs/final_lecturer <promoted feature flags>
.venv/Scripts/python.exe -m training.run_seeds --name final_raw --seeds 0 1 2 3 4 --root runs/final_lecturer <promoted raw flags>
```

Final tables against the other opponents, with the agent's seat rotated across all four positions (list the five run folders; `--baseline` is only used for the difference column):

```
.venv/Scripts/python.exe -m training.score_run runs/final_lecturer/final_feature_seed0 runs/final_lecturer/final_feature_seed1 runs/final_lecturer/final_feature_seed2 runs/final_lecturer/final_feature_seed3 runs/final_lecturer/final_feature_seed4 --opponent random --rotate-seats --baseline 0.7606
.venv/Scripts/python.exe -m training.score_run runs/final_lecturer/final_feature_seed0 runs/final_lecturer/final_feature_seed1 runs/final_lecturer/final_feature_seed2 runs/final_lecturer/final_feature_seed3 runs/final_lecturer/final_feature_seed4 --opponent expert --rotate-seats
```

For reference, the existing feature DQN scores 31.35% against 3 Heuristic and 76.06% against 3 Random over five seeds.

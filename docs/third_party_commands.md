# Third-party benchmark: commands (Alhajry expert)

Run from the project root in an external terminal. Times are estimates from measured speed (200 games take about 20 s for the rule-based agents; DQN scoring has taken about 6-10 minutes per 2,000-game run).

## 1. Anchors (does the port behave like a real expert?), seat 0, 2,000 games each, about 3-4 min each, about 20 min in total

```
.venv/Scripts/python.exe -m training.score_baselines --agent alhajry --opponent heuristic --out runs/third_party/anchors.csv
.venv/Scripts/python.exe -m training.score_baselines --agent alhajry --opponent random --out runs/third_party/anchors.csv
.venv/Scripts/python.exe -m training.score_baselines --agent alhajry --opponent expert --out runs/third_party/anchors.csv
.venv/Scripts/python.exe -m training.score_baselines --agent heuristic --opponent alhajry --out runs/third_party/anchors.csv
.venv/Scripts/python.exe -m training.score_baselines --agent expert --opponent alhajry --out runs/third_party/anchors.csv
.venv/Scripts/python.exe -m training.score_baselines --agent random --opponent alhajry --out runs/third_party/anchors.csv
```

## 2. Main result: feature DQN (5 seeds) vs 3 Alhajry, seats rotated, about 35-50 min

```
.venv/Scripts/python.exe -m training.score_run runs/pilot_features/treatment_move_threat_seed0 runs/pilot_features/treatment_move_threat_seed1 runs/pilot_features/treatment_move_threat_seed2 runs/pilot_features/treatment_move_threat_seed3 runs/pilot_features/treatment_move_threat_seed4 --opponent alhajry --rotate-seats --baseline 0.25 --out runs/third_party/feature_dqn_vs_alhajry.csv
```

The verdict column ("REJECT" etc.) is meaningless here; read the win rates, the mean and the interval against the 25% fair share.

## 3. Contrast: raw-board DQN (5 seeds) vs 3 Alhajry, seats rotated, about 35-50 min

```
.venv/Scripts/python.exe -m training.score_run runs/full/final/dqn_seed0 runs/full/final/dqn_seed1 runs/full/final/dqn_seed2 runs/full/final/dqn_seed3 runs/full/final/dqn_seed4 --opponent alhajry --rotate-seats --baseline 0.25 --out runs/third_party/raw_dqn_vs_alhajry.csv
```

# Ludo DQN: plan for the lecturer's three suggestions (2026-10-05)

## Context

**Lecturer's advice:**
1. Increase the neural network.
2. Add more features and knowledge.
3. Investigate the board representation and the information available to the agent.

**Order agreed with the user:**
1. Board representation.
2. Network size.
3. Targeted features, chosen from the agent's actual mistakes.
4. A cheap screening test before every 2-hour training run.

**Current position:**
- Feature DQN: 31.35% vs 3 Heuristic (5 seeds). Raw-board DQN: 8.7%. Heuristic mirror: 24.2%. Expert: 28.7%.
- Every Feature DQN variant plateaus after about 7,000 episodes.
- Six pilots in a row failed, and two of those failures could have been predicted (see the audit).


### Noise study result (done 2026-10-05; decides what "headroom" can mean)

- **Setup:**
  - 48 real multi-move decisions from Heuristic games.
  - 600 rollouts per candidate move with shared dice; Heuristic plays every seat in the rollouts.
  - Each policy's choice is judged on 300 held-out rollouts.
- **Moves within a decision differ very little:** the best and worst move differ by about 2.3pp in win chance on average (best vs second: median 1.3pp). Part of that gap is noise.
- **Regret per decision** (win chance lost against the best move, held-out):

| Chooser | Regret per decision |
|---|---|
| Random | 1.48pp |
| Heuristic | 0.80pp |
| Feature DQN | 0.78pp |
| Rollout, 300 rollouts | 0.69pp |
| Expert | 0.64pp |

  - Every SE is about 0.2pp, so all competent choosers are statistically the same per decision. Only Random is clearly worse.
  - The Feature DQN agrees with the Heuristic on 50% of these decisions.
- **Cheaper estimators** (race standing after 40 or 100 turns) are slightly less noisy than win/loss, but still cannot separate competent moves.
- **Conclusions:**
  1. Skill shows up only as tiny per-decision edges (well under 1pp) that add up over about 90 decisions a game. This is why DQN vs Heuristic is 31.85% vs 24.2% in full games, yet nothing can be measured per decision.
  2. **Rollout-based ceiling estimates are not feasible here.** Separating 0.2pp edges needs tens of thousands of rollouts per move. The headroom question is closed as "not measurable this way"; the dissertation reports this with the numbers above.
  3. So the **full 2,000-game tournament stays the only real judge**, and cheap screening must test *information and representation* (value prediction), not per-move choices.

## Rules for every step (carried over, plus one new rule)

1. Write the hypothesis and pass mark in `docs/decisions_log.md` before any run.
2. **One change per pilot.**
   - Seed 0, 10,000 episodes, the baseline settings (lr 5e-4, target sync 1,000, epsilon decay 6,000, gamma 0.95, dense reward).
   - Scored on the paired 2,000-game tournament vs 3 Heuristic, final checkpoint.
3. **Decision thresholds:**
   - +3pp over the current best: promote, then confirm on seeds 1 and 2.
   - Within ±3pp: report as "no detectable change".
4. **New: a pre-flight check before any 2-hour run.**
   - Unit tests on hand-built positions prove the change does what is intended. This is the check that would have caught Pilot C.
   - Where relevant, the change must also pass the screening test (Step 1).
5. Report learning curves (per 1,000 episodes) and the agreement diagnostic for every pilot.
6. All new switches default off, so existing checkpoints and configs load unchanged.
7. The user runs long jobs in an external terminal. No git staging or commits. One-line comments only.

## Branch and git handling (user instruction)

- All work happens on a new branch, **`supervisor-improvements`**, created from `main` with `git switch -c supervisor-improvements`.
  - That command carries the current working tree across unchanged: the unstaged Stage 2 files, the untracked `tests/test_potential_reward.py`, and the Stage 1 files the user already staged.
- **I won't stage, commit or push anything.**
  - Every new or changed file stays unstaged.
  - The existing index is left exactly as the user set it (the Stage 1 files stay staged as before, and nothing is added or removed).
- The plan document is saved as **`lecturer-improvements-plan.md`** in the project root, next to `dqn-strong-agent-plan.md`.

## Step 0: Records (no code, no training)

- **`docs/decisions_log.md`:**
  - Step 18: the audit's findings, in short.
  - The noise-study result above.
  - **Correct the Pilot C cause:** the potential gave +1 for leaving base against +6 for advancing 6, so it taught the agent not to leave base. This was a design error, not a weak signal.
- **`docs/dissertation_notes.md`:** add the noise-study table, and "why per-decision skill can't be measured".
- **Plan files:** save this plan as `lecturer-improvements-plan.md` in the project root (done). No other plan file is changed.
- **Scratch scripts:** move `noise_study.py` and `noise_analysis.py` from the scratchpad into `training/noise_study.py` so the result is reproducible. The 600-rollout data stays in `runs/stage1/`.

## Step 1: Screening tool (a quick test of representation and network size)

New `training/screening.py`. It measures how well an input encoding plus network can predict *who wins from a given position*. Each screen takes minutes, not hours.

**Dataset**
- Script: `python -m training.screening collect --games 4000`.
- Plays games with Heuristic players in all seats, plus 10% random moves for variety, using `GameState.step_turn` with seeded dice.
- At every decision of every seat, it stores the raw board (`board.positions`, 16 ints), the acting player, the roll, and whether that player won.
- It stores raw boards, not encodings, so any encoding can be applied later.
- Output: `runs/screening/positions.npz`, about 1.5 million rows.

**Evaluation**
- Script: `python -m training.screening fit --encoding token|track --hidden 128|256|512`.
- Splits by game (70/15/15 train/validation/test), so no game leaks across splits.
- Trains a value network (same MLP shape as `QNetwork`, 1 output, BCE loss) for a fixed number of epochs, with 3 seeds.
- **Reports:**
  - Held-out log-loss and Brier score.
  - Train-vs-validation loss, to show underfitting or overfitting.
  - Two reference rows: a constant 25% prior, and logistic regression on `progress_potential` (from `env/rewards.py`).
- **Pass mark:** a candidate goes to an RL pilot only if its held-out log-loss beats the current token encoding by more than the seed-to-seed SD.
- **Caveat for the write-up:** this screens information content, not RL success. The tournament is still the judge.

**Reuse:** `GameState` (`env/game_state.py`), `HeuristicAgent`, `encode_observation`/`ObservationSpec` (`env/state_encoding.py`), and `QNetwork` with `output_size=1` (`agents/dqn_network.py`).

## Step 2: Board representation (lecturer point 3)

### 2a. Track encoding

- **New `ObservationSpec.board_encoding: "token" | "track"`**, default `"token"`, so the current 64 values stay byte-for-byte unchanged.
- **Track encoding, everything from the agent's viewpoint** (the agent's start square = square 0):
  - **Main track:** for each of the 52 squares, 4 channels: own count, and one count per opponent seat (next, across, previous). Counts are scaled by /2 and capped at 1. That gives 208 values.
  - **Own home column:** 6 squares with own counts, plus opponents' home-column token counts (3 values).
  - **Base and home:** counts per seat, 4 + 4 values.
  - **Own token list:** the 4 own tokens' current features (16 values) are kept, so each output still maps to a real token.
  - **Total:** about 241 values. Exact constants live in `env/state_encoding.py` (`TRACK_ENCODING_SIZE`).
- **Why it should help:** "an opponent is 3 squares behind my token" becomes adjacent squares in the input, instead of seat-offset arithmetic the network must learn.
- **Reuse:**
  - `relative_to_global` and `ENTRY_OFFSETS` in `env/board.py`, rotated by the agent's offset.
  - `BoardState.global_square` and `is_in_home_column`.
  - The move, threat and dice blocks are appended unchanged after it.
- **Wiring:**
  - `observation_size()`, `QNetwork` input size (already taken from `observation_size(spec)` in `agents/dqn_agent.py::_build_network`), and `TrainingConfig.board_encoding`.
  - A `--board-encoding` flag in `training/quick_run.py`.
  - `run_full_evaluation.run_observation_spec` already reads the spec from `config.json`.
- **Pre-flight tests** (`tests/test_track_encoding.py`):
  - Each hand-placed token lands on the expected square and channel, checked for all 4 seats.
  - Rotating the board by one seat and re-encoding from the next player's view gives the same vector.
  - Base and home counts add up to 16 tokens.
  - The default spec is unchanged.
- **Screen it:** track vs token in Step 1, at hidden size 128.
- **RL pilots, only if the screen passes:**
  - **2a-F (feature agent):** track board + move + threat features vs **31.85%**.
  - **2a-R (raw agent, no hand-crafted features):** track board + dice roll vs the token-board raw control (7.50% seed 0). This tests the lecturer's point directly for the "no hand-crafting" DQN, and is reported as its own result.

### 2b. Canonical token order (fixes the "same position, 24 orderings" problem)

- **New switch `ObservationSpec.canonical_token_order`**, default False.
  - Own tokens are sorted by progress (ties broken by token id) before encoding.
  - The per-token blocks (own token list, move features, threat features) are emitted in that order.
- **`DQNAgent`:** maps the action mask into canonical slots and maps the chosen slot back to a token id. It reuses the order function from `env/state_encoding.py` (new `canonical_order(board, player_id)`).
- **Training:** the env returns the order in `info`, and the replay buffer stores canonical-slot actions and masks. The change lives in `training/dqn_trainer.py` and `env/ludo_env.py::_build_info`.
- **Pre-flight tests:**
  - Permuting own token ids on a board gives an identical observation.
  - The agent picks the same physical move.
  - The selected token is always legal.
- **Screen:** not applicable, because value prediction is already order-free.
- **RL pilot:** on top of the best encoding from 2a, against the current best.

## Step 3: Network size (lecturer point 1)

- **The current network** is 108 → 128 → 128 → 4 (about 31,000 weights, 2 hidden ReLU layers). It scores each token's move.
- **Screen:** hidden sizes 128, 256 and 512, for both token and track encodings. Report the train-vs-validation gap; if the bigger network only widens the gap, capacity is not the limit.
- **RL pilots, one change each:**
  - **3a:** hidden 256 on the **current** 108-value model. This answers the lecturer directly: does size alone help?
  - **3b:** hidden 256 (or 512 if the screen favours it) on the best representation from Step 2.
  - The learning rate stays the same. If the curve is still rising at 10,000 episodes, note it, and try 20,000 episodes as a separate single change.
- **No code change** is needed beyond the existing `hidden_size` field and a `--hidden-size` flag in `training/quick_run.py`.

## Step 4: Targeted features (lecturer point 2)

**4a. Error analysis first.** New `training/diagnostics.py`:
- Moves the scratch agreement script (`diag_agree.py`) into the codebase.
- Adds a **loss-pattern report** over 1,000 games of the best agent:
  - Tokens captured on the opponent turn right after the move the agent chose.
  - Captures it skipped.
  - Turns with tokens left in base while holding a 6.
  - Games lost while ahead in the race.
  - Game phase when the agent fell behind.
- Output: a ranked table of mistake categories.

**4b. Candidate features, only for the top 2–3 categories.** Examples, not commitments:
- Race leader's progress and distance to finishing.
- Probability of being captured after the move, from `capturing_rolls` (the graded block tried a version mixed in with redundant values, so this time it is added alone).
- Blockade-on-path flags.

**4c. Process for each candidate:** pre-flight test, then a value-screen (where it is a board-level feature), then its own RL pilot as one change.

## Step 5: Confirm and final evaluation

- Each promoted change is confirmed on seeds 1 and 2, then the final configuration is retrained on 5 seeds.
- **Tables:**
  - vs 3 Heuristic, 3 Random and 3 Expert, 2,000 games per seed, using `training/run_full_evaluation.py` and `training/score_baselines.py`.
  - **Two final agents reported separately:** the best raw (no hand-crafted features) DQN and the best feature DQN.
  - **Ablation:** each adopted change removed in turn (one seed each, labelled as such).
- **Statistics:** the existing `training/statistics.py` functions (seed as the unit, Student-t CI, Holm).

## Compute estimate

- Each RL pilot takes about 2 hours, run by the user externally.
- Planned:
  - Representation: 2a-F, 2a-R and 2b, about 3 runs.
  - Network size: 3a and 3b, about 2 runs.
  - Features: about 2 runs.
  - Confirmations: 2 per promoted change, about 4 to 6 runs.
  - Final: 5 seeds for each of 2 agents, 10 runs.
- **Total:** about 20 to 23 runs, roughly 40 to 46 hours. Screening adds minutes per candidate (data collection about 30 to 60 minutes, once).

## Files

- **Modified:**
  - `env/state_encoding.py`: track encoding, canonical order, spec fields.
  - `env/ludo_env.py`: order in `info`.
  - `agents/dqn_agent.py`: canonical slot mapping.
  - `training/dqn_trainer.py`.
  - `training/config.py`, `training/quick_run.py`: new flags.
  - `docs/decisions_log.md`, `docs/dissertation_notes.md`.
- **New:**
  - `training/screening.py`, `training/diagnostics.py`, `training/noise_study.py`.
  - `tests/test_track_encoding.py`, `tests/test_canonical_order.py`, `tests/test_screening.py`, `tests/test_diagnostics.py`.
  - `lecturer-improvements-plan.md`.
- **Left in place, default off:** the Stage 2 potential-reward code (currently unstaged), kept for the record.

## Verification

- The full `pytest` suite passes after each step.
- With every new switch off, the observation is byte-for-byte unchanged and old checkpoints still load and score 31.85% on the paired seed.
- **Sanity anchors re-checked before trusting new numbers:**
  - Random vs 3 Random ≈ 25%.
  - Heuristic mirror 24.20%.
  - Baseline re-score 31.85%.
- **Screening tool sanity checks:**
  - The constant-prior log-loss matches the formula for p = 0.25.
  - The token encoding beats the constant prior.
  - The train/validation/test split never shares a game.
- **Stop point:** after Step 0 and the Step 1 tool are built, stop for user review before any RL pilot is launched.

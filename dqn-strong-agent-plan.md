# Ludo DQN: updated plan from the current position (2026-09-30)

## Context

The feature DQN (108-value observation) scores 31.35% (5 seeds) against 3 Heuristic and 76% against 3 Random. Four pilots in a row (Huber+clip, batch/replay, shared-token network, graded features) failed to beat it. The training logs and a 300-game behaviour check show why:

- **Every run has converged to the same place.** Win rate, shaped reward (~10 per game) and loss all go flat once exploration ends at episode ~6,000. More episodes of the same setup will not help.
- **The agent is not trained to win.** With gamma 0.95 over ~120 decisions it looks ~20 decisions ahead; the +2 win reward is worth ~0.004 at the start of a game against +1 per capture.
- **The shaping rewards are the Heuristic's priority list plus "don't get captured".** The model copies the Heuristic on captures and home entries, agrees on 52% of real choices, and where it differs it mostly picks a safer move (3,684 safer vs 576 riskier). It is "Heuristic + danger avoidance", and that is what ~31% is.
- **So inputs, network and optimiser were never the bottleneck.** The earlier gamma 0.99 test was on the raw-board model and does not apply to the feature model.

Online research (sources opened where possible):

- No source reports 60%+ for one agent against three competent opponents in 4-player Ludo. Every 60%+ figure is against random opponents, a 2-player game, or both. Against three competent opponents, published results are about 30–41%.
- Our 76% vs 3 Random is already at the top of the published range (59–66%).
- What stronger agents did differently: scored the board **after** each candidate move (afterstates, as TD-Gammon and Alhajry's TD player did), used gamma 0.99 with the win as the dominant reward, trained 10–100x longer, and used self-play or mixed opponents.

**The contact's submission** (`Downloads/Ludo.zip`, one file, `Ludo.py`) was read in full. *(Derived from the contact's code.)*

- What it is: a pygame Ludo game with four separate Q-networks (32 → 64 → 4, about 2,400 weights), one per colour, all learning while they play each other in "Only AI" mode.
- Input: the raw screen pixel coordinates of all 16 tokens. There is no dice roll and no normalisation.
- Training: one update per move from that move alone, no replay buffer, no target network, gamma 0.9, fixed 10% exploration.
- Illegal moves: they are allowed to be chosen, penalised −10, then re-picked.
- Rewards: +100 all home, +50 token home, +25 per move in the home column, +20 capture, +10 safe square, +5 token out of base, +1 forward, −100 no change.
- Faults found in the code:
  - The capture reward never fires, because the `playerKilled` flag is never set.
  - Most rewards are computed for the *next* player, because the move function has already passed the turn on before rewards are calculated.
  - The +5 "left base" reward is paid on every move of an active token, not only when it leaves base.
  - "Moved forward" compares pixel coordinates, so it does not measure progress.
- **There is no evaluation code at all**: no win-rate counting, no tournament, no fixed seeds, no confidence intervals, and none of the "online agents" it was compared against. The ~70% therefore cannot be reproduced or compared; it is most likely against weak opponents in his own game.
- Conclusion: our agent is ahead on every technical axis (legal-move masking, replay, target network, egocentric input with the dice roll, seeded and paired evaluation, statistics). **Two ideas are worth taking**, both already supported by the literature:
  1. Train against copies of itself (his four-colour self-play).
  2. Reward forward progress (his +1 per move and +25 per move in the home column), done properly in Stage 2.
- Nothing else is adopted, and no code is reused.

**Goal:** the strongest honest DQN-family agent we can build, evaluated properly. **Realistic target:** 38–45% against 3 Heuristic, well above anything published for that setup. Time is not a constraint. Engineered features are allowed; the raw-board DQN stays as a separate reported result.

## Rules for every experiment (fixing the earlier mistakes)

1. **Write the hypothesis and pass mark in `docs/decisions_log.md` before the run starts.**
2. **One change per pilot**, seed 0, scored on the paired 2,000-game tournament (`TOURNAMENT_SEED_OFFSET`) against 3 Heuristic.
3. **Noise-aware decisions.**
   - One 2,000-game score has ~±1pp noise; a difference between two runs has ~±1.5pp.
   - A pilot must beat the current best by +3pp to be *promoted*, then be **confirmed on seeds 1 and 2** before it is adopted.
   - Anything within ±3pp is reported as "no detectable change", not as a regression.
4. **Check what was learned, not only the win rate.** For every pilot, record the learning curve per 1,000 episodes, the shaped reward, the share of the return that comes from the win term, and the agreement-with-Heuristic diagnostic (scratchpad `diag_agree.py`, moved into `training/diagnostics.py`).
5. **All new switches default off**, so existing checkpoints and configs load unchanged (existing pattern in `training/config.py`, `env/state_encoding.py`).
6. No git staging or commits; the user handles git. One-line comments only.

## Stage 0: Records (no training)

- **New `docs/dissertation_notes.md`** with:
  - the convergence table and the "wrong objective" diagnosis;
  - the agreement numbers and why each pilot failed;
  - the single-seed noise caution;
  - the critique of the online reward list;
  - the literature comparison table, with each figure's player count and opponent type;
  - the three levels of hand-crafting (policy / inputs / rewards).
- Add a memory pointer to it.
- Add a decisions-log entry for the analysis and for this plan.
- Mark `dqn-improve-plan.md` as superseded, with the four pilot results filled in.

## Stage 1: Two cheap experiments (no training)

1. **Expert opponent** (`agents/expert_agent.py`, reusing `_classify_token` in `agents/heuristic_agent.py` and `env/threats.py`): Alhajry's expert order (release on a 6, defend a threatened token, capture, advance the lead token). Score it as one seat against 3 Heuristic, 2,000 games. This gives a second, stronger benchmark and training opponent.
2. **Headroom estimate** (`agents/rollout_agent.py`): for each legal move, apply it on a board copy (`env.moves.apply_move`) and play the game out N times with Heuristic players in every seat; pick the move with the best win fraction. Score over 500–1,000 games.
   - This is the first real estimate of how far above 31% is reachable, and it fixes the final target.
   - If it cannot clear ~35%, the ceiling is close and the dissertation says so with evidence.

## Stage 2: Fix the objective (the main fix)

Two pilots, run in order:

- **Pilot B: gamma 0.99**, nothing else changed.
- **Pilot C: gamma 0.99 + win-dominant rewards** (`reward_mode="potential"` in `env/rewards.py`, wired in `env/ludo_env.py`).
  - Rewards: win +1, loss −1, and one shaping term, `gamma * Φ(next) − Φ(now)`. Making the win the largest reward matches the contact's ratio (+100 for all tokens home against +20 for a capture). *(Derived from the contact's code.)*
  - `Φ = (own total progress − mean opponent progress) / normaliser`, with `Φ = 0` at the end of the game.
  - This single term pays for moving forward (the contact's progress idea done properly), pays for captures in proportion to what the opponent loses, and charges for being captured in proportion to what we lose.
  - In this form it cannot be farmed and does not change which policy is best at winning.
  - Unit test: the shaping terms sum to `−Φ(start)` over a full game.
- If C wins, add **n-step returns** (n = 3, then 5) in `agents/replay_buffer.py` / `training/dqn_trainer.py` as the next single change.

## Stage 3: Afterstate DQN (strongest support in the literature)

Instead of four outputs `Q(state, token)`, the network scores the board that results from each legal move, `V(board after move)`, and picks the best. It is still a DQN: replay buffer, target network, and the target `r + gamma * max over next legal moves of V_target(next afterstate)`.

- New `agents/afterstate_agent.py` and `network_type="afterstate"`. The simulate-on-a-copy pattern already exists in `env/tactical_features.py` and `agents/lookahead_agent.py`.
- Afterstate input: the 64-value board after the move, plus threat values and race standing for that position (reusing `encode_threat_features` and `_race_standing`). Every token and every game trains the same single scorer.
- Replay stores the candidate afterstates of the next decision (up to 4 per transition).
- Pilot on seed 0 with the best Stage 2 objective.

## Stage 4: Scale up the winner

One change at a time on the best model so far:

1. **Longer training**: 50,000 then 100,000 episodes, with exploration decay and target sync scaled to match.
2. **Opponent pool with self-play** *(self-play idea also derived from the contact's code)*.
   - Each training game draws its three opponents from Heuristic, Expert, Random, and frozen earlier checkpoints of the agent itself.
   - Heuristic stays in the mix so the benchmark score cannot drift.
   - Unlike the contact's version: one learner, frozen copies as opponents, and replay plus a target network.
3. **Optional, reported as a separate row**: one-roll expectimax at play time using the learned value as the evaluator ("DQN + search").

## Optional: playable demo

A simple screen where a person plays against the trained agent, to show the agent in the presentation. It is not used for any reported number. *(Idea derived from the contact's human-vs-AI game modes.)*

## Stage 5: Final evaluation

- Retrain the final configuration on **5 seeds**; score final checkpoints only.
- Tables: vs 3 Random, vs 3 Heuristic, vs 3 Expert, and vs a mixed table. Each is 2,000 games per seed, so 10,000 per agent, matching the proposal.
- Rotate the agent's seat across the four positions (`agent_player_id`); the log records weak evidence of a seat effect.
- Statistics as already built in `training/statistics.py`: the seed is the unit, Student-t interval across seeds, Holm-adjusted tests.
- Ablation table for the final model: each adopted change removed in turn (one seed each, labelled as such).
- Results rows:
  - Random, Heuristic, Expert, Tabular Q, raw-board DQN, Double DQN;
  - feature DQN (31.35%), the final agent, and the final agent + search if built.
- Every number is stated with player count and opponent type.

## Removed from the previous version (unlikely to pay off)

- **Gamma 0.95 + win-dominant rewards pilot:** at gamma 0.95 the win is still invisible; B and C isolate what matters.
- **Heuristic-variant sweep** (trailing-first, safety-first, blockade on/off): replaced by the single Expert opponent. The model already avoids danger and declines blockades, so the sweep answers little.
- **Hidden size 256:** four pilots showed the network is not the bottleneck.
- **2-player tables:** the environment is 4-player throughout; the cost is not justified. The literature comparison is made with player count stated instead.
- **Phase 2.2/2.3 (dueling head, others) and graded-feature leave-one-out.**
- **PPO:** it breaks the proposal's DQN framing.

## Files that change

- `env/rewards.py`, `env/ludo_env.py`: potential reward mode.
- `training/config.py`, `training/quick_run.py`, `training/run_training.py`, `training/run_full_evaluation.py`: pass the new switches through; opponent pool; seat rotation; Expert table.
- `agents/replay_buffer.py`, `training/dqn_trainer.py`, `agents/dqn_agent.py`, `agents/dqn_network.py`: n-step returns, afterstate network and targets.
- New: `agents/expert_agent.py`, `agents/rollout_agent.py`, `agents/afterstate_agent.py`, `training/diagnostics.py`, `docs/dissertation_notes.md`, and a test file for each.
- `docs/decisions_log.md`: one entry per hypothesis and per result.

## Verification

- The full `pytest` suite passes after every stage.
- With all new switches off, the 108-value observation and the dense rewards are byte-for-byte unchanged (existing tests cover this pattern).
- New unit tests:
  - potential shaping telescopes;
  - n-step returns match a hand-computed example;
  - the afterstate agent always picks a legal token and gives identical scores to identical resulting boards;
  - the Expert and rollout agents always pick legal tokens.
- Sanity anchors before trusting any new number:
  - Random vs 3 Random ≈ 25%;
  - Heuristic mirror ≈ 24%;
  - the existing seed-0 baseline re-scores at 31.85% on the same dice seed.
- The user starts long runs in an external terminal with output logged to a file; scoring scripts run afterwards.

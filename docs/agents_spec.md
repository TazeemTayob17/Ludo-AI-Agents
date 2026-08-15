# Agents Specification

This is the fixed agent scope for the project (Step 6 of the build plan). It determines how Steps 7-10 (baselines, DQN architecture, training, evaluation) are scoped. Any change to this list is a decision, not an assumption — it gets logged in `docs/decisions_log.md` before the affected steps are touched.

Status: **confirmed 2026-08-15**.

---

## 1. In scope (core five)

- **Random** — samples uniformly from the legal action mask each decision. No training. The statistical floor baseline; also the no-skill reference point for the 25% win-rate comparison in Step 11's evaluation.
- **Heuristic** — a fixed priority list, no training: capture > exit base > enter home > form/join a blockade > advance the most-advanced token generically. The sanity-check baseline: DQN must clearly beat this, not just Random, to demonstrate it learned something beyond hand-written rules.
- **Tabular Q-learning** — off-policy TD control over a discretized state representation (the discretization scheme is the real design work here, per Step 7.3 — must be fine-grained enough to be a genuinely competitive baseline, not a strawman). The non-deep RL baseline; justifies the "adapting *deep* RL" framing in the title by giving DQN something concrete to outperform.
- **DQN** — the project's main contribution. A neural network trained via experience replay and a target network on the egocentric observation from `env/state_encoding.py`, action-masked throughout (both action selection and the Bellman target, per `docs/env_api.md`).
- **Double DQN** — DQN with the action-selection/action-evaluation decoupling in the Bellman target. Near-zero incremental cost once DQN exists; tests whether overestimation bias measurably affects performance in this environment.

## 2. Explicitly deferred (not in core scope, revisit only if time allows)

- **SARSA** (tabular, on-policy) — cheap to add once Tabular Q-learning's discretization scheme exists, and would set up an on-policy vs off-policy comparison specific to Ludo's capture-risk dynamic. Deferred because it isn't needed for the core contribution.
- **Monte Carlo Tree Search (MCTS)** — a stretch goal, not deferred lightly. Build cost is comparable to DQN itself (tree search with chance nodes for dice, no reusable training-free shortcut), and as an online planning method it costs real compute per decision, which could slow the Step 11 tournament significantly depending on simulation budget. High narrative value (directly addresses "stochasticity" via chance nodes, strong AlphaGo/AlphaZero recognition) but only worth taking on after the core five are trained and evaluated, so it can't put the core contribution at risk.

Both deferred items were considered during Step 6 planning (see `docs/decisions_log.md`), not overlooked.

## 3. Why this scope

The core five is the minimum needed for the dissertation's central comparison: does a DQN (and its Double DQN variant) meaningfully outperform both naive (Random, Heuristic) and classical non-deep (Tabular Q-learning) approaches in a stochastic, multi-agent-adjacent environment. Every agent on the core list is either a required baseline (Random, Heuristic), the contribution itself (DQN), or a near-free extension with real comparative value (Double DQN, Tabular Q-learning). SARSA and MCTS both add genuine value but at a cost (build time, in MCTS's case also evaluation runtime) that isn't justified until the core result exists and is defensible on its own.

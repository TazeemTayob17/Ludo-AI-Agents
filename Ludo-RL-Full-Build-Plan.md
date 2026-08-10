# Ludo RL Project — Full Build Plan (Step-by-Step)

This is the granular, code-level version of the phase plan — written so you can hand each numbered step to Claude Code one at a time, review the output, and move on. Agent count/type decisions are deferred to Step 6 as requested.

---

## STEP 1 — Repo Scaffolding
1.1 Create folder structure (`env/`, `agents/`, `training/`, `evaluation/`, `notebooks/`, `tests/`, `docs/`)
1.2 Set up virtual environment, install `gymnasium`, `numpy`, `torch` (CPU build — no CUDA GPU available), `matplotlib`, `pytest`; pin versions in `requirements.txt`
1.3 Set up `pytest` config and a placeholder passing test
1.4 Create `docs/decisions_log.md` — empty, ready for dated one-line-rationale entries starting Step 2
1.5 Initialize git, commit "Phase 0 scaffold"

**Checkpoint:** empty project runs, `pytest` passes, git history started.

---

## STEP 2 — Rules Specification (docs/rules_spec.md)
2.1 Write out your fixed Ludo ruleset (exact-six-to-exit, three-sixes-forfeit, capture-on-safe-square, blockade behaviour, bonus-roll triggers)
2.2 Ambiguities resolved (classic Ludo ruleset — Ludo variants disagree on all of these, so each is a documented decision, not an assumption):
   - Blockades: 2 same-color tokens on one square form a blockade that fully blocks the square — opponents cannot land on it or pass through it, and the blockade cannot be captured while intact.
   - Exact-count-to-home overshoot: a token needs the exact roll to land on the final home square; if the roll overshoots, that token has no legal move this roll (other tokens may still have legal moves with the same roll).
   - Three-sixes-forfeit: rolling three consecutive 6s in one turn forfeits the entire turn — all three rolls are discarded and no token moves, even though the first two 6s each granted a bonus roll before the bust.
   - Bonus roll on a "dead" 6: rolling a 6 always grants a bonus roll, regardless of whether that 6 resulted in any usable legal move.
2.3 Review it yourself line by line — this is a decisions document, not code, so Claude Code drafts it but you finalize it
2.4 Log each resolved ambiguity above as a dated entry in `docs/decisions_log.md` with a one-line rationale
2.5 Commit as the "contract" file all later steps reference

**Checkpoint:** rules_spec.md is unambiguous and final; every ambiguity above has a documented decision.

---

## STEP 3 — Board & State Representation
3.1 Build token position representation (16 tokens, base/track/home-stretch encoding)
3.2 Build coordinate transform functions (player-relative → absolute track position)
3.3 Define and test **egocentric state encoding**: the DQN's actual input vector must always represent "my tokens" vs "opponents' tokens" relative to the acting seat, never hardcoded to an absolute player index — otherwise the network effectively has to relearn the game per seat. Treat this as its own tested design decision, not an implicit side effect of 3.2's coordinate transforms.
3.4 Unit test transforms against hand-calculated examples, including the egocentric encoding from 3.3
3.5 Commit + review; log the encoding choice in `docs/decisions_log.md`

**Checkpoint:** you can construct any board state manually and the representation matches what you expect, from any seat's perspective.

---

## STEP 4 — Core Rules Engine
4.1 Dice roll function (test for uniform distribution); route dice RNG through the environment's seeded `self.np_random`, not bare `random`/`numpy.random` calls, so seeding actually controls reproducibility
4.2 Legal move generator (given state + roll → legal token indices), tested against every rule in rules_spec.md individually, including the ambiguity resolutions from Step 2.2
4.3 Step function: apply action, resolve captures, safe zones, blockades, turn order, bonus rolls, and:
   - Max-turn truncation: cap episode length and distinguish `terminated` (someone won) from `truncated` (hit the cap), per the Gymnasium API
   - No-legal-move handling: if a player has zero legal moves, the environment resolves/skips that turn internally — the agent is never handed a state with a fully-masked action vector
4.4 Reward function as a standalone configurable dict (+0.5/+0.2/+1.0/+2.0/-1.0/-0.5 or your chosen values — currently placeholders, tunable)
4.5 Reward-function unit tests as their own category, separate from legal-move tests: exact reward-value assertions per transition type (capture, exit-base, enter-home, safe-zone entry, truncation)
4.6 Action mask exposed alongside state
4.7 Write `docs/env_api.md`: the action-mask/`info`-dict contract (action_mask, dice_roll, current_player, captures_this_turn) — one source of truth so the env, DQN, and baseline agents can't drift apart on this
4.8 Review each of 4.1–4.7 individually — this is the highest bug-risk phase, don't batch-review

**Checkpoint:** every rule in your spec has a passing test; no exceptions. Truncation and no-legal-move behavior are both explicitly tested.

---

## STEP 5 — Gymnasium Wrapper + Validation
5.1 Implement `reset()`, `step()`, `render()` (text-based render is enough). `step()` must internally fast-forward through the three opponents' full turns (dice → policy decision → resolution, including bonus-roll sequences) and only return control once it's the agent's turn again or the episode has ended — this turn-cycling loop is the main Ludo-specific wrapper risk, give it its own test (confirm it terminates and doesn't stall when a seat has repeated no-legal-move turns)
5.2 Reward attribution: if an opponent captures the agent's token during the turns fast-forwarded inside a single `step()` call, that event's reward must be folded into the reward returned from the agent's next `step()` — test this explicitly with a constructed scenario
5.3 Smoke test: 1,000 fully random 4-player games to completion, asserting no crashes and no illegal states (token counts always sum correctly per player); run as an actual `pytest`, not a one-off script, so it reruns automatically on engine changes
5.4 Manually play a few games via `render()` to eyeball correctness
5.5 Commit as "Environment v1.0 — validated"

**Checkpoint:** 1,000/1,000 random games complete cleanly. Do not proceed to agents until this passes — bugs here silently corrupt everything downstream.

---

## STEP 6 — Decide Agent Scope *(your decision point)*
6.1 Confirm final agent list: Random / Heuristic / Tabular Q-learning / DQN / **Double DQN (in scope)** — given the low incremental engineering cost on top of a working DQN and the added comparative narrative (does overestimation bias matter in this environment), commit to building it rather than leaving it as a maybe
6.2 Update docs/rules_spec.md or a new docs/agents_spec.md noting exactly what's in scope
6.3 This determines how Steps 7–10 below are scoped — proceed once decided

---

## STEP 7 — Baseline Agents
7.1 Random agent — sample uniformly from action mask
7.2 Heuristic agent — write out priority list explicitly (capture > exit base > enter home > blockade > advance) before coding it, so it's defensible in your write-up
7.3 Tabular Q-learning agent — decide state-hashing/discretization scheme first (raw state space is too large for a table — this is worth a paragraph in your methodology). Document the information loss from discretization explicitly: if it's too aggressive, Tabular-Q becomes a strawman and "DQN beats Tabular-Q" stops being an interesting result — pick a scheme that's a genuinely competitive baseline, then implement
7.4 Round-robin sanity check, as an **automated regression test with a numeric threshold** (e.g., Heuristic win-rate > 40% over N games vs Random), not just an eyeball check — rerun this test whenever the engine changes to catch silent drift; if it fails, stop and debug the engine or heuristic before continuing

**Checkpoint:** baseline hierarchy makes intuitive sense (Random < Heuristic, roughly), enforced by an automated test.

---

## STEP 8 — DQN Architecture
8.1 Network module (3 FC layers + ReLU, 4 output nodes) — unit test on dummy tensors for shape correctness
8.2 Replay buffer — unit test push/sample independently of the environment; store the **action mask for both `s` and `s'`** in each transition (not just `s`), since masking is state-dependent and must be reapplied during training
8.3 Target network + sync-every-N-steps logic
8.4 Masked ε-greedy action selection (critical: never let raw argmax pick a masked-out action)
8.5 Masked Bellman target: `max_a' Q(s', a')` must also mask out illegal actions in the next state `s'`, or the network silently bootstraps off Q-values for actions that could never be taken. Verify this on a hand-constructed transition, not just the loss formula in isolation
8.6 Loss function per your Bellman equation — verify numerically on a tiny toy MDP first

**Checkpoint:** each component passes its own isolated unit test before touching the real environment, including the masked-target case in 8.5.

---

## STEP 9 — Toy Environment Validation
9.1 Build a simplified single-agent, fixed/no-dice version of the environment. Ensure it still **exercises action masking** (e.g., some tokens pre-placed at home/base/mid-track) rather than a trivial always-all-legal setup — otherwise this step validates the Bellman math but not the masking discipline, which is the highest-risk DQN-specific bug
9.2 Train DQN on it, confirm loss decreases and agent reliably reaches home
9.3 Only proceed to full training once this converges — this is your cheapest bug-catching checkpoint

**Checkpoint:** DQN visibly learns in the simplified setting, with masking actually exercised.

---

## STEP 10 — Full Training
10.1 Move training to the full 4-player stochastic environment
10.2 Log everything from the first run: episode reward, win/loss, loss curve, epsilon decay, game length (CSV as source of truth; TensorBoard optional for live monitoring only, since Step 12 regenerates figures from saved CSVs/logs)
10.3 Every training run reads a versioned config file (hyperparameters, reward dict, seed); save that exact config alongside its logs/checkpoints
10.4 Hyperparameter sweep: no GPU is available (CPU-only laptop), so a full 6-way factorial grid is impractical. Scope it to the 2-3 hyperparameters DQN literature identifies as most impactful for stability (learning rate, target-sync frequency, epsilon-decay schedule), sweep those with a small grid or random search (~8-12 configs), hold the rest at literature-typical defaults, and document this scoping decision (hardware-justified, not arbitrary) in `docs/decisions_log.md`
10.5 Train each learning agent (DQN, Double DQN, Tabular-Q) across **multiple seeds (e.g., 5)** — a single trained network's tournament result tells you about that network, not about the method
10.6 Checkpoint models regularly (e.g. every 10k episodes) **and** track best-so-far by evaluation win-rate, since DQN training curves aren't monotonic; embed episode count, config, and git commit hash in checkpoint metadata
10.7 Tag git commits at major milestones (`v0.1-engine-validated`, `v0.2-baselines`, `v0.3-dqn-toy-converged`, `v1.0-full-eval`) so every result traces to an exact commit

**Checkpoint:** stable win rate against baselines across repeated evaluation runs and across seeds, not a single lucky run.

---

## STEP 11 — Evaluation
11.1 Freeze all agents (no more training)
11.2 Run the 10,000-game round-robin tournament, per trained seed from Step 10.5
11.3 Compute win rate, average game length, capture-to-death ratio. Decide the statistical test explicitly rather than leaving it generic: e.g., a binomial/chi-squared test against the 25% no-skill baseline for a 4-player symmetric game, plus confidence intervals; report mean ± std across the multi-seed training runs from 10.5, not a single run's numbers
11.4 Reward-shaping ablation (dense-shaped vs sparse win/lose-only reward) — **planned, not optional**, since reward values are flexible placeholders and this directly interrogates whether your hand-picked shaping is doing the work. DQN vs Double DQN and 2-player vs 4-player performance remain optional ablations if time allows
11.5 Save all results to CSV — never hand-copy terminal output into your report

**Checkpoint:** every metric from your proposal's evaluation protocol has a final number, on disk, with variance across seeds reported.

---

## STEP 12 — Documentation
12.1 Regenerate all figures programmatically from saved CSVs/logs
12.2 Write results mapped directly to your three secondary research questions
12.3 Use `docs/decisions_log.md` as primary source material for the methodology chapter — it already has dated rationale for every non-obvious call made since Step 2
12.4 Update your Wits AI declaration form to reflect Claude Code usage across these steps

**Checkpoint:** report finished, figures reproducible, declaration updated.

---

## OPTIONAL ADD-ON (post-project, after Step 12) — Visual Renderer
Not required for any research question, objective, or evaluation metric — purely a nice-to-have for demos/presentation.

A.1 Build a simple visual board renderer (matplotlib or pygame) that draws token positions from the existing state representation
A.2 Reuse Step 5's `render()` hook rather than modifying the core engine — the visual layer should sit on top of, not inside, the environment
A.3 Optionally add basic animation/replay of a saved game log for demo purposes
A.4 Skip entirely with zero impact on your results if time runs out — this only touches presentation, not methodology

**Checkpoint:** none required — this is scope-flexible and can be dropped at any point.

---

## Working Notes
- Review gates: Steps 4, 8, and 9 are where bugs are most likely and most costly if missed — read every line Claude Code generates there, not just the diff summary.
- Keep rules_spec.md, agents_spec.md, env_api.md, and the reward-shaping dict as fixed reference files Claude Code re-reads before any engine/agent change, so nothing silently drifts from your documented methodology.
- Test categories to keep distinct, not folded into one generic "tests": rules-engine tests (one per rule), reward-function tests, action-mask tests (including the masked-Bellman-target case), environment invariants (the 1,000-game smoke test as a real pytest), agent unit tests (network shape, replay buffer, target sync), and the Heuristic-beats-Random regression gate.
- Commit after every checkpoint, not just at phase ends — gives you clean rollback points if a later step reveals an earlier bug. Tag major milestones per Step 10.7.
- Append to `docs/decisions_log.md` as decisions are made, not retroactively — it's the cheapest insurance for the methodology chapter.

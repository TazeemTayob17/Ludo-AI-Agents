# Dissertation notes: why the feature DQN plateaued at ~31% and what we changed (2026-09-30)

Working notes for the write-up. Numbers come from `docs/decisions_log.md` and `runs/`; nothing here is new data except where marked (agreement diagnostic, 2026-09-30).

## 1. Where we stood

- Feature DQN (108-value observation): **31.35%** mean over 5 seeds against 3 Heuristic (SD 0.97pp, 95% CI [30.15%, 32.55%]); seed 0 = 31.85%. Fair share is 25%; Heuristic mirror is 24.20%.
- Against 3 Random: **76.06%** (5 seeds). Random vs 3 Random: 25.40% (sanity check).
- Raw-board DQN: 8.7%. Ablation (seed 0): control 7.50%, threats-only 9.00%, moves-only 29.85%, both 31.85%.

## 2. Four pilots that failed (seed 0, 2,000 paired games vs 3 Heuristic, baseline 31.85%)

| Pilot | Change | Win rate | vs baseline | Reading |
|---|---|---|---|---|
| 1.1 | Huber loss + gradient clip 10 | 26.85% | -5.00pp | real drop |
| 1.2 | Batch 128, replay 200k | 30.50% | -1.35pp | noise |
| 2.1 | Shared per-token network | 27.25% | -4.60pp | real drop |
| 3 | Graded feature block (144 values) | 28.60% | -3.25pp | borderline drop |

Common factor: all four changed the *network, optimiser or inputs*. None changed the *objective*.

## 3. Diagnosis: the objective is misaligned, not the model

- **Every run converges at the same place.** Win rate, shaped reward (~10 per game) and loss (~0.25) go flat from about episode 7,000, right after epsilon reaches 0.05 at episode 6,000. More episodes of the same setup will not help.
- **The agent is not trained to win.** Gamma 0.95 over ~120 decisions gives a horizon of ~20 decisions. The +2 win reward is worth about 0.004 at the start of a game, against +1 per capture. The discounted return is dominated by captures, exits and safe squares.
- **The shaping rewards are the Heuristic's priority list plus "don't get captured".** The learned policy is "Heuristic + danger avoidance".
- **Agreement diagnostic (300 games, baseline seed 0, won 31.0%):**
  - 74% of decisions had more than one legal move; the DQN agreed with the Heuristic on 52% of those.
  - Where it disagreed, it chose the safer move 3,684 times and the riskier move 576 times.
  - The Heuristic's "blockade" choice became "advance" 1,343 times.
  - Different-token "advance vs advance" accounted for 9,723 disagreements.
  - Captures and home entries almost always matched the Heuristic.
- **Conclusion:** inputs, network size and optimiser were not the bottleneck. A different objective (win-dominant reward, longer horizon) is the untested lever. The earlier gamma 0.99 test was on the raw-board model and does not transfer to the feature model.

## 4. Noise caution (single-seed results)

- One 2,000-game score has about ±1pp of sampling noise; the difference between two runs about ±1.5pp.
- Seed spread for the feature DQN is SD ~1pp, so a single seed-0 pilot cannot resolve differences under ~3pp.
- New rule: promote only at +3pp, confirm on seeds 1 and 2, otherwise report "no detectable change". Do not call within-noise differences regressions.

## 5. Critique of the online reward list

The list found online (large step rewards for exiting base, entering safe squares, a per-step forward bonus, big capture bonuses) is the same pattern we already used, and it has the same flaws:
- It pays for the Heuristic's intermediate goals, so the agent learns to imitate the Heuristic, not to win.
- Reward terms that are not potential-based can be farmed and can change which policy is optimal.
- Fix adopted in the plan: win/loss as the largest reward, plus one potential-based progress term `gamma*Phi(s') - Phi(s)` with `Phi = own progress - mean opponent progress`, which leaves the optimal policy unchanged (Ng, Harada & Russell, 1999).

## 6. Literature comparison (state player count and opponent type with every number)

| Source | Setup | Win rate |
|---|---|---|
| Published agents, 4-player vs 3 competent opponents | various | ~30-41% |
| Published agents vs random opponents | 4-player | 59-66% |
| Alhajry TD(lambda) (secondary source) | vs expert / vs random | ~30% / ~66% |
| Our feature DQN | 4-player vs 3 Heuristic | 31.35% |
| Our feature DQN | 4-player vs 3 Random | 76.06% |
| Contact's "~70%" | unverifiable: no evaluation code, no opponent spec | not comparable |

- No source found reports 60%+ for one agent against three competent opponents in 4-player Ludo. Every 60%+ figure is vs random opponents or in a 2-player game.
- Realistic target against 3 Heuristic: 38-45%.
- Techniques with literature support: afterstate evaluation (TD-Gammon style), gamma ~0.99 with a win-dominant reward, potential-based shaping, much longer training, self-play or opponent pools, shallow expectimax.

## 7. Three levels of hand-crafting (framing for RQ1 and the proposal)

1. **Hand-crafted policy** (the Heuristic, a lookahead player): the behaviour is written by the designer. We use these only as opponents and benchmarks.
2. **Hand-crafted inputs** (move and threat features): the designer chooses what the network sees, and the network learns what to do. This is where our gain came from (7.5% -> 31.85%).
3. **Hand-crafted rewards** (shaping): the designer chooses what is paid. Done badly this smuggles in a policy (our diagnosis); done as potential-based shaping it does not.

The proposal's RQ1 says "without relying on handcrafted heuristics". Our position: no hand-written policy is used; level 2 is state representation (RQ3). The raw-board DQN and the feature DQN are reported as two separate results, and the gap is framed as evidence of how much representation matters.

## 8. The contact's agent (derived from my friend's code; analysed 2026-09-30)

- A single pygame file with four separate Q-networks (32 -> 64 -> 4, ~2,400 weights), one per colour, learning online in self-play. Input is raw pixel coordinates, no dice roll, no replay buffer, no target network, gamma 0.9, fixed epsilon 0.1.
- Rewards: +100 all home, +50 token home, +25 per move in home column, +20 capture, +10 safe, +5 out of base, +1 forward, -100 no change.
- Bugs: the capture flag is never set; rewards are computed for the next player; the +5 is paid on every move; "forward" compares pixel coordinates.
- No evaluation code of any kind, so ~70% cannot be reproduced.
- Adopted ideas (each marked "derived from my friend" in the plan): self-play opponents, a forward-progress reward (done properly as potential shaping), a win-dominant reward ratio, and an optional human-vs-AI demo. No code is reused.

## 9. Link to the research proposal

- RQ1 (DQN with replay, no hand-crafted policy): consistent; the afterstate DQN and the opponent pool keep replay and a target network.
- RQ2 (vs Tabular Q and Heuristic; win rate, game length): add Expert and Random columns, report game length and capture-to-death ratio.
- RQ3 (state representation and dense reward shaping): the move/threat features and the potential-vs-dense reward comparison are direct answers.
- RQ4 (stability via target network and replay): keep both; n-step returns and a longer horizon are tested one at a time.
- Evaluation: 10,000 tournament games per agent (5 seeds x 2,000), seat rotation, seed-level t-intervals and Holm-adjusted tests.
- The proposal's literature review cites Alhajry at ~30% against 3 expert opponents, which matches our 31.35%; this supports reading ~31% as the competent-opponent regime, not a failure.

## 10. Why per-decision skill cannot be measured (noise study, 2026-10-05)

- 48 real multi-move decisions, 600 rollouts per candidate move with shared dice and Heuristic players in every seat; each chooser is judged on 300 held-out rollouts.
- The best and worst move differ by only 2.3pp of win chance on average (best vs second: median 1.3pp).

| Chooser | Regret per decision (pp) |
|---|---|
| Random | 1.48 |
| Heuristic | 0.80 |
| Feature DQN | 0.78 |
| Rollout (300) | 0.69 |
| Expert | 0.64 |

- Every standard error is about 0.2pp, so all competent choosers are tied per decision; skill is the sum of tiny edges over ~90 decisions per game.
- Consequence: a rollout agent cannot estimate the headroom above 31% at any affordable budget, and full-game tournaments are the only reliable judge. Value-prediction screening (who wins from this position) is used for representation and capacity questions instead.
- Pilot C correction: the potential paid +1 for leaving Base against +6 for advancing 6 squares, so it taught the agent not to leave Base (a design error, not a weak signal).

## 11. Representation, capacity and feature screening (2026-10-05)

- Method: a fast value-prediction screen (does the encoding plus a small network predict who wins from a position?) over 1.37M Heuristic self-play decisions, split by game, 3 seeds per cell. It tests information content in minutes instead of hours per reinforcement-learning run, but it cannot see per-move information.
- Findings: canonical token order is the one consistent gain; the track layout helps only without it; 256 or 512 hidden units do not help; the risk and race blocks do not help; race standing alone already gets most of the predictable signal (log-loss 0.5134 against 0.4980 for the best encoding and 0.5780 for the constant prior).
- Error analysis of the feature DQN (400 games): 8.0 own tokens captured per game (Heuristic 9.2), 6.9% of multi-move choices riskier than necessary (Heuristic 13.5%), almost no skipped captures, exits or home entries, and 81% of lost games had a race lead at some point. The agent is already past the obvious mistakes, which is consistent with a plateau near 31% when the opponents are competent and the dice dominate.
- Framing for the lecturer's three suggestions: a bigger network (tested: no gain in the screen), more features (tested: no gain beyond the existing ones), and board representation and information available (the one place with a measurable gain: order-free handling of tokens).

## 12. Pilot outcomes for the lecturer's suggestions (2026-10-14)

- Bigger network (hidden 256, feature DQN, 5 seeds): 26.55% (SD 2.67pp) vs 31.35% (SD 0.97pp) for 128 units, a 4.8pp drop. Matches the value screen's finding that 256 and 512 units give no gain; here a wider network at unchanged settings was worse and less stable.
- Track board (raw DQN, 5 seeds): 7.85% (SD 1.74pp) vs 8.70% (SD 2.03pp), no detectable effect. Canonical token order: -4.40pp on the feature agent, no effect on the raw agent over 3 seeds (Step 21).
- Write-up point: three representation and capacity changes suggested as improvements were each tested against multi-seed controls and none helped. The value-prediction screen predicted gains from canonical order that did not transfer to play, so screening cannot replace the tournament.
- Final feature DQN vs 3 Expert (5 seeds, seats rotated, 2,000 games each): 25.84% (SD 0.71pp, 95% CI [24.96%, 26.72%]), about even with the 25% fair share. Against 3 Heuristic it scores 31.35% and against 3 Random 76.06%, so its advantage shrinks sharply against a stronger opponent that it was not trained against.

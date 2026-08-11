# Decisions Log

Dated, one-line-rationale entries for every non-obvious design decision made during this project (rule interpretations, reward values, discretization schemes, hyperparameter choices, etc.). This is primary source material for the dissertation's methodology chapter — append as decisions are made, not retroactively.

Format: `YYYY-MM-DD — decision — rationale`

## Step 2 — Rules Specification

- 2026-08-11 — Blockades (2+ own tokens on one main-track square) fully block opponent landing and passage, and cannot be captured — chosen as the classic/most common Ludo convention; keeps blockades a meaningful defensive mechanic rather than a cosmetic one.
- 2026-08-11 — Exact-count-to-home required: a token with no exact roll to land precisely on relative_position 57 has no legal move that turn (other tokens may still move) — standard rule, preserves late-game tension rather than letting tokens "cap out" at home.
- 2026-08-11 — Three consecutive sixes forfeits the *entire* turn, including moves already made from the first two sixes, not just the third roll — chosen over the "only discard the third roll" alternative as the more well-known "three sixes = bust" house rule; has a real engine implication (turn must be buffered before commit, see rules_spec.md §9) that was flagged back for review.
- 2026-08-11 — A roll of 6 always grants a bonus roll, even if it resulted in no usable move (e.g. Base already has a token, or the only eligible token is blocked) — bonus-roll eligibility is tied to the die value itself, not to whether the roll was usable, matching standard implementations.
- 2026-08-11 — Board geometry fixed: 52-square shared main track, entry squares spaced 13 apart (global 0/13/26/39), 8 safe squares (4 entry + 4 star squares at +8 from each entry), 57-square total path per token (51 main-track + 6 home-column) — standard classic-Ludo numeric layout; this is the fixed contract Step 3's coordinate transforms implement against.
- 2026-08-11 — No stacking allowed in the home column: a token cannot land on a home-column square already occupied by another of the player's own tokens — blockades are a main-track-only mechanic, home column is a single-file private corridor.
- 2026-08-11 — Bonus rolls are also granted for capturing an opponent token and for a token reaching Home, not only for rolling a 6 (at most one bonus roll per die roll even if multiple conditions are met) — matches the standard three-trigger convention used in most digital Ludo implementations, chosen for a richer/more standard bonus-roll dynamic than a 6-only rule.
- 2026-08-11 — Three-sixes bust will be implemented via buffer-before-commit (simulate the full six-streak before mutating board state) rather than snapshot-and-rollback — avoids needing to un-resolve captures/other state changes if the third six busts the turn.
- 2026-08-11 — Episode/game ends immediately when any one player finishes all 4 tokens (first-to-finish, single winner) — no continued play to rank 2nd-4th place. Chosen over the full-ranking variant for a cleaner binary win/loss terminal signal and lower engine complexity, appropriate for the single-agent-vs-fixed-opponents RL framing.

# Ludo Rules Specification

This is the fixed ruleset contract for the project. Every later step (board representation, rules engine, legal move generation, reward function) implements exactly this document — nothing more, nothing less. If a rule needs to change later, it changes here first, gets a dated entry in `docs/decisions_log.md`, and only then gets implemented.

Status: **finalized and confirmed 2026-08-11** (per Step 2.3 of the build plan — drafted, reviewed line by line, confirmed). This is now the fixed contract Step 3 onward implements against.

---

## 1. Players & Tokens

- 4 players, indexed `0`-`3` (conventionally Red/Green/Yellow/Blue).
- Each player has 4 tokens, indexed `0`-`3`. 16 tokens total.
- Turn order is fixed rotation: Player 0 → 1 → 2 → 3 → 0 → ...
- Each token is always in exactly one of three states: **Base** (not yet in play), **Track** (on the main track or in the player's private home column), or **Home** (finished).

---

## 2. Board & Path Geometry **[CONFIRMED]**

This numeric layout is what Step 3's coordinate transforms will implement against, so it belongs here as the fixed contract, not left to Step 3 to invent.

- **Main track:** 52 squares total, shared by all players, indexed globally `0`-`51`, arranged in a loop.
- **Entry squares:** each player enters the main track at a fixed global square, spaced 13 apart (`52 / 4 players`):
  - Player 0 → global square `0`
  - Player 1 → global square `13`
  - Player 2 → global square `26`
  - Player 3 → global square `39`
- **Per-player relative position:** each token's progress is tracked as a position *relative to its own player's entry square*, not the global square directly:
  - `relative_position 0` = the player's entry square (the first square a token lands on when it exits base)
  - `relative_position 0-50` (51 squares) = travel along the shared main track: `global_square = (player_entry_offset + relative_position) mod 52`
  - `relative_position 51-56` (6 squares) = the player's private home column — no longer on the shared main track, not visible/reachable by any other player
  - `relative_position 57` = **Home** (finished) — reached only by landing exactly on it (see §5.3, exact-count rule)
  - Total path length per token: 57 squares of movement from exiting base to finishing.
- **Safe squares:** 8 global squares where captures never happen — the 4 entry squares plus 4 additional "star" squares, each 8 squares after the corresponding entry square:
  - `{0, 8, 13, 21, 26, 34, 39, 47}` (global track indices)
- The home column (`relative_position 51-56`) and Home (`57`) are always safe by construction — no opponent token can ever be there.

---

## 3. Dice **[CONFIRMED — standard]**

- A single fair six-sided die, values 1-6, uniform distribution, one roll at a time.

---

## 4. Turn Structure

4.1 On a player's turn, roll the die.
4.2 Determine all legal moves for the current player given the roll and board state (§5).
4.3 If there are zero legal moves, the turn passes to the next player automatically — no decision is presented to the acting agent/player for an all-illegal state. **[CONFIRMED]**
4.4 If there is at least one legal move, the player (or agent) chooses one token to move.
4.5 Apply the move and resolve its consequences (capture, blockade formation, reaching Home) — see §6-§9.
4.6 Determine whether a bonus roll is granted (§9) and whether the turn passes to the next player (§10).

---

## 5. Movement Rules

### 5.1 Exiting Base
- A token in Base can only move onto the track by rolling exactly a **6**, which places it at `relative_position 0` (its entry square).
- A player may have more than one token in Base; each 6 rolled lets the player choose *any one* eligible token in Base to exit (subject to legal-move rules generally — if multiple tokens are in Base, exiting any one of them is a distinct legal action).

### 5.2 Main-Track Movement (`relative_position` 0-50)
- A token advances by exactly the die roll: `new_relative_position = relative_position + roll`.
- **Path blocking [CONFIRMED, engine-relevant]:** because blockades fully block passage (§8), a move is illegal if *any* square the token would pass over **or land on** — from its current global square up to and including the destination — is occupied by an opponent blockade (2+ opposing tokens of the same color on one square). This means legal-move generation must check every intermediate square along the path, not just the destination.
- A player's own blockade never blocks that player's own other tokens — a player's tokens may freely pass through or land on a square already occupied by their own token(s), including joining an existing blockade (3 or 4 same-color tokens may co-locate on one square; it's still one blockade for opponent-blocking purposes).

### 5.3 Home Column & Finishing (`relative_position` 51-57) **[CONFIRMED]**
- Once `relative_position` would reach 51 or beyond, the token has left the shared main track and moves solely within its own private home column.
- **Exact count required [CONFIRMED]:** a token finishes only if `relative_position + roll == 57` exactly. If the roll would overshoot 57, that token has no legal move this roll (other tokens may still have legal moves with the same roll).
- **No stacking in the home column [DRAFTED]:** a token may not land on a home-column square already occupied by another of the player's own tokens — that destination is illegal for that token this roll. Blockades are a main-track-only mechanic; the home column is a single-file private corridor with no blockade concept.

---

## 6. Capturing **[CONFIRMED core rule, clarifications DRAFTED]**

- A capture occurs when a token lands on a main-track square occupied by exactly one opponent token, and that square is **not** a safe square (§2).
- The captured token is sent back to its owner's Base (state reset to Base; must re-exit via a future 6).
- A capture can never happen against a blockade (illegal to land there at all, per §5.2) — so at most one opponent token can ever be captured by a single move; there is no multi-capture case.
- The home column and Home are never capturable — no opponent token can ever be there.
- Multiple different-colored single tokens may coexist on the same safe square without capturing each other (§7).

---

## 7. Safe Squares **[CONFIRMED]**

- The 8 global squares listed in §2 (4 entry squares + 4 star squares) are safe: no capture can ever occur there, regardless of how many tokens of how many different colors occupy the square.
- A same-color blockade may still form on a safe square (e.g., a player's own two tokens both sitting on their entry square) — it is simply redundant with the square's inherent safety.

---

## 8. Blockades **[CONFIRMED]**

- Two or more of a player's own tokens on the same main-track square form a blockade.
- A blockade fully blocks that square against every opponent: opponents may neither land on it nor pass over it (§5.2).
- A blockade cannot be captured.
- Blockades do not exist in the home column (§5.3) — only on the shared main track.

---

## 9. Bonus Rolls **[CONFIRMED]**

A bonus roll (the same player rolls and moves again) is granted when any **one** of the following is true for the roll just resolved:
- the roll was a **6** (regardless of whether it resulted in a usable move) **[CONFIRMED]**
- the move captured an opponent token
- the move brought a token Home (finished)

At most **one** bonus roll is granted per die roll, even if multiple of the above are simultaneously true (e.g., a 6 that also captures still grants exactly one bonus roll, not two).

### Three consecutive sixes **[CONFIRMED]**
- Track a `consecutive_sixes` counter: incremented when a roll shows a 6, reset to 0 on any non-6 roll (a capture/home-entry bonus roll that isn't itself a 6 also resets the counter).
- If `consecutive_sixes` reaches 3 in one turn, **the entire turn is forfeited**: all moves made from that turn's rolls (including the first two sixes' moves) are discarded, the board reverts to its state at the start of the turn, and play passes to the next player.
- **Engineering note for Step 4:** because a turn's moves aren't final until we know whether a third consecutive 6 occurs, the engine must either (a) buffer/simulate the whole six-streak before committing any board mutation, or (b) snapshot board state at turn start and roll back on a bust. Buffering before commit is the simpler and recommended approach, particularly since captures would otherwise need to be un-done (opponent token un-captured) on rollback.

---

## 10. Turn End Conditions

- The turn passes to the next player when the most recent roll neither grants a bonus (§9) nor is followed by a legal move being available.
- If a player has zero legal moves for a given roll, the turn resolution for that roll still checks §9 (a 6 still grants a bonus roll even with no usable move — **[CONFIRMED]**); if that roll wasn't a 6 and produced no move, the turn simply passes.
- **Engineering safety note:** bonus rolls from repeated captures are not capped by any explicit game rule (only the three-sixes counter is capped). As a defensive measure against a bug causing an unbounded loop within a single environment `step()` call, Step 4/5 should include a generous internal safety cap on bonus rolls per turn (e.g., 100) that is never expected to trigger under correct rules, purely as a circuit breaker.

---

## 11. Winning Condition & Episode Termination **[CONFIRMED]**

- A player wins by moving all 4 of their tokens to Home (`relative_position 57`).
- **The episode/game ends immediately when any one player achieves this** — first-to-finish, single winner. Other players' unfinished tokens are irrelevant at that point; there is no continued play to rank 2nd/3rd/4th place.
- This is the simplest, most standard framing for a single-agent-vs-fixed-opponents RL setup (clean binary win/loss terminal signal, bounded episode complexity). The alternative (play continues, eliminating/skipping finished players, until all 4 finish, producing a full ranking) was considered and rejected in favor of this simpler framing.

---

## 12. Confirmation Record

All items in this document, including the five points below that were newly drafted (not previously discussed), were reviewed and confirmed on 2026-08-11:

1. Board/path numeric layout (52-square track, 13-square entry spacing, 8 safe squares at the stated positions, 57-square total path) — §2
2. No-stacking rule in the home column — §5.3
3. Bonus roll also granted on captures and on reaching Home, not just on rolling a 6 — §9
4. Buffer-before-commit implementation approach for the three-sixes bust — §9
5. Episode ends at first finisher (no ranking of 2nd-4th place) — §11

This document is now the finalized rules contract. Any future change to a rule here must be re-logged in `docs/decisions_log.md` with a dated rationale before the engine (Step 4+) is updated to match.

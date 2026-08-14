# Environment API Contract

This is the fixed contract for how the rules engine exposes state to any caller (baseline agents, the DQN, and later the Gymnasium wrapper in Step 5). Nothing outside `env/` should inspect `BoardState` directly to make decisions — everything a decision-maker needs is listed here.

## Action space

An action is a single integer `token_id` in `{0, 1, 2, 3}`, meaning "move my token with this id." Which values are actually legal for the current roll is given by the action mask below.

## Action mask

`env.legal_moves.get_action_mask(board, player_id, roll) -> np.ndarray[bool]`, shape `(4,)`. `mask[token_id]` is `True` if that token has a legal move for the given roll, per `docs/rules_spec.md` §5. A masked-out action must never be selected — this applies both at action-selection time and, later (Step 8.5), when computing the Bellman target's `max_a' Q(s', a')` over the next state.

## Per-roll decision inputs

Whenever a decision-maker (agent or scripted policy) is asked to choose a token, it receives:
- `board`: the current `BoardState`
- `player_id`: whose turn it is
- `roll`: the die value just rolled (`env.dice.roll_die`)
- `legal_tokens`: the tuple of legal token ids for this roll (`env.legal_moves.get_legal_tokens`) — equivalent to the `True` entries of the action mask

A decision-maker is only ever called when `legal_tokens` is non-empty — if a roll has zero legal moves, the turn engine (`env.turns.play_turn`) resolves that roll internally (see rules_spec.md §4.3, §10) without asking anyone to choose.

## Move outcome (`env.moves.MoveOutcome`)

Returned by `apply_move` after a chosen token is moved:
- `token_id`: which token moved
- `exited_base`: whether it just left Base
- `captured_player` / `captured_token`: the opponent token sent back to Base, or `None`/`None` if no capture happened
- `reached_home`: whether it just finished

This is the per-move equivalent of an `info` dict's `captures_this_turn` field — a caller accumulating a whole turn's captures should sum `captured_player is not None` across the `moves` list in the `TurnResult` below.

## Turn outcome (`env.turns.TurnResult`)

Returned by `play_turn` after a full player turn (including any bonus rolls) resolves:
- `player_id`
- `rolls`: every die value rolled this turn, in order
- `moves`: the `MoveOutcome` for each roll that had a legal move (skips rolls that had none)
- `busted`: `True` if three consecutive sixes forfeited the whole turn (board reverted to its state at turn start)
- `winner`: `True` if this turn brought the player's last token Home

## Game state (`env.game_state.GameState`)

Owns whole-game progression across many turns:
- `board`, `current_player`, `turn_count`
- `terminated`: `True` once a player has won (per the Gymnasium API distinction — see rules_spec.md §11)
- `truncated`: `True` once `turn_count` reaches `max_turns` without a winner
- `winner`: the winning player's id, once `terminated`

## Reward function (`env.rewards`)

`REWARD_CONFIG` is the single configurable reward dict; `compute_move_reward(outcome, board, player_id)` scores the agent's own move (capture, exit-base, reach-home, safe-square entry). `captured_penalty()`, `win_reward()`, `loss_reward()`, and `truncation_reward()` cover the game-level events that aren't tied to a single move. All values are placeholders, tunable, and subject to the reward-shaping ablation planned in Step 11.

Step 5's Gymnasium wrapper will translate `GameState` + the per-roll decision inputs above directly into `reset()`/`step()`'s `obs`, `reward`, `terminated`, `truncated`, and `info` — this document is the single source of truth for that mapping so the env, DQN, and baseline agents can't drift apart on it.

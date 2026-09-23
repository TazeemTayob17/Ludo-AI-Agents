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

**Caller contract for `reset()`:** Gymnasium's `reset()` signature never reports `terminated`/`truncated` (only `step()`'s does), but `LudoEnv.reset()` can rarely leave an already-truncated game — if the agent's own tokens never roll a 6 for long enough that `max_turns` is reached before its first real decision (negligible at real `max_turns` budgets like 1000, but possible at small ones used for fast testing/evaluation). Any rollout loop must read `env.game.terminated` / `env.game.truncated` right after `reset()` instead of assuming a decision is always pending — every trainer, `evaluate()`, and `training/tournament.py` follow this pattern.

## Observation (`env.state_encoding`)

`encode_observation(board, acting_player, roll=None, spec=None)` returns the egocentric board vector: 4 seats x 4 tokens x 4 features = 64 values, acting player's tokens first. `spec` is an `ObservationSpec` with three switches, all off by default; each one appends a block after the board features, always in this order:

| Switch | Block | Size |
|---|---|---|
| `include_dice_roll` | one-hot of the pending roll (all zeros at a terminal state) | 6 |
| `include_move_features` | per token: `is_legal`, `captures`, `exits_base`, `reaches_home`, `forms_blockade`, `lands_safe`, `escapes_threat`, `ends_in_danger` (all zeros for an illegal token or no pending roll) | 32 |
| `include_threat_features` | per token: `is_threatened`, `opponent_behind_closeness` (nearest opponent within 12 squares), `capture_opportunity` (smallest capturing roll) | 12 |

Use `observation_size(spec)` rather than a literal — it drives both `LudoEnv.observation_space` and `QNetwork`'s input layer. The move features are computed by applying each legal move to a copy of the board with the real `apply_move`, so they can't drift from the rules engine.

With every switch off the observation is byte-for-byte the original 64 values, so checkpoints trained before the switches existed still load. The switches are set per run on `TrainingConfig` (`observation_spec()` bundles them), saved into the run's `config.json`, and read back by `run_full_evaluation.load_trained_policy` / `run_observation_spec` so each agent is always evaluated in an env whose observation format matches the one it was trained on. Tabular Q's `discretize_state` takes none of these blocks, and keeps its original `_is_threatened` check so its existing results stay reproducible.

## Threat checks (`env.threats`)

`capture_distance`, `is_token_threatened`, `target_distance` and `nearest_opponent_behind` answer "who can capture whom on the next single roll" using the real movement rules: an attacker must stay on its main track (an opponent about to turn into its own home column is not a threat), cannot pass or land on a blockade of any other player, and cannot capture on a safe square. They look one roll ahead only — bonus-roll chains are not modelled.

## Seeding and reproducibility

`LudoEnv` uses an injected `rng` when one is supplied, and only otherwise falls back to Gymnasium's `self.np_random`. That fallback is **not** seeded by `reset(seed=None)`, so an env built without `rng=` draws its dice from OS entropy and is not reproducible. Any run whose results are reported must pass an explicit `rng` — `training/run_training.py::build_env` does this, deriving training and evaluation dice streams from the config's seed via `TRAINING_DICE_SEED_OFFSET` / `EVAL_DICE_SEED_OFFSET`, and `training/tournament.py::run_tournament` does it from its own `seed` argument.

## Reward function (`env.rewards`)

`REWARD_CONFIG` is the single configurable reward dict (the "dense" shaping used through Step 10); `SPARSE_REWARD_CONFIG` is Step 11.4's ablation target — every shaping term zeroed, only `win`/`loss` surviving at the same magnitudes. `compute_move_reward(outcome, board, player_id, config=None)` scores the agent's own move (capture, exit-base, reach-home, safe-square entry). `captured_penalty(config=None)`, `win_reward(config=None)`, `loss_reward(config=None)`, and `truncation_reward(config=None)` cover the game-level events that aren't tied to a single move. Every function defaults to `REWARD_CONFIG` when `config` is omitted; `LudoEnv(reward_config=...)` passes its chosen dict to all of them, so a single env instance is consistently dense- or sparse-shaped for its whole lifetime.

## Info dict: capture bookkeeping

Alongside `action_mask`/`dice_roll`/`current_player`, every `info` dict also carries `episode_captures_made` and `episode_times_captured` — running counts, reset each `reset()`, of how many times the agent has captured an opponent token and been captured itself so far this episode. Read from the final `info` after `terminated`/`truncated` to get an episode's totals (used by Step 11.3's capture-to-death ratio metric).

Both counts are consistent with the board: if the agent's turn ends in a three-sixes bust, the board reverts, and any capture made earlier in that same turn is decremented back out of `episode_captures_made`. The matching reward is refunded through `_pending_reward_adjustment` (the accumulator that also carries deferred got-captured penalties), so it lands on the agent's next `step()` — or is paid immediately if the episode ends first.

Step 5's Gymnasium wrapper will translate `GameState` + the per-roll decision inputs above directly into `reset()`/`step()`'s `obs`, `reward`, `terminated`, `truncated`, and `info` — this document is the single source of truth for that mapping so the env, DQN, and baseline agents can't drift apart on it.

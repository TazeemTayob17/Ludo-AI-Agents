# Third-party benchmark agents

Every file here is **our own re-implementation** of an agent published elsewhere, written from the source code or paper and
played under this project's rules and environment. None of the original code is copied, installed or run. Each file
records its source URL and the exact commit it was read from, and states where this project's 4-player rules forced a
change. Results against these agents are "in the style of" the original, not the original code itself.

| File | Source | Commit read | Notes |
|---|---|---|---|
| `alhajry_expert.py` | github.com/Nitinrajgor07/ludo-deep-reinforcement-learning, `phd_ludo_drl/opponents/expert_opponent.py` (MIT); Alhajry, Alvi and Ahmed, IEEE CIG 2012, doi 10.1109/CIG.2012.6374142 | `13edaa1bc17143d34dd60dd47e72253f2878e8a5` | 4-level priority list; the source's 2-player threat test is generalised to 3 opponents |

To remove this benchmark: delete `agents/third_party/`, `tests/test_alhajry_expert.py` and the `alhajry` entries in `training/score_run.py`.

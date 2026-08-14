# Dice rolling for the Ludo engine, seeded through an injected random generator.

from __future__ import annotations

import numpy as np

# Rolls a single fair six-sided die using the given random generator.
def roll_die(rng: np.random.Generator) -> int:
    return int(rng.integers(1, 7))

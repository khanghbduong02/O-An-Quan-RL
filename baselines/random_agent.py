"""Uniform random baseline."""

from __future__ import annotations

import random
from typing import Optional

from engine import Action, State


class RandomAgent:
    """Plays uniformly at random among the legal moves."""

    name = "random"

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = random.Random(seed)

    def choose(self, state: State) -> Action:
        actions = state.legal_actions()
        if not actions:
            raise ValueError("no legal move available")
        return self.rng.choice(actions)

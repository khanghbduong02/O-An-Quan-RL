"""Greedy baseline: take the move that scores the most points right now."""

from __future__ import annotations

import random
from typing import List

from engine import DEFAULT_RULES, Action, Rules, State


class GreedyAgent:
    """Maximises the points earned by the move itself, ties broken at random."""

    name = "greedy"

    def __init__(self, seed: int = 0, rules: Rules = DEFAULT_RULES) -> None:
        self.rng = random.Random(seed)
        self.rules = rules

    def choose(self, state: State) -> Action:
        best_points = -1
        best: List[Action] = []
        for action in state.legal_actions():
            _, result = state.apply(action, self.rules)
            if result.points > best_points:
                best_points = result.points
                best = [action]
            elif result.points == best_points:
                best.append(action)
        if not best:  # pragma: no cover - defensive
            raise ValueError("no legal move available")
        return self.rng.choice(best)

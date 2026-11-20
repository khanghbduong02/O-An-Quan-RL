"""Elo utilities: convert match scores to rating differences and fit a table."""

from __future__ import annotations

import math
from typing import Dict, Iterable, Sequence, Tuple


def elo_difference(score: float) -> float:
    """Rating advantage implied by a score in ``(0, 1)`` against an opponent."""
    if score <= 0.0:
        return -math.inf
    if score >= 1.0:
        return math.inf
    return -400.0 * math.log10(1.0 / score - 1.0)


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def fit_elo(
    names: Sequence[str],
    pairings: Iterable[Tuple[int, int, float, int]],
    iterations: int = 3000,
    learning_rate: float = 4.0,
) -> Dict[str, float]:
    """Fit Elo ratings by gradient ascent on the Bradley-Terry likelihood.

    ``pairings`` holds ``(i, j, score_of_i, games)``.  The mean rating is
    centred on zero, so only differences are meaningful.
    """
    pairings = list(pairings)
    total_games = sum(games for _, _, _, games in pairings) or 1
    ratings = [0.0] * len(names)
    for _ in range(iterations):
        gradient = [0.0] * len(names)
        for i, j, score, games in pairings:
            expected = expected_score(ratings[i], ratings[j])
            delta = games * (score - expected)
            gradient[i] += delta
            gradient[j] -= delta
        for index in range(len(ratings)):
            ratings[index] += learning_rate * gradient[index] / total_games
        mean = sum(ratings) / len(ratings)
        ratings = [rating - mean for rating in ratings]
    return {name: round(rating, 1) for name, rating in zip(names, ratings)}

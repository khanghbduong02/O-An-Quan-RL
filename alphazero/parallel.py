"""Multiprocess self-play.

Self-play is embarrassingly parallel: each worker receives the current network
weights and a seed, plays one game with its own search, and returns the game
record.  With one process per core a training run collects an order of
magnitude more games in the same wall-clock time.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Sequence, Tuple

import numpy as np

from engine import DEFAULT_RULES

from .network import PolicyValueNet
from .self_play import GameRecord, play_game

#: Everything a worker needs for one game.
GameJob = Tuple[Dict[str, np.ndarray], int, int, int, int, int, str]


def play_one(job: GameJob) -> GameRecord:
    """Play a single self-play game (runs in a worker process)."""
    weights, seed, hidden, layers, simulations, temperature_plies, value_target = job
    net = PolicyValueNet(hidden=hidden, layers=layers)
    net.params = {key: np.array(value) for key, value in weights.items()}
    return play_game(
        net,
        rules=DEFAULT_RULES,
        simulations=simulations,
        rng=np.random.default_rng(seed),
        temperature_plies=temperature_plies,
        value_target=value_target,
    )


class SelfPlayPool:
    """Thin wrapper around a process pool of ``play_one`` tasks."""

    def __init__(self, workers: int) -> None:
        if workers < 1:
            raise ValueError("workers must be at least 1")
        self.workers = workers
        self._pool: ProcessPoolExecutor | None = (
            ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
        )

    def play(
        self,
        weights: Dict[str, np.ndarray],
        games: int,
        *,
        hidden: int,
        layers: int,
        simulations: int,
        temperature_plies: int,
        value_target: str,
        seed: int,
    ) -> List[GameRecord]:
        if self._pool is None:
            return [play_one((weights, seed + index, hidden, layers, simulations, temperature_plies, value_target)) for index in range(games)]
        jobs: Sequence[GameJob] = [
            (weights, seed + index, hidden, layers, simulations, temperature_plies, value_target)
            for index in range(games)
        ]
        return list(self._pool.map(play_one, jobs))

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

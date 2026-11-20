"""Self-play game generation: MCTS decides, the network learns from the visit counts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from engine import DEFAULT_RULES, N_ACTIONS, Rules, State, action_from_index, initial_state

from .features import encode, legal_mask
from .mcts import MCTS
from .network import PolicyValueNet


@dataclass
class Sample:
    """One training position."""

    x: np.ndarray
    pi: np.ndarray  # search visit distribution
    mask: np.ndarray
    player: int
    z: float = 0.0  # filled in when the game ends


@dataclass
class GameRecord:
    samples: List[Sample] = field(default_factory=list)
    p0: int = 0
    p1: int = 0
    winner: int = -1  # -1 draw
    plies: int = 0


def _choose_action(pi: np.ndarray, temperature: float, rng: np.random.Generator) -> int:
    if temperature <= 1e-3:
        return int(np.argmax(pi))
    weights = np.power(pi, 1.0 / temperature)
    total = weights.sum()
    if total <= 0.0:  # pragma: no cover - defensive
        return int(np.argmax(pi))
    return int(rng.choice(N_ACTIONS, p=weights / total))


def play_game(
    net: PolicyValueNet,
    *,
    rules: Rules = DEFAULT_RULES,
    simulations: int = 100,
    rng: Optional[np.random.Generator] = None,
    temperature: float = 1.0,
    temperature_plies: int = 8,
    add_noise: bool = True,
    value_target: str = "outcome",
) -> GameRecord:
    """Play one full self-play game and attach value targets."""
    rng = np.random.default_rng(0) if rng is None else rng
    search = MCTS(net, rules=rules, simulations=simulations, rng=rng)
    state = initial_state(rules)
    samples: List[Sample] = []

    while not state.done:
        pi = search.search(state, add_noise=add_noise)
        samples.append(
            Sample(x=encode(state, rules), pi=pi, mask=legal_mask(state), player=state.player)
        )
        temperature_now = temperature if state.ply < temperature_plies else 1e-3
        action = _choose_action(pi, temperature_now, rng)
        state, _ = state.apply(action_from_index(state.player, action), rules)

    p0, p1 = state.final_points(rules)
    for sample in samples:
        mine, theirs = (p0, p1) if sample.player == 0 else (p1, p0)
        if value_target == "score_diff":
            sample.z = float(np.clip((mine - theirs) / 50.0, -1.0, 1.0))
        else:
            sample.z = 0.0 if mine == theirs else (1.0 if mine > theirs else -1.0)

    winner = -1 if p0 == p1 else (0 if p0 > p1 else 1)
    return GameRecord(samples=samples, p0=p0, p1=p1, winner=winner, plies=state.ply)

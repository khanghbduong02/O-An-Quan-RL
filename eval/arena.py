"""Match harness: play two agents, alternating who moves first."""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Any, Dict, Optional

from engine import DEFAULT_RULES, Rules, initial_state


@dataclass
class MatchResult:
    games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0

    @property
    def score(self) -> float:
        """Score for the first agent in [0, 1] (draws count half)."""
        return (self.wins + 0.5 * self.draws) / self.games if self.games else 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "games": self.games,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "score": round(self.score, 4),
        }


def play_game(
    agent_a,
    agent_b,
    *,
    first_player: int = 0,
    rules: Rules = DEFAULT_RULES,
    opening_plies: int = 0,
    opening_rng: Optional[random.Random] = None,
) -> int:
    """Return 0 when ``agent_a`` wins, 1 when ``agent_b`` wins, -1 for a draw.

    ``opening_plies`` random moves are played by both sides before the agents
    take over.  Deep agents are deterministic, so without it every game in a
    match would be the same game.
    """
    state = initial_state(rules)
    if first_player == 1:
        state = replace(state, player=1)
    if opening_plies:
        rng = opening_rng or random.Random(0)
        for _ in range(opening_plies):
            if state.done:
                break
            state, _ = state.apply(rng.choice(state.legal_actions()), rules)
    while not state.done:
        agent = agent_a if state.player == 0 else agent_b
        state, _ = state.apply(agent.choose(state), rules)
    p0, p1 = state.final_points(rules)
    if p0 == p1:
        return -1
    return 0 if p0 > p1 else 1


def play_match(
    agent_a,
    agent_b,
    games: int = 100,
    rules: Rules = DEFAULT_RULES,
    alternate: bool = True,
    opening_plies: int = 0,
    seed: int = 0,
) -> MatchResult:
    """Play ``games`` games; with ``alternate`` the starting player swaps."""
    result = MatchResult()
    opening_rng = random.Random(seed)
    for index in range(games):
        first = index % 2 if alternate else 0
        winner = play_game(
            agent_a,
            agent_b,
            first_player=first,
            rules=rules,
            opening_plies=opening_plies,
            opening_rng=opening_rng,
        )
        result.games += 1
        if winner == -1:
            result.draws += 1
        elif winner == 0:
            result.wins += 1
        else:
            result.losses += 1
    return result

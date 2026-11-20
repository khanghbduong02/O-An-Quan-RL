"""Alpha-beta baseline with a score-difference evaluation.

The game is deterministic and perfect-information, so a plain negamax search
with alpha-beta pruning is a strong classical opponent.  Moves are ordered by
the points they score immediately, which is enough for this branching factor,
and iterative deepening makes a wall-clock limit usable.
"""

from __future__ import annotations

import math
import random
import time
from typing import List, Optional, Sequence, Tuple

from engine import DEFAULT_RULES, PLAYER_CELLS, Action, Rules, State

_INF = math.inf


class _Timeout(Exception):
    pass


class AlphaBetaAgent:
    """Negamax + alpha-beta over the engine."""

    name = "alphabeta"

    def __init__(
        self,
        depth: int = 4,
        time_limit: Optional[float] = None,
        seed: int = 0,
        rules: Rules = DEFAULT_RULES,
        material_weight: float = 0.25,
        threat_weight: float = 0.0,
        check_every: int = 256,
    ) -> None:
        self.depth = depth
        self.time_limit = time_limit
        self.rules = rules
        self.material_weight = material_weight
        #: subtract this fraction of the opponent's best immediate capture from
        #: the leaf evaluation; even-depth leaves otherwise ignore the reply
        self.threat_weight = threat_weight
        self.check_every = check_every
        self.rng = random.Random(seed)
        self._nodes = 0

    # ------------------------------------------------------------------ public
    def choose(self, state: State) -> Action:
        actions = state.legal_actions()
        if not actions:
            raise ValueError("no legal move available")
        if len(actions) == 1:
            return actions[0]

        deadline = None if self.time_limit is None else time.perf_counter() + self.time_limit
        best_action = actions[0]
        self._nodes = 0
        for depth in range(1, self.depth + 1):
            try:
                action, _ = self._root(state, depth, deadline)
            except _Timeout:
                break
            best_action = action
        return best_action

    # -------------------------------------------------------------- evaluation
    def evaluate(self, state: State) -> float:
        """Score difference, board material and the side-to-move's best capture."""
        points = state.points(self.rules)
        mine, theirs = (points[0], points[1]) if state.player == 0 else (points[1], points[0])
        my_side = sum(state.cells[cell] for cell in PLAYER_CELLS[state.player])
        their_side = sum(state.cells[cell] for cell in PLAYER_CELLS[1 - state.player])
        score = (mine - theirs) + self.material_weight * (my_side - their_side)
        if self.threat_weight:
            best_capture = 0
            for action in state.legal_actions():
                _, result = state.apply(action, self.rules)
                if result.points > best_capture:
                    best_capture = result.points
            score -= self.threat_weight * best_capture
        return score

    def _terminal(self, state: State) -> float:
        p0, p1 = state.final_points(self.rules)
        return (p0 - p1) if state.player == 0 else (p1 - p0)

    def _ordered(self, state: State) -> List[Action]:
        scored: List[Tuple[float, Action]] = []
        for action in state.legal_actions():
            _, result = state.apply(action, self.rules)
            scored.append((result.points, action))
        scored.sort(key=lambda item: -item[0])
        return [action for _, action in scored]

    # ----------------------------------------------------------------- search
    def _root(self, state: State, depth: int, deadline: Optional[float]):
        alpha, beta = -_INF, _INF
        best_value = -_INF
        best_actions: List[Action] = []
        for action in self._ordered(state):
            child, _ = state.apply(action, self.rules)
            value = -self._search(child, depth - 1, -beta, -alpha, deadline)
            if value > best_value + 1e-9:
                best_value = value
                best_actions = [action]
            elif value > best_value - 1e-9:
                # Random ties: with many equal-valued moves, always taking the
                # first one makes the agent predictable and measurably weaker.
                best_actions.append(action)
            if value > alpha:
                alpha = value
        assert best_actions
        return self.rng.choice(best_actions), best_value

    def _search(
        self,
        state: State,
        depth: int,
        alpha: float,
        beta: float,
        deadline: Optional[float],
    ) -> float:
        self._nodes += 1
        if (
            deadline is not None
            and self._nodes % self.check_every == 0
            and time.perf_counter() > deadline
        ):
            raise _Timeout

        if state.done:
            return self._terminal(state)
        if depth <= 0:
            return self.evaluate(state)

        best = -_INF
        for action in self._ordered(state):
            child, _ = state.apply(action, self.rules)
            value = -self._search(child, depth - 1, -beta, -alpha, deadline)
            if value > best:
                best = value
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        return best

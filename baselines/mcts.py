"""Plain Monte Carlo tree search with random rollouts (classical baseline)."""

from __future__ import annotations

import math
import random
from typing import List, Optional

from engine import DEFAULT_RULES, Action, Rules, State


class _Node:
    __slots__ = ("state", "parent", "action", "children", "visits", "value", "untried")

    def __init__(self, state: State, parent: Optional["_Node"] = None, action: Optional[Action] = None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: List["_Node"] = []
        self.visits = 0
        self.value = 0.0
        self.untried: List[Action] = list(state.legal_actions()) if not state.done else []


class PlainMCTS:
    """UCB1 tree search; leaf values come from random playouts."""

    name = "mcts"

    def __init__(
        self,
        iterations: int = 200,
        exploration: float = 1.4,
        seed: int = 0,
        rules: Rules = DEFAULT_RULES,
        max_rollout_plies: Optional[int] = None,
    ) -> None:
        self.iterations = iterations
        self.exploration = exploration
        self.rules = rules
        self.max_rollout_plies = max_rollout_plies
        self.rng = random.Random(seed)

    def choose(self, state: State) -> Action:
        if state.done:
            raise ValueError("the game is over")
        root = _Node(state)
        for _ in range(self.iterations):
            node = root
            while not node.untried and node.children:
                node = self._select(node)
            if node.untried:
                action = node.untried.pop(self.rng.randrange(len(node.untried)))
                child_state, _ = node.state.apply(action, self.rules)
                child = _Node(child_state, node, action)
                node.children.append(child)
                node = child
            value = self._rollout(node.state)
            while node is not None:
                node.visits += 1
                node.value += value
                value = -value
                node = node.parent

        if not root.children:  # pragma: no cover - zero iterations
            return self.rng.choice(state.legal_actions())
        return max(root.children, key=lambda child: child.visits).action

    def _select(self, node: _Node) -> _Node:
        log_visits = math.log(max(node.visits, 1))
        best_score = -math.inf
        best_child = node.children[0]
        for child in node.children:
            if child.visits == 0:  # pragma: no cover - expansion covers this
                return child
            exploit = -(child.value / child.visits)  # child's view is the opponent's
            explore = self.exploration * math.sqrt(log_visits / child.visits)
            score = exploit + explore
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def _rollout(self, state: State) -> float:
        """Random playout; the result is scored for the player to move in
        ``state``.  It has to be captured before the playout because a finished
        position keeps ``player`` on whoever moved last, and on an already
        finished position that is the *opponent* of the node's mover."""
        perspective = (1 - state.player) if state.done else state.player
        plies = 0
        while not state.done:
            actions = state.legal_actions()
            state, _ = state.apply(actions[self.rng.randrange(len(actions))], self.rules)
            plies += 1
            if self.max_rollout_plies is not None and plies >= self.max_rollout_plies:
                break
        if state.done:
            p0, p1 = state.final_points(self.rules)
            mine, theirs = (p0, p1) if perspective == 0 else (p1, p0)
        else:
            points = state.points(self.rules)
            mine, theirs = (
                (points[0], points[1]) if perspective == 0 else (points[1], points[0])
            )
        if mine > theirs:
            return 1.0
        if theirs > mine:
            return -1.0
        return 0.0

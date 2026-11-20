"""PUCT Monte Carlo tree search, AlphaZero style.

Each simulation walks from the root to a leaf, evaluates the leaf with the
policy/value network, expands it and backs the value up, flipping the sign at
every level because the players alternate.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np

from engine import DEFAULT_RULES, N_ACTIONS, Rules, State, action_from_index

from .features import legal_mask
from .network import PolicyValueNet


class _Node:
    __slots__ = ("state", "prior", "visits", "value", "children")

    def __init__(self, state: State, prior: float = 1.0) -> None:
        self.state = state
        self.prior = prior
        self.visits = 0
        self.value = 0.0
        self.children: Dict[int, "_Node"] = {}


class MCTS:
    """Search with PUCT selection and network priors."""

    def __init__(
        self,
        net: PolicyValueNet,
        rules: Rules = DEFAULT_RULES,
        simulations: int = 100,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.net = net
        self.rules = rules
        self.simulations = simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.rng = np.random.default_rng(0) if rng is None else rng

    # ------------------------------------------------------------------ search
    def search(self, state: State, add_noise: bool = False) -> np.ndarray:
        """Return the normalised root visit distribution over action indices."""
        if state.done:
            return legal_mask(state)

        root = _Node(state)
        self._expand(root)
        if add_noise and root.children:
            self._add_root_noise(root)

        for _ in range(self.simulations):
            node = root
            path = [node]
            while node.children:
                node = self._select(node)
                path.append(node)
            value = self._expand(node) if not node.state.done else node.state.returns(self.rules)
            for visited in reversed(path):
                visited.visits += 1
                visited.value += value
                value = -value

        visits = np.zeros(N_ACTIONS, dtype=np.float64)
        for action, child in root.children.items():
            visits[action] = child.visits
        total = visits.sum()
        if total <= 0:  # pragma: no cover - happens only with zero simulations
            mask = legal_mask(state)
            return mask / max(mask.sum(), 1.0)
        return visits / total

    def policy(self, state: State, add_noise: bool = False) -> np.ndarray:
        """Search and return priors, for callers that want the tree's opinion."""
        return self.search(state, add_noise=add_noise)

    # ------------------------------------------------------------------- pieces
    def _expand(self, node: _Node) -> float:
        """Create children for ``node`` and return the network value."""
        if node.state.done:
            return node.state.returns(self.rules)
        probs, value = self.net.predict(node.state, self.rules)
        mask = legal_mask(node.state)
        player = node.state.player
        for index in range(N_ACTIONS):
            if mask[index] <= 0.0:
                continue
            child_state, _ = node.state.apply(action_from_index(player, index), self.rules)
            node.children[index] = _Node(child_state, prior=float(probs[index]))
        return float(value)

    def _select(self, node: _Node) -> _Node:
        best_score = -1e18
        best_child: Optional[_Node] = None
        sqrt_visits = math.sqrt(max(node.visits, 1))
        for child in node.children.values():
            # child values are from the child's point of view (the opponent)
            q = -(child.value / child.visits) if child.visits else 0.0
            u = self.c_puct * child.prior * sqrt_visits / (1 + child.visits)
            score = q + u
            if score > best_score:
                best_score = score
                best_child = child
        assert best_child is not None
        return best_child

    def _add_root_noise(self, root: _Node) -> None:
        actions = sorted(root.children)
        noise = self.rng.dirichlet([self.dirichlet_alpha] * len(actions))
        for action, epsilon in zip(actions, noise):
            child = root.children[action]
            child.prior = (1.0 - self.dirichlet_epsilon) * child.prior + (
                self.dirichlet_epsilon * float(epsilon)
            )

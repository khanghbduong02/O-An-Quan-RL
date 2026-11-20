"""Agent that plays a trained checkpoint: policy head alone, or with PUCT search."""

from __future__ import annotations

from typing import Optional

import numpy as np

from alphazero.mcts import MCTS
from alphazero.network import PolicyValueNet
from engine import DEFAULT_RULES, Action, Rules, State, action_from_index


class NetAgent:
    """Wraps a :class:`PolicyValueNet`.

    ``simulations=0`` plays the raw policy (the "no search" arm of the search
    experiment); a positive value runs PUCT before every move.
    """

    name = "net"

    def __init__(
        self,
        net: PolicyValueNet,
        simulations: int = 0,
        seed: int = 0,
        rules: Rules = DEFAULT_RULES,
    ) -> None:
        self.net = net
        self.simulations = simulations
        self.rules = rules
        self.rng = np.random.default_rng(seed)

    def choose(self, state: State) -> Action:
        if self.simulations > 0:
            search = MCTS(
                self.net, rules=self.rules, simulations=self.simulations, rng=self.rng
            )
            visits = search.search(state)
            return action_from_index(state.player, int(np.argmax(visits)))
        probs, _ = self.net.predict(state, self.rules)
        return action_from_index(state.player, int(np.argmax(probs)))

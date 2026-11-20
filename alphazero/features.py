"""State -> feature vector and legal-move mask.

The encoding is absolute (not perspective-normalised) because the action space
already is: action index 0..9 always means "my five squares x two directions",
whatever colour the network is playing.  The player-to-move feature tells the
network whose view it is.
"""

from __future__ import annotations

import numpy as np

from engine import DEFAULT_RULES, N_ACTIONS, Rules, State, action_index

#: 12 stone counts + 2 mandarin flags + 2 captured totals + 2 mandarins taken
#: + player to move + 10 "points this action captures" values
FEATURES = 29


def action_capture_points(state: State, rules: Rules = DEFAULT_RULES) -> np.ndarray:
    """Points each action would capture right now (0 for illegal actions).

    Without this the network has to infer, from raw board counts alone, which
    sowing move lands on a rich square: the policy head cannot express that and
    the value head misses the immediate swing, which makes a short search worse
    than no search at all.
    """
    points = np.zeros(N_ACTIONS, dtype=np.float32)
    for cell, direction in state.legal_actions():
        _, result = state.apply((cell, direction), rules)
        points[action_index(cell, direction)] = result.points
    return points


def encode(state: State, rules: Rules = DEFAULT_RULES) -> np.ndarray:
    """Feature vector for one position."""
    vector = np.zeros(FEATURES, dtype=np.float32)
    vector[0:12] = np.minimum(np.asarray(state.cells, dtype=np.float32), 40.0) / 20.0
    vector[12] = 1.0 if state.quan[0] else 0.0
    vector[13] = 1.0 if state.quan[1] else 0.0
    vector[14] = state.captured[0] / 50.0
    vector[15] = state.captured[1] / 50.0
    vector[16] = state.mandarins_taken[0] / 2.0
    vector[17] = state.mandarins_taken[1] / 2.0
    vector[18] = float(state.player)
    vector[19:29] = np.minimum(action_capture_points(state, rules), 40.0) / 20.0
    return vector


def legal_mask(state: State) -> np.ndarray:
    """1.0 for legal action indices of the player to move, else 0.0."""
    mask = np.zeros(N_ACTIONS, dtype=np.float32)
    for cell, direction in state.legal_actions():
        mask[action_index(cell, direction)] = 1.0
    return mask

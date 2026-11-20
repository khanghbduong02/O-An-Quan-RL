"""O an quan engine: board geometry, rules, state and move generation."""

from .layout import (
    CLOCKWISE,
    COUNTER_CLOCKWISE,
    DIRECTIONS,
    N_ACTIONS,
    N_ACTIONS_PER_PLAYER,
    N_CELLS,
    PLAYER_CELLS,
    QUAN_CELLS,
    action_from_index,
    action_index,
    action_name,
    is_quan_cell,
    live_quan,
    owner,
    quan_index,
    step,
)
from .render import format_board
from .rules import DEFAULT_RULES, Rules
from .state import (
    Action,
    EngineError,
    IllegalMoveError,
    MoveResult,
    State,
    initial_state,
)

__all__ = [
    "Action",
    "CLOCKWISE",
    "COUNTER_CLOCKWISE",
    "DEFAULT_RULES",
    "DIRECTIONS",
    "EngineError",
    "IllegalMoveError",
    "MoveResult",
    "N_ACTIONS",
    "N_ACTIONS_PER_PLAYER",
    "N_CELLS",
    "PLAYER_CELLS",
    "QUAN_CELLS",
    "Rules",
    "State",
    "action_from_index",
    "action_index",
    "action_name",
    "format_board",
    "initial_state",
    "is_quan_cell",
    "live_quan",
    "owner",
    "quan_index",
    "step",
]

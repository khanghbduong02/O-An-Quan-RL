"""Plain-ASCII rendering of a position (console output stays ASCII-safe)."""

from __future__ import annotations

from .layout import QUAN_CELLS, is_quan_cell, quan_index
from .rules import DEFAULT_RULES, Rules
from .state import State

_WIDTH = 7


def _cell_text(state: State, cell: int) -> str:
    label = f"{cell:02d}"
    if is_quan_cell(cell) and state.quan[quan_index(cell)]:
        label += "Q"
    return f"{label}:{state.cells[cell]}"


def format_board(state: State, rules: Rules = DEFAULT_RULES) -> str:
    """Draw the ring, the scores and whose turn it is."""
    column = f"+{'-' * _WIDTH}" * 5 + "+"

    def row(cells) -> str:
        return "|" + "|".join(f"{_cell_text(state, c):^{_WIDTH}}" for c in cells) + "|"

    left = f"{_cell_text(state, QUAN_CELLS[0]):^{_WIDTH}}"
    right = f"{_cell_text(state, QUAN_CELLS[1]):^{_WIDTH}}"
    pad = " " * (_WIDTH + 1)

    points = state.points(rules)
    lines = [
        f"  P0 (squares 1-5): {points[0]} points      "
        f"P1 (squares 7-11): {points[1]} points",
        "",
        pad + column,
        left + " " + row((1, 2, 3, 4, 5)) + " " + right,
        pad + column,
        pad + row((11, 10, 9, 8, 7)),
        pad + column,
        "",
        "  square index:stone count, Q = mandarin still there",
    ]
    if state.done:
        lines.append(f"  game over ({state.end_reason})")
    else:
        lines.append(f"  player {state.player} to move")
    return "\n".join(lines)

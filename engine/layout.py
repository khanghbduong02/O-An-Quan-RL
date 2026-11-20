"""Board geometry for O an quan.

The board is a ring of twelve squares::

                 +-----+-----+-----+-----+-----+
                 |  1  |  2  |  3  |  4  |  5  |     player 0 (top)
    (0) QUAN     +-----+-----+-----+-----+-----+     QUAN (6)
                 | 11  | 10  |  9  |  8  |  7  |     player 1 (bottom)
                 +-----+-----+-----+-----+-----+

Every square belongs to the same ring:

* ``0`` and ``6`` are the two mandarin squares (o quan).
* ``1..5`` are player 0's five citizen squares (o dan), left to right.
* ``7..11`` are player 1's five citizen squares, right to left, so that
  6 -> 7 -> 8 -> ... -> 11 -> 0 -> 1 ... walks the ring.

``CLOCKWISE`` (+1) walks the ring with increasing index; it runs left to
right along the top row and right to left along the bottom row, as drawn
above.  ``COUNTER_CLOCKWISE`` (-1) walks it the other way.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

N_CELLS = 12

#: The two mandarin squares, in ``QUAN_INDEX`` order (left, right).
QUAN_CELLS: Tuple[int, int] = (0, 6)

#: The five citizen squares of each player, in ring order.
PLAYER_CELLS: Tuple[Tuple[int, ...], Tuple[int, ...]] = (
    (1, 2, 3, 4, 5),
    (7, 8, 9, 10, 11),
)

CLOCKWISE = 1
COUNTER_CLOCKWISE = -1
DIRECTIONS: Tuple[int, int] = (CLOCKWISE, COUNTER_CLOCKWISE)

#: A move is ``(cell, direction)``; there are five squares and two
#: directions, so at most ten distinct moves per turn.
N_ACTIONS_PER_PLAYER = 10

#: Size of the policy head: one logit per move, same for both players.
N_ACTIONS = N_ACTIONS_PER_PLAYER

_QUAN_INDEX = {QUAN_CELLS[0]: 0, QUAN_CELLS[1]: 1}


def step(cell: int, direction: int) -> int:
    """Return the square that follows ``cell`` in ``direction``."""
    return (cell + direction) % N_CELLS


def is_quan_cell(cell: int) -> bool:
    """True for the two mandarin squares."""
    return cell in _QUAN_INDEX


def quan_index(cell: int) -> int:
    """Index (0 or 1) of a mandarin square.  Only valid for mandarin cells."""
    return _QUAN_INDEX[cell]


def live_quan(quan: Sequence[bool], cell: int) -> bool:
    """True when ``cell`` still holds its mandarin piece."""
    index = _QUAN_INDEX.get(cell)
    return index is not None and bool(quan[index])


def owner(cell: int) -> Optional[int]:
    """The player controlling ``cell``, or ``None`` for a mandarin square."""
    if cell in PLAYER_CELLS[0]:
        return 0
    if cell in PLAYER_CELLS[1]:
        return 1
    return None


def action_index(cell: int, direction: int) -> int:
    """Compact index in ``0..9`` for a move, independent of the player."""
    player = owner(cell)
    if player is None:
        raise ValueError(f"cell {cell} is not a citizen square")
    local = PLAYER_CELLS[player].index(cell)
    return local * 2 + (0 if direction == CLOCKWISE else 1)


def action_from_index(player: int, index: int) -> Tuple[int, int]:
    """Inverse of :func:`action_index` for the given player."""
    if not 0 <= index < N_ACTIONS_PER_PLAYER:
        raise ValueError(f"action index {index} is out of range")
    cell = PLAYER_CELLS[player][index // 2]
    direction = CLOCKWISE if index % 2 == 0 else COUNTER_CLOCKWISE
    return cell, direction


def action_name(cell: int, direction: int) -> str:
    """Readable move label, e.g. ``"3 cw"``."""
    return f"{cell} {'cw' if direction == CLOCKWISE else 'ccw'}"

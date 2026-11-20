"""Immutable game state and move resolution for O an quan.

The rules implemented here are the ones written down in ``docs/rules.md``.
In short:

* a turn lifts every stone from one of the mover's five squares and sows
  them one by one around the ring (mandarin squares receive stones too);
* if the square after the last stone holds stones *and* is a citizen
  square, they are lifted and the sowing continues in the same direction;
* if it is a mandarin square, the turn ends (a mandarin is never lifted);
* if it is empty, the square after it is captured when it holds anything,
  and captures chain through alternating empty / non-empty squares;
* capturing a mandarin square takes the mandarin piece with its stones and
  the square becomes an ordinary citizen square.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Sequence, Tuple

from .layout import (
    CLOCKWISE,
    COUNTER_CLOCKWISE,
    DIRECTIONS,
    N_CELLS,
    PLAYER_CELLS,
    QUAN_CELLS,
    is_quan_cell,
    live_quan,
    quan_index,
    step,
)
from .rules import DEFAULT_RULES, Rules

#: A move is ``(cell, direction)``.
Action = Tuple[int, int]

#: Hard stop for the sowing loop, so a rule bug surfaces as an error
#: instead of a hang.  A legal turn never comes close to this.
MAX_TURN_STEPS = 2_000


class IllegalMoveError(ValueError):
    """Raised when a move cannot be played in the given state."""


class EngineError(RuntimeError):
    """Raised when the engine detects an internal inconsistency."""


@dataclass(frozen=True)
class MoveResult:
    """What happened during one move, in order (useful for UIs and tests)."""

    player: int
    action: Action
    #: square the turn started from, and how many stones were lifted there
    lifted: Tuple[int, int]
    #: every square that received a stone, in the order the stones fell
    sown: Tuple[int, ...]
    #: captured squares and how many citizen stones each one held
    captures: Tuple[Tuple[int, int], ...]
    #: mandarin squares captured by this move
    mandarins_captured: Tuple[int, ...]
    #: points scored by this move (citizen stones + mandarin value)
    points: int


@dataclass(frozen=True)
class State:
    """A complete, immutable position."""

    #: citizen stones in each of the twelve squares
    cells: Tuple[int, ...]
    #: whether each mandarin piece is still on its square (index 0 -> cell 0,
    #: index 1 -> cell 6)
    quan: Tuple[bool, bool] = (True, True)
    #: which player captured each mandarin square, -1 while it is still there
    quan_captor: Tuple[int, int] = (-1, -1)
    #: citizen stones captured so far by each player
    captured: Tuple[int, int] = (0, 0)
    #: mandarin pieces captured so far by each player
    mandarins_taken: Tuple[int, int] = (0, 0)
    #: player to move
    player: int = 0
    #: number of moves played so far
    ply: int = 0
    #: ply of the last capture, for the no-progress draw rule
    last_capture_ply: int = 0
    #: True once the game is over
    done: bool = False
    #: why the game ended: "both_mandarins", "cannot_refill", "no_progress"
    #: or "ply_limit"
    end_reason: str = ""

    # ------------------------------------------------------------------ setup
    @classmethod
    def from_cells(
        cls,
        cells: Sequence[int],
        *,
        quan: Tuple[bool, bool] = (True, True),
        quan_captor: Tuple[int, int] = (-1, -1),
        captured: Tuple[int, int] = (0, 0),
        mandarins_taken: Tuple[int, int] = (0, 0),
        player: int = 0,
        ply: int = 0,
        last_capture_ply: int = 0,
    ) -> "State":
        """Build a position from a list of twelve stone counts (for tests)."""
        cells = tuple(int(count) for count in cells)
        if len(cells) != N_CELLS:
            raise ValueError(f"expected {N_CELLS} cells, got {len(cells)}")
        if any(count < 0 for count in cells):
            raise ValueError("stone counts must be non-negative")
        if player not in (0, 1):
            raise ValueError(f"bad player {player}")
        return cls(
            cells=cells,
            quan=quan,
            quan_captor=quan_captor,
            captured=captured,
            mandarins_taken=mandarins_taken,
            player=player,
            ply=ply,
            last_capture_ply=last_capture_ply,
        )

    # ------------------------------------------------------------------ queries
    def mandarin_alive(self, cell: int) -> bool:
        """True when ``cell`` is a mandarin square that still holds its piece."""
        return live_quan(self.quan, cell)

    def has_pieces(self, cell: int) -> bool:
        """True when a square holds citizen stones (mandarins do not count)."""
        return self.cells[cell] > 0

    def side_is_empty(self, player: int) -> bool:
        """True when all five of ``player``'s citizen squares are empty."""
        return all(self.cells[cell] == 0 for cell in PLAYER_CELLS[player])

    def legal_actions(self) -> Tuple[Action, ...]:
        """All legal moves for the player to move."""
        if self.done:
            return ()
        return tuple(
            (cell, direction)
            for cell in PLAYER_CELLS[self.player]
            if self.cells[cell] > 0
            for direction in DIRECTIONS
        )

    def total_dan(self) -> int:
        """Citizen stones on the board plus captured ones (an invariant)."""
        return sum(self.cells) + sum(self.captured)

    def points(self, rules: Rules = DEFAULT_RULES) -> Tuple[int, int]:
        """Points from captures only, ignoring stones still on the board."""
        return (
            self.captured[0] + rules.mandarin_value * self.mandarins_taken[0],
            self.captured[1] + rules.mandarin_value * self.mandarins_taken[1],
        )

    def final_points(self, rules: Rules = DEFAULT_RULES) -> Tuple[int, int]:
        """Final score.  Only meaningful once the game is over.

        Stones left in a player's own five squares belong to that player.
        Stones left in a square whose mandarin was captured belong to the
        player who captured that mandarin.  Stones in a mandarin square that
        is still holding its piece are never scored.
        """
        if not self.done:
            raise EngineError("final_points() called on a running game")
        board_p0, board_p1 = self.points(rules)
        for cell in PLAYER_CELLS[0]:
            board_p0 += self.cells[cell]
        for cell in PLAYER_CELLS[1]:
            board_p1 += self.cells[cell]
        for index, cell in enumerate(QUAN_CELLS):
            captor = self.quan_captor[index]
            if captor == 0:
                board_p0 += self.cells[cell]
            elif captor == 1:
                board_p1 += self.cells[cell]
        return board_p0, board_p1

    def returns(self, rules: Rules = DEFAULT_RULES) -> float:
        """Game value for the player to move: +1 win, -1 loss, 0 draw."""
        p0, p1 = self.final_points(rules)
        difference = (p0 - p1) if self.player == 0 else (p1 - p0)
        if difference > 0:
            return 1.0
        if difference < 0:
            return -1.0
        return 0.0

    # -------------------------------------------------------------------- moves
    def apply(
        self, action: Action, rules: Rules = DEFAULT_RULES
    ) -> Tuple["State", MoveResult]:
        """Play ``action`` and return the next position plus a move report."""
        if self.done:
            raise IllegalMoveError("the game is over")
        cell, direction = action
        if direction not in DIRECTIONS:
            raise IllegalMoveError(f"direction must be +1 or -1, got {direction}")
        if cell not in PLAYER_CELLS[self.player]:
            raise IllegalMoveError(f"square {cell} is not one of your squares")
        if self.cells[cell] == 0:
            raise IllegalMoveError(f"square {cell} is empty")

        player = self.player
        cells: List[int] = list(self.cells)
        quan: List[bool] = list(self.quan)
        quan_captor: List[int] = list(self.quan_captor)
        captured: List[int] = list(self.captured)
        mandarins_taken: List[int] = list(self.mandarins_taken)
        sown: List[int] = []

        def sow(start: int, stones: int) -> int:
            """Drop ``stones`` one per square after ``start``; return the last."""
            position = start
            for _ in range(stones):
                position = step(position, direction)
                cells[position] += 1
                sown.append(position)
            return position

        lifted = cells[cell]
        cells[cell] = 0
        position = sow(cell, lifted)

        captures: List[Tuple[int, int]] = []
        mandarins_captured: List[int] = []

        for _ in range(MAX_TURN_STEPS):
            following = step(position, direction)
            if cells[following] > 0:
                if is_quan_cell(following):
                    break  # a mandarin square is never lifted from: turn ends
                # Ordinary square holding stones: lift them and keep sowing.
                stones = cells[following]
                cells[following] = 0
                position = sow(following, stones)
                continue
            # An empty square (a mandarin square without citizen stones counts
            # as empty here): the move can now only end in captures.
            while True:
                target = step(following, direction)
                if cells[target] == 0:
                    break  # two empty squares in a row: nothing to take
                if (
                    is_quan_cell(target)
                    and live_quan(quan, target)
                    and cells[target] < rules.mandarin_capture_min
                ):
                    break  # a "quan non" is not rich enough to be taken
                gained = cells[target]
                cells[target] = 0
                captured[player] += gained
                captures.append((target, gained))
                if is_quan_cell(target) and live_quan(quan, target):
                    quan[quan_index(target)] = False
                    quan_captor[quan_index(target)] = player
                    mandarins_taken[player] += 1
                    mandarins_captured.append(target)
                following = step(target, direction)
                if cells[following] > 0:
                    break  # sowing never resumes after a capture
            break
        else:  # pragma: no cover - defensive
            raise EngineError("sowing did not terminate; rules bug")

        ply_now = self.ply + 1
        state = State(
            cells=tuple(cells),
            quan=tuple(quan),
            quan_captor=tuple(quan_captor),
            captured=tuple(captured),
            mandarins_taken=tuple(mandarins_taken),
            player=player,
            ply=ply_now,
            last_capture_ply=ply_now if captures else self.last_capture_ply,
        )
        result = MoveResult(
            player=player,
            action=action,
            lifted=(cell, lifted),
            sown=tuple(sown),
            captures=tuple(captures),
            mandarins_captured=tuple(mandarins_captured),
            points=(
                sum(gained for _, gained in captures)
                + rules.mandarin_value * len(mandarins_captured)
            ),
        )
        return _end_move(state, rules), result


def _end_move(state: State, rules: Rules) -> State:
    """Check the end conditions and hand the turn to the other player."""
    if not any(state.quan):
        return replace(state, done=True, end_reason="both_mandarins")
    if state.ply - state.last_capture_ply >= rules.no_progress_plies:
        return replace(state, done=True, end_reason="no_progress")
    if state.ply >= rules.ply_limit:
        return replace(state, done=True, end_reason="ply_limit")

    following = 1 - state.player
    cells = list(state.cells)
    captured = list(state.captured)
    side = PLAYER_CELLS[following]
    if all(cells[cell] == 0 for cell in side):
        # Refill: place one captured stone per owned square, then move.
        needed = len(side)
        if captured[following] < needed:
            return replace(
                state, player=following, done=True, end_reason="cannot_refill"
            )
        captured[following] -= needed
        for cell in side:
            cells[cell] = 1
    return replace(
        state, cells=tuple(cells), captured=tuple(captured), player=following
    )


def initial_state(rules: Rules = DEFAULT_RULES) -> State:
    """The standard starting position; player 0 moves first."""
    cells = [0] * N_CELLS
    for player_cells in PLAYER_CELLS:
        for cell in player_cells:
            cells[cell] = rules.stones_per_square
    return State(cells=tuple(cells))


__all__ = [
    "Action",
    "CLOCKWISE",
    "COUNTER_CLOCKWISE",
    "EngineError",
    "IllegalMoveError",
    "MoveResult",
    "State",
    "initial_state",
]

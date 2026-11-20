"""Rule tests for the O an quan engine.

Every worked example in ``docs/rules.md`` appears here.  Run with pytest
(``python -m pytest tests -q``) or directly with ``python tests/test_engine.py``;
no third-party package is required.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import (  # noqa: E402  (import after the sys.path fix)
    CLOCKWISE,
    COUNTER_CLOCKWISE,
    DEFAULT_RULES,
    PLAYER_CELLS,
    Rules,
    State,
    action_from_index,
    action_index,
    initial_state,
)
from engine.state import IllegalMoveError  # noqa: E402

CW = CLOCKWISE
CCW = COUNTER_CLOCKWISE


def make(
    cells,
    *,
    quan=(True, True),
    quan_captor=(-1, -1),
    captured=(0, 0),
    mandarins_taken=(0, 0),
    player=0,
) -> State:
    """Build a test position from twelve stone counts."""
    return State.from_cells(
        cells,
        quan=quan,
        quan_captor=quan_captor,
        captured=captured,
        mandarins_taken=mandarins_taken,
        player=player,
    )


# --------------------------------------------------------------------- setup


def test_initial_position():
    state = initial_state()
    assert list(state.cells) == [0, 5, 5, 5, 5, 5, 0, 5, 5, 5, 5, 5]
    assert state.quan == (True, True)
    assert state.player == 0
    assert not state.done
    assert state.total_dan() == 50


def test_legal_actions_at_setup():
    state = initial_state()
    actions = state.legal_actions()
    assert len(actions) == 10
    assert all(cell in PLAYER_CELLS[0] for cell, _ in actions)
    assert len(set(actions)) == 10


def test_illegal_moves_are_rejected():
    state = initial_state()
    for action in [(0, CW), (6, CW), (7, CW)]:  # not the mover's squares
        try:
            state.apply(action)
        except IllegalMoveError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"{action} should have been rejected")
    for action in [(1, 0), (1, 2)]:  # bad direction
        try:
            state.apply(action)
        except IllegalMoveError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"{action} should have been rejected")
    empty = make([0] * 12)
    try:
        empty.apply((1, CW))
    except IllegalMoveError:
        pass
    else:  # pragma: no cover
        raise AssertionError("sowing from an empty square should be rejected")


def test_no_moves_after_the_game_is_over():
    state = make([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    state = State.from_cells(state.cells, quan=(False, False), player=0)
    over = state.__class__(
        cells=state.cells,
        quan=(False, False),
        player=0,
        done=True,
        end_reason="both_mandarins",
    )
    assert over.legal_actions() == ()
    try:
        over.apply((1, CW))
    except IllegalMoveError:
        pass
    else:  # pragma: no cover
        raise AssertionError("moves after the end should be rejected")


# ------------------------------------------------------- worked examples


def test_example_1_opening_move():
    """Example 1: the opening move chains once and captures six stones."""
    state = initial_state()
    nxt, result = state.apply((1, CW))

    assert result.lifted == (1, 5)
    assert result.sown == (2, 3, 4, 5, 6, 8, 9, 10, 11, 0)
    assert result.captures == ((2, 6),)
    assert result.points == 6

    assert list(nxt.cells) == [1, 0, 0, 6, 6, 6, 1, 0, 6, 6, 6, 6]
    assert nxt.captured == (6, 0)
    assert nxt.player == 1
    assert not nxt.done
    assert nxt.total_dan() == 50


def test_example_2_continuation_ends_at_a_mandarin():
    """Example 2: sowing continues once, then stops in front of a mandarin."""
    state = make([0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0])
    nxt, result = state.apply((1, CW))

    assert result.sown == (2, 4, 5)
    assert result.captures == ()
    assert list(nxt.cells) == [0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0]
    assert nxt.player == 1


def test_example_3_capture_a_mandarin():
    """Example 3: a capture takes the mandarin and the five stones with it."""
    state = make([0, 3, 0, 0, 2, 0, 5, 0, 0, 0, 0, 0])
    nxt, result = state.apply((1, CW))

    assert result.captures == ((6, 5),)
    assert result.mandarins_captured == (6,)
    assert result.points == 15  # 5 stones + 10 for the mandarin

    assert list(nxt.cells) == [0, 0, 1, 1, 3, 0, 0, 0, 0, 0, 0, 0]
    assert nxt.quan == (True, False)
    assert nxt.quan_captor == (-1, 0)
    assert nxt.mandarins_taken == (1, 0)
    assert nxt.points() == (15, 0)
    assert nxt.total_dan() == state.total_dan()


def test_example_4_capture_chain():
    """Example 4: alternating empty / occupied squares chain two captures."""
    state = make(
        [0, 2, 0, 0, 0, 1, 0, 2, 0, 3, 0, 1],
        quan=(False, False),
        quan_captor=(1, 0),
    )
    nxt, result = state.apply((5, CW))

    assert result.sown == (6, 8, 9)
    assert result.captures == ((11, 1), (1, 2))
    assert result.points == 3
    assert list(nxt.cells) == [0, 0, 0, 0, 0, 0, 1, 0, 1, 4, 0, 0]
    assert nxt.captured == (3, 0)
    assert nxt.total_dan() == state.total_dan()


def test_example_5_forced_refill():
    """Example 5: an emptied side is refilled from captured stones."""
    state = make(
        [0, 4, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        quan=(True, False),
        quan_captor=(-1, 0),
        captured=(0, 5),
    )
    nxt, result = state.apply((1, CW))

    assert result.captures == ((7, 1),)
    assert list(nxt.cells) == [0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1]
    assert nxt.captured == (1, 0)  # the refill spent player 1's five stones
    assert nxt.player == 1
    assert not nxt.done
    assert nxt.total_dan() == 10  # 5 stones started on the board, 5 were captured


def test_example_6_both_mandarins_and_final_score():
    """Example 6: taking the last mandarin ends the game and scores it."""
    state = make(
        [5, 9, 0, 0, 0, 0, 2, 3, 0, 0, 0, 0],
        quan=(True, False),
        quan_captor=(-1, 1),
        mandarins_taken=(0, 1),
    )
    nxt, result = state.apply((1, CW))

    assert result.mandarins_captured == (0,)
    assert result.points == 16  # 5 stones + 1 chained stone + 10 for the mandarin
    assert list(nxt.cells) == [0, 0, 0, 1, 1, 1, 3, 4, 1, 1, 1, 0]
    assert nxt.done
    assert nxt.end_reason == "both_mandarins"
    assert nxt.final_points() == (19, 20)
    assert nxt.returns() == -1.0


# ------------------------------------------------------------- end of game


def test_game_ends_when_a_player_cannot_refill():
    state = make(
        [0, 4, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        quan=(True, False),
        quan_captor=(-1, 0),
        captured=(0, 2),  # only two stones: cannot fill five squares
    )
    nxt, _ = state.apply((1, CW))

    assert nxt.done
    assert nxt.end_reason == "cannot_refill"
    assert nxt.player == 1
    assert nxt.legal_actions() == ()


def test_ply_limit_ends_the_game():
    rules = Rules(ply_limit=1)
    nxt, _ = initial_state().apply((1, CW), rules)

    assert nxt.done
    assert nxt.end_reason == "ply_limit"


def test_mandarin_value_follows_the_rules():
    state = make([0, 3, 0, 0, 2, 0, 5, 0, 0, 0, 0, 0])
    five_point = Rules(mandarin_value=5)
    nxt, result = state.apply((1, CW), five_point)

    assert result.mandarins_captured == (6,)
    assert nxt.points(five_point) == (10, 0)  # 5 stones + 5 for the mandarin


# -------------------------------------------------------------- invariants


def test_action_encoding_round_trip():
    for player in (0, 1):
        actions = set()
        for index in range(10):
            cell, direction = action_from_index(player, index)
            assert cell in PLAYER_CELLS[player]
            assert action_index(cell, direction) == index
            actions.add((cell, direction))
        assert len(actions) == 10


def test_random_games_conserve_stones_and_terminate():
    for seed in range(50):
        rng = random.Random(seed)
        state = initial_state()
        plies = 0
        while not state.done:
            assert state.total_dan() == 50, f"seed {seed}: stones not conserved"
            actions = state.legal_actions()
            assert actions, f"seed {seed}: running game without legal moves"
            state, _ = state.apply(rng.choice(actions))
            plies += 1
            assert plies <= DEFAULT_RULES.ply_limit
        assert state.total_dan() == 50
        low, high = state.final_points()
        total_mandarins = sum(state.mandarins_taken)
        assert low + high <= 50 + DEFAULT_RULES.mandarin_value * total_mandarins
        if all(taken == 1 for taken in state.mandarins_taken):
            assert low + high == 50 + 2 * DEFAULT_RULES.mandarin_value


def test_returns_matches_the_final_score():
    for seed in range(10):
        rng = random.Random(seed)
        state = initial_state()
        while not state.done:
            state, _ = state.apply(rng.choice(state.legal_actions()))
        p0, p1 = state.final_points()
        mover = state.player
        expected = 0.0
        if p0 != p1:
            winner = 0 if p0 > p1 else 1
            expected = 1.0 if winner == mover else -1.0
        assert state.returns() == expected


def test_a_bare_mandarin_is_not_a_capture_target():
    """Sowing three stones puts one on 4; 5 is empty and 6 holds only the mandarin."""
    state = make([0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    nxt, result = state.apply((1, CW))

    assert result.captures == ()
    assert result.mandarins_captured == ()
    assert result.points == 0
    assert nxt.quan == (True, True)


def test_a_rich_mandarin_can_be_captured():
    state = make([0, 3, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0])
    nxt, result = state.apply((1, CW))

    assert result.captures == ((6, 5),)
    assert result.mandarins_captured == (6,)
    assert result.points == 15  # five stones + the mandarin
    assert nxt.quan == (True, False)


def test_capture_chain_runs_through_a_bare_mandarin():
    """A bare mandarin square is empty, so the cell beyond it can be taken."""
    # Four stones from square 1 land on 2, 3, 4, 5; square 6 (bare mandarin) is the
    # empty square, so square 7 is captured through it.
    state = make([0, 4, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0])
    nxt, result = state.apply((1, CW))

    assert result.captures == ((7, 3),)
    assert result.mandarins_captured == ()
    assert nxt.quan == (True, True)


def test_a_young_mandarin_is_not_capturable_by_default():
    """Quan non: fewer than five citizen stones is not rich enough."""
    state = make([0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0])
    nxt, result = state.apply((1, CW))

    assert result.mandarins_captured == ()
    assert result.captures == ()
    assert nxt.quan == (True, True)


def test_mandarin_capture_threshold_is_configurable():
    state = make([0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0])
    lenient = Rules(mandarin_capture_min=4)
    nxt, result = state.apply((1, CW), lenient)

    assert result.mandarins_captured == (6,)
    assert result.captures == ((6, 4),)


def test_no_progress_rule_ends_a_looping_game():
    state = State.from_cells(
        [0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0], player=0, ply=119, last_capture_ply=0
    )
    nxt, result = state.apply((1, CW))

    assert result.captures == ()
    assert nxt.done
    assert nxt.end_reason == "no_progress"


def test_both_directions_are_playable_through_a_mandarin():
    """Stones are sown into mandarin squares from either direction."""
    state = make([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    clockwise, _ = state.apply((5, CW))
    assert clockwise.cells[6] == 1

    counter, _ = state.apply((5, CCW))
    assert counter.cells[4] == 1


if __name__ == "__main__":
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\n{len(tests)} tests passed")

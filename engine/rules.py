"""Rule parameters.

``docs/rules.md`` fixes one ruleset.  The handful of knobs local variants
disagree about live here so later experiments can vary them explicitly
instead of silently changing the engine.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rules:
    """Parameters of the ruleset used by the engine."""

    #: citizen stones (dan) placed in each of the ten citizen squares
    stones_per_square: int = 5
    #: points awarded for capturing a mandarin piece (variants use 5 or 10)
    mandarin_value: int = 10
    #: a mandarin square can only be captured once it holds at least this many
    #: citizen stones (*quan non* rule).  Never less than 1: a bare mandarin is
    #: not a capture target.  With 1 a mandarin falls after a single stone is
    #: sown into it and games end in a handful of moves, so 5 is the default.
    mandarin_capture_min: int = 5
    #: draw when no citizen stone has been captured for this many plies (stones
    #: can otherwise circulate forever, like a fifty-move rule; a normal game
    #: lasts about twenty plies, so sixty without a capture is a clear loop)
    no_progress_plies: int = 60
    #: final safety net: a game still running after this many plies is a draw
    ply_limit: int = 1000


DEFAULT_RULES = Rules()

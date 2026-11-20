"""Common interface for agents (baselines now, learned policies later)."""

from __future__ import annotations

from typing import Protocol

from engine import Action, State


class Agent(Protocol):
    """Anything that can pick a legal move for the player to move."""

    name: str

    def choose(self, state: State) -> Action:
        """Return one of ``state.legal_actions()``."""

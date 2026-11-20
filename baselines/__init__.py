"""Baseline opponents."""

from .alphabeta import AlphaBetaAgent
from .base import Agent
from .greedy import GreedyAgent
from .mcts import PlainMCTS
from .net_agent import NetAgent
from .random_agent import RandomAgent

__all__ = [
    "Agent",
    "AlphaBetaAgent",
    "GreedyAgent",
    "NetAgent",
    "PlainMCTS",
    "RandomAgent",
]

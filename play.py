"""Play O an quan in the terminal.

    python play.py                                   # you (P0) vs a random opponent
    python play.py --p1 net                          # you vs the trained checkpoint
    python play.py --p0 human --p1 human             # two humans, hot seat
    python play.py --p0 random --p1 random --seed 3  # watch two random players
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Sequence

from baselines import NetAgent, RandomAgent
from baselines.base import Agent
from engine import (
    CLOCKWISE,
    COUNTER_CLOCKWISE,
    DEFAULT_RULES,
    State,
    action_name,
    format_board,
    initial_state,
)

HELP_TEXT = """\
  move:     <square> <direction>      e.g. "3 cw" or "5 ccw"
            cw  = clockwise (+1), ccw = counter-clockwise (-1)
  commands: moves   list your legal moves
            board   redraw the board
            quit    stop playing
"""


def parse_move(text: str):
    """Turn ``"3 cw"`` (or ``"3 1"``, ``"3 cc"`` ...) into an action."""
    parts = text.replace(",", " ").split()
    if len(parts) != 2:
        raise ValueError('expected "<square> <direction>", e.g. "3 cw"')
    try:
        cell = int(parts[0])
    except ValueError:
        raise ValueError(f"{parts[0]!r} is not a square number") from None
    word = parts[1].lower()
    if word in ("cw", "c", "+", "+1", "1", "clockwise"):
        direction = CLOCKWISE
    elif word in ("ccw", "cc", "-", "-1", "counterclockwise", "counter-clockwise"):
        direction = COUNTER_CLOCKWISE
    else:
        raise ValueError("direction must be 'cw' or 'ccw'")
    return cell, direction


class HumanAgent:
    """Reads moves from stdin."""

    name = "human"

    def choose(self, state: State):
        legal = state.legal_actions()
        listing = ", ".join(action_name(*action) for action in legal)
        while True:
            try:
                text = input(f"  P{state.player} move> ").strip()
            except EOFError:
                print()
                raise SystemExit(0) from None
            word = text.lower()
            if word in ("q", "quit", "exit"):
                raise SystemExit(0)
            if word in ("h", "help", "?"):
                print(HELP_TEXT)
                continue
            if word in ("m", "moves"):
                print(f"  legal moves: {listing}")
                continue
            if word in ("b", "board"):
                print(format_board(state))
                continue
            try:
                action = parse_move(text)
            except ValueError as error:
                print(f"  {error}")
                continue
            if action not in legal:
                print(f"  illegal move; legal moves: {listing}")
                continue
            return action


def describe(action) -> str:
    cell, direction = action
    word = "clockwise" if direction == CLOCKWISE else "counter-clockwise"
    return f"square {cell} {word}"


def describe_result(result) -> str:
    bits = [f"P{result.player} lifts {result.lifted[1]} from square {result.lifted[0]}"]
    if result.captures:
        total = sum(stone for _, stone in result.captures)
        squares = ", ".join(str(cell) for cell, _ in result.captures)
        bits.append(f"captures {total} stone(s) at square {squares}")
    if result.mandarins_captured:
        squares = ", ".join(str(cell) for cell in result.mandarins_captured)
        bits.append(f"takes the MANDARIN at square {squares}")
    bits.append(f"+{result.points} points")
    return "  " + "; ".join(bits)


def make_agent(
    kind: str, seed: int, checkpoint: Optional[str] = None, simulations: int = 0
) -> Agent:
    if kind == "human":
        return HumanAgent()
    if kind == "net":
        from alphazero import PolicyValueNet

        if not checkpoint or not Path(checkpoint).exists():
            raise SystemExit(
                f"checkpoint not found: {checkpoint!r} "
                "(train one with: python train.py --run runs/base_az)"
            )
        return NetAgent(PolicyValueNet.load(checkpoint), simulations=simulations, seed=seed)
    return RandomAgent(seed)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Play O an quan in the terminal.")
    parser.add_argument("--p0", default="human", choices=("human", "random", "net"))
    parser.add_argument("--p1", default="random", choices=("human", "random", "net"))
    parser.add_argument("--ckpt", default="runs/base_model.npz", help="net checkpoint")
    parser.add_argument(
        "--sims",
        type=int,
        default=0,
        help="MCTS simulations for net players (0 = raw policy, which is stronger here)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    rules = DEFAULT_RULES
    state = initial_state(rules)
    agents: Dict[int, Agent] = {
        0: make_agent(args.p0, args.seed, args.ckpt, args.sims),
        1: make_agent(args.p1, args.seed + 1, args.ckpt, args.sims),
    }
    human_at_the_table = any(agent.name == "human" for agent in agents.values())

    print("O an quan - type 'help' at the prompt for the move syntax")
    print(format_board(state, rules))

    while not state.done:
        agent = agents[state.player]
        action = agent.choose(state)
        if agent.name != "human":
            print(f"\n  P{state.player} ({agent.name}) plays {describe(action)}")
        state, result = state.apply(action, rules)
        print(describe_result(result))
        if human_at_the_table or state.done:
            print(format_board(state, rules))

    p0, p1 = state.final_points(rules)
    print(f"\n  final score: P0 {p0} - {p1} P1   (ended by {state.end_reason})")
    if p0 == p1:
        print("  draw")
    else:
        print(f"  player {0 if p0 > p1 else 1} wins")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

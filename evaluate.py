"""Evaluate a checkpoint against baselines (or against another checkpoint).

    python evaluate.py --ckpt runs/az_base/final.npz --sims 64 --games 40
    python evaluate.py --ckpt runs/base/final.npz --opponents net:runs/mine/final.npz

Opponent names: ``random``, ``greedy``, ``alphabetaN`` (depth N),
``mctsN`` (N playouts) and ``net:PATH`` (another checkpoint, policy only).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Sequence

from alphazero import PolicyValueNet
from baselines import AlphaBetaAgent, GreedyAgent, NetAgent, PlainMCTS, RandomAgent
from eval.arena import play_match
from eval.elo import elo_difference


def build_opponent(name: str, seed: int):
    if name == "random":
        return RandomAgent(seed)
    if name == "greedy":
        return GreedyAgent(seed)
    if name.startswith("alphabeta"):
        depth = int(name[len("alphabeta") :] or 4)
        return AlphaBetaAgent(depth=depth, seed=seed)
    if name.startswith("mcts"):
        iterations = int(name[len("mcts") :] or 200)
        return PlainMCTS(iterations=iterations, seed=seed)
    if name.startswith("net:"):
        return NetAgent(PolicyValueNet.load(name[len("net:") :]), simulations=0, seed=seed)
    raise SystemExit(f"unknown opponent {name!r}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint.")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--sims", type=int, default=64, help="MCTS simulations per move")
    parser.add_argument(
        "--policy-only", action="store_true", help="play the raw policy, no search"
    )
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument(
        "--opening-plies",
        type=int,
        default=2,
        help="random plies before the agents play, for game-to-game variety",
    )
    parser.add_argument("--opponents", default="random,greedy,alphabeta4,mcts100")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="write results as JSON here")
    args = parser.parse_args(argv)

    net = PolicyValueNet.load(args.ckpt)
    agent = NetAgent(
        net, simulations=0 if args.policy_only else args.sims, seed=args.seed
    )
    label = f"{Path(args.ckpt).stem} ({'policy only' if args.policy_only else f'{args.sims} sims'})"
    print(f"Evaluating {label} over {args.games} games per opponent\n", flush=True)

    table = {}
    header = f"  {'opponent':<16}{'W':>4}{'L':>4}{'D':>4}{'score':>8}{'elo':>9}{'time':>8}"
    print(header)
    for name in [part.strip() for part in args.opponents.split(",") if part.strip()]:
        opponent = build_opponent(name, args.seed + 7)
        started = time.perf_counter()
        match = play_match(
            agent,
            opponent,
            games=args.games,
            seed=args.seed,
            opening_plies=args.opening_plies,
        )
        elapsed = time.perf_counter() - started
        delta = elo_difference(match.score)
        table[name] = {
            **match.as_dict(),
            "elo_diff": None if delta in (float("inf"), float("-inf")) else round(delta, 1),
            "seconds": round(elapsed, 1),
        }
        delta_text = "   inf" if delta == float("inf") else f"{delta:+8.0f}"
        print(
            f"  {name:<16}{match.wins:>4}{match.losses:>4}{match.draws:>4}"
            f"{match.score:>8.3f}{delta_text:>9}{elapsed:>7.1f}s",
            flush=True,
        )

    output = Path(args.out) if args.out else Path(args.ckpt).with_name("eval.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(args.ckpt),
        "policy_only": args.policy_only,
        "simulations": 0 if args.policy_only else args.sims,
        "games_per_opponent": args.games,
        "opening_plies": args.opening_plies,
        "seed": args.seed,
        "results": table,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

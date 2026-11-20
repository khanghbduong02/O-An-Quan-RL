"""Supervised warm start for the policy/value network.

Cold-start AlphaZero on this game deadlocks: with a near-uniform policy the
self-play games are long and noisy, the value head cannot resolve a ~10 point
capture from its own error, the search therefore sees no advantage in
capturing, and the policy stays near-uniform.  Imitating a heuristic teacher
for a few thousand games gives the network a capture-seeking prior and a value
that tracks the score, after which AlphaZero search has something to improve on.

    python pretrain.py --run runs/pretrain_greedy --games 8000 --epochs 3
"""

from __future__ import annotations

import argparse
import time
from typing import Optional, Sequence

import numpy as np

from alphazero.features import encode, legal_mask
from alphazero.network import PolicyValueNet
from baselines import GreedyAgent, RandomAgent
from engine import DEFAULT_RULES, N_ACTIONS, action_index, initial_state


def collect_games(games: int, seed: int, mix_random_plies: bool = True):
    """Play greedy-vs-greedy games and record (state, teacher move, outcome)."""
    rng = np.random.default_rng(seed)
    greedy = [GreedyAgent(seed=int(rng.integers(1_000_000))), GreedyAgent(seed=int(rng.integers(1_000_000)))]
    explorer = RandomAgent(seed=int(rng.integers(1_000_000)))
    xs, pis, masks, zs = [], [], [], []
    for _ in range(games):
        state = initial_state()
        if mix_random_plies:
            for _ in range(int(rng.integers(0, 3))):
                if state.done:
                    break
                state, _ = state.apply(explorer.choose(state))
        rows = []
        while not state.done:
            action = greedy[state.player].choose(state)
            pi = np.zeros(N_ACTIONS, dtype=np.float32)
            pi[action_index(*action)] = 1.0
            rows.append((encode(state), pi, legal_mask(state), state.player))
            state, _ = state.apply(action)
        p0, p1 = state.final_points()
        for x, pi, mask, player in rows:
            mine, theirs = (p0, p1) if player == 0 else (p1, p0)
            xs.append(x)
            pis.append(pi)
            masks.append(mask)
            zs.append(float(np.clip((mine - theirs) / 50.0, -1.0, 1.0)))
    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(pis, dtype=np.float32),
        np.asarray(masks, dtype=np.float32),
        np.asarray(zs, dtype=np.float32),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Supervised warm start from a greedy teacher.")
    parser.add_argument("--run", default="runs/pretrain_greedy", help="output directory")
    parser.add_argument("--games", type=int, default=8000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    start = time.perf_counter()
    x, pi, mask, z = collect_games(args.games, args.seed)
    print(f"collected {len(x)} positions from {args.games} greedy games in {time.perf_counter()-start:.1f}s", flush=True)

    net = PolicyValueNet(hidden=args.hidden, layers=args.layers, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    for epoch in range(args.epochs):
        order = rng.permutation(len(x))
        policy_loss = value_loss = 0.0
        batches = 0
        for offset in range(0, len(order), args.batch):
            idx = order[offset : offset + args.batch]
            stats = net.train_batch(x[idx], pi[idx], z[idx], mask[idx], lr=args.lr)
            policy_loss += stats["policy_loss"]
            value_loss += stats["value_loss"]
            batches += 1
        print(
            f"epoch {epoch+1}/{args.epochs}: policy {policy_loss/max(batches,1):.3f} "
            f"value {value_loss/max(batches,1):.3f}",
            flush=True,
        )

    import os

    os.makedirs(args.run, exist_ok=True)
    path = os.path.join(args.run, "final.npz")
    net.save(path)
    print(f"saved warm-start checkpoint to {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

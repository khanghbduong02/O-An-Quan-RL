"""Train the policy-gradient agent (A2C self-play, no search).

    python train_a2c.py --run runs/a2c_base --iters 40 --games 16
"""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

from ppo import A2CConfig, A2CTrainer


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="A2C self-play training.")
    parser.add_argument("--run", default="runs/a2c_base")
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--games", type=int, default=16, dest="games_per_iter")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--batch", type=int, default=128, dest="batch_size")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--time-budget", type=float, default=None)
    args = parser.parse_args(argv)

    config = A2CConfig(
        run_dir=args.run,
        iterations=args.iters,
        games_per_iter=args.games_per_iter,
        hidden=args.hidden,
        layers=args.layers,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        entropy_coef=args.entropy_coef,
        seed=args.seed,
        resume=args.resume,
        time_budget=args.time_budget,
    )
    print(
        f"Training A2C on {config.run_dir}: {config.iterations} iters x "
        f"{config.games_per_iter} games, no search",
        flush=True,
    )
    path = A2CTrainer(config).run()
    print(f"\nsaved final checkpoint to {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

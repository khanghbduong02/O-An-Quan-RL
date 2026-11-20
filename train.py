"""Train an AlphaZero-style agent by self-play.

    python train.py --run runs/az_base --iters 20 --games 32 --sims 64
    python train.py --run runs/az_base --resume runs/az_base/latest.npz --iters 10
"""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

from alphazero import TrainConfig, Trainer


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="AlphaZero-style self-play training.")
    parser.add_argument("--run", default="runs/az_base", help="output directory")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--games", type=int, default=32, dest="games_per_iter")
    parser.add_argument("--sims", type=int, default=64, dest="simulations")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=128, dest="batch_size")
    parser.add_argument("--steps", type=int, default=40, dest="train_steps")
    parser.add_argument("--buffer", type=int, default=400, dest="buffer_games")
    parser.add_argument("--temp-plies", type=int, default=8, dest="temperature_plies")
    parser.add_argument(
        "--value-target", default="outcome", choices=("outcome", "score_diff")
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", default=None, help="checkpoint to continue from")
    parser.add_argument(
        "--time-budget", type=float, default=None, help="stop after this many seconds"
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="processes used for self-play"
    )
    args = parser.parse_args(argv)

    config = TrainConfig(
        run_dir=args.run,
        iterations=args.iters,
        games_per_iter=args.games_per_iter,
        simulations=args.simulations,
        hidden=args.hidden,
        layers=args.layers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        train_steps=args.train_steps,
        buffer_games=args.buffer_games,
        temperature_plies=args.temperature_plies,
        value_target=args.value_target,
        seed=args.seed,
        resume=args.resume,
        time_budget=args.time_budget,
        workers=args.workers,
    )
    print(
        f"Training AlphaZero on {config.run_dir}: "
        f"{config.iterations} iters x {config.games_per_iter} games, "
        f"{config.simulations} sims, net {config.hidden}x{config.layers}",
        flush=True,
    )
    path = Trainer(config).run()
    print(f"\nsaved final checkpoint to {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

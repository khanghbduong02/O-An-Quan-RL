"""AlphaZero training loop: self-play games, replay buffer, gradient steps."""

from __future__ import annotations

import csv
import json
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from engine import DEFAULT_RULES

from .network import PolicyValueNet
from .parallel import SelfPlayPool
from .self_play import GameRecord, Sample, play_game


@dataclass
class TrainConfig:
    """Everything that defines a training run (saved next to the checkpoints)."""

    run_dir: str = "runs/az_base"
    iterations: int = 20
    games_per_iter: int = 32
    simulations: int = 64
    hidden: int = 128
    layers: int = 2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    train_steps: int = 40
    buffer_games: int = 400
    temperature_plies: int = 8
    value_target: str = "outcome"
    seed: int = 0
    checkpoint_every: int = 1
    resume: Optional[str] = None
    time_budget: Optional[float] = None
    #: self-play worker processes (1 = single process, no pool)
    workers: int = 1


class Trainer:
    """Owns the network, the buffer and the run directory."""

    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self.run_dir = Path(config.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(config.seed)

        if config.resume:
            self.net = PolicyValueNet.load(config.resume, seed=config.seed)
            self.net.lr = config.lr
            self.net.weight_decay = config.weight_decay
        else:
            self.net = PolicyValueNet(
                hidden=config.hidden,
                layers=config.layers,
                seed=config.seed,
                lr=config.lr,
                weight_decay=config.weight_decay,
            )
        self.buffer: deque = deque(maxlen=config.buffer_games)
        self.pool = SelfPlayPool(config.workers) if config.workers > 1 else None
        self.history_path = self.run_dir / "history.csv"
        (self.run_dir / "config.json").write_text(
            json.dumps(asdict(config), indent=2), encoding="utf-8"
        )

    # ------------------------------------------------------------------ helpers
    def _samples(self) -> List[Sample]:
        return [sample for game in self.buffer for sample in game]

    def _batch(self, samples: List[Sample], size: int):
        indices = self.rng.integers(0, len(samples), size=size)
        chosen = [samples[index] for index in indices]
        return (
            np.stack([sample.x for sample in chosen]),
            np.stack([sample.pi for sample in chosen]),
            np.array([sample.z for sample in chosen], dtype=np.float64),
            np.stack([sample.mask for sample in chosen]),
        )

    # ------------------------------------------------------------------- phases
    def self_play(self) -> List[GameRecord]:
        if self.pool is not None:
            records = self.pool.play(
                self.net.params,
                self.config.games_per_iter,
                hidden=self.net.hidden,
                layers=self.net.layers,
                simulations=self.config.simulations,
                temperature_plies=self.config.temperature_plies,
                value_target=self.config.value_target,
                seed=int(self.rng.integers(0, 2**31 - 1)),
            )
        else:
            records = [
                play_game(
                    self.net,
                    rules=DEFAULT_RULES,
                    simulations=self.config.simulations,
                    rng=self.rng,
                    temperature_plies=self.config.temperature_plies,
                    value_target=self.config.value_target,
                )
                for _ in range(self.config.games_per_iter)
            ]
        for record in records:
            self.buffer.append(record.samples)
        return records

    def learn(self) -> dict:
        samples = self._samples()
        losses = {"policy_loss": [], "value_loss": []}
        if len(samples) >= self.config.batch_size:
            for _ in range(self.config.train_steps):
                inputs, pi, z, mask = self._batch(samples, self.config.batch_size)
                result = self.net.train_batch(inputs, pi, z, mask)
                losses["policy_loss"].append(result["policy_loss"])
                losses["value_loss"].append(result["value_loss"])
        return losses

    def train_iteration(self, iteration: int) -> dict:
        start = time.perf_counter()
        records = self.self_play()
        selfplay_time = time.perf_counter() - start

        start = time.perf_counter()
        losses = self.learn()
        train_time = time.perf_counter() - start

        winners = [record.winner for record in records]
        stats = {
            "iteration": iteration,
            "games": len(records),
            "plies_mean": float(np.mean([record.plies for record in records])),
            "p0_wins": sum(1 for winner in winners if winner == 0),
            "p1_wins": sum(1 for winner in winners if winner == 1),
            "draws": sum(1 for winner in winners if winner == -1),
            "score_diff_mean": float(
                np.mean([abs(record.p0 - record.p1) for record in records])
            ),
            "policy_loss": float(np.mean(losses["policy_loss"]))
            if losses["policy_loss"]
            else float("nan"),
            "value_loss": float(np.mean(losses["value_loss"]))
            if losses["value_loss"]
            else float("nan"),
            "buffer_positions": len(self._samples()),
            "selfplay_s": round(selfplay_time, 1),
            "train_s": round(train_time, 1),
        }
        self._log(stats)
        return stats

    def _log(self, stats: dict) -> None:
        write_header = not self.history_path.exists()
        with self.history_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(stats))
            if write_header:
                writer.writeheader()
            writer.writerow(stats)
        print(
            "  iter {iteration:3d} | plies {plies_mean:5.1f} | W/L/D {p0_wins}/{p1_wins}/{draws}"
            " | pol {policy_loss:.3f} val {value_loss:.3f} | {selfplay_s:5.1f}s + {train_s:4.1f}s".format(
                **stats
            ),
            flush=True,
        )

    # -------------------------------------------------------------- checkpoints
    def save(self, iteration: int, name: Optional[str] = None) -> str:
        path = self.net.save(self.run_dir / (name or f"iter_{iteration:03d}.npz"))
        self.net.save(self.run_dir / "latest.npz")
        return path

    def run(self) -> str:
        started = time.perf_counter()
        iteration = 0
        for iteration in range(1, self.config.iterations + 1):
            print(f"\niteration {iteration}/{self.config.iterations}", flush=True)
            self.train_iteration(iteration)
            if self.config.checkpoint_every and iteration % self.config.checkpoint_every == 0:
                self.save(iteration)
            if (
                self.config.time_budget is not None
                and time.perf_counter() - started > self.config.time_budget
            ):
                print(f"time budget ({self.config.time_budget:.0f}s) reached", flush=True)
                break
        return self.save(iteration, name="final.npz")

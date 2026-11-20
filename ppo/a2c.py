"""Self-play advantage actor-critic: the brief's Agent A ("PPO or A2C").

No search is used: the policy samples moves, the game result gives the value
target, and the advantage is ``z - V(s)``.  Because the only reward is at the
end of the game, no bootstrapping is needed.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from alphazero.features import encode, legal_mask
from alphazero.network import PolicyValueNet
from engine import DEFAULT_RULES, Rules, action_from_index, initial_state


@dataclass
class A2CConfig:
    run_dir: str = "runs/a2c_base"
    iterations: int = 30
    games_per_iter: int = 16
    hidden: int = 128
    layers: int = 2
    lr: float = 2e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    epochs: int = 2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    temperature: float = 1.0
    seed: int = 0
    resume: Optional[str] = None
    time_budget: Optional[float] = None


def sample_action(probs: np.ndarray, temperature: float, rng: np.random.Generator) -> int:
    if temperature <= 1e-3:
        return int(np.argmax(probs))
    weights = np.power(probs, 1.0 / temperature)
    total = weights.sum()
    if total <= 0.0:  # pragma: no cover - defensive
        return int(np.argmax(probs))
    return int(rng.choice(len(probs), p=weights / total))


def play_policy_game(net: PolicyValueNet, rules: Rules = DEFAULT_RULES, rng=None, temperature: float = 1.0):
    """One self-play game driven purely by the policy head."""
    rng = np.random.default_rng(0) if rng is None else rng
    state = initial_state(rules)
    steps: List[dict] = []
    while not state.done:
        probs, value = net.predict(state, rules)
        action = sample_action(probs, temperature, rng)
        steps.append(
            {
                "x": encode(state, rules),
                "mask": legal_mask(state),
                "action": action,
                "player": state.player,
                "value": value,
            }
        )
        state, _ = state.apply(action_from_index(state.player, action), rules)

    p0, p1 = state.final_points(rules)
    for step in steps:
        mine, theirs = (p0, p1) if step["player"] == 0 else (p1, p0)
        step["z"] = 1.0 if mine > theirs else (-1.0 if theirs > mine else 0.0)
        step["advantage"] = step["z"] - step["value"]
    return steps, p0, p1


class A2CTrainer:
    def __init__(self, config: A2CConfig) -> None:
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
        self.history_path = self.run_dir / "history.csv"
        (self.run_dir / "config.json").write_text(
            json.dumps(asdict(config), indent=2), encoding="utf-8"
        )

    def train_iteration(self, iteration: int) -> dict:
        started = time.perf_counter()
        steps: List[dict] = []
        results = []
        for _ in range(self.config.games_per_iter):
            game_steps, p0, p1 = play_policy_game(
                self.net, DEFAULT_RULES, self.rng, self.config.temperature
            )
            steps.extend(game_steps)
            results.append((p0, p1))
        collect_time = time.perf_counter() - started

        inputs = np.stack([step["x"] for step in steps])
        actions = np.array([step["action"] for step in steps], dtype=np.int64)
        values_target = np.array([step["z"] for step in steps], dtype=np.float64)
        masks = np.stack([step["mask"] for step in steps])
        advantages = np.array([step["advantage"] for step in steps], dtype=np.float64)
        if advantages.size and advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        started = time.perf_counter()
        policy_losses: List[float] = []
        value_losses: List[float] = []
        entropies: List[float] = []
        for _ in range(self.config.epochs):
            order = self.rng.permutation(len(inputs))
            for start in range(0, len(inputs), self.config.batch_size):
                batch = order[start : start + self.config.batch_size]
                result = self.net.train_a2c_batch(
                    inputs[batch],
                    actions[batch],
                    advantages[batch],
                    values_target[batch],
                    masks[batch],
                    entropy_coef=self.config.entropy_coef,
                    value_coef=self.config.value_coef,
                )
                policy_losses.append(result["policy_loss"])
                value_losses.append(result["value_loss"])
                entropies.append(result["entropy"])
        train_time = time.perf_counter() - started

        winners = [
            0 if p0 > p1 else (1 if p1 > p0 else -1) for p0, p1 in results
        ]
        stats = {
            "iteration": iteration,
            "games": len(results),
            "plies_mean": len(steps) / max(len(results), 1),
            "p0_wins": sum(1 for winner in winners if winner == 0),
            "p1_wins": sum(1 for winner in winners if winner == 1),
            "draws": sum(1 for winner in winners if winner == -1),
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
            "collect_s": round(collect_time, 1),
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
            " | pol {policy_loss:+.3f} val {value_loss:.3f} ent {entropy:.2f}"
            " | {collect_s:5.1f}s + {train_s:4.1f}s".format(**stats),
            flush=True,
        )

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
            self.save(iteration)
            if (
                self.config.time_budget is not None
                and time.perf_counter() - started > self.config.time_budget
            ):
                print(f"time budget ({self.config.time_budget:.0f}s) reached", flush=True)
                break
        return self.save(iteration, name="final.npz")

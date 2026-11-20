"""Tests for the baselines, the match harness and the training loops."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from alphazero import PolicyValueNet, TrainConfig, Trainer, play_game
from baselines import AlphaBetaAgent, GreedyAgent, NetAgent, PlainMCTS, RandomAgent
from engine import State, initial_state
from eval.arena import play_match
from eval.elo import elo_difference, fit_elo
from ppo import A2CConfig, A2CTrainer

# A position where square 1 throws three stones onto 2, 3, 4, leaving 5 empty
# in front of the rich mandarin square 6: capturing it is worth 15.
CAPTURE_POSITION = [0, 3, 0, 0, 2, 0, 5, 0, 0, 0, 0, 0]


def test_greedy_agent_takes_the_mandarin():
    state = State.from_cells(CAPTURE_POSITION)
    action = GreedyAgent(seed=0).choose(state)
    _, result = state.apply(action)
    assert result.points == 15, result


def test_alpha_beta_finds_the_mandarin():
    state = State.from_cells(CAPTURE_POSITION)
    action = AlphaBetaAgent(depth=2, seed=0).choose(state)
    _, result = state.apply(action)
    assert result.points == 15, result


def test_alpha_beta_returns_legal_moves():
    state = initial_state()
    agent = AlphaBetaAgent(depth=2, time_limit=5.0, seed=0)
    assert agent.choose(state) in state.legal_actions()


def test_plain_mcts_returns_legal_moves():
    state = initial_state()
    agent = PlainMCTS(iterations=8, seed=0)
    assert agent.choose(state) in state.legal_actions()


def test_net_agent_returns_legal_moves():
    net = PolicyValueNet(hidden=8, seed=0)
    state = initial_state()
    assert NetAgent(net, simulations=0).choose(state) in state.legal_actions()
    assert NetAgent(net, simulations=4, seed=0).choose(state) in state.legal_actions()


def test_arena_counts_every_game():
    match = play_match(RandomAgent(1), RandomAgent(2), games=4)
    assert match.games == 4
    assert match.wins + match.losses + match.draws == 4
    assert 0.0 <= match.score <= 1.0


def test_elo_helpers():
    assert abs(elo_difference(0.5)) < 1e-9
    assert 65.0 < elo_difference(0.6) < 75.0
    # a beats b always and c 90% of the time; b and c draw, so c (who loses
    # less often to a) must come out ahead of b.
    ratings = fit_elo(
        ["a", "b", "c"],
        [(0, 1, 1.0, 40), (1, 2, 0.5, 40), (0, 2, 0.9, 40)],
    )
    assert ratings["a"] > ratings["c"] > ratings["b"], ratings


def test_self_play_records_match_the_game():
    net = PolicyValueNet(hidden=8, seed=0)
    record = play_game(net, simulations=4, rng=np.random.default_rng(0))
    assert record.plies == len(record.samples)
    assert record.plies > 0
    for sample in record.samples:
        assert sample.pi.shape == (10,)
        assert abs(float(sample.pi.sum()) - 1.0) < 1e-6
        assert sample.z in (-1.0, 0.0, 1.0)


def test_alpha_zero_trainer_smoke():
    with tempfile.TemporaryDirectory() as directory:
        config = TrainConfig(
            run_dir=directory,
            iterations=1,
            games_per_iter=2,
            simulations=4,
            hidden=8,
            batch_size=8,
            train_steps=2,
            buffer_games=8,
        )
        trainer = Trainer(config)
        stats = trainer.train_iteration(1)
        assert stats["games"] == 2
        path = trainer.save(1, name="final.npz")
        net = PolicyValueNet.load(path)
    probs, _ = net.predict(initial_state())
    assert abs(float(probs.sum()) - 1.0) < 1e-6


def test_a2c_trainer_smoke():
    with tempfile.TemporaryDirectory() as directory:
        config = A2CConfig(
            run_dir=directory,
            iterations=1,
            games_per_iter=2,
            hidden=8,
            batch_size=16,
            epochs=1,
        )
        trainer = A2CTrainer(config)
        stats = trainer.train_iteration(1)
        assert stats["games"] == 2
        path = trainer.save(1, name="final.npz")
        net = PolicyValueNet.load(path)
    probs, _ = net.predict(initial_state())
    assert abs(float(probs.sum()) - 1.0) < 1e-6


if __name__ == "__main__":
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\n{len(tests)} tests passed")

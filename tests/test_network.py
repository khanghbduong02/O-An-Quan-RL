"""Numerical checks for the NumPy policy/value network.

Backpropagation is verified against central finite differences, so a silent
gradient bug cannot survive the test suite.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from alphazero.features import FEATURES, legal_mask  # noqa: E402
from alphazero.network import (  # noqa: E402
    PolicyValueNet,
    masked_entropy,
    masked_softmax,
)
from engine import N_ACTIONS, initial_state  # noqa: E402


def fake_batch(seed: int = 0, count: int = 3):
    rng = np.random.default_rng(seed)
    inputs = rng.normal(size=(count, FEATURES))
    mask = np.ones((count, N_ACTIONS))
    mask[0, ::2] = 0.0
    target = rng.dirichlet(np.ones(N_ACTIONS), size=count) * mask
    target /= target.sum(axis=1, keepdims=True)
    values = rng.uniform(-1.0, 1.0, size=count)
    return inputs, target, values, mask


def test_masked_softmax_respects_legality():
    logits = np.array([[0.0] * 10])
    mask = np.zeros((1, 10))
    mask[0, [1, 3]] = 1.0
    probs = masked_softmax(logits, mask)
    assert np.allclose(probs, np.array([[0, 0.5, 0, 0.5, 0, 0, 0, 0, 0, 0]]))


def test_supervised_gradients_match_finite_differences():
    inputs, target, values_target, mask = fake_batch()
    net = PolicyValueNet(hidden=8, layers=2, seed=1, weight_decay=0.0)
    count = inputs.shape[0]

    def loss() -> float:
        logits, values, _ = net.forward(inputs)
        probs = masked_softmax(logits, mask)
        return float(
            -(target * np.log(np.maximum(probs, 1e-12))).sum() / count
            + ((values - values_target) ** 2).mean()
        )

    logits, values, cache = net.forward(inputs)
    probs = masked_softmax(logits, mask)
    grads = net._backprop(cache, (probs - target) / count, 2.0 * (values - values_target) / count)

    rng = np.random.default_rng(3)
    epsilon = 1e-5
    for key, param in net.params.items():
        for _ in range(3):
            index = tuple(int(rng.integers(0, size)) for size in param.shape)
            original = param[index]
            param[index] = original + epsilon
            up = loss()
            param[index] = original - epsilon
            down = loss()
            param[index] = original
            numeric = (up - down) / (2 * epsilon)
            analytic = float(grads[key][index])
            assert abs(numeric - analytic) <= 1e-5 + 1e-4 * abs(numeric), (key, numeric, analytic)


def test_a2c_gradients_match_finite_differences():
    inputs, _, values_target, mask = fake_batch(seed=2)
    net = PolicyValueNet(hidden=8, layers=2, seed=2, weight_decay=0.0)
    rng = np.random.default_rng(5)
    actions = np.array([0 if mask[row, 0] else 1 for row in range(inputs.shape[0])])
    advantages = rng.normal(size=inputs.shape[0])
    entropy_coef, value_coef = 0.05, 0.5
    rows = np.arange(len(actions))

    def loss() -> float:
        logits, values, _ = net.forward(inputs)
        probs = masked_softmax(logits, mask)
        logp = np.log(np.maximum(probs, 1e-12))
        return float(
            -(advantages * logp[rows, actions]).mean()
            + value_coef * ((values - values_target) ** 2).mean()
            - entropy_coef * masked_entropy(probs, mask).mean()
        )

    logits, values, cache = net.forward(inputs)
    probs = masked_softmax(logits, mask)
    logp = np.log(np.maximum(probs, 1e-12))
    d_logits, d_values, _ = net._a2c_gradients(
        probs, logp, values, actions, advantages, values_target, mask, entropy_coef, value_coef
    )
    grads = net._backprop(cache, d_logits, d_values)

    rng = np.random.default_rng(11)
    epsilon = 1e-5
    for key, param in net.params.items():
        for _ in range(2):
            index = tuple(int(rng.integers(0, size)) for size in param.shape)
            original = param[index]
            param[index] = original + epsilon
            up = loss()
            param[index] = original - epsilon
            down = loss()
            param[index] = original
            numeric = (up - down) / (2 * epsilon)
            analytic = float(grads[key][index])
            assert abs(numeric - analytic) <= 1e-5 + 1e-4 * abs(numeric), (key, numeric, analytic)


def test_predictions_are_legal_and_normalised():
    net = PolicyValueNet(hidden=16, seed=0)
    state = initial_state()
    probs, value = net.predict(state)
    assert abs(float(probs.sum()) - 1.0) < 1e-6
    mask = legal_mask(state)
    assert np.all(probs[mask == 0] == 0.0)
    assert np.count_nonzero(probs) == 10
    assert -1.0 <= value <= 1.0


def test_checkpoint_round_trip():
    net = PolicyValueNet(hidden=16, seed=7)
    with tempfile.TemporaryDirectory() as directory:
        path = net.save(Path(directory) / "ckpt.npz")
        loaded = PolicyValueNet.load(path)
    state = initial_state()
    first, value_first = net.predict(state)
    second, value_second = loaded.predict(state)
    assert np.allclose(first, second)
    assert abs(value_first - value_second) < 1e-9


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

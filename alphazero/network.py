"""Small policy/value network in NumPy (no deep-learning framework needed).

Architecture: a shared ReLU MLP trunk with a policy head (10 logits, masked to
legal moves) and a value head (tanh, value of the position for the player to
move).  Training uses AdamW on the sum of a policy cross-entropy and a value
mean-squared error, computed with hand-written backpropagation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from engine import DEFAULT_RULES, N_ACTIONS, Rules, State

from .features import FEATURES, encode, legal_mask

_NEG = -1e9


def masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Softmax over legal actions only.  ``logits``/``mask`` are (N, A)."""
    masked = np.where(mask > 0, logits, _NEG)
    masked = masked - masked.max(axis=1, keepdims=True)
    weights = np.exp(masked) * (mask > 0)
    total = weights.sum(axis=1, keepdims=True)
    return weights / np.maximum(total, 1e-12)


def masked_entropy(probs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Entropy per row, ignoring illegal actions."""
    logp = np.log(np.maximum(probs, 1e-12)) * (mask > 0)
    return -(probs * logp).sum(axis=1)


class PolicyValueNet:
    """Two-headed MLP with manual gradients and an AdamW optimiser."""

    def __init__(
        self,
        hidden: int = 128,
        layers: int = 2,
        seed: int = 0,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> None:
        if layers < 1:
            raise ValueError("at least one hidden layer is required")
        self.hidden = hidden
        self.layers = layers
        self.lr = lr
        self.weight_decay = weight_decay

        rng = np.random.default_rng(seed)
        sizes = [FEATURES] + [hidden] * layers
        params: Dict[str, np.ndarray] = {}
        for index in range(layers):
            scale = np.sqrt(2.0 / sizes[index])
            params[f"W{index + 1}"] = rng.normal(0.0, scale, (sizes[index], sizes[index + 1]))
            params[f"b{index + 1}"] = np.zeros(sizes[index + 1])
        # Small policy head: start close to a uniform prior.
        params["Wp"] = rng.normal(0.0, 0.01, (hidden, N_ACTIONS))
        params["bp"] = np.zeros(N_ACTIONS)
        params["Wv"] = rng.normal(0.0, np.sqrt(1.0 / hidden), (hidden, 1))
        params["bv"] = np.zeros(1)
        self.params = params

        self._m = {key: np.zeros_like(value) for key, value in params.items()}
        self._v = {key: np.zeros_like(value) for key, value in params.items()}
        self._updates = 0

    # ------------------------------------------------------------- inference
    def forward(self, inputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Return (policy logits, values, cache for backprop)."""
        cache: dict = {"inputs": inputs, "pre": [], "post": []}
        hidden = inputs
        for index in range(1, self.layers + 1):
            pre = hidden @ self.params[f"W{index}"] + self.params[f"b{index}"]
            hidden = np.maximum(pre, 0.0)
            cache["pre"].append(pre)
            cache["post"].append(hidden)
        cache["hidden"] = hidden
        logits = hidden @ self.params["Wp"] + self.params["bp"]
        values = np.tanh(hidden @ self.params["Wv"] + self.params["bv"])[:, 0]
        return logits, values, cache

    def predict(
        self, state: State, rules: Rules = DEFAULT_RULES
    ) -> Tuple[np.ndarray, float]:
        """Legal-move prior and value for the player to move."""
        logits, values, _ = self.forward(encode(state, rules)[None, :])
        probs = masked_softmax(logits, legal_mask(state)[None, :])[0]
        return probs, float(values[0])

    def predict_many(self, states, rules: Rules = DEFAULT_RULES):
        """Batched variant used by evaluation code."""
        inputs = np.stack([encode(state, rules) for state in states])
        masks = np.stack([legal_mask(state) for state in states])
        logits, values, _ = self.forward(inputs)
        return masked_softmax(logits, masks), values

    # -------------------------------------------------------------- training
    def _backprop(self, cache: dict, d_logits: np.ndarray, d_values: np.ndarray):
        """Gradients for all parameters given head gradients."""
        hidden = cache["hidden"]
        values = np.tanh(hidden @ self.params["Wv"] + self.params["bv"])[:, 0]
        grads: Dict[str, np.ndarray] = {}
        grads["Wp"] = hidden.T @ d_logits
        grads["bp"] = d_logits.sum(axis=0)
        d_hidden = d_logits @ self.params["Wp"].T
        d_pre_value = (d_values * (1.0 - values**2))[:, None]
        grads["Wv"] = hidden.T @ d_pre_value
        grads["bv"] = d_pre_value.sum(axis=0)
        d_hidden = d_hidden + d_pre_value @ self.params["Wv"].T
        for index in range(self.layers, 0, -1):
            d_pre = d_hidden * (cache["pre"][index - 1] > 0.0)
            previous = cache["post"][index - 2] if index > 1 else cache["inputs"]
            grads[f"W{index}"] = previous.T @ d_pre
            grads[f"b{index}"] = d_pre.sum(axis=0)
            d_hidden = d_pre @ self.params[f"W{index}"].T
        return grads

    def _adam(self, grads: Dict[str, np.ndarray], lr: Optional[float] = None) -> None:
        lr = self.lr if lr is None else lr
        self._updates += 1
        bias1 = 1.0 - 0.9**self._updates
        bias2 = 1.0 - 0.999**self._updates
        for key, grad in grads.items():
            m = self._m[key]
            v = self._v[key]
            m *= 0.9
            m += 0.1 * grad
            v *= 0.999
            v += 0.001 * grad * grad
            step = (m / bias1) / (np.sqrt(v / bias2) + 1e-8)
            self.params[key] -= lr * (step + self.weight_decay * self.params[key])

    def train_batch(
        self,
        inputs: np.ndarray,
        policy_target: np.ndarray,
        value_target: np.ndarray,
        mask: np.ndarray,
        lr: Optional[float] = None,
    ) -> dict:
        """One supervised step on (state, visit distribution, outcome)."""
        count = inputs.shape[0]
        logits, values, cache = self.forward(inputs)
        probs = masked_softmax(logits, mask)
        logp = np.log(np.maximum(probs, 1e-12))
        policy_loss = float(-(policy_target * logp).sum() / count)
        value_loss = float(((values - value_target) ** 2).mean())

        d_logits = (probs - policy_target) / count
        d_values = 2.0 * (values - value_target) / count
        self._adam(self._backprop(cache, d_logits, d_values), lr)
        return {"policy_loss": policy_loss, "value_loss": value_loss}

    def train_a2c_batch(
        self,
        inputs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
        value_targets: np.ndarray,
        mask: np.ndarray,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        lr: Optional[float] = None,
    ) -> dict:
        """One actor-critic step on on-policy samples (see ``ppo/a2c.py``)."""
        count = inputs.shape[0]
        logits, values, cache = self.forward(inputs)
        probs = masked_softmax(logits, mask)
        logp = np.log(np.maximum(probs, 1e-12))
        policy_loss = float(-(advantages * logp[np.arange(count), actions]).mean())
        value_loss = float(((values - value_targets) ** 2).mean())
        d_logits, d_values, entropy = self._a2c_gradients(
            probs,
            logp,
            values,
            actions,
            advantages,
            value_targets,
            mask,
            entropy_coef,
            value_coef,
        )
        self._adam(self._backprop(cache, d_logits, d_values), lr)
        return {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
        }

    @staticmethod
    def _a2c_gradients(
        probs: np.ndarray,
        logp: np.ndarray,
        values: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
        value_targets: np.ndarray,
        mask: np.ndarray,
        entropy_coef: float,
        value_coef: float,
    ):
        """Head gradients of -A*log pi(a) + c_v*(v-z)^2 - c_e*H(pi)."""
        count = probs.shape[0]
        entropy_rows = masked_entropy(probs, mask)
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(count), actions] = 1.0
        d_logits = -(advantages[:, None] * (one_hot - probs)) / count
        # d(-H)/dz = +p * (log p + H)  (entropy is maximised, so it enters J
        # with a minus sign; the two minus signs cancel)
        d_logits += entropy_coef * (probs * (logp + entropy_rows[:, None])) / count
        d_values = value_coef * 2.0 * (values - value_targets) / count
        return d_logits, d_values, float(entropy_rows.mean())

    # ------------------------------------------------------------ persistence
    def save(self, path) -> str:
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "hidden": self.hidden,
            "layers": self.layers,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "features": FEATURES,
            "actions": N_ACTIONS,
        }
        np.savez(
            path,
            **{key: value for key, value in self.params.items()},
            meta=np.array(json.dumps(meta)),
        )
        return str(path)

    @classmethod
    def load(cls, path, seed: int = 0) -> "PolicyValueNet":
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["meta"]))
        if meta.get("features") != FEATURES or meta.get("actions") != N_ACTIONS:
            raise ValueError("checkpoint was built for a different encoding")
        net = cls(
            hidden=int(meta["hidden"]),
            layers=int(meta["layers"]),
            seed=seed,
            lr=float(meta.get("lr", 1e-3)),
            weight_decay=float(meta.get("weight_decay", 1e-4)),
        )
        for key in net.params:
            net.params[key] = np.array(data[key], dtype=np.float64)
        return net

"""AlphaZero-style agent: NumPy policy/value network, PUCT search, self-play."""

from .features import FEATURES, encode, legal_mask
from .mcts import MCTS
from .network import PolicyValueNet, masked_softmax
from .self_play import GameRecord, Sample, play_game
from .trainer import TrainConfig, Trainer

__all__ = [
    "FEATURES",
    "GameRecord",
    "MCTS",
    "PolicyValueNet",
    "Sample",
    "TrainConfig",
    "Trainer",
    "encode",
    "legal_mask",
    "masked_softmax",
    "play_game",
]

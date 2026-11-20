"""Self-play policy-gradient agent (A2C; the brief allows PPO or A2C)."""

from .a2c import A2CConfig, A2CTrainer, play_policy_game, sample_action

__all__ = ["A2CConfig", "A2CTrainer", "play_policy_game", "sample_action"]

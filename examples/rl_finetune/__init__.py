"""
RL Fine-tuning for SmolVLA Policy

This package provides tools for fine-tuning SmolVLA policies using residual RL methods.
"""

from .residual_smolvla import ResidualSmolVLAPolicy, ResidualMLP
from .rl_finetune_smolvla import ResidualRLTrainer, SimpleRewardFunction

__all__ = [
    "ResidualSmolVLAPolicy",
    "ResidualMLP",
    "ResidualRLTrainer",
    "SimpleRewardFunction",
]



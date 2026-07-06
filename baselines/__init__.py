"""Simple Quoridor baseline players."""

from .greedy import GreedyPlayer
from .minimax import MinimaxPlayer
from .random_player import RandomPlayer

__all__ = ["RandomPlayer", "GreedyPlayer", "MinimaxPlayer"]

"""Uniform random legal baseline."""

from __future__ import annotations

import random

import quoridor_engine as qe

from .common import legal_actions


class RandomPlayer:
    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def select_action(self, state: qe.GameState) -> int:
        actions = legal_actions(state)
        if not actions:
            raise ValueError("cannot select an action from a terminal state")
        return self.rng.choice(actions)

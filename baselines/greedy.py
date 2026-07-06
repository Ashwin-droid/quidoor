"""Greedy shortest-path baseline."""

from __future__ import annotations

import quoridor_engine as qe

from .common import clone_apply, legal_actions


class GreedyPlayer:
    """Always step to a pawn move that minimizes own shortest path length."""

    def select_action(self, state: qe.GameState) -> int:
        player = state.to_move
        actions = legal_actions(state)
        pawn_actions = [a for a in actions if a < 12]
        if not pawn_actions:
            if not actions:
                raise ValueError("cannot select an action from a terminal state")
            return actions[0]

        def score(action: int) -> tuple[int, int]:
            child = clone_apply(state, action)
            return child.shortest_path_len(player), action

        return min(pawn_actions, key=score)

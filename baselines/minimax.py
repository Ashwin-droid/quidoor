"""Two-ply alpha-beta baseline with a shortest-path-difference heuristic."""

from __future__ import annotations

import math

import quoridor_engine as qe

from .common import clone_apply, evaluate_shortest_path_diff, legal_actions


class MinimaxPlayer:
    def __init__(self, depth: int = 2) -> None:
        if depth < 1:
            raise ValueError("depth must be at least 1")
        self.depth = depth

    def select_action(self, state: qe.GameState) -> int:
        root_player = state.to_move
        actions = legal_actions(state)
        if not actions:
            raise ValueError("cannot select an action from a terminal state")

        best_action = actions[0]
        best_score = -math.inf
        alpha = -math.inf
        beta = math.inf
        for action in self._ordered_actions(state, actions, root_player):
            child = clone_apply(state, action)
            score = self._search(child, self.depth - 1, root_player, alpha, beta)
            if score > best_score:
                best_score = score
                best_action = action
            alpha = max(alpha, best_score)
        return best_action

    def _search(
        self,
        state: qe.GameState,
        depth: int,
        root_player: int,
        alpha: float,
        beta: float,
    ) -> float:
        winner = state.winner()
        if winner is not None or depth == 0:
            return evaluate_shortest_path_diff(state, root_player)

        actions = legal_actions(state)
        if not actions:
            return evaluate_shortest_path_diff(state, root_player)

        if state.to_move == root_player:
            value = -math.inf
            for action in self._ordered_actions(state, actions, root_player):
                value = max(
                    value,
                    self._search(clone_apply(state, action), depth - 1, root_player, alpha, beta),
                )
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value

        value = math.inf
        for action in self._ordered_actions(state, actions, root_player):
            value = min(
                value,
                self._search(clone_apply(state, action), depth - 1, root_player, alpha, beta),
            )
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value

    def _ordered_actions(
        self,
        state: qe.GameState,
        actions: list[int],
        root_player: int,
    ) -> list[int]:
        reverse = state.to_move == root_player

        def rank(action: int) -> float:
            return evaluate_shortest_path_diff(clone_apply(state, action), root_player)

        return sorted(actions, key=rank, reverse=reverse)

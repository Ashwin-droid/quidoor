"""Dead-simple terminal play against a baseline."""

from __future__ import annotations

import argparse
import sys

import quoridor_engine as qe

from .common import action_to_label, parse_action, render_board
from .greedy import GreedyPlayer
from .minimax import MinimaxPlayer
from .random_player import RandomPlayer


def make_player(name: str, seed: int | None):
    if name == "random":
        return RandomPlayer(seed=seed)
    if name == "greedy":
        return GreedyPlayer()
    if name in {"minimax", "minimax-2"}:
        return MinimaxPlayer(depth=2)
    raise ValueError(f"unknown baseline `{name}`")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", choices=["random", "greedy", "minimax-2"], default="greedy")
    parser.add_argument("--human", type=int, choices=[0, 1], default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-plies", type=int, default=200)
    args = parser.parse_args(argv)

    baseline = make_player(args.baseline, args.seed)
    state = qe.GameState()

    while True:
        print(render_board(state))
        winner = state.winner()
        if winner is not None:
            print(f"winner: player {winner}")
            return 0
        if state.ply >= args.max_plies:
            print("ply cap reached")
            return 0

        if state.to_move != args.human:
            action = baseline.select_action(state)
            print(f"{args.baseline}: {action_to_label(action, state)} ({action})")
            state.apply(action)
            continue

        try:
            raw = input("move> ")
        except EOFError:
            print("input closed")
            return 0
        if raw.strip().lower() in {"q", "quit", "exit"}:
            print("bye")
            return 0
        try:
            action = parse_action(raw, state)
        except ValueError as exc:
            print(exc, file=sys.stderr)
            continue
        state.apply(action)


if __name__ == "__main__":
    raise SystemExit(main())

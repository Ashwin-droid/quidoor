"""Quick round-robin among the fixed baselines."""

from __future__ import annotations

import argparse
from collections import defaultdict

import quoridor_engine as qe

from .greedy import GreedyPlayer
from .minimax import MinimaxPlayer
from .random_player import RandomPlayer


def make_players(seed: int):
    return {
        "random": lambda offset=0: RandomPlayer(seed=seed + offset),
        "greedy": lambda offset=0: GreedyPlayer(),
        "minimax-2": lambda offset=0: MinimaxPlayer(depth=2),
    }


def play_game(p0, p1, max_plies: int = 200) -> int | None:
    state = qe.GameState()
    while state.winner() is None and state.ply < max_plies:
        player = p0 if state.to_move == 0 else p1
        state.apply(player.select_action(state))
    return state.winner()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--games", type=int, default=4, help="games per ordered matchup")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-plies", type=int, default=200)
    args = parser.parse_args(argv)

    factories = make_players(args.seed)
    names = list(factories)
    wins: dict[str, int] = defaultdict(int)
    losses: dict[str, int] = defaultdict(int)
    draws: dict[str, int] = defaultdict(int)

    for i, p0_name in enumerate(names):
        for p1_name in names:
            if p0_name == p1_name:
                continue
            for game_idx in range(args.games):
                p0 = factories[p0_name](1000 * i + game_idx)
                p1 = factories[p1_name](2000 * i + game_idx)
                winner = play_game(p0, p1, max_plies=args.max_plies)
                if winner is None:
                    draws[p0_name] += 1
                    draws[p1_name] += 1
                elif winner == 0:
                    wins[p0_name] += 1
                    losses[p1_name] += 1
                else:
                    wins[p1_name] += 1
                    losses[p0_name] += 1

    print("player      wins  losses  draws")
    print("-------------------------------")
    for name in names:
        print(f"{name:<10} {wins[name]:>4} {losses[name]:>7} {draws[name]:>6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

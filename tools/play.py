#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import quoridor_engine as qe
from baselines import GreedyPlayer, MinimaxPlayer, RandomPlayer
from baselines.common import (
    action_to_label,
    evaluate_position,
    format_eval,
    parse_action,
    render_board,
)
from trainer.arena import (
    StubEvaluator,
    evaluate_state,
    load_checkpoint_evaluator,
    load_config_for_cli,
    resolve_step,
    search_root,
    top_actions,
)
from trainer.config import Config


def make_baseline(name: str, seed: int):
    if name == "random":
        return RandomPlayer(seed=seed)
    if name == "greedy":
        return GreedyPlayer()
    if name in {"minimax", "minimax-2"}:
        return MinimaxPlayer(depth=2)
    raise ValueError(f"unknown baseline `{name}`")


def print_net_view(state: qe.GameState, evaluator, config: Config) -> None:
    policy, value = evaluate_state(state, evaluator, config)
    items = top_actions(policy, state, k=5)
    joined = ", ".join(
        f"{item['label']}:{item['prob']:.3f}" for item in items
    )
    print(f"eval {format_eval(evaluate_position(state))}")
    print(f"net[{evaluator.name}] value={value:+.3f} top5: {joined}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play terminal Quoridor against a checkpoint, stub, or baseline.")
    parser.add_argument("--run", type=Path, default=None, help="Run directory containing ckpt/<step>")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step; defaults to latest/max step")
    parser.add_argument(
        "--opponent",
        choices=["checkpoint", "stub", "random", "greedy", "minimax-2"],
        default="checkpoint",
    )
    parser.add_argument("--human", type=int, choices=[0, 1], default=0)
    parser.add_argument("--sims", type=int, default=64, help="MCTS sims for checkpoint/stub opponent")
    parser.add_argument("--ply-cap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.device:
        os.environ["QUIDOOR_ARENA_DEVICE"] = str(args.device)

    run_dir = args.run
    step = resolve_step(run_dir, args.step)
    if run_dir is not None:
        config = load_config_for_cli(run_dir, step)
    else:
        config = Config()
    config.mcts.sims_per_move = max(1, int(args.sims))
    config.selfplay.leaf_batch_size = max(1, int(args.sims))
    config.rules.ply_cap = int(args.ply_cap)
    config.net.compile = False

    if args.opponent == "stub" or run_dir is None:
        evaluator = StubEvaluator(name="stub_eval")
    else:
        evaluator = load_checkpoint_evaluator(run_dir, step, config)
    analysis_evaluator = evaluator

    baseline = None
    if args.opponent in {"random", "greedy", "minimax-2"}:
        baseline = make_baseline(args.opponent, args.seed)

    state = qe.GameState()
    print("Moves: pawn destination like e2, pawn code like N/S/E/W, wall like he3 or vf4, or action index.")
    print("Commands: q, quit, exit.")
    print()

    while True:
        print(render_board(state))
        winner = state.winner()
        if winner is not None:
            print(f"winner: player {winner}")
            return 0
        if int(state.ply) >= int(config.rules.ply_cap):
            print("ply cap reached")
            return 0

        print_net_view(state, analysis_evaluator, config)

        if int(state.to_move) == int(args.human):
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
            print(f"human: {action_to_label(action, state)} ({action})")
            state.apply(int(action))
            print()
            continue

        if baseline is not None:
            action = int(baseline.select_action(state))
            print(f"{args.opponent}: {action_to_label(action, state)} ({action})")
        else:
            search = search_root(state, evaluator, config, seed=args.seed + int(state.ply))
            action = int(search.action)
            mcts_line = ", ".join(
                f"{item['label']}:{item['prob']:.3f}" for item in search.top_actions
            )
            print(f"{evaluator.name} MCTS: {mcts_line}")
            print(f"{evaluator.name}: {action_to_label(action, state)} ({action})")
        state.apply(action)
        print()


if __name__ == "__main__":
    raise SystemExit(main())

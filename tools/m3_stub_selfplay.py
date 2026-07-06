#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

import quoridor_engine as qe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drive M3 Rust SelfPlayManager with the built-in stub evaluator."
    )
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--sims", type=int, default=24)
    parser.add_argument("--concurrent", type=int, default=4)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--timeout-ms", type=int, default=2)
    parser.add_argument("--ply-cap", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--extra-planes", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = qe.EngineConfig(
        sims_per_move=args.sims,
        fast_sims_per_move=args.sims,
        concurrent_games=args.concurrent,
        leaf_batch_size=args.batch,
        collect_timeout_ms=args.timeout_ms,
        ply_cap=args.ply_cap,
        extra_planes=args.extra_planes,
    )
    sp = qe.SelfPlayManager(cfg, num_games=args.games, seed=args.seed)

    games = 0
    positions = 0
    leaves = 0
    empty_polls = 0
    started = time.perf_counter()

    while not sp.done():
        planes, leaf_ids, masks = sp.collect_leaves(
            max_batch=args.batch, timeout_ms=args.timeout_ms
        )
        del masks
        batch = int(planes.shape[0])
        if batch:
            policy_logits, values = qe.stub_eval(planes)
            sp.submit(leaf_ids, policy_logits, values)
            leaves += batch
        else:
            empty_polls += 1

        for states, policies, masks, to_move, outcome in sp.drain_finished_games():
            del policies, masks, to_move, outcome
            games += 1
            positions += int(states.shape[0])

    for states, policies, masks, to_move, outcome in sp.drain_finished_games():
        del policies, masks, to_move, outcome
        games += 1
        positions += int(states.shape[0])

    elapsed = max(time.perf_counter() - started, 1.0e-9)
    print(
        f"completed {games} games in {elapsed:.3f}s "
        f"({games / elapsed:.2f} games/sec, {positions / elapsed:.1f} positions/sec)"
    )
    print(
        f"positions={positions} leaves={leaves} empty_polls={empty_polls} "
        f"sims_per_move={args.sims} batch={args.batch} concurrent={args.concurrent}"
    )


if __name__ == "__main__":
    main()

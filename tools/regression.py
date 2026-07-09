#!/usr/bin/env python3
"""Checkpoint regression gate: play the latest checkpoint against the previous one.

Exits 0 when the latest checkpoint holds its ground, 1 on a detected
regression, 2 when the matchup cannot be set up (e.g. no previous
checkpoint). Results are also appended under `<run>/regression/`.

    python tools/regression.py --run runs/run1                # latest vs previous, 50 games
    python tools/regression.py --run runs/run1 --games 20 --sims 64
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trainer.arena import (
    SearchAgent,
    find_previous_checkpoint_step,
    load_checkpoint_evaluator,
    load_config_for_cli,
    resolve_step,
    run_matchup_series,
    write_json_atomic,
)


def wilson_interval(score: float, games: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a win rate (draws counted as half wins)."""

    if games <= 0:
        return 0.0, 1.0
    denom = 1.0 + z * z / games
    center = (score + z * z / (2 * games)) / denom
    spread = z * math.sqrt(score * (1.0 - score) / games + z * z / (4 * games * games)) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regression-test the latest checkpoint against the previous one.")
    parser.add_argument("--run", required=True, type=Path, help="Run directory, e.g. runs/run1")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint under test (default: latest)")
    parser.add_argument("--baseline-step", type=int, default=None, help="Reference checkpoint (default: previous)")
    parser.add_argument("--games", type=int, default=50, help="Games to play, colors balanced (default: 50)")
    parser.add_argument("--sims", type=int, default=64, help="MCTS sims per move for both sides (default: 64)")
    parser.add_argument("--ply-cap", type=int, default=None)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.45,
        help="Fail when the latest checkpoint scores below this rate (draws = 0.5; default: 0.45)",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.device:
        os.environ["QUIDOOR_ARENA_DEVICE"] = str(args.device)

    run_dir = Path(args.run)
    step = resolve_step(run_dir, args.step)
    baseline_step = (
        int(args.baseline_step)
        if args.baseline_step is not None
        else find_previous_checkpoint_step(run_dir, step)
    )
    if baseline_step is None:
        print(f"no previous checkpoint before step {step} in {run_dir}/ckpt — nothing to compare")
        return 2
    if int(baseline_step) == int(step):
        print(f"baseline step equals test step ({step}) — nothing to compare")
        return 2

    config = load_config_for_cli(run_dir, step)
    config.mcts.sims_per_move = max(1, int(args.sims))
    config.selfplay.leaf_batch_size = max(1, int(args.sims))
    if args.ply_cap is not None:
        config.rules.ply_cap = int(args.ply_cap)
    config.net.compile = False

    latest = load_checkpoint_evaluator(run_dir, step, config)
    baseline = load_checkpoint_evaluator(run_dir, baseline_step, config)
    for evaluator, label in ((latest, step), (baseline, baseline_step)):
        if evaluator.kind != "checkpoint":
            print(f"checkpoint {label} did not load a model ({getattr(evaluator, 'detail', None)})")
            return 2

    games = max(1, int(args.games))
    print(f"regression gate: {latest.name} vs {baseline.name}, {games} games, {args.sims} sims/move")
    started = time.time()
    result = run_matchup_series(
        latest.name,
        lambda state, seed: SearchAgent(latest, config, state, seed),
        baseline.name,
        lambda state, seed: SearchAgent(baseline, config, state, seed),
        games,
        config,
        seed=int(args.seed),
    )
    duration = time.time() - started

    score = (result.wins_a + 0.5 * result.draws) / games
    low, high = wilson_interval(score, games)
    passed = score >= float(args.threshold)
    verdict = "PASS" if passed else "REGRESSION"

    summary = {
        "version": 1,
        "run": str(run_dir),
        "step": int(step),
        "baseline_step": int(baseline_step),
        "games": games,
        "sims": int(args.sims),
        "wins": result.wins_a,
        "losses": result.wins_b,
        "draws": result.draws,
        "score": round(score, 4),
        "score_ci95": [round(low, 4), round(high, 4)],
        "threshold": float(args.threshold),
        "verdict": verdict,
        "duration_sec": round(duration, 1),
        "finished_at": time.time(),
    }
    out_path = run_dir / "regression" / f"{int(step)}_vs_{int(baseline_step)}.json"
    write_json_atomic(out_path, summary)

    print(
        f"{latest.name}: {result.wins_a} wins, {result.wins_b} losses, {result.draws} draws "
        f"→ score {score:.1%} (95% CI {low:.1%}–{high:.1%}) in {duration:.0f}s"
    )
    if high < 0.5:
        print("note: the confidence interval sits entirely below 50% — this is a clear regression")
    elif low > 0.5:
        print("note: the confidence interval sits entirely above 50% — clear improvement")
    print(f"verdict: {verdict} (threshold {args.threshold:.0%}), saved {out_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

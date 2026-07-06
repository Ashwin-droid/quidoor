#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from tensorboard.backend.event_processing import event_accumulator
except Exception:  # pragma: no cover - depends on optional local tooling
    event_accumulator = None  # type: ignore[assignment]


PREFERRED_SCALARS = (
    "training/loss",
    "training/policy_loss",
    "training/value_loss",
    "training/policy_entropy",
    "training/lr",
    "selfplay/games_per_hour",
    "selfplay/positions_per_sec",
    "selfplay/avg_game_length",
    "selfplay/draw_rate",
    "selfplay/mcts_sims_per_sec",
    "strength/elo",
    "arena/win_rate_vs_previous",
    "arena/win_rate_vs_random",
    "arena/win_rate_vs_greedy",
    "arena/win_rate_vs_minimax-2",
)


def load_status(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "status.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def latest_scalars(run_dir: Path, limit: int = 16) -> tuple[list[dict[str, Any]], str | None]:
    tb_dir = run_dir / "tb"
    if event_accumulator is None:
        return [], "tensorboard is not importable"
    if not tb_dir.exists():
        return [], "no tb/ directory"

    try:
        accumulator = event_accumulator.EventAccumulator(
            str(tb_dir),
            size_guidance={event_accumulator.SCALARS: 1},
        )
        accumulator.Reload()
    except Exception as exc:
        return [], f"could not read tensorboard events: {exc}"

    tags = list(accumulator.Tags().get("scalars", []))
    selected = [tag for tag in PREFERRED_SCALARS if tag in tags]
    selected.extend(tag for tag in tags if tag not in selected)
    out: list[dict[str, Any]] = []
    for tag in selected[:limit]:
        events = accumulator.Scalars(tag)
        if not events:
            continue
        event = events[-1]
        out.append(
            {
                "tag": tag,
                "step": int(event.step),
                "value": float(event.value),
                "wall_time": float(event.wall_time),
            }
        )
    return out, None


def fmt_value(value: Any) -> str:
    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value:.4g}"
        return str(value)
    return str(value)


def fmt_age(timestamp: Any) -> str:
    try:
        age = max(0.0, time.time() - float(timestamp))
    except Exception:
        return "unknown"
    if age < 60:
        return f"{age:.0f}s ago"
    if age < 3600:
        return f"{age / 60:.1f}m ago"
    return f"{age / 3600:.1f}h ago"


def render_status(run_dir: Path, status: dict[str, Any] | None) -> list[str]:
    lines = [f"Run: {run_dir}"]
    if status is None:
        lines.append("status.json: missing")
        return lines

    state = status.get("state", status.get("phase", "unknown"))
    phase = status.get("phase", state)
    lines.append(f"State: {state}    Phase: {phase}")

    fields = (
        ("last_checkpoint", "Last checkpoint"),
        ("training_step", "Training step"),
        ("steps", "Steps"),
        ("games_completed", "Games"),
        ("positions_generated", "Positions"),
        ("buffer_fill", "Buffer fill"),
        ("buffer_size", "Buffer size"),
        ("eta", "ETA"),
        ("shutdown", "Shutdown"),
    )
    for key, label in fields:
        if key in status:
            lines.append(f"{label}: {fmt_value(status[key])}")
    timestamp = status.get("updated_at") or status.get("timestamp") or status.get("time")
    if timestamp is not None:
        lines.append(f"Updated: {fmt_age(timestamp)}")
    return lines


def render_scalars(scalars: list[dict[str, Any]], note: str | None) -> list[str]:
    lines = ["", "Latest TensorBoard Scalars:"]
    if note:
        lines.append(f"  {note}")
    if not scalars:
        if not note:
            lines.append("  none found")
        return lines
    width = max(len(item["tag"]) for item in scalars)
    for item in scalars:
        lines.append(
            f"  {item['tag']:<{width}}  step={item['step']:<8d} value={fmt_value(item['value'])}"
        )
    return lines


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print a one-screen Quoridor run summary.")
    parser.add_argument("run", type=Path, help="Run directory")
    parser.add_argument("--scalars", type=int, default=16, help="Maximum scalar tags to print")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = args.run
    status = load_status(run_dir)
    scalars, note = latest_scalars(run_dir, limit=max(0, int(args.scalars)))
    print("\n".join(render_status(run_dir, status) + render_scalars(scalars, note)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

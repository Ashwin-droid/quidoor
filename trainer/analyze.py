from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

import quoridor_engine as qe
from baselines.common import parse_action

from .arena import (
    RootSearchResult,
    load_config_for_cli,
    load_checkpoint_evaluator,
    search_root,
    top_actions,
    write_json_atomic,
)
from .config import Config


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROBE_PATH = ROOT / "probes" / "quoridor_probes.yaml"


def load_probe_suite(path: Path = DEFAULT_PROBE_PATH) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        suite = yaml.safe_load(handle)
    if not isinstance(suite, dict):
        raise ValueError("probe suite must be a YAML mapping")
    positions = suite.get("positions")
    if not isinstance(positions, list) or not positions:
        raise ValueError("probe suite must contain a non-empty positions list")
    return suite


def run_probe_suite(run_dir: Path, ckpt_step: int, config: Config) -> dict[str, Any]:
    run_dir = Path(run_dir)
    suite = load_probe_suite()
    evaluator = load_checkpoint_evaluator(run_dir, ckpt_step, config)
    started_at = time.time()

    results: list[dict[str, Any]] = []
    pass_count = 0
    check_count = 0
    for idx, item in enumerate(suite["positions"]):
        if not isinstance(item, dict):
            raise ValueError(f"probe entry {idx} is not a mapping")
        probe_id = str(item.get("id", f"probe_{idx}"))
        position = str(item["position"])
        state = qe.GameState.from_string(position)
        search = search_root(state, evaluator, config, seed=int(ckpt_step) * 10_000 + idx)
        assertions = list(item.get("assertions", []))
        assertion_results = [
            evaluate_assertion(assertion, state, search)
            for assertion in assertions
            if isinstance(assertion, dict)
        ]
        check_count += len(assertion_results)
        pass_count += sum(1 for result in assertion_results if result["passed"])
        results.append(
            {
                "id": probe_id,
                "theme": item.get("theme"),
                "description": item.get("description"),
                "position": state.to_string(),
                "to_move": int(state.to_move),
                "value": float(search.value),
                "best_action": {
                    "action": int(search.action),
                    "label": search.top_actions[0]["label"] if search.top_actions else None,
                },
                "top_actions": search.top_actions,
                "assertions": assertion_results,
            }
        )

    payload = {
        "version": 1,
        "suite": suite.get("suite", "quoridor_probes"),
        "step": int(ckpt_step),
        "run_dir": str(run_dir),
        "evaluator": {
            "name": evaluator.name,
            "kind": evaluator.kind,
            "detail": getattr(evaluator, "detail", None),
        },
        "started_at": started_at,
        "finished_at": time.time(),
        "summary": {
            "positions": len(results),
            "assertions": check_count,
            "soft_passes": pass_count,
            "soft_failures": check_count - pass_count,
        },
        "results": results,
    }
    out_dir = run_dir / "analysis" / f"ckpt_{int(ckpt_step)}"
    write_json_atomic(out_dir / "probes.json", payload)
    return payload


def evaluate_assertion(
    assertion: dict[str, Any],
    state: qe.GameState,
    search: RootSearchResult,
) -> dict[str, Any]:
    kind = str(assertion.get("type", "unknown"))
    result: dict[str, Any] = {
        "type": kind,
        "note": assertion.get("note"),
        "passed": False,
        "observed": None,
        "expected": assertion,
    }

    try:
        if kind == "top_action_type":
            expected = str(assertion["expected"])
            actual = action_type(search.action)
            result["observed"] = actual
            result["passed"] = actual == expected
        elif kind == "action_type_mass":
            expected = str(assertion["expected"])
            minimum = float(assertion.get("min", 0.0))
            mass = policy_mass_for_type(search.policy, expected)
            result["observed"] = mass
            result["passed"] = mass >= minimum
        elif kind == "action_in_top_k":
            k = int(assertion.get("k", 5))
            labels = [str(label) for label in assertion.get("actions", [])]
            expected_actions = {parse_action(label, state) for label in labels}
            observed_actions = {entry["action"] for entry in top_actions(search.policy, state, k=k)}
            result["observed"] = sorted(observed_actions)
            result["passed"] = bool(expected_actions & observed_actions)
        elif kind == "value":
            op = str(assertion.get("op", "gt"))
            threshold = float(assertion.get("threshold", 0.0))
            result["observed"] = float(search.value)
            if op == "gt":
                result["passed"] = search.value > threshold
            elif op == "lt":
                result["passed"] = search.value < threshold
            elif op == "abs_lt":
                result["passed"] = abs(search.value) < threshold
            else:
                result["error"] = f"unknown value op `{op}`"
        else:
            result["error"] = f"unknown assertion type `{kind}`"
    except Exception as exc:
        result["error"] = str(exc)
    return result


def action_type(action: int) -> str:
    return "pawn" if int(action) < 12 else "wall"


def policy_mass_for_type(policy: np.ndarray, expected: str) -> float:
    arr = np.asarray(policy, dtype=np.float64)
    if expected == "pawn":
        return float(np.sum(arr[:12]))
    if expected == "wall":
        return float(np.sum(arr[12:]))
    raise ValueError(f"unknown action type `{expected}`")


def dump_opening_book(
    run_dir: Path,
    ckpt_steps: Iterable[int],
    config: Config,
    *,
    top_k_positions: int = 8,
) -> dict[str, Any]:
    """Dump machine-readable opening policies for checkpoints.

    Replay shards persist encoded planes plus a notation sidecar, so this miner
    consumes the sidecar when present and otherwise records a clear note while
    still dumping the start position.
    """

    run_dir = Path(run_dir)
    mined_positions, mining_notes = mine_frequent_opening_positions(
        run_dir, top_k=top_k_positions
    )
    start = qe.GameState().to_string()
    positions = [{"id": "start", "position": start, "source": "start"}]
    positions.extend(mined_positions)

    all_policies: list[np.ndarray] = []
    all_visits: list[np.ndarray] = []
    checkpoint_payloads: list[dict[str, Any]] = []
    for step in ckpt_steps:
        evaluator = load_checkpoint_evaluator(run_dir, int(step), config)
        entry = {
            "step": int(step),
            "evaluator": {
                "name": evaluator.name,
                "kind": evaluator.kind,
                "detail": getattr(evaluator, "detail", None),
            },
            "positions": [],
        }
        for pos_idx, pos in enumerate(positions):
            state = qe.GameState.from_string(pos["position"])
            search = search_root(state, evaluator, config, seed=int(step) * 100_000 + pos_idx)
            all_policies.append(search.policy.astype(np.float32, copy=True))
            all_visits.append(search.visits.astype(np.float32, copy=True))
            entry["positions"].append(
                {
                    **pos,
                    "value": float(search.value),
                    "best_action": int(search.action),
                    "top_actions": search.top_actions,
                    "policy_index": len(all_policies) - 1,
                }
            )
        checkpoint_payloads.append(entry)

    out_dir = run_dir / "analysis" / "openings"
    payload = {
        "version": 1,
        "run_dir": str(run_dir),
        "created_at": time.time(),
        "positions": positions,
        "mining_notes": mining_notes,
        "checkpoints": checkpoint_payloads,
        "arrays": {
            "npz": str(out_dir / "opening_book.npz"),
            "policy_shape": [len(all_policies), config.rules.num_actions],
            "visits_shape": [len(all_visits), config.rules.num_actions],
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "opening_book.npz",
        policies=np.stack(all_policies).astype(np.float32)
        if all_policies
        else np.zeros((0, config.rules.num_actions), dtype=np.float32),
        visits=np.stack(all_visits).astype(np.float32)
        if all_visits
        else np.zeros((0, config.rules.num_actions), dtype=np.float32),
    )
    write_json_atomic(out_dir / "opening_book.json", payload)
    return payload


def mine_frequent_opening_positions(
    run_dir: Path,
    *,
    top_k: int = 8,
    ply_min: int = 2,
    ply_max: int = 8,
) -> tuple[list[dict[str, Any]], list[str]]:
    counter: Counter[str] = Counter()
    notes: list[str] = []
    shard_paths = list((Path(run_dir) / "ckpt").glob("*/buffer/*.npz"))
    if not shard_paths:
        return [], ["no replay buffer shards found"]

    for shard_path in shard_paths:
        try:
            with np.load(shard_path, allow_pickle=False) as data:
                key = first_present_key(
                    data.files,
                    ("position_strings", "states_text", "positions", "position"),
                )
                if key is None:
                    notes.append(f"{shard_path}: no position-string field")
                    continue
                raw_positions = data[key]
                for raw in raw_positions:
                    text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
                    if not text.startswith("Q1 "):
                        continue
                    ply = extract_ply(text)
                    if ply_min <= ply <= ply_max:
                        qe.GameState.from_string(text)
                        counter[text] += 1
        except Exception as exc:
            notes.append(f"{shard_path}: {exc}")

    positions = [
        {
            "id": f"frequent_ply_{extract_ply(position)}_{idx}",
            "position": position,
            "source": "buffer",
            "count": int(count),
            "ply": extract_ply(position),
        }
        for idx, (position, count) in enumerate(counter.most_common(top_k), start=1)
    ]
    if not positions and not notes:
        notes.append("no ply-2..8 position strings found in replay shards")
    return positions, notes


def first_present_key(keys: Iterable[str], candidates: Iterable[str]) -> str | None:
    key_set = set(keys)
    for candidate in candidates:
        if candidate in key_set:
            return candidate
    return None


def extract_ply(position: str) -> int:
    match = re.search(r"(?:^|\s)ply=(\d+)(?:\s|$)", position)
    if not match:
        return -1
    return int(match.group(1))


def parse_steps(text: str | None, run_dir: Path, fallback_step: int | None) -> list[int]:
    if text:
        return [int(part.strip()) for part in text.split(",") if part.strip()]
    if fallback_step is not None:
        return [int(fallback_step)]
    ckpt_root = Path(run_dir) / "ckpt"
    steps = sorted(int(path.name) for path in ckpt_root.iterdir() if path.name.isdigit()) if ckpt_root.exists() else []
    return steps


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Quoridor checkpoint analysis probes.")
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--steps", default=None, help="Comma-separated steps for opening dumps")
    parser.add_argument("--mode", choices=["probes", "openings", "all"], default="probes")
    parser.add_argument("--sims", type=int, default=None)
    parser.add_argument("--top-k-positions", type=int, default=8)
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.device:
        os.environ["QUIDOOR_ARENA_DEVICE"] = str(args.device)
    config_step = args.step
    if config_step is None and args.steps:
        parsed_steps = parse_steps(args.steps, args.run, None)
        config_step = parsed_steps[0] if parsed_steps else None
    config = load_config_for_cli(args.run, config_step) if config_step is not None else Config()
    if args.sims is not None:
        config.mcts.sims_per_move = int(args.sims)
        config.selfplay.leaf_batch_size = max(1, int(args.sims))

    outputs: dict[str, Any] = {}
    if args.mode in {"probes", "all"}:
        if args.step is None:
            raise SystemExit("--step is required for probe mode")
        outputs["probes"] = run_probe_suite(args.run, int(args.step), config)
    if args.mode in {"openings", "all"}:
        steps = parse_steps(args.steps, args.run, args.step)
        outputs["openings"] = dump_opening_book(
            args.run,
            steps,
            config,
            top_k_positions=int(args.top_k_positions),
        )
    print(json.dumps(outputs, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

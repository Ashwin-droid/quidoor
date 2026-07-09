from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

import numpy as np
import torch

import quoridor_engine as qe
from baselines import GreedyPlayer, MinimaxPlayer, RandomPlayer
from baselines.common import action_to_label

from .config import Config
from .model import Inferencer, load_model


BASELINE_ELO_ANCHORS: dict[str, float] = {
    "baseline:random": -800.0,
    "baseline:greedy": -400.0,
    "baseline:minimax-2": 0.0,
}


class PolicyValueEvaluator(Protocol):
    name: str
    kind: str

    def infer(
        self, planes: np.ndarray, masks: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return policy logits and values from the current-player perspective."""


@dataclass(slots=True)
class StubEvaluator:
    name: str = "stub_eval"
    kind: str = "stub"
    detail: str | None = None

    def infer(
        self, planes: np.ndarray, masks: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        del masks
        return qe.stub_eval(np.asarray(planes, dtype=np.float32))


@dataclass(slots=True)
class ModelEvaluator:
    name: str
    inferencer: Inferencer
    kind: str = "checkpoint"
    detail: str | None = None

    def infer(
        self, planes: np.ndarray, masks: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.inferencer.infer(planes, masks)


@dataclass(slots=True)
class RootSearchResult:
    action: int
    visits: np.ndarray
    policy: np.ndarray
    value: float
    top_actions: list[dict[str, Any]]


@dataclass(slots=True)
class MatchupResult:
    player_a: str
    player_b: str
    games: int
    wins_a: int
    wins_b: int
    draws: int
    player_a_as_p0: int
    player_a_as_p1: int

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


class ArenaAgent(Protocol):
    name: str

    def select_action(self, state: qe.GameState) -> int:
        ...

    def observe_action(self, action: int, state_after: qe.GameState) -> None:
        ...


class SearchAgent:
    def __init__(
        self,
        evaluator: PolicyValueEvaluator,
        config: Config,
        initial_state: qe.GameState,
        seed: int,
    ) -> None:
        self.name = evaluator.name
        self.evaluator = evaluator
        self.config = config
        self.engine_config = make_engine_config(config)
        self.rng = random.Random(seed)
        self._next_seed = seed
        self.mcts = qe.BatchedMcts(
            self.engine_config, initial_state.clone(), seed=int(self._bump_seed())
        )

    def select_action(self, state: qe.GameState) -> int:
        drive_mcts(self.mcts, self.evaluator, self.config)
        opening_moves = int(getattr(self.config.eval, "opening_temperature_moves", 0))
        temperature = float(getattr(self.config.mcts, "temperature", 1.0))
        if state.ply < opening_moves and temperature > 1.0e-6:
            visits = np.asarray(self.mcts.root_visits(), dtype=np.float64)
            sampled = sample_from_visits(visits, state, temperature, self.rng)
            if sampled is not None:
                return int(sampled)
        return int(self.mcts.best_action())

    def observe_action(self, action: int, state_after: qe.GameState) -> None:
        try:
            self.mcts.advance(int(action))
        except Exception:
            self.mcts = qe.BatchedMcts(
                self.engine_config,
                state_after.clone(),
                seed=int(self._bump_seed()),
            )

    def _bump_seed(self) -> int:
        self._next_seed = (self._next_seed + 0x9E3779B97F4A7C15) & ((1 << 63) - 1)
        return self._next_seed


class BaselineAgent:
    def __init__(self, name: str, seed: int) -> None:
        self.name = f"baseline:{name}"
        if name == "random":
            self.player = RandomPlayer(seed=seed)
        elif name == "greedy":
            self.player = GreedyPlayer()
        elif name in {"minimax", "minimax-2"}:
            self.player = MinimaxPlayer(depth=2)
            self.name = "baseline:minimax-2"
        else:
            raise ValueError(f"unknown baseline `{name}`")

    def select_action(self, state: qe.GameState) -> int:
        return int(self.player.select_action(state))

    def observe_action(self, action: int, state_after: qe.GameState) -> None:
        del action, state_after


def make_engine_config(config: Config) -> qe.EngineConfig:
    return qe.EngineConfig(**config.to_engine_kwargs())


def load_checkpoint_evaluator(
    run_dir: Path,
    ckpt_step: int,
    config: Config,
    device: torch.device | str | None = None,
) -> PolicyValueEvaluator:
    """Load a checkpoint evaluator, falling back to the Rust stub evaluator.

    This handles the early-run edge cases where the trainer has not produced a
    checkpoint yet, or where a checkpoint explicitly marks itself as stub-only.
    """

    run_dir = Path(run_dir)
    ckpt_dir = run_dir / "ckpt" / str(int(ckpt_step))
    uses_stub, reason = checkpoint_uses_stub(ckpt_dir)
    if uses_stub:
        return StubEvaluator(name=f"ckpt_{ckpt_step}:stub", detail=reason)

    device_obj = torch.device(device or default_device())
    try:
        model = load_model(ckpt_dir, config, device_obj)
        inferencer = Inferencer(model, config)
        return ModelEvaluator(
            name=f"ckpt_{ckpt_step}",
            inferencer=inferencer,
            detail=f"{ckpt_dir} on {device_obj}",
        )
    except Exception as exc:
        return StubEvaluator(name=f"ckpt_{ckpt_step}:stub", detail=f"load failed: {exc}")


def checkpoint_uses_stub(ckpt_dir: Path) -> tuple[bool, str]:
    state_path = Path(ckpt_dir) / "state.pt"
    if not state_path.exists():
        return True, f"missing checkpoint state at {state_path}"

    try:
        state = torch.load(state_path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older torch versions
        state = torch.load(state_path, map_location="cpu")
    except Exception as exc:
        return True, f"could not read checkpoint state: {exc}"

    if isinstance(state, dict):
        for key in ("uses_stub", "stub", "stub_eval"):
            if bool(state.get(key, False)):
                return True, f"checkpoint field {key}=true"
        if "model_state" in state:
            model_state = state.get("model_state")
            if model_state is None:
                return True, "checkpoint model_state is null"
            if isinstance(model_state, dict) and len(model_state) > 0:
                return False, "model_state present"
            return True, "checkpoint model_state is empty"
        if state and all(torch.is_tensor(value) for value in state.values()):
            return False, "raw model state dict present"
    return True, "checkpoint does not contain a recognizable model_state"


def default_device() -> str:
    requested = os.environ.get("QUIDOOR_DEVICE") or os.environ.get("QUIDOOR_ARENA_DEVICE")
    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def drive_mcts(
    mcts: qe.BatchedMcts,
    evaluator: PolicyValueEvaluator,
    config: Config,
    *,
    max_empty_polls: int = 10_000,
) -> None:
    batch_size = max(1, int(config.selfplay.leaf_batch_size))
    timeout_ms = max(0, int(config.selfplay.collect_timeout_ms))
    empty_polls = 0

    while not bool(mcts.ready()):
        planes, leaf_ids, masks = mcts.collect_leaves(batch_size, timeout_ms)
        batch = int(np.asarray(leaf_ids).shape[0])
        if batch == 0:
            empty_polls += 1
            if empty_polls > max_empty_polls:
                raise RuntimeError(f"MCTS stalled while serving {evaluator.name}")
            continue

        logits, values = evaluator.infer(
            np.asarray(planes, dtype=np.float32),
            np.asarray(masks).astype(np.bool_, copy=False),
        )
        mcts.submit(
            np.asarray(leaf_ids, dtype=np.int64),
            np.asarray(logits, dtype=np.float32),
            np.asarray(values, dtype=np.float32),
        )
        empty_polls = 0


def evaluate_state(
    state: qe.GameState,
    evaluator: PolicyValueEvaluator,
    config: Config,
) -> tuple[np.ndarray, float]:
    planes = np.asarray(state.encode(config.net.extra_planes), dtype=np.float32)[None, ...]
    masks = np.asarray(state.legal_mask()).astype(np.bool_, copy=False)[None, ...]
    logits, values = evaluator.infer(planes, masks)
    policy = masked_softmax(np.asarray(logits[0], dtype=np.float32), masks[0])
    return policy, float(values[0])


def net_eval_score(
    state: qe.GameState,
    evaluator: PolicyValueEvaluator,
    config: Config,
) -> float | None:
    """Model's continuous eval from player 0's perspective, when available.

    Returns None for stub evaluators and for checkpoints that predate the
    score head (whose head weights would be uninitialized noise).
    """

    inferencer = getattr(evaluator, "inferencer", None)
    if inferencer is None:
        return None
    if not getattr(getattr(inferencer, "model", None), "score_head_loaded", False):
        return None
    planes = np.asarray(state.encode(config.net.extra_planes), dtype=np.float32)[None, ...]
    masks = np.asarray(state.legal_mask()).astype(np.bool_, copy=False)[None, ...]
    _, _, scores = inferencer.infer_with_score(planes, masks)
    score = float(scores[0])
    return score if int(state.to_move) == 0 else -score


def search_root(
    state: qe.GameState,
    evaluator: PolicyValueEvaluator,
    config: Config,
    *,
    seed: int = 0,
) -> RootSearchResult:
    mcts = qe.BatchedMcts(make_engine_config(config), state.clone(), seed=int(seed))
    drive_mcts(mcts, evaluator, config)
    visits = np.asarray(mcts.root_visits(), dtype=np.float32)
    policy = normalize_policy(visits)
    action = int(mcts.best_action())
    if float(policy.sum()) <= 0.0:
        policy = np.zeros(config.rules.num_actions, dtype=np.float32)
        policy[action] = 1.0
    _, value = evaluate_state(state, evaluator, config)
    return RootSearchResult(
        action=action,
        visits=visits,
        policy=policy,
        value=value,
        top_actions=top_actions(policy, state, k=5),
    )


def normalize_policy(weights: np.ndarray) -> np.ndarray:
    arr = np.asarray(weights, dtype=np.float32)
    total = float(np.sum(arr))
    if total <= 0.0 or not math.isfinite(total):
        return np.zeros_like(arr, dtype=np.float32)
    return (arr / total).astype(np.float32, copy=False)


def masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    legal = np.asarray(mask, dtype=bool)
    out = np.zeros_like(logits, dtype=np.float64)
    if not np.any(legal):
        return out.astype(np.float32)
    legal_logits = logits[legal]
    finite = np.isfinite(legal_logits)
    if not np.any(finite):
        out[legal] = 1.0 / float(np.count_nonzero(legal))
        return out.astype(np.float32)
    max_logit = float(np.max(legal_logits[finite]))
    exp = np.zeros_like(legal_logits, dtype=np.float64)
    exp[finite] = np.exp(legal_logits[finite] - max_logit)
    denom = float(np.sum(exp))
    if denom <= 0.0 or not math.isfinite(denom):
        out[legal] = 1.0 / float(np.count_nonzero(legal))
    else:
        out[legal] = exp / denom
    return out.astype(np.float32)


def top_actions(policy: np.ndarray, state: qe.GameState, k: int = 5) -> list[dict[str, Any]]:
    legal = np.asarray(state.legal_mask()).astype(bool, copy=False)
    scores = np.asarray(policy, dtype=np.float64).copy()
    scores[~legal] = -1.0
    order = np.argsort(scores)[::-1]
    out: list[dict[str, Any]] = []
    for action in order:
        if len(out) >= k:
            break
        if not legal[action]:
            continue
        out.append(
            {
                "action": int(action),
                "label": action_to_label(int(action), state),
                "prob": float(max(scores[action], 0.0)),
            }
        )
    return out


def sample_from_visits(
    visits: np.ndarray,
    state: qe.GameState,
    temperature: float,
    rng: random.Random,
) -> int | None:
    legal = np.asarray(state.legal_mask()).astype(bool, copy=False)
    weights = np.asarray(visits, dtype=np.float64).copy()
    weights[~legal] = 0.0
    if temperature <= 1.0e-6:
        return None
    positive = weights > 0.0
    if not np.any(positive):
        return None
    weights[positive] = np.power(weights[positive], 1.0 / temperature)
    total = float(np.sum(weights))
    if total <= 0.0 or not math.isfinite(total):
        return None
    sample = rng.random() * total
    for action, weight in enumerate(weights):
        sample -= float(weight)
        if sample <= 0.0:
            return int(action)
    return int(np.argmax(weights))


def run_arena(run_dir: Path, ckpt_step: int, config: Config) -> dict[str, Any]:
    run_dir = Path(run_dir)
    started_at = time.time()
    current = load_checkpoint_evaluator(run_dir, ckpt_step, config)
    summary: dict[str, Any] = {
        "version": 1,
        "step": int(ckpt_step),
        "current_player": current.name,
        "current_kind": current.kind,
        "current_detail": getattr(current, "detail", None),
        "started_at": started_at,
        "finished_at": None,
        "duration_sec": None,
        "previous": None,
        "baselines": {},
        "matchups": [],
    }

    seed_base = int(ckpt_step) * 1_000_003 + 17
    previous_step = find_previous_checkpoint_step(run_dir, ckpt_step)
    previous_games = int(config.eval.games_vs_previous)
    if previous_games > 0 and previous_step is not None:
        previous = load_checkpoint_evaluator(run_dir, previous_step, config)
        result = run_matchup_series(
            current.name,
            lambda state, seed: SearchAgent(current, config, state, seed),
            previous.name,
            lambda state, seed: SearchAgent(previous, config, state, seed),
            previous_games,
            config,
            seed=seed_base + 101,
        )
        summary["previous"] = {
            "step": int(previous_step),
            "opponent": previous.name,
            "opponent_kind": previous.kind,
            "result": result.asdict(),
        }
        summary["matchups"].append(result.asdict())
    elif previous_games > 0:
        summary["previous"] = {
            "skipped": True,
            "reason": "no previous checkpoint",
        }

    baseline_games = int(config.eval.games_vs_baseline)
    if baseline_games > 0:
        for offset, baseline in enumerate(("random", "greedy", "minimax-2")):
            baseline_player_name = f"baseline:{baseline}"
            result = run_matchup_series(
                current.name,
                lambda state, seed, ev=current: SearchAgent(ev, config, state, seed),
                baseline_player_name,
                lambda state, seed, b=baseline: BaselineAgent(b, seed),
                baseline_games,
                config,
                seed=seed_base + 1_000 + offset * 10_000,
            )
            summary["baselines"][baseline] = result.asdict()
            summary["matchups"].append(result.asdict())

    summary["finished_at"] = time.time()
    summary["duration_sec"] = float(summary["finished_at"] - started_at)
    elo_payload = update_elo_file(run_dir, summary)
    summary["elo"] = {
        "path": str(run_dir / "elo.json"),
        "players": elo_payload.get("players", {}),
    }
    return summary


def run_matchup_series(
    player_a: str,
    make_a: Callable[[qe.GameState, int], ArenaAgent],
    player_b: str,
    make_b: Callable[[qe.GameState, int], ArenaAgent],
    games: int,
    config: Config,
    *,
    seed: int,
) -> MatchupResult:
    games = max(0, int(games))
    wins_a = 0
    wins_b = 0
    draws = 0
    a_as_p0 = 0
    a_as_p1 = 0
    color_schedule = balanced_color_schedule(games)
    for game_idx, a_color in enumerate(color_schedule):
        game_seed = seed + game_idx * 97
        state = qe.GameState()
        agent_a = make_a(state, game_seed + 11)
        agent_b = make_b(state, game_seed + 53)
        if a_color == 0:
            p0, p1 = agent_a, agent_b
            a_as_p0 += 1
        else:
            p0, p1 = agent_b, agent_a
            a_as_p1 += 1
        outcome_p0 = play_one_game(p0, p1, config)
        if outcome_p0 == 0:
            draws += 1
        else:
            a_won = outcome_p0 == 1 if a_color == 0 else outcome_p0 == -1
            if a_won:
                wins_a += 1
            else:
                wins_b += 1

    return MatchupResult(
        player_a=player_a,
        player_b=player_b,
        games=games,
        wins_a=wins_a,
        wins_b=wins_b,
        draws=draws,
        player_a_as_p0=a_as_p0,
        player_a_as_p1=a_as_p1,
    )


def balanced_color_schedule(games: int) -> list[int]:
    first_half = (games + 1) // 2
    return [0] * first_half + [1] * (games - first_half)


def play_one_game(p0: ArenaAgent, p1: ArenaAgent, config: Config) -> int:
    state = qe.GameState()
    agents = [p0, p1]
    ply_cap = int(config.rules.ply_cap)
    while True:
        winner = state.winner()
        if winner is not None:
            return 1 if int(winner) == 0 else -1
        if int(state.ply) >= ply_cap:
            return 0
        actor = agents[int(state.to_move)]
        action = int(actor.select_action(state))
        state.apply(action)
        for agent in agents:
            agent.observe_action(action, state)


def find_previous_checkpoint_step(run_dir: Path, ckpt_step: int) -> int | None:
    ckpt_root = Path(run_dir) / "ckpt"
    if not ckpt_root.exists():
        return None
    steps: list[int] = []
    for child in ckpt_root.iterdir():
        if child.is_dir() and child.name.isdigit():
            step = int(child.name)
            if step < int(ckpt_step):
                steps.append(step)
    return max(steps) if steps else None


def update_elo_file(run_dir: Path, arena_summary: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    elo_path = run_dir / "elo.json"
    existing: dict[str, Any] = {}
    if elo_path.exists():
        with elo_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)

    history = list(existing.get("arena_history", existing.get("history", [])))
    history.append(arena_summary_without_elo(arena_summary))
    matchups = collect_matchups(history)
    players = fit_elo_ladder(matchups, anchors=BASELINE_ELO_ANCHORS)
    ladder = [
        {"name": name, "elo": float(elo)}
        for name, elo in sorted(players.items(), key=lambda item: (-item[1], item[0]))
    ]
    payload = {
        "version": 1,
        "updated_at": time.time(),
        "players": {name: {"elo": float(elo)} for name, elo in sorted(players.items())},
        "ladder": ladder,
        "arena_history": history,
    }
    write_json_atomic(elo_path, payload)
    return payload


def arena_summary_without_elo(summary: dict[str, Any]) -> dict[str, Any]:
    out = dict(summary)
    out.pop("elo", None)
    return out


def collect_matchups(history: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    matchups: list[dict[str, Any]] = []
    for entry in history:
        for matchup in entry.get("matchups", []):
            if matchup.get("games", 0) > 0:
                matchups.append(dict(matchup))
    return matchups


def fit_elo_ladder(
    matchups: Iterable[dict[str, Any] | MatchupResult],
    anchors: dict[str, float] | None = None,
) -> dict[str, float]:
    """Fit a Bradley-Terry style Elo ladder from pairwise game results.

    The implementation converts observed scores into Elo-difference equations
    with Jeffreys smoothing, then solves the weighted least-squares system. That
    gives deterministic, small-data-friendly logistic Elo estimates without
    requiring scipy.
    """

    anchors = dict(anchors or {})
    rows: list[tuple[str, str, float, float]] = []
    players: set[str] = set(anchors)
    for raw in matchups:
        matchup = raw.asdict() if isinstance(raw, MatchupResult) else dict(raw)
        games = int(matchup.get("games", 0))
        if games <= 0:
            continue
        a = str(matchup["player_a"])
        b = str(matchup["player_b"])
        wins_a = float(matchup.get("wins_a", 0))
        wins_b = float(matchup.get("wins_b", 0))
        draws = float(matchup.get("draws", max(0.0, games - wins_a - wins_b)))
        score_a = wins_a + 0.5 * draws
        p = (score_a + 0.5) / (games + 1.0)
        p = min(max(p, 1.0e-6), 1.0 - 1.0e-6)
        elo_diff = 400.0 * math.log10(p / (1.0 - p))
        rows.append((a, b, elo_diff, float(games)))
        players.add(a)
        players.add(b)

    unknown = sorted(player for player in players if player not in anchors)
    if not unknown:
        return {name: float(value) for name, value in sorted(anchors.items())}

    index = {player: idx for idx, player in enumerate(unknown)}
    matrix: list[list[float]] = []
    target: list[float] = []
    for a, b, elo_diff, games in rows:
        coeff = [0.0] * len(unknown)
        rhs = elo_diff
        if a in anchors:
            rhs -= anchors[a]
        else:
            coeff[index[a]] += 1.0
        if b in anchors:
            rhs += anchors[b]
        else:
            coeff[index[b]] -= 1.0
        if any(abs(value) > 0.0 for value in coeff):
            weight = math.sqrt(max(games, 1.0))
            matrix.append([weight * value for value in coeff])
            target.append(weight * rhs)

    if not matrix or not anchors:
        matrix.append([1.0] * len(unknown))
        target.append(0.0)

    solution, *_ = np.linalg.lstsq(
        np.asarray(matrix, dtype=np.float64),
        np.asarray(target, dtype=np.float64),
        rcond=None,
    )
    elos = {name: float(value) for name, value in anchors.items()}
    for player, idx in index.items():
        elos[player] = float(solution[idx])
    return {name: round(value, 3) for name, value in sorted(elos.items())}


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def resolve_step(run_dir: Path | None, requested: int | None) -> int:
    """Resolve a checkpoint step for CLI tools, defaulting to the latest one."""

    if requested is not None:
        return int(requested)
    if run_dir is None:
        return 0
    run_dir = Path(run_dir)
    latest = run_dir / "ckpt" / "latest"
    if latest.exists():
        try:
            target = latest.resolve()
            if target.name.isdigit():
                return int(target.name)
        except Exception:
            try:
                text = latest.read_text(encoding="utf-8").strip()
                if text.isdigit():
                    return int(text)
            except Exception:
                pass
    ckpt_root = run_dir / "ckpt"
    if ckpt_root.exists():
        steps = sorted(int(path.name) for path in ckpt_root.iterdir() if path.name.isdigit())
        if steps:
            return steps[-1]
    return 0


def load_config_for_cli(run_dir: Path, ckpt_step: int) -> Config:
    config = Config()
    state_path = Path(run_dir) / "ckpt" / str(int(ckpt_step)) / "state.pt"
    if not state_path.exists():
        return config
    try:
        state = torch.load(state_path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover
        state = torch.load(state_path, map_location="cpu")
    except Exception:
        return config
    if isinstance(state, dict) and isinstance(state.get("config"), dict):
        apply_config_dict(config, state["config"])
    return config


def apply_config_dict(config: Config, data: dict[str, Any]) -> None:
    for key, value in data.items():
        if not hasattr(config, key):
            continue
        current = getattr(config, key)
        if is_dataclass(current) and isinstance(value, dict):
            for child_key, child_value in value.items():
                if hasattr(current, child_key):
                    setattr(current, child_key, child_value)
        else:
            setattr(config, key, value)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Quoridor checkpoint arena evaluation.")
    parser.add_argument("--run", required=True, type=Path, help="Run directory, e.g. runs/my-run")
    parser.add_argument("--step", required=True, type=int, help="Checkpoint step to evaluate")
    parser.add_argument("--sims", type=int, default=None, help="Override MCTS sims per move")
    parser.add_argument("--games-vs-previous", type=int, default=None)
    parser.add_argument("--games-vs-baseline", type=int, default=None)
    parser.add_argument("--ply-cap", type=int, default=None)
    parser.add_argument("--device", default=None, help="Set QUIDOOR_ARENA_DEVICE for this run")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.device:
        os.environ["QUIDOOR_ARENA_DEVICE"] = str(args.device)
    config = load_config_for_cli(args.run, args.step)
    if args.sims is not None:
        config.mcts.sims_per_move = int(args.sims)
        config.selfplay.leaf_batch_size = max(config.selfplay.leaf_batch_size, int(args.sims))
    if args.games_vs_previous is not None:
        config.eval.games_vs_previous = int(args.games_vs_previous)
    if args.games_vs_baseline is not None:
        config.eval.games_vs_baseline = int(args.games_vs_baseline)
    if args.ply_cap is not None:
        config.rules.ply_cap = int(args.ply_cap)

    summary = run_arena(args.run, args.step, config)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

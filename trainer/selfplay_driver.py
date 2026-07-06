from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np

from .config import Config
from .model import Inferencer
from .replay_buffer import ReplayBuffer


class ShutdownLike(Protocol):
    shutdown_state: str
    fast_abort_requested: bool
    draining: bool

    def poll_stop_file(self) -> None:
        ...


@dataclass(slots=True)
class SelfPlayMetrics:
    games: int = 0
    positions: int = 0
    draws: int = 0
    leaf_evals: int = 0
    leaf_batches: int = 0
    empty_polls: int = 0
    batch_capacity: int = 0
    game_lengths: list[int] = field(default_factory=list)
    policy_entropy_sum: float = 0.0
    policy_entropy_count: int = 0
    elapsed_seconds: float = 0.0

    @property
    def games_per_hour(self) -> float:
        return self.games / max(self.elapsed_seconds, 1.0e-9) * 3600.0

    @property
    def positions_per_sec(self) -> float:
        return self.positions / max(self.elapsed_seconds, 1.0e-9)

    @property
    def leaf_batch_fill_ratio(self) -> float:
        if self.batch_capacity == 0:
            return 0.0
        return self.leaf_evals / self.batch_capacity

    @property
    def draw_rate(self) -> float:
        return self.draws / self.games if self.games else 0.0

    @property
    def avg_game_length(self) -> float:
        return float(np.mean(self.game_lengths)) if self.game_lengths else 0.0

    @property
    def avg_root_policy_entropy(self) -> float:
        if self.policy_entropy_count == 0:
            return 0.0
        return self.policy_entropy_sum / self.policy_entropy_count

    @property
    def mcts_sims_per_sec(self) -> float:
        return self.leaf_evals / max(self.elapsed_seconds, 1.0e-9)

    def as_dict(self) -> dict[str, float | int]:
        return {
            "games": self.games,
            "positions": self.positions,
            "draws": self.draws,
            "leaf_evals": self.leaf_evals,
            "leaf_batches": self.leaf_batches,
            "empty_polls": self.empty_polls,
            "games_per_hour": self.games_per_hour,
            "positions_per_sec": self.positions_per_sec,
            "leaf_batch_fill_ratio": self.leaf_batch_fill_ratio,
            "draw_rate": self.draw_rate,
            "avg_game_length": self.avg_game_length,
            "avg_root_policy_entropy": self.avg_root_policy_entropy,
            "mcts_sims_per_sec": self.mcts_sims_per_sec,
            "elapsed_seconds": self.elapsed_seconds,
        }


Evaluator = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
TickCallback = Callable[[SelfPlayMetrics], None]


def inferencer_evaluator(inferencer: Inferencer) -> Evaluator:
    def evaluate(planes: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return inferencer.infer(planes, masks)

    return evaluate


def stub_evaluator() -> Evaluator:
    import quoridor_engine as qe

    def evaluate(planes: np.ndarray, _masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        logits, values = qe.stub_eval(planes)
        return np.asarray(logits, dtype=np.float32), np.asarray(values, dtype=np.float32)

    return evaluate


def run_selfplay_phase(
    *,
    config: Config,
    buffer: ReplayBuffer,
    evaluator: Evaluator,
    num_games: int | None = None,
    seed: int = 0,
    shutdown: ShutdownLike | None = None,
    tick_callback: TickCallback | None = None,
    logger: Any | None = None,
) -> SelfPlayMetrics:
    import quoridor_engine as qe

    target_games = int(num_games if num_games is not None else config.selfplay.phase_games)
    max_batch = int(config.selfplay.leaf_batch_size)
    timeout_ms = int(config.selfplay.collect_timeout_ms)
    concurrent = max(1, int(config.selfplay.concurrent_games))
    engine_cfg = qe.EngineConfig(**config.to_engine_kwargs())
    metrics = SelfPlayMetrics()
    started = time.perf_counter()

    while metrics.games < target_games:
        if shutdown is not None:
            shutdown.poll_stop_file()
            if shutdown.fast_abort_requested:
                break
            if shutdown.draining:
                break

        chunk_games = min(concurrent, target_games - metrics.games)
        if chunk_games <= 0:
            break
        if logger is not None:
            logger.event(
                "selfplay_chunk_started",
                games=chunk_games,
                completed=metrics.games,
                target=target_games,
            )
        manager = qe.SelfPlayManager(
            engine_cfg, num_games=chunk_games, seed=int(seed + metrics.games)
        )
        stop_spawning_requested = False

        while not manager.done():
            if shutdown is not None:
                shutdown.poll_stop_file()
                if shutdown.fast_abort_requested:
                    metrics.elapsed_seconds = time.perf_counter() - started
                    return metrics
                if shutdown.draining and not stop_spawning_requested:
                    stop_spawning = getattr(manager, "stop_spawning", None)
                    if callable(stop_spawning):
                        stop_spawning()
                    stop_spawning_requested = True

            planes, leaf_ids, masks = manager.collect_leaves(
                max_batch=max_batch, timeout_ms=timeout_ms
            )
            batch = int(planes.shape[0])
            if batch:
                policy_logits, values = evaluator(planes, masks)
                manager.submit(
                    leaf_ids,
                    np.asarray(policy_logits, dtype=np.float32),
                    np.asarray(values, dtype=np.float32),
                )
                metrics.leaf_evals += batch
                metrics.leaf_batches += 1
                metrics.batch_capacity += max_batch
            else:
                metrics.empty_polls += 1

            _drain_finished(manager, buffer, metrics)
            metrics.elapsed_seconds = time.perf_counter() - started
            if tick_callback is not None:
                tick_callback(metrics)

        _drain_finished(manager, buffer, metrics)
        metrics.elapsed_seconds = time.perf_counter() - started
        if tick_callback is not None:
            tick_callback(metrics)

        if shutdown is not None:
            shutdown.poll_stop_file()
            if shutdown.fast_abort_requested or shutdown.draining:
                break

    metrics.elapsed_seconds = time.perf_counter() - started
    if logger is not None:
        logger.event("selfplay_phase_finished", **metrics.as_dict())
    return metrics


def _drain_finished(manager: Any, buffer: ReplayBuffer, metrics: SelfPlayMetrics) -> None:
    records = manager.drain_finished_games()
    if not records:
        return
    for record in records:
        states, policies, masks, to_move, outcome, position_strings = (
            _unpack_finished_record(record)
        )
        states_np = np.asarray(states, dtype=np.float32)
        policies_np = np.asarray(policies, dtype=np.float32)
        masks_np = np.asarray(masks).astype(np.bool_, copy=False)
        to_move_np = np.asarray(to_move, dtype=np.uint8)
        buffer.add_finished_game(
            states_np,
            policies_np,
            masks_np,
            to_move_np,
            outcome,
            position_strings=position_strings,
        )
        length = int(states_np.shape[0])
        metrics.games += 1
        metrics.positions += length
        metrics.draws += int(outcome == 0)
        metrics.game_lengths.append(length)
        entropy = _policy_entropy(policies_np)
        metrics.policy_entropy_sum += float(entropy.sum())
        metrics.policy_entropy_count += int(entropy.shape[0])


def _unpack_finished_record(
    record: Any,
) -> tuple[Any, Any, Any, Any, Any, Any | None]:
    if isinstance(record, dict):
        return (
            record["states"],
            record["policies"],
            record["masks"],
            record["to_move"],
            record["outcome"],
            record.get("position_strings"),
        )

    position_strings = getattr(record, "position_strings", None)
    items = tuple(record)
    if len(items) == 6:
        states, policies, masks, to_move, outcome, tuple_positions = items
        if position_strings is None:
            position_strings = tuple_positions
    elif len(items) == 5:
        states, policies, masks, to_move, outcome = items
    else:
        raise ValueError(f"finished-game records must have 5 or 6 fields, got {len(items)}")
    return states, policies, masks, to_move, outcome, position_strings


def _policy_entropy(policies: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(policies, dtype=np.float32), 1.0e-12, 1.0)
    entropy = -(clipped * np.log(clipped)).sum(axis=1)
    return entropy.astype(np.float64, copy=False)

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import Config

BOARD = 9
NUM_ACTIONS = 140
BASE_PLANES = 6
EXTRA_PLANES = 8
WALL_ANCHORS = 64
ANCHOR_WIDTH = 8
POSITION_STRING_DTYPE = np.dtype("<U512")

PAWN_MIRROR = np.array([0, 1, 3, 2, 4, 5, 7, 6, 9, 8, 11, 10], dtype=np.int64)


def _build_mirror_actions() -> np.ndarray:
    mapping = np.empty(NUM_ACTIONS, dtype=np.int64)
    mapping[:12] = PAWN_MIRROR
    for anchor in range(WALL_ANCHORS):
        row = anchor // ANCHOR_WIDTH
        col = anchor % ANCHOR_WIDTH
        mirrored_anchor = row * ANCHOR_WIDTH + (ANCHOR_WIDTH - 1 - col)
        mapping[12 + anchor] = 12 + mirrored_anchor
        mapping[76 + anchor] = 76 + mirrored_anchor
    return mapping


MIRROR_ACTIONS = _build_mirror_actions()


def eval_score_targets(
    z: np.ndarray | Iterable[float],
    position_strings: np.ndarray | Iterable[str] | None = None,
) -> np.ndarray:
    """Continuous eval targets in moves-ahead units, side-to-move perspective.

    Positions where the race is provably decided get the exact mate score
    (+/-(40 - distance), always outside +/-20) regardless of how the game
    actually ended. Everything else blends the shortest-path heuristic with
    the game outcome half-and-half, so positions from lost games are pulled
    negative and the network is penalised for claiming wins (or mates) for
    the side that went on to lose. Both components are bounded so undecided
    targets stay inside the +/-20 non-mate region. Mirroring does not affect
    targets: scores are left-right symmetric.
    """

    from baselines.common import (
        EVAL_SCORE_CLAMP as _clamp,
        forced_win,
        heuristic_score_moves,
        mate_score,
    )

    z_np = np.asarray(z, dtype=np.float32)
    targets = ((0.5 * _clamp) * z_np).astype(np.float32)
    if position_strings is None:
        return targets

    import quoridor_engine as qe

    out = targets.copy()
    for idx, text in enumerate(np.asarray(position_strings).reshape(-1)):
        text = str(text)
        if not text:
            continue
        try:
            state = qe.GameState.from_string(text)
        except Exception:
            continue
        mover = int(state.to_move)
        mate = forced_win(state)
        if mate is not None:
            winner, distance = mate
            score = mate_score(distance)
            out[idx] = score if winner == mover else -score
        else:
            heuristic = float(
                np.clip(heuristic_score_moves(state, mover), -_clamp, _clamp)
            )
            out[idx] = 0.5 * heuristic + (0.5 * _clamp) * float(z_np[idx])
    return out.astype(np.float32)


def mirror_policy(policy: np.ndarray) -> np.ndarray:
    arr = np.asarray(policy)
    if arr.shape[-1] != NUM_ACTIONS:
        raise ValueError(f"policy last dimension must be {NUM_ACTIONS}, got {arr.shape}")
    out = np.empty_like(arr)
    out[..., MIRROR_ACTIONS] = arr
    return out


def mirror_planes(planes: np.ndarray) -> np.ndarray:
    arr = np.asarray(planes)
    single = arr.ndim == 3
    if single:
        arr = arr[None, ...]
    if arr.ndim != 4 or arr.shape[-2:] != (BOARD, BOARD):
        raise ValueError(f"planes must have shape [C,9,9] or [B,C,9,9], got {planes.shape}")

    out = np.empty_like(arr)
    for plane in range(arr.shape[1]):
        if plane == 3:
            out[:, plane] = 0
            out[:, plane, :, : BOARD - 1] = arr[:, plane, :, : BOARD - 1][
                :, :, ::-1
            ]
        else:
            out[:, plane] = arr[:, plane, :, ::-1]
    return out[0] if single else out


def mirror_sample(
    planes: np.ndarray, policy: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return mirror_planes(planes), mirror_policy(policy), mirror_policy(mask)


class ReplayBuffer:
    """Ring buffer of AlphaZero training positions with shard persistence."""

    def __init__(self, config: Config, seed: int | None = None) -> None:
        self.config = config
        self.capacity = int(config.training.replay_buffer_size)
        if self.capacity <= 0:
            raise ValueError("replay_buffer_size must be positive")
        self.board = config.rules.board
        self.num_actions = config.rules.num_actions
        self.num_planes = EXTRA_PLANES if config.net.extra_planes else BASE_PLANES
        if self.board != BOARD or self.num_actions != NUM_ACTIONS:
            raise ValueError("ReplayBuffer currently supports the fixed 9x9/140 action game")

        self.planes = np.zeros(
            (self.capacity, self.num_planes, self.board, self.board), dtype=np.float32
        )
        self.policies = np.zeros((self.capacity, self.num_actions), dtype=np.float32)
        self.masks = np.zeros((self.capacity, self.num_actions), dtype=np.bool_)
        self.z = np.zeros((self.capacity,), dtype=np.float32)
        self.eval_score = np.zeros((self.capacity,), dtype=np.float32)
        self.position_strings = np.full(
            (self.capacity,), "", dtype=POSITION_STRING_DTYPE
        )

        self.size = 0
        self.write_pos = 0
        self.total_written = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        planes: np.ndarray,
        policies: np.ndarray,
        masks: np.ndarray,
        z: np.ndarray | Iterable[float] | float,
        position_strings: np.ndarray | Iterable[str] | None = None,
    ) -> None:
        planes_np, policies_np, masks_np, z_np = self._normalize_batch(
            planes, policies, masks, z
        )
        original_count = planes_np.shape[0]
        if original_count == 0:
            return
        position_strings_np = self._normalize_position_strings(
            position_strings, original_count
        )
        eval_score_np = eval_score_targets(z_np, position_strings_np)

        mirror_probability = float(self.config.training.mirror_probability)
        if mirror_probability > 0:
            do_mirror = self.rng.random(original_count) < mirror_probability
            if np.any(do_mirror):
                planes_np = planes_np.copy()
                policies_np = policies_np.copy()
                masks_np = masks_np.copy()
                planes_np[do_mirror] = mirror_planes(planes_np[do_mirror])
                policies_np[do_mirror] = mirror_policy(policies_np[do_mirror])
                masks_np[do_mirror] = mirror_policy(masks_np[do_mirror])

        self._write_batch(
            planes_np,
            policies_np,
            masks_np,
            z_np,
            position_strings_np,
            total_increment=original_count,
            eval_score=eval_score_np,
        )

    def add_finished_game(
        self,
        states: np.ndarray,
        policies: np.ndarray,
        masks: np.ndarray,
        to_move: np.ndarray,
        outcome: int | float,
        position_strings: np.ndarray | Iterable[str] | None = None,
    ) -> None:
        to_move_np = np.asarray(to_move, dtype=np.uint8)
        outcome_f = float(outcome)
        z = np.where(to_move_np == 0, outcome_f, -outcome_f).astype(np.float32)
        self.add(states, policies, masks, z, position_strings=position_strings)

    def add_finished_games(self, records: Iterable[Any]) -> None:
        for record in records:
            if isinstance(record, dict):
                self.add_finished_game(
                    record["states"],
                    record["policies"],
                    record["masks"],
                    record["to_move"],
                    record["outcome"],
                    position_strings=record.get("position_strings"),
                )
            else:
                position_strings = getattr(record, "position_strings", None)
                items = tuple(record)
                if len(items) == 6:
                    states, policies, masks, to_move, outcome, tuple_positions = items
                    if position_strings is None:
                        position_strings = tuple_positions
                elif len(items) == 5:
                    states, policies, masks, to_move, outcome = items
                else:
                    raise ValueError(
                        f"finished-game records must have 5 or 6 fields, got {len(items)}"
                    )
                self.add_finished_game(
                    states,
                    policies,
                    masks,
                    to_move,
                    outcome,
                    position_strings=position_strings,
                )

    def sample(self, batch_size: int | None = None) -> dict[str, np.ndarray]:
        if self.size == 0:
            raise ValueError("cannot sample from an empty replay buffer")
        size = int(batch_size or self.config.training.batch_size)
        if size <= 0:
            raise ValueError("batch_size must be positive")
        indices = self.rng.integers(0, self.size, size=size)
        return {
            "planes": self.planes[indices].copy(),
            "policy": self.policies[indices].copy(),
            "mask": self.masks[indices].copy(),
            "z": self.z[indices].copy(),
            "eval_score": self.eval_score[indices].copy(),
        }

    sample_batch = sample

    def as_arrays(self) -> dict[str, np.ndarray]:
        indices = self._logical_indices()
        return {
            "planes": self.planes[indices].copy(),
            "policy": self.policies[indices].copy(),
            "mask": self.masks[indices].copy(),
            "z": self.z[indices].copy(),
            "eval_score": self.eval_score[indices].copy(),
            "position_strings": self.position_strings[indices].copy(),
        }

    def stats(self) -> dict[str, int]:
        return {
            "capacity": self.capacity,
            "size": self.size,
            "write_pos": self.write_pos,
            "total_written": self.total_written,
        }

    def save_to(self, path: Path, shard_size: int | None = None) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        shard_size = int(shard_size or self.config.training.replay_shard_size)
        if shard_size <= 0:
            raise ValueError("shard_size must be positive")

        existing = _read_manifest_if_present(path / "manifest.json")
        generation = int(existing.get("generation", -1)) + 1 if existing else 0

        indices = self._logical_indices()
        shards: list[dict[str, Any]] = []
        for shard_idx, start in enumerate(range(0, self.size, shard_size)):
            stop = min(start + shard_size, self.size)
            chunk_indices = indices[start:stop]
            filename = f"shard_{generation:06d}_{shard_idx:04d}.npz"
            final_path = path / filename
            tmp_path = path / f".{filename}.tmp"
            with tmp_path.open("wb") as handle:
                np.savez(
                    handle,
                    planes=self.planes[chunk_indices],
                    policy=self.policies[chunk_indices],
                    mask=self.masks[chunk_indices],
                    z=self.z[chunk_indices],
                    eval_score=self.eval_score[chunk_indices],
                    position_strings=self.position_strings[chunk_indices],
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, final_path)
            _fsync_dir(path)
            shards.append({"file": filename, "count": int(stop - start)})

        manifest = {
            "version": 1,
            "generation": generation,
            "capacity": self.capacity,
            "size": self.size,
            "total_written": self.total_written,
            "board": self.board,
            "num_actions": self.num_actions,
            "num_planes": self.num_planes,
            "shard_size": shard_size,
            "position_string_field": "position_strings",
            "shards": shards,
        }
        _write_json_atomic(path / "manifest.json", manifest)

    @classmethod
    def load_from(cls, path: Path, config: Config, seed: int | None = None) -> "ReplayBuffer":
        path = Path(path)
        with (path / "manifest.json").open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        buffer = cls(config, seed=seed)
        if int(manifest.get("board", BOARD)) != buffer.board:
            raise ValueError("buffer board size is incompatible with config")
        if int(manifest.get("num_actions", NUM_ACTIONS)) != buffer.num_actions:
            raise ValueError("buffer action size is incompatible with config")
        if int(manifest.get("num_planes", buffer.num_planes)) != buffer.num_planes:
            raise ValueError("buffer plane count is incompatible with config")

        shards = list(manifest.get("shards", []))
        loaded = 0
        for shard_idx, shard in enumerate(shards):
            is_final = shard_idx == len(shards) - 1
            try:
                planes, policies, masks, z, eval_score, position_strings = _load_shard(
                    path / shard["file"]
                )
            except Exception:
                if is_final:
                    break
                raise
            expected = int(shard.get("count", planes.shape[0]))
            if planes.shape[0] != expected:
                if is_final:
                    break
                raise ValueError(
                    f"shard {shard['file']} count mismatch: {planes.shape[0]} != {expected}"
                )
            if eval_score is None:
                # Shard predates the eval-score field: rebuild targets from
                # the stored outcomes and position strings.
                eval_score = eval_score_targets(z, position_strings)
            buffer._write_batch(
                planes,
                policies,
                masks,
                z,
                position_strings,
                total_increment=0,
                eval_score=eval_score,
            )
            loaded += planes.shape[0]

        buffer.total_written = loaded
        return buffer

    def _normalize_batch(
        self,
        planes: np.ndarray,
        policies: np.ndarray,
        masks: np.ndarray,
        z: np.ndarray | Iterable[float] | float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        planes_np = np.asarray(planes, dtype=np.float32)
        policies_np = np.asarray(policies, dtype=np.float32)
        masks_np = np.asarray(masks).astype(np.bool_, copy=False)
        z_np = np.asarray(z, dtype=np.float32)

        if planes_np.ndim == 3:
            planes_np = planes_np[None, ...]
        if policies_np.ndim == 1:
            policies_np = policies_np[None, ...]
        if masks_np.ndim == 1:
            masks_np = masks_np[None, ...]
        if z_np.ndim == 0:
            z_np = z_np[None]

        expected_planes = (self.num_planes, self.board, self.board)
        if planes_np.ndim != 4 or planes_np.shape[1:] != expected_planes:
            raise ValueError(
                f"planes must have shape [B,{self.num_planes},{self.board},{self.board}], "
                f"got {planes_np.shape}"
            )
        if policies_np.shape != (planes_np.shape[0], self.num_actions):
            raise ValueError(
                f"policies must have shape [B,{self.num_actions}], got {policies_np.shape}"
            )
        if masks_np.shape != (planes_np.shape[0], self.num_actions):
            raise ValueError(
                f"masks must have shape [B,{self.num_actions}], got {masks_np.shape}"
            )
        if z_np.shape != (planes_np.shape[0],):
            raise ValueError(f"z must have shape [B], got {z_np.shape}")
        return planes_np, policies_np, masks_np, z_np

    def _normalize_position_strings(
        self,
        position_strings: np.ndarray | Iterable[str] | None,
        count: int,
    ) -> np.ndarray:
        if position_strings is None:
            return np.full((count,), "", dtype=POSITION_STRING_DTYPE)
        arr = np.asarray(position_strings, dtype=POSITION_STRING_DTYPE)
        if arr.ndim == 0:
            arr = arr[None]
        if arr.shape != (count,):
            raise ValueError(f"position_strings must have shape [B], got {arr.shape}")
        return arr

    def _write_batch(
        self,
        planes: np.ndarray,
        policies: np.ndarray,
        masks: np.ndarray,
        z: np.ndarray,
        position_strings: np.ndarray | None,
        total_increment: int,
        eval_score: np.ndarray | None = None,
    ) -> None:
        count = planes.shape[0]
        if count == 0:
            return
        if position_strings is None:
            position_strings = np.full((count,), "", dtype=POSITION_STRING_DTYPE)
        else:
            position_strings = np.asarray(position_strings, dtype=POSITION_STRING_DTYPE)
        if eval_score is None:
            eval_score = eval_score_targets(z, position_strings)
        else:
            eval_score = np.asarray(eval_score, dtype=np.float32)
        if count > self.capacity:
            planes = planes[-self.capacity :]
            policies = policies[-self.capacity :]
            masks = masks[-self.capacity :]
            z = z[-self.capacity :]
            eval_score = eval_score[-self.capacity :]
            position_strings = position_strings[-self.capacity :]
            count = self.capacity

        start = 0
        while start < count:
            writable = min(count - start, self.capacity - self.write_pos)
            end = start + writable
            target = slice(self.write_pos, self.write_pos + writable)
            self.planes[target] = planes[start:end]
            self.policies[target] = policies[start:end]
            self.masks[target] = masks[start:end]
            self.z[target] = z[start:end]
            self.eval_score[target] = eval_score[start:end]
            self.position_strings[target] = position_strings[start:end]
            self.write_pos = (self.write_pos + writable) % self.capacity
            self.size = min(self.capacity, self.size + writable)
            start = end

        self.total_written += int(total_increment)

    def _logical_indices(self) -> np.ndarray:
        if self.size < self.capacity:
            return np.arange(self.size)
        return np.concatenate(
            (
                np.arange(self.write_pos, self.capacity),
                np.arange(0, self.write_pos),
            )
        )


def _load_shard(
    path: Path,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None
]:
    with np.load(path, allow_pickle=False) as data:
        planes = np.asarray(data["planes"], dtype=np.float32)
        policies = np.asarray(data["policy"], dtype=np.float32)
        masks = np.asarray(data["mask"]).astype(np.bool_, copy=False)
        z = np.asarray(data["z"], dtype=np.float32)
        eval_score = (
            np.asarray(data["eval_score"], dtype=np.float32)
            if "eval_score" in data.files
            else None
        )
        position_key = _first_present_key(
            data.files, ("position_strings", "states_text", "positions", "position")
        )
        position_strings = (
            np.asarray(data[position_key], dtype=POSITION_STRING_DTYPE)
            if position_key is not None
            else None
        )
    return planes, policies, masks, z, eval_score, position_strings


def _first_present_key(keys: Iterable[str], candidates: Iterable[str]) -> str | None:
    key_set = set(keys)
    for candidate in candidates:
        if candidate in key_set:
            return candidate
    return None


def _read_manifest_if_present(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)
    _fsync_dir(path.parent)


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

from __future__ import annotations

import os
import random
import shutil
import signal
import time
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import Config
from .logging_utils import fsync_dir, read_json_if_present, utc_now_iso, write_json_atomic
from .replay_buffer import ReplayBuffer

BREAKING_CONFIG_PATHS = {
    ("rules", "board"),
    ("rules", "num_actions"),
    ("net", "blocks"),
    ("net", "channels"),
    ("net", "extra_planes"),
}


class ShutdownController:
    def __init__(self, run_dir: Path, logger: Any | None = None) -> None:
        self.run_dir = Path(run_dir)
        self.stop_path = self.run_dir / "STOP"
        self.logger = logger
        self.signal_count = 0
        self.shutdown_state = "running"
        self.exit_code = 0
        self.stop_reason: str | None = None
        self._previous_handlers: dict[int, Any] = {}
        self._installed = False

    def install(self) -> None:
        if self._installed:
            return
        for sig in (signal.SIGINT, signal.SIGTERM):
            self._previous_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, self._handle_signal)
        self._installed = True

    def restore(self) -> None:
        if not self._installed:
            return
        for sig, handler in self._previous_handlers.items():
            signal.signal(sig, handler)
        self._installed = False

    @property
    def draining(self) -> bool:
        return self.shutdown_state == "draining"

    @property
    def fast_abort_requested(self) -> bool:
        return self.shutdown_state == "fast_abort"

    @property
    def immediate_abort_requested(self) -> bool:
        return self.shutdown_state == "immediate_abort"

    def poll_stop_file(self) -> None:
        if self.stop_path.exists() and self.signal_count == 0:
            self.request_stop("STOP file")

    def request_stop(self, reason: str) -> None:
        if self.signal_count == 0:
            self.signal_count = 1
            self.shutdown_state = "draining"
            self.stop_reason = reason
            self._log("shutdown_draining", reason=reason)
        elif self.signal_count == 1:
            self.signal_count = 2
            self.shutdown_state = "fast_abort"
            self.exit_code = 130
            self.stop_reason = reason
            self._log("shutdown_fast_abort", reason=reason)
        else:
            self.signal_count = 3
            self.shutdown_state = "immediate_abort"
            self._log("shutdown_immediate_abort", reason=reason)
            os._exit(130)

    def mark_stopped_clean(self) -> None:
        self.shutdown_state = "stopped_clean"
        self.exit_code = 0

    def _handle_signal(self, signum: int, _frame: Any) -> None:
        self.request_stop(signal.Signals(signum).name)

    def _log(self, event: str, **fields: Any) -> None:
        if self.logger is not None:
            self.logger.event(event, **fields)


def save_checkpoint_atomic(
    *,
    run_dir: Path,
    step: int,
    config: Config,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    buffer: ReplayBuffer,
    counters: dict[str, int],
    logger: Any | None = None,
    fast: bool = False,
) -> Path:
    run_dir = Path(run_dir)
    ckpt_root = run_dir / "ckpt"
    ckpt_root.mkdir(parents=True, exist_ok=True)
    target = ckpt_root / str(int(step))
    tmp = ckpt_root / f".tmp_{int(step)}_{os.getpid()}_{time.time_ns()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    state = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "counters": {key: int(value) for key, value in counters.items()},
        "config": config.asdict(),
        "rng": capture_rng_state(),
        "elo": read_json_if_present(run_dir / "elo.json"),
        "created_at": utc_now_iso(),
        "fast": bool(fast),
    }
    torch.save(state, tmp / "state.pt")
    if fast:
        (tmp / "buffer").mkdir()
        write_json_atomic(
            tmp / "buffer" / "manifest.json",
            {
                "version": 1,
                "fast_checkpoint": True,
                "size": len(buffer),
                "total_written": int(buffer.stats()["total_written"]),
                "shards": [],
            },
        )
    else:
        buffer.save_to(tmp / "buffer", shard_size=config.training.replay_shard_size)
    fsync_dir(tmp)

    if target.exists():
        shutil.rmtree(target)
    os.replace(tmp, target)
    fsync_dir(ckpt_root)
    update_latest_symlink(ckpt_root, target.name)
    apply_retention(ckpt_root, config.checkpoint.keep_last, config.checkpoint.keep_every)
    if logger is not None:
        logger.event(
            "checkpoint_saved",
            step=int(step),
            path=target,
            fast=bool(fast),
            buffer_size=len(buffer),
            counters=counters,
        )
    return target


def load_checkpoint(
    resume_path: Path,
    requested_config: Config | None = None,
    *,
    device: torch.device | str = "cpu",
    seed: int | None = None,
) -> dict[str, Any]:
    ckpt_dir = resolve_checkpoint_dir(Path(resume_path))
    try:
        state: dict[str, Any] = torch.load(
            ckpt_dir / "state.pt", map_location=device, weights_only=False
        )
    except TypeError:
        state = torch.load(ckpt_dir / "state.pt", map_location=device)
    saved_config = config_from_dict(state.get("config", {}))
    config = requested_config or saved_config
    validate_resume_config(saved_config, config)
    buffer_dir = ckpt_dir / "buffer"
    try:
        buffer = ReplayBuffer.load_from(buffer_dir, config, seed=seed)
    except Exception:
        buffer = ReplayBuffer(config, seed=seed)
    counters = {
        "games": int(state.get("counters", {}).get("games", 0)),
        "steps": int(state.get("counters", {}).get("steps", 0)),
        "positions": int(state.get("counters", {}).get("positions", 0)),
    }
    return {
        "ckpt_dir": ckpt_dir,
        "state": state,
        "config": config,
        "saved_config": saved_config,
        "buffer": buffer,
        "counters": counters,
    }


def resolve_checkpoint_dir(path: Path) -> Path:
    path = Path(path)
    if (path / "state.pt").exists():
        return path
    latest = path / "ckpt" / "latest"
    if latest.exists() or latest.is_symlink():
        target = os.readlink(latest) if latest.is_symlink() else latest.read_text().strip()
        ckpt_dir = latest.parent / target
        if (ckpt_dir / "state.pt").exists():
            return ckpt_dir
    ckpt_root = path / "ckpt"
    candidates = [
        child
        for child in ckpt_root.iterdir()
        if child.is_dir() and child.name.isdigit() and (child / "state.pt").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"no checkpoint found under {path}")
    return max(candidates, key=lambda child: int(child.name))


def update_latest_symlink(ckpt_root: Path, target_name: str) -> None:
    latest = Path(ckpt_root) / "latest"
    tmp = Path(ckpt_root) / f".latest.{os.getpid()}.tmp"
    try:
        if tmp.exists() or tmp.is_symlink():
            tmp.unlink()
        os.symlink(target_name, tmp)
        os.replace(tmp, latest)
    except OSError:
        write_json_atomic(latest, {"target": target_name})
    fsync_dir(Path(ckpt_root))


def apply_retention(ckpt_root: Path, keep_last: int, keep_every: int) -> None:
    dirs = sorted(
        [child for child in Path(ckpt_root).iterdir() if child.is_dir() and child.name.isdigit()],
        key=lambda child: int(child.name),
    )
    if not dirs:
        return
    keep: set[Path] = set(dirs[-max(keep_last, 0) :])
    if keep_every > 0:
        keep.update(child for child in dirs if int(child.name) % keep_every == 0)
    latest = Path(ckpt_root) / "latest"
    if latest.is_symlink():
        keep.add(latest.parent / os.readlink(latest))
    for child in dirs:
        if child not in keep:
            shutil.rmtree(child, ignore_errors=True)
    fsync_dir(Path(ckpt_root))


def config_from_dict(data: dict[str, Any]) -> Config:
    return _dataclass_from_dict(Config, data)


def validate_resume_config(saved: Config, requested: Config) -> list[str]:
    saved_dict = saved.asdict()
    requested_dict = requested.asdict()
    breaking: list[str] = []
    warnings: list[str] = []
    for path, saved_value, requested_value in _walk_config(saved_dict, requested_dict):
        if saved_value == requested_value:
            continue
        dotted = ".".join(path)
        if path in BREAKING_CONFIG_PATHS:
            breaking.append(dotted)
        else:
            warnings.append(dotted)
    if breaking:
        raise ValueError(
            "resume config is incompatible for breaking fields: "
            + ", ".join(sorted(breaking))
        )
    return warnings


def capture_rng_state() -> dict[str, Any]:
    rng: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng["torch_cuda"] = torch.cuda.get_rng_state_all()
    return rng


def restore_rng_state(rng: dict[str, Any] | None) -> None:
    if not rng:
        return
    try:
        random.setstate(rng["python"])
    except Exception:
        pass
    try:
        np.random.set_state(rng["numpy"])
    except Exception:
        pass
    try:
        torch.set_rng_state(rng["torch"])
    except Exception:
        pass
    if torch.cuda.is_available() and "torch_cuda" in rng:
        try:
            torch.cuda.set_rng_state_all(rng["torch_cuda"])
        except Exception:
            pass


def _dataclass_from_dict(cls: type[Any], data: dict[str, Any]) -> Any:
    obj = cls()
    for field in fields(cls):
        if field.name not in data:
            continue
        current = getattr(obj, field.name)
        value = data[field.name]
        if is_dataclass(current) and isinstance(value, dict):
            setattr(obj, field.name, _dataclass_from_dict(type(current), value))
        else:
            setattr(obj, field.name, value)
    return obj


def _walk_config(
    saved: dict[str, Any], requested: dict[str, Any], prefix: tuple[str, ...] = ()
) -> list[tuple[tuple[str, ...], Any, Any]]:
    paths: list[tuple[tuple[str, ...], Any, Any]] = []
    keys = set(saved) | set(requested)
    for key in keys:
        saved_value = saved.get(key)
        requested_value = requested.get(key)
        path = (*prefix, str(key))
        if isinstance(saved_value, dict) and isinstance(requested_value, dict):
            paths.extend(_walk_config(saved_value, requested_value, path))
        else:
            paths.append((path, saved_value, requested_value))
    return paths

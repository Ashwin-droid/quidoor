from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from trainer.config import Config
from trainer.lifecycle import ShutdownController, load_checkpoint, save_checkpoint_atomic
from trainer.logging_utils import StatusWriter
from trainer.model import QuoridorNet
from trainer.replay_buffer import ReplayBuffer
from trainer.selfplay_driver import run_selfplay_phase, stub_evaluator
from trainer.train import build_optimizer, build_scheduler, run_training


def tiny_config() -> Config:
    cfg = Config()
    cfg.net.blocks = 1
    cfg.net.channels = 8
    cfg.net.bf16 = False
    cfg.net.compile = False
    cfg.net.fixed_batch_size = 4
    cfg.mcts.sims_per_move = 2
    cfg.mcts.fast_sims_per_move = 2
    cfg.selfplay.concurrent_games = 1
    cfg.selfplay.leaf_batch_size = 4
    cfg.selfplay.collect_timeout_ms = 0
    cfg.selfplay.phase_games = 1
    cfg.rules.ply_cap = 8
    cfg.training.replay_buffer_size = 64
    cfg.training.replay_shard_size = 16
    cfg.training.batch_size = 4
    cfg.training.train_steps_per_phase = 1
    cfg.training.planned_train_steps = 8
    cfg.training.mirror_probability = 0.0
    cfg.checkpoint.autosave_minutes = 0
    cfg.checkpoint.keep_last = 5
    cfg.checkpoint.keep_every = 10
    return cfg


def make_batch(count: int, cfg: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    planes = np.zeros((count, 6, 9, 9), dtype=np.float32)
    policy = np.zeros((count, 140), dtype=np.float32)
    mask = np.zeros((count, 140), dtype=np.bool_)
    z = np.linspace(-1.0, 1.0, count, dtype=np.float32)
    for idx in range(count):
        planes[idx, 0, idx % 9, idx % 9] = 1.0
        action = idx % cfg.rules.num_actions
        policy[idx, action] = 1.0
        mask[idx, action] = True
    return planes, policy, mask, z


def test_checkpoint_round_trip_restores_counters_and_buffer(tmp_path: Path) -> None:
    cfg = tiny_config()
    model = QuoridorNet(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    buffer = ReplayBuffer(cfg, seed=11)
    buffer.add(*make_batch(6, cfg))
    counters = {"games": 3, "steps": 2, "positions": 6}

    save_checkpoint_atomic(
        run_dir=tmp_path,
        step=counters["steps"],
        config=cfg,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        buffer=buffer,
        counters=counters,
    )

    loaded = load_checkpoint(tmp_path, device=torch.device("cpu"), seed=12)
    assert loaded["counters"] == counters
    assert len(loaded["buffer"]) == len(buffer)
    expected = buffer.as_arrays()
    actual = loaded["buffer"].as_arrays()
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])


def test_stop_file_drains_current_selfplay_chunk_without_starting_next(tmp_path: Path) -> None:
    cfg = tiny_config()
    cfg.selfplay.phase_games = 3
    cfg.selfplay.concurrent_games = 1
    cfg.rules.ply_cap = 6
    buffer = ReplayBuffer(cfg, seed=13)
    shutdown = ShutdownController(tmp_path)
    ticks = 0

    def tick(_metrics) -> None:
        nonlocal ticks
        if ticks == 0:
            (tmp_path / "STOP").touch()
        ticks += 1

    metrics = run_selfplay_phase(
        config=cfg,
        buffer=buffer,
        evaluator=stub_evaluator(),
        seed=14,
        shutdown=shutdown,
        tick_callback=tick,
    )

    assert shutdown.draining
    assert metrics.games == 1
    assert 0 < len(buffer) == metrics.positions


def test_status_json_is_atomically_rewritten_as_valid_json(tmp_path: Path) -> None:
    writer = StatusWriter(tmp_path, interval_seconds=0)
    for idx in range(20):
        writer.write({"phase": "test", "idx": idx, "nested": {"ok": True}}, force=True)
        payload = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
        assert payload["idx"] == idx
        assert payload["nested"]["ok"] is True
    assert not list(tmp_path.glob(".status.json*.tmp"))


def test_tiny_end_to_end_training_run_completes(tmp_path: Path) -> None:
    cfg = tiny_config()
    result = run_training(
        cfg,
        run_dir=tmp_path / "run",
        device=torch.device("cpu"),
        seed=15,
        max_phases=1,
        use_stub_selfplay=True,
        run_hooks=False,
        status_interval_seconds=0,
        console=False,
    )

    assert result.exit_code == 0
    assert result.counters["games"] == 1
    assert result.counters["steps"] == 1
    assert result.last_checkpoint is not None
    assert (result.last_checkpoint / "state.pt").exists()
    status = json.loads((tmp_path / "run" / "status.json").read_text(encoding="utf-8"))
    assert status["phase"] == "completed"
    assert status["state"] == "completed"

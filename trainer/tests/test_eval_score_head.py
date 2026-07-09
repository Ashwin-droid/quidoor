from __future__ import annotations

import numpy as np
import quoridor_engine as qe
import torch

from baselines.common import format_net_score, mate_score
from trainer.arena import ModelEvaluator, net_eval_score
from trainer.config import Config
from trainer.model import Inferencer, QuoridorNet
from trainer.replay_buffer import ReplayBuffer, eval_score_targets
from trainer.train import build_optimizer, build_scheduler, train_one_step


START = "Q1 t=0 ply=0 p0=e1 p1=e9 w=10,10 h=- v=-"
# Player 0 one step from goal on their own move: forced mate in 1.
MATE_P0 = "Q1 t=0 ply=14 p0=e8 p1=f3 w=10,10 h=- v=-"


def tiny_config() -> Config:
    cfg = Config()
    cfg.net.blocks = 1
    cfg.net.channels = 16
    cfg.net.bf16 = False
    cfg.net.compile = False
    cfg.net.fixed_batch_size = 4
    cfg.training.batch_size = 8
    cfg.training.replay_buffer_size = 64
    cfg.training.mirror_probability = 0.0
    return cfg


def test_targets_without_strings_are_scaled_outcomes() -> None:
    targets = eval_score_targets(np.array([1.0, 0.0, -1.0], dtype=np.float32))
    assert targets.tolist() == [10.0, 0.0, -10.0]


def test_targets_blend_heuristic_and_outcome() -> None:
    # Start position heuristic is 0, so the target is purely the outcome half.
    targets = eval_score_targets(
        np.array([1.0, -1.0], dtype=np.float32), np.array([START, START])
    )
    assert targets.tolist() == [10.0, -10.0]
    assert np.all(np.abs(targets) <= 20.0)


def test_mate_targets_override_outcome_and_flip_perspective() -> None:
    # The race says player 0 (the mover) mates in 1; even if the game was
    # eventually lost, the target stays in the positive mate region.
    targets = eval_score_targets(
        np.array([-1.0], dtype=np.float32), np.array([MATE_P0])
    )
    assert targets[0] == mate_score(1) == 39.0

    # Same board with player 1 to move: no forced mate, so the blend applies.
    undecided = "Q1 t=1 ply=15 p0=e8 p1=f3 w=10,10 h=- v=-"
    blended = eval_score_targets(np.array([1.0], dtype=np.float32), np.array([undecided]))
    assert abs(blended[0]) <= 20.0


def test_format_net_score_maps_regions() -> None:
    assert format_net_score(0.0) == "0"
    assert format_net_score(1.25) == "+125"
    assert format_net_score(-3.0) == "-300"
    assert format_net_score(39.0) == "+M1"
    assert format_net_score(-35.4) == "-M5"


def test_buffer_stores_and_roundtrips_eval_scores(tmp_path) -> None:
    cfg = tiny_config()
    buffer = ReplayBuffer(cfg, seed=3)
    planes = np.zeros((2, 6, 9, 9), dtype=np.float32)
    policies = np.zeros((2, 140), dtype=np.float32)
    policies[:, 0] = 1.0
    masks = np.ones((2, 140), dtype=np.bool_)
    buffer.add(
        planes,
        policies,
        masks,
        np.array([-1.0, 1.0], dtype=np.float32),
        position_strings=np.array([MATE_P0, START]),
    )
    assert buffer.eval_score[:2].tolist() == [39.0, 10.0]
    assert set(buffer.sample(4)) >= {"planes", "policy", "mask", "z", "eval_score"}

    buffer.save_to(tmp_path)
    restored = ReplayBuffer.load_from(tmp_path, cfg, seed=3)
    assert restored.eval_score[:2].tolist() == [39.0, 10.0]


def test_legacy_shards_without_eval_score_are_rederived(tmp_path) -> None:
    cfg = tiny_config()
    buffer = ReplayBuffer(cfg, seed=3)
    planes = np.zeros((1, 6, 9, 9), dtype=np.float32)
    policies = np.zeros((1, 140), dtype=np.float32)
    policies[:, 0] = 1.0
    masks = np.ones((1, 140), dtype=np.bool_)
    buffer.add(
        planes,
        policies,
        masks,
        np.array([-1.0], dtype=np.float32),
        position_strings=np.array([MATE_P0]),
    )
    buffer.save_to(tmp_path)

    # Strip the eval_score field to simulate a pre-upgrade shard.
    shard = next(tmp_path.glob("shard_*.npz"))
    with np.load(shard, allow_pickle=False) as data:
        stripped = {key: data[key] for key in data.files if key != "eval_score"}
    np.savez(shard, **stripped)

    restored = ReplayBuffer.load_from(tmp_path, cfg, seed=3)
    assert restored.eval_score[0] == 39.0


def test_train_step_reports_score_loss() -> None:
    cfg = tiny_config()
    model = QuoridorNet(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    buffer = ReplayBuffer(cfg, seed=5)
    state = qe.GameState()
    planes = np.repeat(
        np.asarray(state.encode(False), dtype=np.float32)[None, ...], 8, axis=0
    )
    mask = np.repeat(
        np.asarray(state.legal_mask()).astype(np.bool_)[None, ...], 8, axis=0
    )
    policies = np.zeros((8, 140), dtype=np.float32)
    policies[:, int(np.flatnonzero(mask[0])[0])] = 1.0
    buffer.add(
        planes,
        policies,
        mask,
        np.ones((8,), dtype=np.float32),
        position_strings=np.array([START] * 8),
    )

    metrics = train_one_step(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        buffer=buffer,
        config=cfg,
        device=torch.device("cpu"),
    )
    assert np.isfinite(metrics["eval_score_loss"])
    assert np.isfinite(metrics["eval_score_mae"])
    assert metrics["total_loss"] >= metrics["policy_loss"] + metrics["value_loss"]


def test_net_eval_score_perspective_and_gating() -> None:
    cfg = tiny_config()
    model = QuoridorNet(cfg).eval()
    evaluator = ModelEvaluator(name="test", inferencer=Inferencer(model, cfg))

    state = qe.GameState()
    score_p0 = net_eval_score(state, evaluator, cfg)
    assert score_p0 is not None

    flipped = qe.GameState.from_string("Q1 t=1 ply=1 p0=e2 p1=e9 w=10,10 h=- v=-")
    assert net_eval_score(flipped, evaluator, cfg) is not None

    model.score_head_loaded = False
    assert net_eval_score(state, evaluator, cfg) is None

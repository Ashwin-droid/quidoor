from __future__ import annotations

import json

import numpy as np

from trainer.analyze import mine_frequent_opening_positions
from trainer.config import Config
from trainer.replay_buffer import ReplayBuffer


def config_with_capacity(capacity: int) -> Config:
    cfg = Config()
    cfg.training.replay_buffer_size = capacity
    cfg.training.replay_shard_size = 2
    cfg.training.batch_size = 2
    cfg.training.mirror_probability = 0.0
    return cfg


def make_batch(cfg: Config, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    planes = np.zeros((count, 6, 9, 9), dtype=np.float32)
    policies = np.zeros((count, 140), dtype=np.float32)
    masks = np.zeros((count, 140), dtype=np.bool_)
    z = np.arange(count, dtype=np.float32)
    for idx in range(count):
        planes[idx, 0, 0, 0] = idx
        policies[idx, idx % 140] = 1.0
        masks[idx, : idx % 11 + 1] = True
    return planes, policies, masks, z


def test_ring_buffer_keeps_most_recent_positions() -> None:
    cfg = config_with_capacity(3)
    buffer = ReplayBuffer(cfg, seed=1)
    planes, policies, masks, z = make_batch(cfg, 5)

    buffer.add(planes, policies, masks, z)

    snapshot = buffer.as_arrays()
    assert len(buffer) == 3
    np.testing.assert_array_equal(snapshot["z"], np.array([2, 3, 4], dtype=np.float32))
    np.testing.assert_array_equal(snapshot["planes"][:, 0, 0, 0], [2, 3, 4])
    assert buffer.stats()["total_written"] == 5


def test_sample_batch_shapes() -> None:
    cfg = config_with_capacity(5)
    buffer = ReplayBuffer(cfg, seed=2)
    buffer.add(*make_batch(cfg, 5))

    batch = buffer.sample_batch(4)

    assert batch["planes"].shape == (4, 6, 9, 9)
    assert batch["policy"].shape == (4, 140)
    assert batch["mask"].shape == (4, 140)
    assert batch["z"].shape == (4,)


def test_shard_save_load_round_trip(tmp_path) -> None:
    cfg = config_with_capacity(10)
    buffer = ReplayBuffer(cfg, seed=3)
    buffer.add(*make_batch(cfg, 5))
    buffer_dir = tmp_path / "buffer"

    buffer.save_to(buffer_dir)
    loaded = ReplayBuffer.load_from(buffer_dir, cfg, seed=4)

    expected = buffer.as_arrays()
    actual = loaded.as_arrays()
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])


def test_position_strings_round_trip_and_feed_opening_miner(tmp_path) -> None:
    cfg = config_with_capacity(10)
    buffer = ReplayBuffer(cfg, seed=7)
    planes, policies, masks, z = make_batch(cfg, 3)
    position_strings = np.array(
        [
            "Q1 t=0 ply=0 p0=e1 p1=e9 w=10,10 h=- v=-",
            "Q1 t=0 ply=2 p0=e2 p1=e8 w=10,10 h=- v=-",
            "Q1 t=0 ply=4 p0=e3 p1=e7 w=10,10 h=- v=-",
        ]
    )
    buffer.add(planes, policies, masks, z, position_strings=position_strings)
    buffer_dir = tmp_path / "ckpt" / "0" / "buffer"

    buffer.save_to(buffer_dir)
    loaded = ReplayBuffer.load_from(buffer_dir, cfg, seed=8)

    np.testing.assert_array_equal(
        loaded.as_arrays()["position_strings"], position_strings
    )
    with np.load(buffer_dir / "shard_000000_0000.npz", allow_pickle=False) as shard:
        assert "position_strings" in shard.files

    positions, notes = mine_frequent_opening_positions(tmp_path, top_k=4)
    assert notes == []
    assert [item["position"] for item in positions] == position_strings[1:].tolist()
    assert [item["ply"] for item in positions] == [2, 4]


def test_torn_final_shard_does_not_corrupt_earlier_shards(tmp_path) -> None:
    cfg = config_with_capacity(10)
    buffer = ReplayBuffer(cfg, seed=5)
    buffer.add(*make_batch(cfg, 5))
    buffer_dir = tmp_path / "buffer"
    buffer.save_to(buffer_dir)

    manifest_path = buffer_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    final_shard = buffer_dir / manifest["shards"][-1]["file"]
    final_shard.write_bytes(b"torn npz")

    loaded = ReplayBuffer.load_from(buffer_dir, cfg, seed=6)
    snapshot = loaded.as_arrays()

    assert len(loaded) == 4
    np.testing.assert_array_equal(snapshot["z"], np.array([0, 1, 2, 3], dtype=np.float32))

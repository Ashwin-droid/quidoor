from __future__ import annotations

import numpy as np

from trainer.config import Config
from trainer.replay_buffer import (
    MIRROR_ACTIONS,
    ReplayBuffer,
    mirror_planes,
    mirror_policy,
)


def test_mirror_action_map_is_an_involution() -> None:
    np.testing.assert_array_equal(MIRROR_ACTIONS[MIRROR_ACTIONS], np.arange(140))
    assert MIRROR_ACTIONS[2] == 3
    assert MIRROR_ACTIONS[3] == 2
    assert MIRROR_ACTIONS[8] == 9
    assert MIRROR_ACTIONS[12] == 19
    assert MIRROR_ACTIONS[76 + 3 * 8 + 2] == 76 + 3 * 8 + 5


def test_plane_and_policy_mirror_are_involutions_for_encoded_layout() -> None:
    rng = np.random.default_rng(9)
    planes = rng.standard_normal((8, 9, 9), dtype=np.float32)
    planes[3, :, 8] = 0.0
    policy = rng.random(140, dtype=np.float32)
    mask = rng.random(140) > 0.4

    np.testing.assert_allclose(mirror_planes(mirror_planes(planes)), planes)
    np.testing.assert_allclose(mirror_policy(mirror_policy(policy)), policy)
    np.testing.assert_array_equal(mirror_policy(mirror_policy(mask)), mask)


def test_buffer_applies_forced_mirror_at_write_time() -> None:
    cfg = Config()
    cfg.training.replay_buffer_size = 4
    cfg.training.mirror_probability = 1.0
    buffer = ReplayBuffer(cfg, seed=10)

    planes = np.zeros((1, 6, 9, 9), dtype=np.float32)
    planes[0, 0, 2, 1] = 1.0
    planes[0, 3, 4, 2] = 1.0
    policy = np.zeros((1, 140), dtype=np.float32)
    policy[0, 2] = 0.25
    policy[0, 12] = 0.75
    mask = np.zeros((1, 140), dtype=np.bool_)
    mask[0, [2, 12]] = True
    z = np.array([1.0], dtype=np.float32)

    buffer.add(planes, policy, mask, z)
    snapshot = buffer.as_arrays()

    np.testing.assert_array_equal(snapshot["planes"][0], mirror_planes(planes[0]))
    np.testing.assert_array_equal(snapshot["policy"][0], mirror_policy(policy[0]))
    np.testing.assert_array_equal(snapshot["mask"][0], mirror_policy(mask[0]))
    np.testing.assert_array_equal(snapshot["z"], z)

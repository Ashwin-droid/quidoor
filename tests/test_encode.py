from __future__ import annotations

import numpy as np
import quoridor_engine as qe

from tests.reference import BOARD, ReferenceState


def mirror_encoded_lr(encoded: np.ndarray) -> np.ndarray:
    out = np.zeros_like(encoded)
    planes = encoded.shape[0]
    for p in range(planes):
        for r in range(BOARD):
            for c in range(BOARD):
                if p == 3:
                    if c < BOARD - 1:
                        out[p, r, BOARD - 2 - c] = encoded[p, r, c]
                else:
                    out[p, r, BOARD - 1 - c] = encoded[p, r, c]
    return out


def test_notation_round_trips() -> None:
    positions = [
        "Q1 t=0 ply=0 p0=e1 p1=e9 w=10,10 h=- v=-",
        "Q1 t=1 ply=8 p0=d3 p1=f7 w=8,8 h=c2,f5 v=g1,b6",
    ]
    for text in positions:
        state = qe.GameState.from_string(text)
        assert state.to_string() == text
        assert qe.GameState.from_string(state.to_string()).to_string() == text


def test_canonical_action_involution() -> None:
    for action in range(qe.NUM_ACTIONS):
        assert qe.canonical_action(action, 0) == action
        mapped = qe.canonical_action(action, 1)
        assert qe.canonical_action(mapped, 1) == action


def test_mirror_action_involution() -> None:
    for action in range(qe.NUM_ACTIONS):
        mapped = qe.mirror_action(action)
        assert qe.mirror_action(mapped) == action


def test_canonical_encoding_rotates_player_one_to_move() -> None:
    state = qe.GameState.from_string("Q1 t=1 ply=1 p0=e2 p1=e9 w=10,10 h=- v=-")
    encoded = state.encode(False)
    assert encoded.shape == (6, 9, 9)
    assert encoded[0, 0, 4] == 1.0
    assert encoded[1, 7, 4] == 1.0
    assert np.allclose(encoded[4], 1.0)
    assert np.allclose(encoded[5], 1.0)


def test_mirror_state_encoding_consistency() -> None:
    text = "Q1 t=0 ply=8 p0=d3 p1=f7 w=8,8 h=c2,f5 v=g1,b6"
    state = qe.GameState.from_string(text)
    mirrored_text = ReferenceState.from_string(text).mirror_lr().to_string()
    mirrored = qe.GameState.from_string(mirrored_text)

    encoded = state.encode(False)
    mirrored_encoded = mirrored.encode(False)
    assert np.array_equal(mirrored_encoded, mirror_encoded_lr(encoded))

    mask = state.legal_mask()
    mirrored_mask = mirrored.legal_mask()
    for action in range(qe.NUM_ACTIONS):
        assert bool(mask[action]) == bool(mirrored_mask[qe.mirror_action(action)])


def test_extra_planes_shape_and_notation_round_trip() -> None:
    state = qe.GameState()
    assert state.encode(True).shape == (8, 9, 9)
    assert qe.GameState.from_string(state.to_string()).to_string() == state.to_string()

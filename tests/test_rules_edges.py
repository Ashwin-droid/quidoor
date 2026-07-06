from __future__ import annotations

import numpy as np
import pytest
import quoridor_engine as qe

from tests.reference import wall_action


def legal_actions(state: qe.GameState) -> set[int]:
    return {int(a) for a in np.flatnonzero(state.legal_mask())}


def test_straight_jump_and_blocked_diagonals() -> None:
    straight = qe.GameState.from_string("Q1 t=0 ply=0 p0=e5 p1=e6 w=10,10 h=- v=-")
    assert 5 in legal_actions(straight)
    assert 1 not in legal_actions(straight)
    assert 10 not in legal_actions(straight)
    assert 11 not in legal_actions(straight)

    blocked = qe.GameState.from_string("Q1 t=0 ply=1 p0=e5 p1=e6 w=9,10 h=e6 v=-")
    actions = legal_actions(blocked)
    assert 5 not in actions
    assert {10, 11}.issubset(actions)

    one_diag_blocked = qe.GameState.from_string(
        "Q1 t=0 ply=2 p0=e5 p1=e6 w=9,9 h=e6 v=e5"
    )
    actions = legal_actions(one_diag_blocked)
    assert 5 not in actions
    assert 10 not in actions
    assert 11 in actions


def test_board_edge_behind_opponent_enables_diagonals() -> None:
    state = qe.GameState.from_string("Q1 t=0 ply=0 p0=e8 p1=e9 w=10,10 h=- v=-")
    actions = legal_actions(state)
    assert 5 not in actions
    assert {10, 11}.issubset(actions)


def test_wall_overlap_and_cross_matrix_through_binding() -> None:
    h = qe.GameState.from_string("Q1 t=0 ply=1 p0=e1 p1=e9 w=9,10 h=d4 v=-")
    h_actions = legal_actions(h)
    assert wall_action("h", "d4") not in h_actions
    assert wall_action("h", "c4") not in h_actions
    assert wall_action("h", "e4") not in h_actions
    assert wall_action("v", "d4") not in h_actions
    assert wall_action("h", "f4") in h_actions
    assert wall_action("v", "e4") in h_actions

    v = qe.GameState.from_string("Q1 t=0 ply=1 p0=e1 p1=e9 w=9,10 h=- v=d4")
    v_actions = legal_actions(v)
    assert wall_action("v", "d4") not in v_actions
    assert wall_action("v", "d3") not in v_actions
    assert wall_action("v", "d5") not in v_actions
    assert wall_action("h", "d4") not in v_actions
    assert wall_action("v", "d6") in v_actions
    assert wall_action("h", "d5") in v_actions


def test_path_blocking_wall_rejected_including_placer_trap() -> None:
    state = qe.GameState.from_string(
        "Q1 t=0 ply=5 p0=e1 p1=e9 w=5,10 h=a4,c4,e4 v=h1,h3"
    )
    assert wall_action("h", "g4") not in legal_actions(state)


def test_no_walls_left_removes_wall_actions_only() -> None:
    state = qe.GameState.from_string(
        "Q1 t=0 ply=28 p0=g1 p1=f6 w=0,0 "
        "h=a1,e1,e4,b5,d6,c7,h7 "
        "v=d1,g1,a2,b3,d3,f4,a6,b6,e6,h6,a8,d8,f8"
    )
    actions = legal_actions(state)
    assert any(a < 12 for a in actions)
    assert all(a < 12 for a in actions)


def test_illegal_apply_raises_value_error() -> None:
    state = qe.GameState()
    with pytest.raises(ValueError):
        state.apply(0)

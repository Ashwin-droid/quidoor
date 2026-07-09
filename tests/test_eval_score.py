from __future__ import annotations

import quoridor_engine as qe

from baselines.common import (
    MATE_CP,
    EvalScore,
    evaluate_position,
    forced_win,
    format_eval,
)


# Twenty vertical walls stacked in the a-c and g-h files: they only block
# horizontal movement, so straight runs along the e-file keep their
# manhattan shortest-path lengths.
V20 = "a1,a3,a5,a7,b1,b3,b5,b7,c1,c3,c5,c7,g1,g3,g5,g7,h1,h3,h5,h7"
V10 = "a1,a3,a5,a7,b1,b3,b5,b7,c1,c3"


def state(text: str) -> qe.GameState:
    return qe.GameState.from_string(text)


def test_start_position_is_exactly_zero() -> None:
    score = evaluate_position(qe.GameState())
    assert score == EvalScore(kind="cp", cp=0)
    assert format_eval(score) == "0"
    assert forced_win(qe.GameState()) is None


def test_cp_is_integer_path_and_wall_difference() -> None:
    ahead = state("Q1 t=1 ply=1 p0=e2 p1=e9 w=10,10 h=- v=-")
    assert evaluate_position(ahead) == EvalScore(kind="cp", cp=100)
    assert format_eval(evaluate_position(ahead)) == "+100"

    # One player-0 wall spent far away from both paths: -25 for the wall.
    walls = state("Q1 t=0 ply=2 p0=e2 p1=e8 w=9,10 h=a1 v=-")
    assert evaluate_position(walls) == EvalScore(kind="cp", cp=-25)
    assert format_eval(evaluate_position(walls)) == "-25"


def test_mate_when_wallless_opponent_loses_race() -> None:
    pos = state(f"Q1 t=0 ply=30 p0=e8 p1=b8 w=10,0 h=- v={V10}")
    assert int(pos.shortest_path_len(0)) == 1
    assert forced_win(pos) == (0, 1)
    score = evaluate_position(pos)
    assert score == EvalScore(kind="mate", cp=MATE_CP, mate_in=1, winner=0)
    assert format_eval(score) == "+M1"


def test_mate_when_wallless_mover_is_strictly_behind() -> None:
    pos = state(f"Q1 t=0 ply=30 p0=e4 p1=e2 w=0,10 h=- v={V10}")
    assert int(pos.shortest_path_len(0)) == 5
    assert int(pos.shortest_path_len(1)) == 1
    assert forced_win(pos) == (1, 1)
    score = evaluate_position(pos)
    assert score == EvalScore(kind="mate", cp=-MATE_CP, mate_in=1, winner=1)
    assert format_eval(score) == "-M1"


def test_equal_pure_race_goes_to_the_mover() -> None:
    base = f"p0=e4 p1=e6 w=0,0 h=- v={V20}"
    p0_moves = state(f"Q1 t=0 ply=40 {base}")
    assert int(p0_moves.shortest_path_len(0)) == 5
    assert int(p0_moves.shortest_path_len(1)) == 5
    assert forced_win(p0_moves) == (0, 5)
    assert format_eval(evaluate_position(p0_moves)) == "+M5"

    p1_moves = state(f"Q1 t=1 ply=41 {base}")
    assert forced_win(p1_moves) == (1, 5)
    assert format_eval(evaluate_position(p1_moves)) == "-M5"


def test_mover_one_step_from_goal_mates_despite_walls() -> None:
    # Opponent holds all 10 walls, but they never get a turn.
    pos = state("Q1 t=0 ply=14 p0=e8 p1=f3 w=10,10 h=- v=-")
    assert forced_win(pos) == (0, 1)
    assert format_eval(evaluate_position(pos)) == "+M1"

    # Same square, opponent to move: not decided.
    pos = state("Q1 t=1 ply=15 p0=e8 p1=f3 w=10,10 h=- v=-")
    assert forced_win(pos) is None


def test_no_mate_claim_while_leader_can_still_be_walled() -> None:
    # Player 1 is one step from goal but player 0 still holds 10 walls, so
    # the race is not provably over.
    pos = state(f"Q1 t=0 ply=30 p0=e4 p1=e2 w=10,0 h=- v={V10}")
    assert forced_win(pos) is None
    assert evaluate_position(pos).kind == "cp"


def test_finished_game_is_mate_in_zero() -> None:
    won = state("Q1 t=1 ply=17 p0=e9 p1=d5 w=10,10 h=- v=-")
    assert won.winner() == 0
    assert forced_win(won) == (0, 0)
    assert format_eval(evaluate_position(won)) == "+M0"

    lost = state("Q1 t=0 ply=18 p0=d5 p1=e1 w=10,10 h=- v=-")
    assert lost.winner() == 1
    assert format_eval(evaluate_position(lost)) == "-M0"

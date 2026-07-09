"""Shared helpers for small baseline players and terminal tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import quoridor_engine as qe


BOARD = 9
NUM_ACTIONS = 140
PAWN_DELTAS: tuple[tuple[int, int], ...] = (
    (-1, 0),
    (1, 0),
    (0, 1),
    (0, -1),
    (-2, 0),
    (2, 0),
    (0, 2),
    (0, -2),
    (-1, 1),
    (-1, -1),
    (1, 1),
    (1, -1),
)
PAWN_NAMES = ("N", "S", "E", "W", "NN", "SS", "EE", "WW", "NE", "NW", "SE", "SW")


class Player(Protocol):
    def select_action(self, state: qe.GameState) -> int:
        """Return one legal action index for the given state."""


def legal_actions(state: qe.GameState) -> list[int]:
    return [int(a) for a in np.flatnonzero(state.legal_mask())]


def clone_apply(state: qe.GameState, action: int) -> qe.GameState:
    child = state.clone()
    child.apply(int(action))
    return child


def cell_to_coord(row: int, col: int) -> str:
    return f"{chr(ord('a') + col)}{row + 1}"


def coord_to_cell(text: str) -> tuple[int, int]:
    text = text.strip().lower()
    if len(text) != 2 or not ("a" <= text[0] <= "i") or not ("1" <= text[1] <= "9"):
        raise ValueError(f"bad cell coordinate `{text}`")
    return ord(text[1]) - ord("1"), ord(text[0]) - ord("a")


def anchor_to_coord(anchor: int) -> str:
    return f"{chr(ord('a') + anchor % 8)}{anchor // 8 + 1}"


def coord_to_anchor(text: str) -> int:
    text = text.strip().lower()
    if len(text) != 2 or not ("a" <= text[0] <= "h") or not ("1" <= text[1] <= "8"):
        raise ValueError(f"bad wall anchor `{text}`")
    return (ord(text[1]) - ord("1")) * 8 + (ord(text[0]) - ord("a"))


def pawn_destination(state: qe.GameState, action: int) -> tuple[int, int] | None:
    if not 0 <= action < 12:
        return None
    row, col = state.pawn(state.to_move)
    dr, dc = PAWN_DELTAS[action]
    nr, nc = row + dr, col + dc
    if 0 <= nr < BOARD and 0 <= nc < BOARD:
        return nr, nc
    return None


def action_to_label(action: int, state: qe.GameState | None = None) -> str:
    action = int(action)
    if action < 12:
        if state is not None:
            dest = pawn_destination(state, action)
            if dest is not None:
                return cell_to_coord(*dest)
        return PAWN_NAMES[action]
    if action < 76:
        return f"h{anchor_to_coord(action - 12)}"
    return f"v{anchor_to_coord(action - 76)}"


def parse_action(text: str, state: qe.GameState) -> int:
    raw = text.strip().lower().replace(":", " ").replace("-", " ")
    if not raw:
        raise ValueError("empty move")

    legal = set(legal_actions(state))

    if raw.isdigit():
        action = int(raw)
        if action in legal:
            return action
        raise ValueError(f"action {action} is not legal")

    compact = raw.replace(" ", "")
    if compact.upper() in PAWN_NAMES:
        action = PAWN_NAMES.index(compact.upper())
        if action in legal:
            return action
        raise ValueError(f"pawn move {compact.upper()} is not legal")

    if compact[0:1] in {"h", "v"} and len(compact) == 3:
        anchor = coord_to_anchor(compact[1:])
        action = (12 + anchor) if compact[0] == "h" else (76 + anchor)
        if action in legal:
            return action
        raise ValueError(f"wall move {compact} is not legal")

    # Pawn moves are entered as destination cell coordinates, e.g. e2.
    target = coord_to_cell(compact)
    matches = [
        action
        for action in legal
        if action < 12 and pawn_destination(state, action) == target
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"cell move {compact} is not legal")
    raise ValueError(f"cell move {compact} is ambiguous")


def parse_wall_sets(state: qe.GameState) -> tuple[set[int], set[int]]:
    fields: dict[str, str] = {}
    for token in state.to_string().split():
        if "=" in token:
            key, value = token.split("=", 1)
            fields[key] = value

    def parse_list(value: str) -> set[int]:
        if value == "-":
            return set()
        return {coord_to_anchor(item) for item in value.split(",")}

    return parse_list(fields["h"]), parse_list(fields["v"])


def render_board(state: qe.GameState) -> str:
    """Render a compact ASCII board with walls and coordinate labels."""

    hwalls, vwalls = parse_wall_sets(state)
    p0 = state.pawn(0)
    p1 = state.pawn(1)
    lines = ["    a b c d e f g h i"]
    for r in range(BOARD - 1, -1, -1):
        cell_parts: list[str] = []
        for c in range(BOARD):
            token = "."
            if (r, c) == p0:
                token = "0"
            elif (r, c) == p1:
                token = "1"
            cell_parts.append(token)
            if c < BOARD - 1:
                blocked = (r < 8 and (r * 8 + c) in vwalls) or (r > 0 and ((r - 1) * 8 + c) in vwalls)
                cell_parts.append("|" if blocked else " ")
        lines.append(f"{r + 1:>2}  {''.join(cell_parts)}  {r + 1}")
        if r > 0:
            sep_parts: list[str] = []
            for c in range(BOARD):
                blocked = ((r - 1) * 8 + c) in hwalls if c < 8 else False
                blocked = blocked or (((r - 1) * 8 + c - 1) in hwalls if c > 0 else False)
                sep_parts.append("-" if blocked else " ")
                if c < BOARD - 1:
                    sep_parts.append("+")
            lines.append(f"    {''.join(sep_parts)}")
    lines.append("    a b c d e f g h i")
    lines.append(
        f"to_move={state.to_move} ply={state.ply} "
        f"walls=({state.walls_left(0)},{state.walls_left(1)})"
    )
    return "\n".join(lines)


CP_PER_MOVE = 100
CP_PER_WALL = 25
MATE_CP = 10_000

# Continuous score space for the network's eval head, in moves-ahead units:
# |score| <= 20 is a normal evaluation, anything beyond is a forced mate with
# the depth encoding distance — mate in n maps to +/-(40 - n), so mate-in-1
# is ~39 and a very distant mate saturates at 21.
EVAL_SCORE_CLAMP = 20.0
MATE_SCORE_BASE = 40.0
MATE_SCORE_MAX_DISTANCE = 19


@dataclass(frozen=True, slots=True)
class EvalScore:
    """Stockfish-style position score from player 0's perspective.

    `kind` is `"cp"` for a heuristic score in centi-moves (one full move of
    shortest-path advantage = 100) or `"mate"` for a forced win. For mates,
    `winner` is the winning player and `mate_in` counts the winner's own
    remaining moves; `cp` is clamped to +/-MATE_CP so the score stays usable
    as a bar/graph value.
    """

    kind: str
    cp: int
    mate_in: int | None = None
    winner: int | None = None


def forced_win(state: qe.GameState) -> tuple[int, int] | None:
    """Return `(winner, moves)` when the position is a decided race, else None.

    A finished game is a win in 0. Otherwise the position is provably decided
    once the trailing side has no walls left to slow the leader: the leader
    just walks their shortest path. This uses the standard Quoridor race
    assumption that a wall-less opponent's pawn cannot delay the runner (a
    facing pawn is jumped over, at worst breaking even).
    """

    winner = state.winner()
    if winner is not None:
        return int(winner), 0

    mover = int(state.to_move)
    other = mover ^ 1
    d_mover = int(state.shortest_path_len(mover))
    d_other = int(state.shortest_path_len(other))

    # One step from goal on your own turn: no opponent reply intervenes.
    if d_mover == 1:
        return mover, 1

    # Mover wins the race on ties because they move first from here.
    if int(state.walls_left(other)) == 0 and d_mover <= d_other:
        return mover, d_mover
    if int(state.walls_left(mover)) == 0 and d_other < d_mover:
        return other, d_other
    return None


def evaluate_position(state: qe.GameState) -> EvalScore:
    """Score a position like Stockfish: integer centi-moves, mates as Mxx.

    The heuristic score is symmetric and deterministic — the starting
    position is exactly 0 — combining shortest-path difference (100 per
    move) and walls in hand (25 per wall), always from player 0's view.
    """

    mate = forced_win(state)
    if mate is not None:
        winner, distance = mate
        sign = 1 if winner == 0 else -1
        return EvalScore(kind="mate", cp=sign * MATE_CP, mate_in=distance, winner=winner)

    path_term = int(state.shortest_path_len(1)) - int(state.shortest_path_len(0))
    wall_term = int(state.walls_left(0)) - int(state.walls_left(1))
    cp = CP_PER_MOVE * path_term + CP_PER_WALL * wall_term
    return EvalScore(kind="cp", cp=int(cp))


def format_eval(score: EvalScore) -> str:
    if score.kind == "mate":
        sign = "+" if score.winner == 0 else "-"
        return f"{sign}M{score.mate_in}"
    if score.cp == 0:
        return "0"
    return f"{score.cp:+d}"


def mate_score(mate_in: int) -> float:
    """Map a mate distance into the continuous score space (always > 20)."""

    return MATE_SCORE_BASE - min(max(int(mate_in), 0), MATE_SCORE_MAX_DISTANCE)


def heuristic_score_moves(state: qe.GameState, player: int) -> float:
    """Shortest-path eval in moves-ahead units from `player`'s perspective."""

    player = int(player)
    opponent = player ^ 1
    path_term = int(state.shortest_path_len(opponent)) - int(state.shortest_path_len(player))
    wall_term = int(state.walls_left(player)) - int(state.walls_left(opponent))
    return float(path_term) + (CP_PER_WALL / CP_PER_MOVE) * float(wall_term)


def format_net_score(score: float) -> str:
    """Render a continuous net score like the deterministic eval text."""

    if abs(score) > EVAL_SCORE_CLAMP:
        distance = max(0, round(MATE_SCORE_BASE - abs(score)))
        return f"{'+' if score > 0 else '-'}M{distance}"
    cp = round(score * CP_PER_MOVE)
    return "0" if cp == 0 else f"{cp:+d}"


def evaluate_shortest_path_diff(state: qe.GameState, player: int) -> float:
    winner = state.winner()
    if winner is not None:
        return 10_000.0 if winner == player else -10_000.0

    own = state.shortest_path_len(player)
    opp = state.shortest_path_len(player ^ 1)
    wall_term = 0.25 * (state.walls_left(player) - state.walls_left(player ^ 1))
    return 10.0 * (opp - own) + wall_term

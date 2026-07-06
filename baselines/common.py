"""Shared helpers for small baseline players and terminal tools."""

from __future__ import annotations

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


def evaluate_shortest_path_diff(state: qe.GameState, player: int) -> float:
    winner = state.winner()
    if winner is not None:
        return 10_000.0 if winner == player else -10_000.0

    own = state.shortest_path_len(player)
    opp = state.shortest_path_len(player ^ 1)
    wall_term = 0.25 * (state.walls_left(player) - state.walls_left(player ^ 1))
    return 10.0 * (opp - own) + wall_term

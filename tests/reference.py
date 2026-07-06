"""Naive pure-Python Quoridor rules reference.

This module is deliberately independent of the Rust engine. It models the
SPEC.md rules directly and uses the same 140-action space:

0..11      pawn moves [N, S, E, W, NN, SS, EE, WW, NE, NW, SE, SW]
12..75     horizontal walls, anchor = action - 12
76..139    vertical walls, anchor = action - 76

The implementation favors simple geometry and BFS over cleverness. Bitboards
are used only as a compact way to store wall anchor sets.
"""

from __future__ import annotations

from functools import lru_cache
from dataclasses import dataclass
from typing import Iterable


BOARD = 9
CELLS = BOARD * BOARD
WALL_GRID = BOARD - 1
WALL_ANCHORS = WALL_GRID * WALL_GRID
NUM_ACTIONS = 140
WALLS_PER_PLAYER = 10

HORIZONTAL = "h"
VERTICAL = "v"

PAWN_DELTAS: tuple[tuple[int, int], ...] = (
    (-1, 0),  # N
    (1, 0),  # S
    (0, 1),  # E
    (0, -1),  # W
    (-2, 0),  # NN
    (2, 0),  # SS
    (0, 2),  # EE
    (0, -2),  # WW
    (-1, 1),  # NE
    (-1, -1),  # NW
    (1, 1),  # SE
    (1, -1),  # SW
)

DIRS: tuple[tuple[int, int], ...] = (
    (-1, 0),
    (1, 0),
    (0, 1),
    (0, -1),
)
DIR_N, DIR_S, DIR_E, DIR_W = range(4)

NO_CELL = -1


def cell(row: int, col: int) -> int:
    return row * BOARD + col


def row(cell_idx: int) -> int:
    return cell_idx // BOARD


def col(cell_idx: int) -> int:
    return cell_idx % BOARD


def anchor(row_idx: int, col_idx: int) -> int:
    return row_idx * WALL_GRID + col_idx


def anchor_row(anchor_idx: int) -> int:
    return anchor_idx // WALL_GRID


def anchor_col(anchor_idx: int) -> int:
    return anchor_idx % WALL_GRID


def in_board(row_idx: int, col_idx: int) -> bool:
    return 0 <= row_idx < BOARD and 0 <= col_idx < BOARD


def in_anchor_grid(row_idx: int, col_idx: int) -> bool:
    return 0 <= row_idx < WALL_GRID and 0 <= col_idx < WALL_GRID


def bit(anchor_idx: int) -> int:
    return 1 << anchor_idx


def has_anchor(bits: int, row_idx: int, col_idx: int) -> bool:
    return in_anchor_grid(row_idx, col_idx) and (bits & bit(anchor(row_idx, col_idx))) != 0


def cell_to_coord(cell_idx: int) -> str:
    return f"{chr(ord('a') + col(cell_idx))}{row(cell_idx) + 1}"


def coord_to_cell(text: str) -> int:
    if len(text) != 2 or not ("a" <= text[0] <= "i") or not ("1" <= text[1] <= "9"):
        raise ValueError(f"bad cell `{text}`")
    return cell(ord(text[1]) - ord("1"), ord(text[0]) - ord("a"))


def anchor_to_coord(anchor_idx: int) -> str:
    return f"{chr(ord('a') + anchor_col(anchor_idx))}{anchor_row(anchor_idx) + 1}"


def coord_to_anchor(text: str) -> int:
    if len(text) != 2 or not ("a" <= text[0] <= "h") or not ("1" <= text[1] <= "8"):
        raise ValueError(f"bad wall anchor `{text}`")
    return anchor(ord(text[1]) - ord("1"), ord(text[0]) - ord("a"))


def wall_bits_to_string(bits: int) -> str:
    coords = [anchor_to_coord(a) for a in range(WALL_ANCHORS) if bits & bit(a)]
    return ",".join(coords) if coords else "-"


def parse_wall_bits(text: str) -> int:
    if text == "-":
        return 0
    bits = 0
    for item in text.split(","):
        a = coord_to_anchor(item)
        if bits & bit(a):
            raise ValueError(f"duplicate wall anchor `{item}`")
        bits |= bit(a)
    return bits


def iter_wall_anchors(bits: int) -> Iterable[int]:
    for a in range(WALL_ANCHORS):
        if bits & bit(a):
            yield a


@dataclass
class ReferenceState:
    pawns: list[int]
    hwalls: int = 0
    vwalls: int = 0
    walls_left: list[int] | None = None
    to_move: int = 0
    ply: int = 0

    def __post_init__(self) -> None:
        if self.walls_left is None:
            self.walls_left = [WALLS_PER_PLAYER, WALLS_PER_PLAYER]
        self.validate()

    @classmethod
    def start(cls) -> "ReferenceState":
        return cls(pawns=[cell(0, 4), cell(8, 4)])

    @classmethod
    def from_string(cls, text: str) -> "ReferenceState":
        fields: dict[str, str] = {}
        saw_version = False
        for token in text.split():
            if token == "Q1":
                saw_version = True
                continue
            if "=" not in token:
                raise ValueError(f"bad token `{token}`")
            key, value = token.split("=", 1)
            fields[key] = value
        if not saw_version:
            raise ValueError("missing Q1 version marker")

        required = {"t", "ply", "p0", "p1", "w", "h", "v"}
        missing = sorted(required - set(fields))
        if missing:
            raise ValueError(f"missing fields: {', '.join(missing)}")

        w0, w1 = fields["w"].split(",", 1)
        return cls(
            pawns=[coord_to_cell(fields["p0"]), coord_to_cell(fields["p1"])],
            hwalls=parse_wall_bits(fields["h"]),
            vwalls=parse_wall_bits(fields["v"]),
            walls_left=[int(w0), int(w1)],
            to_move=int(fields["t"]),
            ply=int(fields["ply"]),
        )

    def clone(self) -> "ReferenceState":
        assert self.walls_left is not None
        return self._new_unchecked(
            pawns=self.pawns.copy(),
            hwalls=self.hwalls,
            vwalls=self.vwalls,
            walls_left=self.walls_left.copy(),
            to_move=self.to_move,
            ply=self.ply,
        )

    @classmethod
    def _new_unchecked(
        cls,
        *,
        pawns: list[int],
        hwalls: int,
        vwalls: int,
        walls_left: list[int],
        to_move: int,
        ply: int,
    ) -> "ReferenceState":
        state = cls.__new__(cls)
        state.pawns = pawns
        state.hwalls = hwalls
        state.vwalls = vwalls
        state.walls_left = walls_left
        state.to_move = to_move
        state.ply = ply
        return state

    def key(self) -> tuple[int, int, int, int, int, int, int, int]:
        assert self.walls_left is not None
        return (
            self.pawns[0],
            self.pawns[1],
            self.hwalls,
            self.vwalls,
            self.walls_left[0],
            self.walls_left[1],
            self.to_move,
            self.ply,
        )

    def to_string(self) -> str:
        assert self.walls_left is not None
        return (
            f"Q1 t={self.to_move} ply={self.ply} "
            f"p0={cell_to_coord(self.pawns[0])} p1={cell_to_coord(self.pawns[1])} "
            f"w={self.walls_left[0]},{self.walls_left[1]} "
            f"h={wall_bits_to_string(self.hwalls)} v={wall_bits_to_string(self.vwalls)}"
        )

    def validate(self) -> None:
        assert self.walls_left is not None
        if self.to_move not in (0, 1):
            raise ValueError("to_move must be 0 or 1")
        if not (0 <= self.ply <= 65535):
            raise ValueError("ply out of range")
        if any(p < 0 or p >= CELLS for p in self.pawns):
            raise ValueError("pawn out of board")
        if self.pawns[0] == self.pawns[1]:
            raise ValueError("pawns may not occupy the same cell")
        if any(w < 0 or w > WALLS_PER_PLAYER for w in self.walls_left):
            raise ValueError("wall counts must be between 0 and 10")
        placed = self.hwalls.bit_count() + self.vwalls.bit_count()
        spent = (WALLS_PER_PLAYER - self.walls_left[0]) + (WALLS_PER_PLAYER - self.walls_left[1])
        if placed != spent:
            raise ValueError(f"placed wall count {placed} does not match spent wall count {spent}")
        if not all_walls_non_conflicting(self.hwalls, self.vwalls):
            raise ValueError("wall layout has overlap or crossing")
        if self.shortest_path_len(0) is None or self.shortest_path_len(1) is None:
            raise ValueError("wall layout blocks a player path")

    def winner(self) -> int | None:
        if row(self.pawns[0]) == BOARD - 1:
            return 0
        if row(self.pawns[1]) == 0:
            return 1
        return None

    def pawn(self, player: int) -> tuple[int, int]:
        if player not in (0, 1):
            raise ValueError("player must be 0 or 1")
        return row(self.pawns[player]), col(self.pawns[player])

    def legal_mask(self) -> list[bool]:
        mask = [False] * NUM_ACTIONS
        if self.winner() is not None:
            return mask

        self._add_pawn_moves(mask)

        assert self.walls_left is not None
        if self.walls_left[self.to_move] > 0:
            for a in range(WALL_ANCHORS):
                if self.can_place_wall(HORIZONTAL, a):
                    mask[12 + a] = True
                if self.can_place_wall(VERTICAL, a):
                    mask[76 + a] = True
        return mask

    def legal_actions(self) -> list[int]:
        return [idx for idx, legal in enumerate(self.legal_mask()) if legal]

    def can_place_wall(self, orientation: str, anchor_idx: int) -> bool:
        assert self.walls_left is not None
        if self.walls_left[self.to_move] <= 0:
            return False
        if not (0 <= anchor_idx < WALL_ANCHORS):
            return False
        if wall_conflicts(self.hwalls, self.vwalls, orientation, anchor_idx):
            return False

        new_hwalls = self.hwalls
        new_vwalls = self.vwalls
        if orientation == HORIZONTAL:
            new_hwalls |= bit(anchor_idx)
        elif orientation == VERTICAL:
            new_vwalls |= bit(anchor_idx)
        else:
            raise ValueError(f"bad wall orientation `{orientation}`")

        return (
            shortest_path_len_bits(new_hwalls, new_vwalls, self.pawns[0], 0) is not None
            and shortest_path_len_bits(new_hwalls, new_vwalls, self.pawns[1], 1) is not None
        )

    def apply(self, action: int) -> None:
        if action < 0 or action >= NUM_ACTIONS or not self.legal_mask()[action]:
            raise ValueError(f"illegal action {action}")
        self.apply_known_legal(action)

    def apply_known_legal(self, action: int) -> None:
        """Apply an action the caller has already obtained from legal_actions."""

        if action < 12:
            dest = self.pawn_destination_for_action(action)
            assert dest is not None
            self.pawns[self.to_move] = dest
        elif action < 76:
            self.hwalls |= bit(action - 12)
            assert self.walls_left is not None
            self.walls_left[self.to_move] -= 1
        else:
            self.vwalls |= bit(action - 76)
            assert self.walls_left is not None
            self.walls_left[self.to_move] -= 1

        self.to_move ^= 1
        self.ply += 1

    def pawn_destination_for_action(self, action: int) -> int | None:
        if not (0 <= action < 12):
            return None
        dr, dc = PAWN_DELTAS[action]
        p = self.pawns[self.to_move]
        nr, nc = row(p) + dr, col(p) + dc
        return cell(nr, nc) if in_board(nr, nc) else None

    def shortest_path_len(self, player: int) -> int | None:
        if player not in (0, 1):
            raise ValueError("player must be 0 or 1")
        return shortest_path_len_bits(self.hwalls, self.vwalls, self.pawns[player], player)

    def mirror_lr(self) -> "ReferenceState":
        """Return the exact left-right mirror of this absolute position."""

        assert self.walls_left is not None
        return ReferenceState(
            pawns=[mirror_cell_lr(self.pawns[0]), mirror_cell_lr(self.pawns[1])],
            hwalls=mirror_wall_bits_lr(self.hwalls),
            vwalls=mirror_wall_bits_lr(self.vwalls),
            walls_left=self.walls_left.copy(),
            to_move=self.to_move,
            ply=self.ply,
        )

    def _add_pawn_moves(self, mask: list[bool]) -> None:
        pawn_idx = self.pawns[self.to_move]
        opponent = self.pawns[self.to_move ^ 1]
        for direction, simple_action, jump_action in (
            (DIR_N, 0, 4),
            (DIR_S, 1, 5),
            (DIR_E, 2, 6),
            (DIR_W, 3, 7),
        ):
            adjacent = neighbor_if_open_bits(self.hwalls, self.vwalls, pawn_idx, direction)
            if adjacent is None:
                continue
            if adjacent != opponent:
                mask[simple_action] = True
                continue

            straight_jump = neighbor_if_open_bits(self.hwalls, self.vwalls, opponent, direction)
            if straight_jump is not None:
                mask[jump_action] = True
            else:
                for diag_direction, diag_action in diagonal_options(direction):
                    if neighbor_if_open_bits(self.hwalls, self.vwalls, opponent, diag_direction) is not None:
                        mask[diag_action] = True


def diagonal_options(direction: int) -> tuple[tuple[int, int], tuple[int, int]]:
    if direction == DIR_N:
        return ((DIR_E, 8), (DIR_W, 9))
    if direction == DIR_S:
        return ((DIR_E, 10), (DIR_W, 11))
    if direction == DIR_E:
        return ((DIR_N, 8), (DIR_S, 10))
    if direction == DIR_W:
        return ((DIR_N, 9), (DIR_S, 11))
    raise ValueError(f"bad direction {direction}")


def neighbor_if_open_bits(hwalls: int, vwalls: int, from_cell: int, direction: int) -> int | None:
    nxt = EDGE_NEIGHBORS[from_cell][direction]
    if nxt == NO_CELL:
        return None
    if (hwalls & EDGE_HMASKS[from_cell][direction]) != 0:
        return None
    if (vwalls & EDGE_VMASKS[from_cell][direction]) != 0:
        return None
    return nxt


def edge_blocked_bits(hwalls: int, vwalls: int, from_cell: int, to_cell: int) -> bool:
    fr, fc = row(from_cell), col(from_cell)
    tr, tc = row(to_cell), col(to_cell)
    dr, dc = tr - fr, tc - fc
    if abs(dr) + abs(dc) != 1:
        return True

    if dr == 1:
        return has_anchor(hwalls, fr, fc - 1) or has_anchor(hwalls, fr, fc)
    if dr == -1:
        return has_anchor(hwalls, tr, fc - 1) or has_anchor(hwalls, tr, fc)
    if dc == 1:
        return has_anchor(vwalls, fr - 1, fc) or has_anchor(vwalls, fr, fc)
    if dc == -1:
        return has_anchor(vwalls, fr - 1, tc) or has_anchor(vwalls, fr, tc)
    raise AssertionError("unreachable cardinal edge case")


@lru_cache(maxsize=1_000_000)
def shortest_path_len_bits(hwalls: int, vwalls: int, start: int, player: int) -> int | None:
    goal_row = BOARD - 1 if player == 0 else 0
    queue = [0] * CELLS
    dist = [0] * CELLS
    seen = [False] * CELLS
    head = 0
    tail = 1
    queue[0] = start
    seen[start] = True

    while head < tail:
        current = queue[head]
        head += 1
        if row(current) == goal_row:
            return dist[current]
        for direction in range(4):
            nxt = neighbor_if_open_bits(hwalls, vwalls, current, direction)
            if nxt is not None and not seen[nxt]:
                seen[nxt] = True
                dist[nxt] = dist[current] + 1
                queue[tail] = nxt
                tail += 1
    return None


def wall_conflicts(hwalls: int, vwalls: int, orientation: str, anchor_idx: int) -> bool:
    if not (0 <= anchor_idx < WALL_ANCHORS):
        return True

    if orientation == HORIZONTAL:
        return (hwalls & H_CONFLICT_H[anchor_idx]) != 0 or (vwalls & H_CONFLICT_V[anchor_idx]) != 0
    if orientation == VERTICAL:
        return (hwalls & V_CONFLICT_H[anchor_idx]) != 0 or (vwalls & V_CONFLICT_V[anchor_idx]) != 0
    raise ValueError(f"bad wall orientation `{orientation}`")


def all_walls_non_conflicting(hwalls: int, vwalls: int) -> bool:
    for a in iter_wall_anchors(hwalls):
        if wall_conflicts(hwalls & ~bit(a), vwalls, HORIZONTAL, a):
            return False
    for a in iter_wall_anchors(vwalls):
        if wall_conflicts(hwalls, vwalls & ~bit(a), VERTICAL, a):
            return False
    return True


def mirror_cell_lr(cell_idx: int) -> int:
    return cell(row(cell_idx), BOARD - 1 - col(cell_idx))


def mirror_anchor_lr(anchor_idx: int) -> int:
    return anchor(anchor_row(anchor_idx), WALL_GRID - 1 - anchor_col(anchor_idx))


def mirror_wall_bits_lr(bits: int) -> int:
    out = 0
    for a in iter_wall_anchors(bits):
        out |= bit(mirror_anchor_lr(a))
    return out


def wall_action(orientation: str, anchor_coord: str) -> int:
    a = coord_to_anchor(anchor_coord)
    if orientation == HORIZONTAL:
        return 12 + a
    if orientation == VERTICAL:
        return 76 + a
    raise ValueError(f"bad wall orientation `{orientation}`")


def perft(state: ReferenceState, depth: int, memo: dict[tuple[tuple[int, ...], int], int] | None = None) -> int:
    """Count legal move sequences of exactly ``depth`` plies from ``state``."""

    if depth == 0:
        return 1
    if memo is None:
        memo = {}
    key = (state.key(), depth)
    if key in memo:
        return memo[key]

    total = 0
    for action in state.legal_actions():
        child = state.clone()
        child.apply_known_legal(action)
        total += perft(child, depth - 1, memo)
    memo[key] = total
    return total


def build_edge_tables() -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    neighbors = [[NO_CELL for _ in range(4)] for _ in range(CELLS)]
    hmasks = [[0 for _ in range(4)] for _ in range(CELLS)]
    vmasks = [[0 for _ in range(4)] for _ in range(CELLS)]

    def add_h(mask: int, r: int, c: int) -> int:
        return mask | bit(anchor(r, c)) if in_anchor_grid(r, c) else mask

    def add_v(mask: int, r: int, c: int) -> int:
        return add_h(mask, r, c)

    for r in range(BOARD):
        for c in range(BOARD):
            idx = cell(r, c)
            for direction, (dr, dc) in enumerate(DIRS):
                nr, nc = r + dr, c + dc
                if not in_board(nr, nc):
                    continue
                neighbors[idx][direction] = cell(nr, nc)
                hmask = 0
                vmask = 0
                if direction == DIR_N:
                    hmask = add_h(hmask, r - 1, c - 1)
                    hmask = add_h(hmask, r - 1, c)
                elif direction == DIR_S:
                    hmask = add_h(hmask, r, c - 1)
                    hmask = add_h(hmask, r, c)
                elif direction == DIR_E:
                    vmask = add_v(vmask, r - 1, c)
                    vmask = add_v(vmask, r, c)
                elif direction == DIR_W:
                    vmask = add_v(vmask, r - 1, c - 1)
                    vmask = add_v(vmask, r, c - 1)
                hmasks[idx][direction] = hmask
                vmasks[idx][direction] = vmask

    return neighbors, hmasks, vmasks


def build_wall_conflict_tables() -> tuple[list[int], list[int], list[int], list[int]]:
    h_conflict_h = [0] * WALL_ANCHORS
    h_conflict_v = [0] * WALL_ANCHORS
    v_conflict_h = [0] * WALL_ANCHORS
    v_conflict_v = [0] * WALL_ANCHORS

    for a in range(WALL_ANCHORS):
        r, c = anchor_row(a), anchor_col(a)

        hmask = bit(a)
        if c > 0:
            hmask |= bit(anchor(r, c - 1))
        if c + 1 < WALL_GRID:
            hmask |= bit(anchor(r, c + 1))
        h_conflict_h[a] = hmask
        h_conflict_v[a] = bit(a)

        vmask = bit(a)
        if r > 0:
            vmask |= bit(anchor(r - 1, c))
        if r + 1 < WALL_GRID:
            vmask |= bit(anchor(r + 1, c))
        v_conflict_h[a] = bit(a)
        v_conflict_v[a] = vmask

    return h_conflict_h, h_conflict_v, v_conflict_h, v_conflict_v


EDGE_NEIGHBORS, EDGE_HMASKS, EDGE_VMASKS = build_edge_tables()
H_CONFLICT_H, H_CONFLICT_V, V_CONFLICT_H, V_CONFLICT_V = build_wall_conflict_tables()

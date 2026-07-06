"""Per-checkpoint visual artifacts for the Quoridor AlphaZero run.

Public contract (frozen, see CONTRACTS.md):

    generate_checkpoint_artifacts(run_dir: Path, ckpt_step: int, config: Config) -> None

Writes ``runs/<run_id>/viz/ckpt_<step>/`` (opening heatmaps, value-calibration plot,
sample-game SVG strips, concept-probe report + a per-checkpoint page) and refreshes
``runs/<run_id>/index.html`` -- the chronological "watch the meta evolve" page.

Everything is dark-background / light-foreground and self-contained (no external assets),
so ``python -m http.server`` in the run dir serves it directly.

Also runnable standalone::

    python -m trainer.viz --run runs/<id> --step N [--sims 64 --games 4 --plies 40]

The board SVG is hand-written (no extra deps); calibration uses matplotlib if present and
soft-fails to a placeholder otherwise. Every artifact is wrapped so a single failure
degrades to a placeholder card rather than taking down the whole page -- viz is derived,
abandonable data (SPEC 7.3).
"""

from __future__ import annotations

import argparse
import html
import json
import math
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

from .config import Config

try:  # engine is a hard dependency of viz, but keep import guarded for tooling
    import quoridor_engine as qe  # type: ignore

    _ENGINE_OK = True
    _ENGINE_ERR: Optional[str] = None
except Exception as exc:  # pragma: no cover - engine should be built
    qe = None  # type: ignore
    _ENGINE_OK = False
    _ENGINE_ERR = f"{type(exc).__name__}: {exc}"


Evaluator = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]

# ---------------------------------------------------------------------------
# Constants: geometry, action encoding, palette
# ---------------------------------------------------------------------------

BOARD = 9
NUM_ACTIONS = 140
ANCHOR_W = 8

# Absolute pawn-move deltas (row, col), indices 0..11 -- must match engine/movegen.rs.
PAWN_DELTAS: tuple[tuple[int, int], ...] = (
    (-1, 0), (1, 0), (0, 1), (0, -1),      # N, S, E, W  (engine naming)
    (-2, 0), (2, 0), (0, 2), (0, -2),      # jumps
    (-1, 1), (-1, -1), (1, 1), (1, -1),    # diagonals
)

# Dark theme palette (light foreground on dark background everywhere).
BG = "#0d1117"
PANEL = "#161b22"
PANEL_2 = "#1c2430"
GRID = "#39424e"
CELL = "#141b24"
INK = "#e6edf3"
MUTED = "#8b98a9"
FAINT = "#5b6675"
P0 = "#54c7ec"   # player 0 pawn (cyan)
P1 = "#ff9e64"   # player 1 pawn (orange)
MOVE_HUE = "#3fb950"   # pawn-move policy (green)
WALL_HUE = "#a371f7"   # wall policy ghost (violet)
WALL_SOLID = "#c9a15a"  # physically-placed walls (tan)
POS_VAL = "#3fb950"
NEG_VAL = "#f85149"

_MIN_PROB = 0.012  # below this a policy component is not drawn (declutter)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _esc(text: Any) -> str:
    return html.escape(str(text), quote=True)


def _file_label(col: int) -> str:
    return chr(ord("a") + col)


def _anchor_rc(label: str) -> tuple[int, int]:
    """'b3' -> (row=2, col=1) in the 8x8 anchor grid."""
    col = ord(label[0]) - ord("a")
    row = int(label[1:]) - 1
    return row, col


def _parse_walls(state: "qe.GameState") -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Return (h_anchors, v_anchors) parsed from the engine notation string."""
    h: list[tuple[int, int]] = []
    v: list[tuple[int, int]] = []
    for tok in state.to_string().split():
        if tok.startswith("h=") and tok[2:] != "-":
            for a in tok[2:].split(","):
                if a:
                    h.append(_anchor_rc(a))
        elif tok.startswith("v=") and tok[2:] != "-":
            for a in tok[2:].split(","):
                if a:
                    v.append(_anchor_rc(a))
    return h, v


def _clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


# ---------------------------------------------------------------------------
# Board geometry -> SVG
# ---------------------------------------------------------------------------

@dataclass
class Geo:
    ox: float
    oy: float
    cell: float

    @property
    def size(self) -> float:
        return BOARD * self.cell

    def cx(self, col: int) -> float:
        return self.ox + col * self.cell

    def cy(self, row: int) -> float:
        # row 0 (rank 1) at the bottom -> screen y grows downward
        return self.oy + (BOARD - 1 - row) * self.cell

    def center(self, row: int, col: int) -> tuple[float, float]:
        return self.cx(col) + self.cell / 2, self.cy(row) + self.cell / 2


def _rect(x, y, w, h, fill, *, stroke=None, sw=1.0, rx=0.0, opacity=1.0) -> str:
    s = (
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
        f'fill="{fill}" opacity="{opacity:.3f}"'
    )
    if rx:
        s += f' rx="{rx:.2f}" ry="{rx:.2f}"'
    if stroke:
        s += f' stroke="{stroke}" stroke-width="{sw:.2f}"'
    return s + "/>"


def _text(x, y, s, *, fill=INK, size=12, anchor="middle", weight="400", opacity=1.0, family="monospace") -> str:
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" fill="{fill}" font-size="{size}" '
        f'font-family="{family}" font-weight="{weight}" opacity="{opacity:.3f}" '
        f'text-anchor="{anchor}" dominant-baseline="middle">{_esc(s)}</text>'
    )


def _wall_rect(geo: Geo, orientation: str, row: int, col: int, fill: str, opacity: float, rx: float) -> str:
    t = geo.cell * 0.17
    if orientation == "h":
        y = geo.cy(row) - t / 2
        x = geo.cx(col)
        return _rect(x, y, 2 * geo.cell, t, fill, opacity=opacity, rx=rx)
    # vertical
    x = geo.cx(col + 1) - t / 2
    y = geo.cy(row + 1)
    return _rect(x, y, t, 2 * geo.cell, fill, opacity=opacity, rx=rx)


def _arrow(x1, y1, x2, y2, color, width, opacity) -> str:
    ang = math.atan2(y2 - y1, x2 - x1)
    # stop the shaft a little short so the head sits cleanly
    hl = max(6.0, width * 2.6)
    sx, sy = x2 - hl * 0.9 * math.cos(ang), y2 - hl * 0.9 * math.sin(ang)
    left = (x2 - hl * math.cos(ang - 0.5), y2 - hl * math.sin(ang - 0.5))
    right = (x2 - hl * math.cos(ang + 0.5), y2 - hl * math.sin(ang + 0.5))
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{sx:.2f}" y2="{sy:.2f}" '
        f'stroke="{color}" stroke-width="{width:.2f}" stroke-linecap="round" opacity="{opacity:.3f}"/>'
        f'<polygon points="{x2:.2f},{y2:.2f} {left[0]:.2f},{left[1]:.2f} {right[0]:.2f},{right[1]:.2f}" '
        f'fill="{color}" opacity="{opacity:.3f}"/>'
    )


def _board_group(
    state: "qe.GameState",
    geo: Geo,
    *,
    policy: Optional[np.ndarray] = None,
    mode: str = "full",      # 'full' | 'topk'
    top_k: int = 3,
    labels: bool = True,
    annotate: bool = True,
) -> str:
    """Render one board (with optional policy overlay) as an SVG group string."""
    parts: list[str] = ["<g>"]
    n = BOARD
    c = geo.cell

    # board backing
    parts.append(_rect(geo.ox, geo.oy, geo.size, geo.size, CELL, rx=c * 0.12))

    # goal-row tints (p0 -> row 8 top, p1 -> row 0 bottom)
    parts.append(_rect(geo.cx(0), geo.cy(n - 1), geo.size, c, P0, opacity=0.10))
    parts.append(_rect(geo.cx(0), geo.cy(0), geo.size, c, P1, opacity=0.10))

    # cells
    for row in range(n):
        for col in range(n):
            parts.append(
                _rect(geo.cx(col), geo.cy(row), c, c, CELL, stroke=GRID, sw=1.0)
            )

    # coordinate labels
    if labels:
        for col in range(n):
            x = geo.cx(col) + c / 2
            parts.append(_text(x, geo.oy + geo.size + c * 0.32, _file_label(col), fill=MUTED, size=c * 0.26))
        for row in range(n):
            y = geo.cy(row) + c / 2
            parts.append(_text(geo.ox - c * 0.32, y, str(row + 1), fill=MUTED, size=c * 0.26))

    # placed walls
    hwalls, vwalls = _parse_walls(state)
    for (r, cc) in hwalls:
        parts.append(_wall_rect(geo, "h", r, cc, WALL_SOLID, 0.95, rx=c * 0.05))
    for (r, cc) in vwalls:
        parts.append(_wall_rect(geo, "v", r, cc, WALL_SOLID, 0.95, rx=c * 0.05))

    # policy overlay
    if policy is not None:
        parts.append(_policy_overlay(state, geo, policy, mode, top_k, annotate))

    # pawns
    for player, color in ((0, P0), (1, P1)):
        pr, pc = state.pawn(player)
        px, py = geo.center(pr, pc)
        moving = player == state.to_move
        parts.append(
            f'<circle cx="{px:.2f}" cy="{py:.2f}" r="{c*0.30:.2f}" fill="{color}" '
            f'stroke="{INK if moving else BG}" stroke-width="{2.2 if moving else 1.2:.2f}"/>'
        )
        parts.append(_text(px, py, f"P{player}", fill=BG, size=c * 0.24, weight="700"))

    parts.append("</g>")
    return "".join(parts)


def _policy_overlay(
    state: "qe.GameState",
    geo: Geo,
    policy: np.ndarray,
    mode: str,
    top_k: int,
    annotate: bool,
) -> str:
    p = np.asarray(policy, dtype=np.float64).reshape(-1)
    total = float(p.sum())
    if total <= 0:
        return ""
    probs = p / total
    pmax = float(probs.max())
    if pmax <= 0:
        return ""

    if mode == "topk":
        order = np.argsort(probs)[::-1]
        chosen = [int(a) for a in order[:top_k] if probs[int(a)] > 0]
    else:
        chosen = [int(a) for a in range(NUM_ACTIONS) if probs[a] >= _MIN_PROB]

    c = geo.cell
    pr, pc = state.pawn(state.to_move)
    src = geo.center(pr, pc)
    out: list[str] = ["<g>"]

    for a in chosen:
        prob = float(probs[a])
        alpha = 0.20 + 0.72 * _clamp01(prob / pmax)
        if a < 12:
            dr, dc = PAWN_DELTAS[a]
            drow, dcol = pr + dr, pc + dc
            if not (0 <= drow < BOARD and 0 <= dcol < BOARD):
                continue
            out.append(_rect(geo.cx(dcol), geo.cy(drow), c, c, MOVE_HUE, opacity=alpha * 0.5, rx=c * 0.1))
            dx, dy = geo.center(drow, dcol)
            out.append(_arrow(src[0], src[1], dx, dy, MOVE_HUE, width=2.0 + 3.0 * (prob / pmax), opacity=alpha))
            if annotate and prob >= 0.03:
                out.append(_text(dx, dy + c * 0.30, f"{prob*100:.0f}", fill=MOVE_HUE, size=c * 0.20, weight="700"))
        else:
            if a < 76:
                orient, anchor = "h", a - 12
            else:
                orient, anchor = "v", a - 76
            r, cc = anchor // ANCHOR_W, anchor % ANCHOR_W
            out.append(_wall_rect(geo, orient, r, cc, WALL_HUE, alpha, rx=c * 0.06))
            if annotate and prob >= 0.03:
                if orient == "h":
                    lx, ly = geo.cx(cc) + c, geo.cy(r)
                else:
                    lx, ly = geo.cx(cc + 1), geo.cy(r + 1) + c
                out.append(_text(lx, ly, f"{prob*100:.0f}", fill=WALL_HUE, size=c * 0.19, weight="700"))
    out.append("</g>")
    return "".join(out)


def _value_bar(x: float, y: float, w: float, value: float, *, height: float = 8.0) -> str:
    v = max(-1.0, min(1.0, float(value)))
    mid = x + w / 2
    half = w / 2
    parts = [_rect(x, y, w, height, PANEL_2, rx=height / 2, stroke=GRID, sw=0.8)]
    if v >= 0:
        parts.append(_rect(mid, y, half * v, height, POS_VAL, rx=height / 2))
    else:
        parts.append(_rect(mid + half * v, y, -half * v, height, NEG_VAL, rx=height / 2))
    parts.append(f'<line x1="{mid:.2f}" y1="{y-1:.2f}" x2="{mid:.2f}" y2="{y+height+1:.2f}" stroke="{FAINT}" stroke-width="1"/>')
    return "".join(parts)


def _svg_doc(width: float, height: float, body: str, *, bg: str = BG) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {height:.0f}" '
        f'width="{width:.0f}" height="{height:.0f}" font-family="ui-monospace, monospace">'
        f'{_rect(0, 0, width, height, bg)}{body}</svg>'
    )


# ---------------------------------------------------------------------------
# Heatmap: one full board + policy + legend
# ---------------------------------------------------------------------------

def render_heatmap_svg(state: "qe.GameState", policy: np.ndarray, title: str, subtitle: str) -> str:
    cell = 48.0
    pad_l = 34.0
    pad_top = 54.0
    board = BOARD * cell
    width = pad_l + board + 22
    legend_h = 74.0
    height = pad_top + board + 26 + legend_h

    geo = Geo(pad_l, pad_top, cell)
    body: list[str] = []
    body.append(_text(pad_l, 22, title, fill=INK, size=17, anchor="start", weight="700"))
    body.append(_text(pad_l, 40, subtitle, fill=MUTED, size=12, anchor="start"))
    body.append(_board_group(state, geo, policy=policy, mode="full", labels=True, annotate=True))

    # legend
    ly = pad_top + board + 30
    lx = pad_l
    body.append(_text(lx, ly, "policy", fill=MUTED, size=11, anchor="start", weight="700"))
    body.append(_rect(lx, ly + 10, 16, 12, MOVE_HUE, opacity=0.85, rx=2))
    body.append(_text(lx + 22, ly + 16, "pawn move (cell + arrow)", fill=INK, size=11, anchor="start"))
    body.append(_rect(lx, ly + 30, 16, 12, WALL_HUE, opacity=0.85, rx=2))
    body.append(_text(lx + 22, ly + 36, "wall placement (groove glyph)", fill=INK, size=11, anchor="start"))
    body.append(f'<circle cx="{lx+8:.1f}" cy="{ly+56:.1f}" r="6" fill="{P0}"/>')
    body.append(_text(lx + 22, ly + 56, "P0 -> rank 9", fill=INK, size=11, anchor="start"))
    body.append(f'<circle cx="{lx+150:.1f}" cy="{ly+56:.1f}" r="6" fill="{P1}"/>')
    body.append(_text(lx + 164, ly + 56, "P1 -> rank 1", fill=INK, size=11, anchor="start"))
    body.append(_text(width - 12, ly + 56, "opacity ~ probability", fill=FAINT, size=10, anchor="end"))

    return _svg_doc(width, height, "".join(body))


# ---------------------------------------------------------------------------
# Sample-game strip
# ---------------------------------------------------------------------------

def render_game_strip_svg(frames: list[dict], *, outcome_label: str) -> str:
    cell = 22.0
    board = BOARD * cell
    tile_w = board + 16
    tile_h = board + 44
    per_row = 8
    n = len(frames)
    rows = max(1, math.ceil(n / per_row))
    width = per_row * tile_w + 24
    header = 34.0
    height = header + rows * (tile_h + 12) + 12

    body: list[str] = []
    body.append(_text(14, 22, f"self-play game  |  {outcome_label}  |  {n} plies shown", fill=INK, size=14, anchor="start", weight="700"))

    for i, fr in enumerate(frames):
        r, cidx = divmod(i, per_row)
        tx = 12 + cidx * tile_w
        ty = header + r * (tile_h + 12)
        body.append(_rect(tx, ty, tile_w, tile_h, PANEL, rx=6, stroke=PANEL_2, sw=1))
        geo = Geo(tx + 8, ty + 6, cell)
        state = qe.GameState.from_string(fr["state"])
        body.append(_board_group(state, geo, policy=fr["visits"], mode="topk", top_k=3, labels=False, annotate=False))
        # move label + value bar
        mover = state.to_move
        body.append(_text(tx + 8, ty + board + 16, f"ply {fr['ply']}  P{mover}", fill=MUTED, size=10, anchor="start"))
        body.append(_value_bar(tx + 8, ty + board + 24, board, fr["value"], height=7))
        body.append(_text(tx + tile_w - 6, ty + board + 16, f"v={fr['value']:+.2f}", fill=INK, size=10, anchor="end"))
    return _svg_doc(width, height, "".join(body), bg=BG)


# ---------------------------------------------------------------------------
# MCTS + evaluator plumbing
# ---------------------------------------------------------------------------

def _make_evaluator(ckpt_dir: Path, config: Config) -> tuple[Evaluator, str]:
    """Return (evaluator, source_label). Falls back to stub_eval on any model failure."""
    try:
        import torch  # noqa: F811

        from .model import Inferencer, load_model

        device = torch.device("cpu")
        model = load_model(ckpt_dir, config, device)
        inf = Inferencer(model, config)

        def ev(planes: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return inf.infer(np.asarray(planes, dtype=np.float32), masks)

        return ev, "model"
    except Exception as exc:  # noqa: BLE001 - fall back to the Rust stub evaluator
        note = f"stub_eval ({type(exc).__name__})"

        def ev(planes: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            logits, values = qe.stub_eval(np.asarray(planes, dtype=np.float32))
            return np.asarray(logits, dtype=np.float32), np.asarray(values, dtype=np.float32)

        return ev, note


def _run_mcts(engine_cfg: "qe.EngineConfig", state: "qe.GameState", evaluator: Evaluator, seed: int) -> np.ndarray:
    """Run a single-game batched MCTS to completion and return absolute root visit counts."""
    m = qe.BatchedMcts(engine_cfg, state, seed=seed)
    guard = 0
    while not m.ready() and guard < 200000:
        guard += 1
        planes, leaf_ids, masks = m.collect_leaves(1024, 2)
        ids = np.asarray(leaf_ids, dtype=np.int64)
        if ids.shape[0] == 0:
            continue
        logits, values = evaluator(np.asarray(planes, dtype=np.float32), np.asarray(masks))
        m.submit(ids, np.ascontiguousarray(logits, dtype=np.float32), np.ascontiguousarray(values, dtype=np.float32))
    return np.asarray(m.root_visits(), dtype=np.float64)


def _predict_value(state: "qe.GameState", evaluator: Evaluator, extra_planes: bool) -> float:
    planes = np.asarray(state.encode(extra_planes), dtype=np.float32)[None, ...]
    masks = np.asarray(state.legal_mask())[None, ...]
    _, values = evaluator(planes, masks)
    return float(np.asarray(values).reshape(-1)[0])


@dataclass
class SelfPlayResult:
    games: list[dict]                 # {frames:[...], outcome:int|None, plies:int}
    calibration: list[tuple[float, float]]  # (predicted_value, z)
    opening_positions: list[str]      # frequent ply-2/ply-4 notation strings


def _selfplay(
    engine_cfg: "qe.EngineConfig",
    evaluator: Evaluator,
    *,
    num_games: int,
    max_plies: int,
    extra_planes: bool,
    temperature_moves: int,
    seed: int,
) -> SelfPlayResult:
    rng = np.random.default_rng(seed)
    games: list[dict] = []
    calibration: list[tuple[float, float]] = []
    opening_counter: Counter[str] = Counter()

    for g in range(num_games):
        state = qe.GameState()
        frames: list[dict] = []
        pending: list[tuple[float, int]] = []  # (pred_value, to_move) per ply
        outcome: Optional[int] = None
        while True:
            if state.winner() is not None:
                outcome = int(state.winner())
                break
            if state.ply >= max_plies:
                outcome = None  # ply-cap draw
                break
            visits = _run_mcts(engine_cfg, state, evaluator, seed=seed * 1009 + g * 31 + state.ply)
            pred_v = _predict_value(state, evaluator, extra_planes)
            frames.append({"state": state.to_string(), "visits": visits, "value": pred_v, "ply": int(state.ply)})
            pending.append((pred_v, int(state.to_move)))
            if state.ply in (2, 4):
                opening_counter[state.to_string()] += 1

            total = visits.sum()
            if total <= 0:
                legal = np.asarray(state.legal_mask()).astype(bool)
                action = int(np.flatnonzero(legal)[0])
            elif state.ply < temperature_moves:
                dist = visits / total
                dist = dist / dist.sum()  # guard against float drift for np.choice
                action = int(rng.choice(NUM_ACTIONS, p=dist))
            else:
                action = int(np.argmax(visits))
            state.apply(action)

        # derive per-position z from the eventual outcome
        for pred_v, tm in pending:
            if outcome is None:
                z = 0.0
            else:
                z = 1.0 if outcome == tm else -1.0
            calibration.append((pred_v, z))

        games.append({"frames": frames, "outcome": outcome, "plies": len(frames)})

    opening_positions = [pos for pos, _ in opening_counter.most_common(8)]
    return SelfPlayResult(games=games, calibration=calibration, opening_positions=opening_positions)


# ---------------------------------------------------------------------------
# Calibration plot (matplotlib, dark) with graceful fallback
# ---------------------------------------------------------------------------

def render_calibration(pairs: Sequence[tuple[float, float]], out_path: Path) -> bool:
    if len(pairs) < 4:
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    preds = np.array([p for p, _ in pairs], dtype=np.float64)
    outs = np.array([z for _, z in pairs], dtype=np.float64)
    edges = np.linspace(-1.0, 1.0, 11)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means: list[float] = []
    counts: list[int] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (preds >= lo) & (preds < hi if hi < 1.0 else preds <= hi)
        counts.append(int(m.sum()))
        means.append(float(outs[m].mean()) if m.any() else np.nan)

    fig, ax = plt.subplots(figsize=(6.2, 5.0), dpi=110)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ax2 = ax.twinx()
    ax2.bar(centers, counts, width=(edges[1] - edges[0]) * 0.9, color=WALL_HUE, alpha=0.22, zorder=1)
    ax2.set_ylabel("samples per bucket", color=MUTED)
    ax2.tick_params(colors=MUTED)
    for sp in ax2.spines.values():
        sp.set_color(GRID)

    ax.plot([-1, 1], [-1, 1], "--", color=FAINT, linewidth=1.2, label="ideal", zorder=2)
    cx = [c for c, mv in zip(centers, means) if not math.isnan(mv)]
    cy = [mv for mv in means if not math.isnan(mv)]
    ax.plot(cx, cy, "-o", color=P0, linewidth=2.0, markersize=6, label="observed", zorder=3)

    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("predicted value (side to move)", color=INK)
    ax.set_ylabel("mean eventual outcome", color=INK)
    ax.set_title("Value calibration", color=INK, fontsize=13)
    ax.tick_params(colors=MUTED)
    ax.grid(True, color=GRID, alpha=0.3)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    leg = ax.legend(loc="upper left", facecolor=PANEL, edgecolor=GRID)
    for t in leg.get_texts():
        t.set_color(INK)

    fig.tight_layout()
    fig.savefig(out_path, facecolor=BG)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Concept-probe report (soft-fail)
# ---------------------------------------------------------------------------

def _run_probes(run_dir: Path, ckpt_step: int, config: Config) -> tuple[Optional[dict], str]:
    try:
        from .analyze import run_probe_suite  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return None, f"probe suite unavailable ({type(exc).__name__})"
    try:
        result = run_probe_suite(run_dir, ckpt_step, config)
        return dict(result) if isinstance(result, dict) else {"result": result}, "ok"
    except Exception as exc:  # noqa: BLE001
        return None, f"probe suite raised {type(exc).__name__}: {exc}"


def _probe_rows(result: dict, prev: Optional[dict]) -> list[tuple[str, Optional[bool], Optional[float], Optional[float], str]]:
    """Normalize probe output into (name, passed, value, prev_value, detail) rows.

    Understands trainer/analyze.py's schema (``results`` list with per-probe
    ``assertions``) and falls back to a generic ``probes`` dict/list.
    """
    def _prev_values() -> dict[str, float]:
        out: dict[str, float] = {}
        if isinstance(prev, dict):
            for item in prev.get("results", []) or []:
                if isinstance(item, dict) and isinstance(item.get("value"), (int, float)):
                    out[str(item.get("id"))] = float(item["value"])
        return out

    rows: list[tuple[str, Optional[bool], Optional[float], Optional[float], str]] = []
    results = result.get("results")
    if isinstance(results, list) and results:
        prev_vals = _prev_values()
        for item in results:
            if not isinstance(item, dict):
                continue
            name = str(item.get("id", item.get("theme", "?")))
            assertions = [a for a in item.get("assertions", []) if isinstance(a, dict)]
            flags = [a.get("passed") for a in assertions if isinstance(a.get("passed"), bool)]
            passed: Optional[bool] = all(flags) if flags else None
            value = item.get("value")
            cur = float(value) if isinstance(value, (int, float)) else None
            failed_notes = "; ".join(
                str(a.get("note", a.get("type", ""))) for a in assertions if a.get("passed") is False
            )
            detail = failed_notes or str(item.get("description", ""))
            rows.append((name, passed, cur, prev_vals.get(name), detail))
        return rows

    # generic fallback: {"probes": {...}} / {"probes": [...]} / flat dict
    items = result.get("probes", result)
    iterable = items.items() if isinstance(items, dict) else enumerate(items if isinstance(items, list) else [items])
    for key, val in iterable:
        if isinstance(val, dict):
            name = str(val.get("name", val.get("id", key)))
            passed = val.get("pass", val.get("passed"))
            passed = passed if isinstance(passed, bool) else None
            cur = None
            for k in ("value", "score", "margin"):
                if isinstance(val.get(k), (int, float)):
                    cur = float(val[k])
                    break
            detail = str(val.get("detail", val.get("note", val.get("description", ""))))
        else:
            name, passed, cur, detail = str(key), None, None, str(val)
        rows.append((name, passed, cur, None, detail))
    return rows


def _probe_html(result: Optional[dict], note: str, prev: Optional[dict]) -> str:
    if result is None:
        return (
            '<div class="probe-card unavailable">'
            f'<div class="probe-title">Concept probes unavailable</div>'
            f'<div class="muted">{_esc(note)}</div></div>'
        )
    parts: list[str] = []
    summary = result.get("summary")
    if isinstance(summary, dict):
        sp = summary.get("soft_passes")
        sf = summary.get("soft_failures")
        if isinstance(sp, int) and isinstance(sf, int):
            total = sp + sf
            parts.append(
                f'<div class="muted" style="margin-bottom:10px">soft assertions: '
                f'<span class="badge pass">{sp} pass</span> '
                f'<span class="badge drift">{sf} fail</span> of {total}</div>'
            )

    body_rows: list[str] = []
    for name, passed, cur, prev_val, detail in _probe_rows(result, prev):
        if passed is True:
            badge = '<span class="badge pass">pass</span>'
        elif passed is False:
            badge = '<span class="badge drift">drift</span>'
        else:
            badge = '<span class="badge neutral">info</span>'
        val_txt = "" if cur is None else f"{cur:+.3f}"
        if cur is not None and prev_val is not None:
            val_txt += f' <span class="delta">&Delta; {cur - prev_val:+.3f}</span>'
        body_rows.append(
            f'<tr><td class="mono">{_esc(name)}</td><td>{badge}</td>'
            f'<td class="mono">{val_txt}</td>'
            f'<td class="muted">{_esc(detail)}</td></tr>'
        )
    parts.append(
        '<table class="probe-table"><thead><tr>'
        '<th>probe</th><th>status</th><th>value vs prev</th><th>detail</th>'
        '</tr></thead><tbody>' + "".join(body_rows) + "</tbody></table>"
    )
    return "".join(parts)


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------

_CSS = """
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { margin: 0; background: #0d1117; color: #e6edf3;
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; line-height: 1.5; }
a { color: #54c7ec; text-decoration: none; }
a:hover { text-decoration: underline; }
.wrap { max-width: 1180px; margin: 0 auto; padding: 32px 24px 80px; }
h1 { font-size: 26px; font-weight: 700; margin: 0 0 4px; letter-spacing: -0.01em; }
h2 { font-size: 18px; font-weight: 600; margin: 40px 0 14px; color: #e6edf3;
  border-bottom: 1px solid #21262d; padding-bottom: 8px; }
.sub { color: #8b98a9; font-size: 14px; margin-bottom: 24px; }
.muted { color: #8b98a9; }
.mono { font-family: ui-monospace, monospace; }
.grid { display: grid; gap: 16px; }
.cards { grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); }
.heatgrid { grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); }
.card { background: #161b22; border: 1px solid #21262d; border-radius: 12px;
  overflow: hidden; transition: border-color .15s, transform .15s; }
.card:hover { border-color: #54c7ec; transform: translateY(-2px); }
.card .body { padding: 14px 16px; }
.card .title { font-weight: 600; font-size: 15px; }
.card .meta { color: #8b98a9; font-size: 12.5px; margin-top: 4px; }
.thumb { display: block; width: 100%; background: #0d1117; border-bottom: 1px solid #21262d; }
.tag { display: inline-block; font-family: ui-monospace, monospace; font-size: 11px;
  padding: 2px 8px; border-radius: 999px; background: #1c2430; color: #8b98a9; margin-right: 6px; }
.tag.model { background: #14351f; color: #3fb950; }
.tag.stub { background: #35240f; color: #ff9e64; }
.scroll { overflow-x: auto; border: 1px solid #21262d; border-radius: 12px; background: #0d1117; }
.scroll svg, .heat svg { display: block; }
.heat { background: #0d1117; border: 1px solid #21262d; border-radius: 12px; overflow: hidden; }
img.plot { max-width: 100%; border: 1px solid #21262d; border-radius: 12px; background: #0d1117; }
.probe-table { width: 100%; border-collapse: collapse; font-size: 13.5px; }
.probe-table th { text-align: left; color: #8b98a9; font-weight: 600; padding: 8px 10px;
  border-bottom: 1px solid #21262d; }
.probe-table td { padding: 8px 10px; border-bottom: 1px solid #161b22; vertical-align: top; }
.badge { font-family: ui-monospace, monospace; font-size: 11px; padding: 2px 8px; border-radius: 6px; }
.badge.pass { background: #14351f; color: #3fb950; }
.badge.drift { background: #3d1418; color: #f85149; }
.badge.neutral { background: #1c2430; color: #8b98a9; }
.delta { color: #8b98a9; font-family: ui-monospace, monospace; font-size: 11px; margin-left: 6px; }
.probe-card { background: #161b22; border: 1px solid #21262d; border-radius: 12px; padding: 18px; }
.probe-card.unavailable { border-style: dashed; }
.probe-title { font-weight: 600; margin-bottom: 4px; }
.spark { background: #161b22; border: 1px solid #21262d; border-radius: 12px; padding: 16px 18px; margin-bottom: 8px; }
.placeholder { background: #161b22; border: 1px dashed #30363d; border-radius: 12px;
  padding: 28px; text-align: center; color: #8b98a9; }
footer { margin-top: 60px; color: #5b6675; font-size: 12px; text-align: center; }
"""


def _placeholder(msg: str) -> str:
    return f'<div class="placeholder">{_esc(msg)}</div>'


def _page(title: str, body: str) -> str:
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{_esc(title)}</title><style>{_CSS}</style></head>"
        f"<body><div class=\"wrap\">{body}</div>"
        '<footer>Quoridor AlphaZero &middot; auto-generated viz &middot; dark-on-light-foreground</footer>'
        "</body></html>"
    )


def _write_checkpoint_page(
    viz_dir: Path,
    ckpt_step: int,
    source: str,
    heatmaps: list[tuple[str, str]],
    calib_ok: bool,
    strips: list[str],
    probe_html: str,
) -> None:
    tag = f'<span class="tag {"model" if source == "model" else "stub"}">{_esc(source)}</span>'
    parts: list[str] = []
    parts.append(f"<h1>Checkpoint {ckpt_step}</h1>")
    parts.append(f'<div class="sub">{tag} &middot; <a href="../../index.html">&larr; run overview</a></div>')

    parts.append("<h2>Opening heatmaps</h2>")
    if heatmaps:
        parts.append('<div class="grid heatgrid">')
        for fname, _title in heatmaps:
            parts.append(f'<div class="heat"><img class="thumb" src="{_esc(fname)}" alt="{_esc(_title)}"></div>')
        parts.append("</div>")
    else:
        parts.append(_placeholder("No opening heatmaps were produced."))

    parts.append("<h2>Value calibration</h2>")
    if calib_ok:
        parts.append('<img class="plot" src="calibration.png" alt="value calibration">')
    else:
        parts.append(_placeholder("Not enough self-play value/outcome pairs to plot calibration."))

    parts.append("<h2>Sample self-play games</h2>")
    if strips:
        for s in strips:
            parts.append(f'<div class="scroll"><img src="{_esc(s)}" alt="sample game strip"></div>')
    else:
        parts.append(_placeholder("No sample games were rendered."))

    parts.append("<h2>Concept probes</h2>")
    parts.append(probe_html)

    (viz_dir / "index.html").write_text(_page(f"Checkpoint {ckpt_step}", "".join(parts)), encoding="utf-8")


# ---------------------------------------------------------------------------
# Elo sparkline (best-effort parse of elo.json)
# ---------------------------------------------------------------------------

def _load_elo(run_dir: Path) -> list[tuple[int, float]]:
    path = run_dir / "elo.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    points: list[tuple[int, float]] = []

    def _consume(seq: Any) -> None:
        if not isinstance(seq, list):
            return
        for item in seq:
            if isinstance(item, dict):
                step = item.get("step", item.get("ckpt", item.get("checkpoint")))
                elo = item.get("elo", item.get("rating", item.get("score")))
                if isinstance(step, (int, float)) and isinstance(elo, (int, float)):
                    points.append((int(step), float(elo)))

    if isinstance(data, dict):
        for key in ("ladder", "history", "ratings", "points"):
            if key in data:
                _consume(data[key])
        if not points and all(isinstance(v, (int, float)) for v in data.values()):
            for k, v in data.items():
                try:
                    points.append((int(k), float(v)))
                except (ValueError, TypeError):
                    pass
    elif isinstance(data, list):
        _consume(data)

    points.sort(key=lambda t: t[0])
    return points


def _elo_sparkline_svg(points: list[tuple[int, float]]) -> str:
    if len(points) < 2:
        return ""
    w, h, pad = 640.0, 90.0, 12.0
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    xr = (x1 - x0) or 1
    yr = (y1 - y0) or 1

    def px(x):
        return pad + (x - x0) / xr * (w - 2 * pad)

    def py(y):
        return h - pad - (y - y0) / yr * (h - 2 * pad)

    pts = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in points)
    dots = "".join(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="2.6" fill="{P0}"/>' for x, y in points)
    body = (
        f'<polyline points="{pts}" fill="none" stroke="{P0}" stroke-width="2"/>' + dots +
        _text(pad, py(y1) - 2, f"{y1:.0f}", fill=MUTED, size=10, anchor="start") +
        _text(pad, py(y0) + 8, f"{y0:.0f}", fill=MUTED, size=10, anchor="start")
    )
    return _svg_doc(w, h, body, bg=PANEL)


# ---------------------------------------------------------------------------
# Run index ("watch the meta evolve")
# ---------------------------------------------------------------------------

def refresh_run_index(run_dir: Path) -> None:
    run_dir = Path(run_dir)
    viz_root = run_dir / "viz"
    entries: list[dict] = []
    if viz_root.exists():
        for d in viz_root.iterdir():
            if not (d.is_dir() and d.name.startswith("ckpt_")):
                continue
            try:
                step = int(d.name[len("ckpt_"):])
            except ValueError:
                continue
            meta = {}
            mpath = d / "meta.json"
            if mpath.exists():
                try:
                    meta = json.loads(mpath.read_text(encoding="utf-8"))
                except Exception:
                    meta = {}
            thumb = None
            for cand in ("heatmap_00_start.svg", "index.html"):
                if (d / cand).exists() and cand.endswith(".svg"):
                    thumb = f"viz/{d.name}/{cand}"
                    break
            entries.append({"step": step, "dir": d.name, "meta": meta, "thumb": thumb})
    entries.sort(key=lambda e: e["step"])

    elo = _load_elo(run_dir)
    elo_by_step = {s: v for s, v in elo}

    parts: list[str] = []
    parts.append(f"<h1>{_esc(run_dir.name)}</h1>")
    parts.append(
        '<div class="sub">Watch the opening meta evolve &mdash; one card per checkpoint, '
        "newest ideas as training progresses.</div>"
    )

    spark = _elo_sparkline_svg(elo)
    if spark:
        parts.append(f'<div class="spark"><div class="muted" style="margin-bottom:8px">Elo ladder</div>{spark}</div>')

    parts.append("<h2>Checkpoints</h2>")
    if entries:
        parts.append('<div class="grid cards">')
        for e in reversed(entries):  # newest first
            meta = e["meta"]
            source = meta.get("source", "?")
            src_cls = "model" if source == "model" else "stub"
            when = meta.get("generated_at", "")
            elo_txt = ""
            if e["step"] in elo_by_step:
                elo_txt = f' &middot; Elo {elo_by_step[e["step"]]:.0f}'
            thumb_html = (
                f'<img class="thumb" src="{_esc(e["thumb"])}" alt="checkpoint {e["step"]} opening">'
                if e["thumb"]
                else '<div class="thumb" style="height:120px"></div>'
            )
            link = f'viz/{e["dir"]}/index.html'
            parts.append(
                f'<a class="card" href="{_esc(link)}">{thumb_html}'
                f'<div class="body"><div class="title">Checkpoint {e["step"]}{elo_txt}</div>'
                f'<div class="meta"><span class="tag {src_cls}">{_esc(source)}</span>'
                f'{_esc(when)}</div></div></a>'
            )
        parts.append("</div>")
    else:
        parts.append(_placeholder("No checkpoint artifacts yet."))

    (run_dir / "index.html").write_text(_page(run_dir.name, "".join(parts)), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_checkpoint_artifacts(
    run_dir: Path,
    ckpt_step: int,
    config: Config,
    *,
    sims: Optional[int] = None,
    selfplay_sims: Optional[int] = None,
    num_games: Optional[int] = None,
    max_plies: Optional[int] = None,
    seed: int = 0,
) -> None:
    """Generate all per-checkpoint visual artifacts and refresh the run index.

    Contract-frozen positional signature ``(run_dir, ckpt_step, config)``; the trailing
    keyword-only args exist purely to let the standalone CLI and tests dial down the
    compute. Individual artifacts soft-fail to placeholders; the run index is always
    refreshed so a partial checkpoint still shows up.
    """
    run_dir = Path(run_dir)
    ckpt_dir = run_dir / "ckpt" / str(ckpt_step)
    viz_dir = run_dir / "viz" / f"ckpt_{ckpt_step}"
    viz_dir.mkdir(parents=True, exist_ok=True)

    if not _ENGINE_OK:
        (viz_dir / "index.html").write_text(
            _page(f"Checkpoint {ckpt_step}", _placeholder(f"quoridor_engine unavailable: {_ENGINE_ERR}")),
            encoding="utf-8",
        )
        refresh_run_index(run_dir)
        return

    heat_sims = int(sims if sims is not None else config.mcts.sims_per_move)
    sp_sims = int(selfplay_sims if selfplay_sims is not None else max(1, min(config.mcts.fast_sims_per_move, 200)))
    n_games = int(num_games if num_games is not None else 6)
    plies_cap = int(max_plies if max_plies is not None else min(config.rules.ply_cap, 80))
    extra = bool(config.net.extra_planes)

    evaluator, source = _make_evaluator(ckpt_dir, config)

    engine_kwargs = config.to_engine_kwargs()
    heat_cfg = qe.EngineConfig(**{**engine_kwargs, "sims_per_move": heat_sims})
    sp_cfg = qe.EngineConfig(**{**engine_kwargs, "sims_per_move": sp_sims})

    heatmaps: list[tuple[str, str]] = []
    strips: list[str] = []
    calib_ok = False

    # --- self-play pass: mining + sample games + calibration ---------------
    try:
        sp = _selfplay(
            sp_cfg,
            evaluator,
            num_games=n_games,
            max_plies=plies_cap,
            extra_planes=extra,
            temperature_moves=max(4, config.eval.opening_temperature_moves),
            seed=seed + ckpt_step,
        )
    except Exception:
        traceback.print_exc()
        sp = SelfPlayResult(games=[], calibration=[], opening_positions=[])

    # --- opening heatmaps ---------------------------------------------------
    try:
        positions: list[tuple[str, str, str]] = []  # (notation, title, subtitle)
        start = qe.GameState()
        positions.append((start.to_string(), "Start position", "root policy, no noise"))
        for i, notation in enumerate(sp.opening_positions):
            try:
                st = qe.GameState.from_string(notation)
            except Exception:
                continue
            positions.append((notation, f"Frequent opening #{i+1}", f"ply {st.ply}, P{st.to_move} to move"))

        for idx, (notation, title, subtitle) in enumerate(positions):
            st = qe.GameState.from_string(notation)
            if st.winner() is not None:
                continue
            visits = _run_mcts(heat_cfg, st, evaluator, seed=seed + ckpt_step * 7 + idx)
            svg = render_heatmap_svg(st, visits, f"{title}  ({heat_sims} sims)", subtitle)
            label = "start" if idx == 0 else f"pos{idx}"
            fname = f"heatmap_{idx:02d}_{label}.svg"
            (viz_dir / fname).write_text(svg, encoding="utf-8")
            heatmaps.append((fname, title))
    except Exception:
        traceback.print_exc()

    # --- sample-game strips -------------------------------------------------
    try:
        for gi, game in enumerate(sp.games[:5]):
            frames = game["frames"][:48]
            if not frames:
                continue
            outcome = game["outcome"]
            if outcome is None:
                label = f"draw (ply cap {plies_cap})"
            else:
                label = f"P{outcome} wins in {game['plies']} plies"
            svg = render_game_strip_svg(frames, outcome_label=label)
            fname = f"game_{gi:02d}.svg"
            (viz_dir / fname).write_text(svg, encoding="utf-8")
            strips.append(fname)
    except Exception:
        traceback.print_exc()

    # --- calibration --------------------------------------------------------
    try:
        calib_ok = render_calibration(sp.calibration, viz_dir / "calibration.png")
    except Exception:
        traceback.print_exc()
        calib_ok = False

    # --- concept probes -----------------------------------------------------
    probe_result, probe_note = _run_probes(run_dir, ckpt_step, config)
    if probe_result is not None:
        try:
            (viz_dir / "probes.json").write_text(json.dumps(probe_result, indent=2, default=str), encoding="utf-8")
        except Exception:
            pass
    prev = _load_prev_probes(run_dir, ckpt_step)
    probe_html = _probe_html(probe_result, probe_note, prev)

    # --- pages --------------------------------------------------------------
    _write_checkpoint_page(viz_dir, ckpt_step, source, heatmaps, calib_ok, strips, probe_html)

    meta = {
        "step": ckpt_step,
        "source": source,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "heatmaps": [f for f, _ in heatmaps],
        "games": strips,
        "calibration": calib_ok,
        "probes": probe_result is not None,
        "heat_sims": heat_sims,
    }
    (viz_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    refresh_run_index(run_dir)


def _load_prev_probes(run_dir: Path, ckpt_step: int) -> Optional[dict]:
    viz_root = run_dir / "viz"
    if not viz_root.exists():
        return None
    prev_steps = []
    for d in viz_root.iterdir():
        if d.is_dir() and d.name.startswith("ckpt_"):
            try:
                s = int(d.name[len("ckpt_"):])
            except ValueError:
                continue
            if s < ckpt_step and (d / "probes.json").exists():
                prev_steps.append(s)
    if not prev_steps:
        return None
    prev_dir = viz_root / f"ckpt_{max(prev_steps)}"
    try:
        return json.loads((prev_dir / "probes.json").read_text(encoding="utf-8"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Generate per-checkpoint Quoridor viz artifacts.")
    ap.add_argument("--run", required=True, type=Path, help="run directory (runs/<id>)")
    ap.add_argument("--step", required=True, type=int, help="checkpoint step")
    ap.add_argument("--sims", type=int, default=48, help="MCTS sims per heatmap position")
    ap.add_argument("--selfplay-sims", type=int, default=24, help="MCTS sims during self-play mining")
    ap.add_argument("--games", type=int, default=4, help="self-play games to generate")
    ap.add_argument("--plies", type=int, default=40, help="max plies per self-play game")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    config = Config()
    generate_checkpoint_artifacts(
        args.run,
        args.step,
        config,
        sims=args.sims,
        selfplay_sims=args.selfplay_sims,
        num_games=args.games,
        max_plies=args.plies,
        seed=args.seed,
    )
    print(f"wrote artifacts to {args.run}/viz/ckpt_{args.step}/ and refreshed {args.run}/index.html")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())

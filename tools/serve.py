#!/usr/bin/env python3
"""Local web UI for playing Quoridor against a model checkpoint.

Serves `web/index.html` plus a small JSON API on top of the arena search
stack. Zero dependencies beyond what the trainer already uses.

    python tools/serve.py                 # newest run, latest checkpoint
    python tools/serve.py --run runs/run1 --step 3000 --sims 256
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import secrets
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

import quoridor_engine as qe
from baselines.common import (
    action_to_label,
    evaluate_position,
    format_eval,
    format_net_score,
    parse_wall_sets,
    pawn_destination,
)
from trainer.arena import (
    SearchAgent,
    StubEvaluator,
    evaluate_state,
    load_checkpoint_evaluator,
    load_config_for_cli,
    net_eval_score,
    normalize_policy,
    resolve_step,
    top_actions,
)
from trainer.config import Config

WEB_DIR = ROOT / "web"
MAX_SESSIONS = 32

# One lock for all engine work: the server is a local single-user tool and
# neither the MCTS bindings nor the shared torch model need to run
# concurrently across sessions.
ENGINE_LOCK = threading.Lock()
SESSIONS: dict[str, "Session"] = {}
SESSIONS_LOCK = threading.Lock()

APP: dict[str, Any] = {}


class Session:
    def __init__(self, human: int, sims: int) -> None:
        self.lock = threading.Lock()
        self.human = int(human)
        self.created = time.time()
        self.config: Config = copy.deepcopy(APP["config"])
        self.config.mcts.sims_per_move = max(1, int(sims))
        self.config.selfplay.leaf_batch_size = max(1, int(sims))
        self.state = qe.GameState()
        self.agent = SearchAgent(
            APP["evaluator"], self.config, self.state, seed=secrets.randbits(48)
        )
        self.history: list[dict[str, Any]] = []
        self.search_top: list[dict[str, Any]] = []

    def record(self, by: str, action: int, label: str) -> dict[str, Any]:
        move = {"by": by, "action": int(action), "label": label, "ply": int(self.state.ply)}
        self.history.append(move)
        return move

    def engine_to_move(self) -> bool:
        return (
            self.state.winner() is None
            and int(self.state.ply) < int(self.config.rules.ply_cap)
            and int(self.state.to_move) != self.human
        )

    def play_engine_move(self) -> dict[str, Any]:
        label_state = self.state.clone()
        with ENGINE_LOCK:
            action = int(self.agent.select_action(self.state))
            visits = np.asarray(self.agent.mcts.root_visits(), dtype=np.float32)
        policy = normalize_policy(visits)
        self.search_top = top_actions(policy, label_state, k=5)
        move = self.record("engine", action, action_to_label(action, label_state))
        self.state.apply(action)
        self.agent.observe_action(action, self.state)
        return move

    def play_human_move(self, action: int) -> dict[str, Any]:
        label = action_to_label(action, self.state)
        move = self.record("human", action, label)
        self.state.apply(int(action))
        self.agent.observe_action(int(action), self.state)
        return move


def legal_moves_payload(state: qe.GameState) -> list[dict[str, Any]]:
    moves: list[dict[str, Any]] = []
    for action in (int(a) for a in np.flatnonzero(state.legal_mask())):
        entry: dict[str, Any] = {"action": action, "label": action_to_label(action, state)}
        if action < 12:
            entry["kind"] = "pawn"
            dest = pawn_destination(state, action)
            entry["dest"] = list(dest) if dest is not None else None
        elif action < 76:
            entry["kind"] = "hwall"
            entry["anchor"] = action - 12
        else:
            entry["kind"] = "vwall"
            entry["anchor"] = action - 76
        moves.append(entry)
    return moves


def state_payload(session: Session, moves: list[dict[str, Any]]) -> dict[str, Any]:
    state = session.state
    winner = state.winner()
    draw = winner is None and int(state.ply) >= int(session.config.rules.ply_cap)
    score = evaluate_position(state)
    with ENGINE_LOCK:
        _, net_value = evaluate_state(state, APP["evaluator"], session.config)
        net_score = net_eval_score(state, APP["evaluator"], session.config)

    hwalls, vwalls = parse_wall_sets(state)
    return {
        "position": state.to_string(),
        "human": session.human,
        "to_move": int(state.to_move),
        "ply": int(state.ply),
        "winner": None if winner is None else int(winner),
        "draw": bool(draw),
        "pawns": [list(state.pawn(0)), list(state.pawn(1))],
        "walls_left": [int(state.walls_left(0)), int(state.walls_left(1))],
        "hwalls": sorted(hwalls),
        "vwalls": sorted(vwalls),
        "legal": [] if winner is not None or draw else legal_moves_payload(state),
        "eval": {
            "kind": score.kind,
            "cp": score.cp,
            "mate_in": score.mate_in,
            "winner": score.winner,
            "text": format_eval(score),
        },
        "net": {
            "name": APP["evaluator"].name,
            "value": float(net_value),
            "score": None if net_score is None else float(net_score),
            "score_text": None if net_score is None else format_net_score(net_score),
        },
        "search_top": session.search_top,
        "moves": moves,
        "history": list(session.history),
        "engine": {
            "name": APP["evaluator"].name,
            "kind": APP["evaluator"].kind,
            "sims": int(session.config.mcts.sims_per_move),
        },
    }


def create_session(human: int, sims: int) -> tuple[str, Session]:
    session = Session(human=human, sims=sims)
    session_id = secrets.token_hex(8)
    with SESSIONS_LOCK:
        while len(SESSIONS) >= MAX_SESSIONS:
            oldest = min(SESSIONS, key=lambda key: SESSIONS[key].created)
            del SESSIONS[oldest]
        SESSIONS[session_id] = session
    return session_id, session


def get_session(session_id: str) -> Session | None:
    with SESSIONS_LOCK:
        return SESSIONS.get(str(session_id))


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:  # quieter default log
        sys.stderr.write("[serve] %s\n" % (fmt % args))

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, code: int, payload: dict[str, Any]) -> None:
        self._send(code, json.dumps(payload).encode("utf-8"), "application/json")

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        data = json.loads(raw.decode("utf-8"))
        return data if isinstance(data, dict) else {}

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in {"/", "/index.html"}:
            page = (WEB_DIR / "index.html").read_bytes()
            self._send(200, page, "text/html; charset=utf-8")
            return
        if path == "/api/meta":
            self._send_json(
                200,
                {
                    "engine": APP["evaluator"].name,
                    "kind": APP["evaluator"].kind,
                    "run": str(APP["run_dir"]) if APP["run_dir"] else None,
                    "step": APP["step"],
                    "default_sims": int(APP["config"].mcts.sims_per_move),
                },
            )
            return
        self._send_json(404, {"error": f"unknown path {path}"})

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        try:
            body = self._read_json()
        except (ValueError, UnicodeDecodeError) as exc:
            self._send_json(400, {"error": f"bad JSON body: {exc}"})
            return

        try:
            if path == "/api/new":
                self.handle_new(body)
            elif path == "/api/move":
                self.handle_move(body)
            else:
                self._send_json(404, {"error": f"unknown path {path}"})
        except Exception as exc:  # surface as JSON, keep the server alive
            self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})

    def handle_new(self, body: dict[str, Any]) -> None:
        human = int(body.get("human", 0))
        if human not in (0, 1):
            self._send_json(400, {"error": "human must be 0 or 1"})
            return
        sims = int(body.get("sims") or APP["config"].mcts.sims_per_move)
        sims = min(max(sims, 1), 4096)
        session_id, session = create_session(human, sims)
        with session.lock:
            moves: list[dict[str, Any]] = []
            if session.engine_to_move():
                moves.append(session.play_engine_move())
            payload = state_payload(session, moves)
        payload["session"] = session_id
        self._send_json(200, payload)

    def handle_move(self, body: dict[str, Any]) -> None:
        session = get_session(str(body.get("session", "")))
        if session is None:
            self._send_json(404, {"error": "unknown or expired session; start a new game"})
            return
        with session.lock:
            state = session.state
            if state.winner() is not None:
                self._send_json(409, {"error": "game is over"})
                return
            if int(state.ply) >= int(session.config.rules.ply_cap):
                self._send_json(409, {"error": "ply cap reached"})
                return
            if int(state.to_move) != session.human:
                self._send_json(409, {"error": "not your turn"})
                return
            try:
                action = int(body.get("action"))
            except (TypeError, ValueError):
                self._send_json(400, {"error": "action must be an integer"})
                return
            legal = {int(a) for a in np.flatnonzero(state.legal_mask())}
            if action not in legal:
                self._send_json(400, {"error": f"action {action} is not legal"})
                return

            moves = [session.play_human_move(action)]
            if session.engine_to_move():
                moves.append(session.play_engine_move())
            payload = state_payload(session, moves)
        payload["session"] = str(body.get("session"))
        self._send_json(200, payload)


def find_default_run(root: Path) -> Path | None:
    runs_dir = root / "runs"
    if not runs_dir.exists():
        return None
    candidates = [
        child
        for child in runs_dir.iterdir()
        if child.is_dir() and (child / "ckpt").exists()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda child: child.stat().st_mtime)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a web UI to play against a checkpoint.")
    parser.add_argument("--run", type=Path, default=None, help="Run directory (default: newest under runs/)")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step (default: latest)")
    parser.add_argument("--sims", type=int, default=256, help="Default MCTS sims per engine move")
    parser.add_argument("--ply-cap", type=int, default=200)
    parser.add_argument("--stub", action="store_true", help="Use the stub evaluator instead of a checkpoint")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.device:
        os.environ["QUIDOOR_ARENA_DEVICE"] = str(args.device)

    run_dir = args.run if args.run is not None else find_default_run(ROOT)
    step = resolve_step(run_dir, args.step)
    config = load_config_for_cli(run_dir, step) if run_dir is not None else Config()
    config.mcts.sims_per_move = max(1, int(args.sims))
    config.selfplay.leaf_batch_size = max(1, int(args.sims))
    config.rules.ply_cap = int(args.ply_cap)
    config.net.compile = False

    if args.stub or run_dir is None:
        evaluator = StubEvaluator(name="stub_eval")
    else:
        evaluator = load_checkpoint_evaluator(run_dir, step, config)

    APP["config"] = config
    APP["evaluator"] = evaluator
    APP["run_dir"] = run_dir
    APP["step"] = int(step)

    detail = getattr(evaluator, "detail", None)
    print(f"engine: {evaluator.name} ({evaluator.kind})" + (f" — {detail}" if detail else ""))
    print(f"run: {run_dir} step: {step} sims: {config.mcts.sims_per_move}")
    server = ThreadingHTTPServer((args.host, int(args.port)), Handler)
    print(f"play at http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

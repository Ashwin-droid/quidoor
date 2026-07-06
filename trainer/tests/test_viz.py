from __future__ import annotations

import json
from pathlib import Path
from xml.dom import minidom

import numpy as np
import pytest

import quoridor_engine as qe

from trainer import viz
from trainer.config import Config


# ---------------------------------------------------------------------------
# Fixtures: synthetic run dir + random-weight checkpoint (CONTRACTS layout)
# ---------------------------------------------------------------------------

def _write_checkpoint(run_dir: Path, step: int, config: Config) -> None:
    import torch

    from trainer.model import QuoridorNet

    ckpt_dir = run_dir / "ckpt" / str(step)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(step + 1)
    model = QuoridorNet(config)
    torch.save(
        {
            "model_state": model.state_dict(),
            "counters": {"games": 0, "steps": step, "positions": 0},
            "config": config.asdict(),
        },
        ckpt_dir / "state.pt",
    )


@pytest.fixture()
def fast_config() -> Config:
    return Config()


def _generate(run_dir: Path, step: int, config: Config, **kw) -> None:
    params = dict(sims=6, selfplay_sims=4, num_games=2, max_plies=10, seed=1)
    params.update(kw)
    viz.generate_checkpoint_artifacts(run_dir, step, config, **params)


def _assert_svg_wellformed(path: Path) -> None:
    doc = minidom.parseString(path.read_text(encoding="utf-8"))
    assert doc.documentElement.tagName == "svg"


# ---------------------------------------------------------------------------
# Geometry / helper unit tests
# ---------------------------------------------------------------------------

def test_geometry_row0_is_at_bottom():
    geo = viz.Geo(0.0, 0.0, 10.0)
    # rank 1 (row 0) should render lower on screen (larger y) than rank 9 (row 8)
    assert geo.cy(0) > geo.cy(8)
    assert geo.cx(0) < geo.cx(8)


def test_anchor_parsing():
    assert viz._anchor_rc("a1") == (0, 0)
    assert viz._anchor_rc("b3") == (2, 1)
    assert viz._anchor_rc("h8") == (7, 7)


def test_wall_parsing_roundtrip():
    state = qe.GameState.from_string("Q1 t=0 ply=4 p0=e2 p1=e8 w=9,9 h=c3 v=f6")
    h, v = viz._parse_walls(state)
    assert (2, 2) in h  # c3 -> row 2, col 2
    assert (5, 5) in v  # f6 -> row 5, col 5


def test_pawn_deltas_match_engine_action_count():
    assert len(viz.PAWN_DELTAS) == 12
    # engine advance action for start (P0 moves toward row 8) is action 1 (delta +row)
    assert viz.PAWN_DELTAS[1] == (1, 0)


# ---------------------------------------------------------------------------
# Board rendering
# ---------------------------------------------------------------------------

def test_heatmap_svg_wellformed_and_has_pawns():
    state = qe.GameState()
    policy = np.zeros(viz.NUM_ACTIONS, dtype=np.float64)
    policy[1] = 5.0   # a pawn move
    policy[12] = 2.0  # an h-wall
    policy[76] = 1.0  # a v-wall
    svg = viz.render_heatmap_svg(state, policy, "Start (test)", "sub")
    doc = minidom.parseString(svg)
    assert doc.documentElement.tagName == "svg"
    assert "P0" in svg and "P1" in svg
    # both a move-hue and a wall-hue element should be present
    assert viz.MOVE_HUE in svg
    assert viz.WALL_HUE in svg


def test_game_strip_wellformed():
    state = qe.GameState()
    frames = [{"state": state.to_string(), "visits": np.eye(viz.NUM_ACTIONS)[1] * 3, "value": 0.25, "ply": 0}]
    svg = viz.render_game_strip_svg(frames, outcome_label="draw")
    minidom.parseString(svg)  # raises if malformed
    assert "v=+0.25" in svg


# ---------------------------------------------------------------------------
# End-to-end artifact generation
# ---------------------------------------------------------------------------

def test_generate_with_model_checkpoint(tmp_path, fast_config):
    run_dir = tmp_path / "runs" / "r1"
    _write_checkpoint(run_dir, 10, fast_config)
    _generate(run_dir, 10, fast_config)

    viz_dir = run_dir / "viz" / "ckpt_10"
    assert (viz_dir / "heatmap_00_start.svg").exists()
    assert (viz_dir / "index.html").exists()
    assert (viz_dir / "meta.json").exists()
    assert (run_dir / "index.html").exists()

    # every produced SVG must be well-formed XML
    svgs = list(viz_dir.glob("*.svg"))
    assert svgs, "expected at least the start heatmap"
    for s in svgs:
        _assert_svg_wellformed(s)

    meta = json.loads((viz_dir / "meta.json").read_text())
    assert meta["source"] == "model"
    assert meta["step"] == 10

    run_index = (run_dir / "index.html").read_text()
    assert "Checkpoint 10" in run_index
    assert "r1" in run_index


def test_generate_falls_back_to_stub_without_checkpoint(tmp_path, fast_config):
    run_dir = tmp_path / "runs" / "r2"
    (run_dir / "ckpt" / "5").mkdir(parents=True)  # no state.pt
    _generate(run_dir, 5, fast_config)

    meta = json.loads((run_dir / "viz" / "ckpt_5" / "meta.json").read_text())
    assert meta["source"].startswith("stub")
    assert (run_dir / "viz" / "ckpt_5" / "heatmap_00_start.svg").exists()


def test_probe_placeholder_when_analyze_missing(tmp_path, fast_config):
    run_dir = tmp_path / "runs" / "r3"
    (run_dir / "ckpt" / "1").mkdir(parents=True)
    _generate(run_dir, 1, fast_config)
    page = (run_dir / "viz" / "ckpt_1" / "index.html").read_text()
    # analyze.run_probe_suite may or may not exist; either way the section renders
    assert "Concept probes" in page


def test_index_lists_multiple_checkpoints_chronologically(tmp_path, fast_config):
    run_dir = tmp_path / "runs" / "r4"
    for step in (2, 4):
        (run_dir / "ckpt" / str(step)).mkdir(parents=True)
        _generate(run_dir, step, fast_config)
    index = (run_dir / "index.html").read_text()
    assert "Checkpoint 2" in index
    assert "Checkpoint 4" in index
    # newest first: ckpt 4 should appear before ckpt 2 in the document
    assert index.index("Checkpoint 4") < index.index("Checkpoint 2")


def test_elo_sparkline_rendered(tmp_path, fast_config):
    run_dir = tmp_path / "runs" / "r5"
    (run_dir / "ckpt" / "3").mkdir(parents=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "elo.json").write_text(
        json.dumps({"ladder": [{"step": 1, "elo": 0}, {"step": 3, "elo": 120}]})
    )
    _generate(run_dir, 3, fast_config)
    index = (run_dir / "index.html").read_text()
    assert "Elo ladder" in index
    assert "polyline" in index  # sparkline drawn


def test_calibration_png_written_with_enough_data(tmp_path, fast_config):
    run_dir = tmp_path / "runs" / "r6"
    (run_dir / "ckpt" / "7").mkdir(parents=True)
    _generate(run_dir, 7, fast_config, num_games=3, max_plies=14)
    # calibration is best-effort; if matplotlib produced it, it must be a real file
    png = run_dir / "viz" / "ckpt_7" / "calibration.png"
    meta = json.loads((run_dir / "viz" / "ckpt_7" / "meta.json").read_text())
    if meta["calibration"]:
        assert png.exists() and png.stat().st_size > 0


def test_load_elo_parses_multiple_shapes(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "elo.json").write_text(json.dumps([{"step": 1, "rating": 10.0}, {"step": 2, "rating": 22.0}]))
    pts = viz._load_elo(run_dir)
    assert pts == [(1, 10.0), (2, 22.0)]

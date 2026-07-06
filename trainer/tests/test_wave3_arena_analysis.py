from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import yaml

import quoridor_engine as qe
from trainer.analyze import run_probe_suite
from trainer.arena import fit_elo_ladder, run_arena
from trainer.config import Config
from trainer.model import QuoridorNet


ROOT = Path(__file__).resolve().parents[2]


def tiny_config() -> Config:
    cfg = Config()
    cfg.mcts.sims_per_move = 2
    cfg.mcts.fast_sims_per_move = 2
    cfg.selfplay.leaf_batch_size = 4
    cfg.selfplay.collect_timeout_ms = 0
    cfg.rules.ply_cap = 4
    cfg.eval.games_vs_previous = 0
    cfg.eval.games_vs_baseline = 1
    cfg.eval.opening_temperature_moves = 0
    cfg.net.blocks = 1
    cfg.net.channels = 8
    cfg.net.bf16 = False
    cfg.net.compile = False
    cfg.net.fixed_batch_size = 4
    return cfg


def write_synthetic_checkpoint(run_dir: Path, step: int, cfg: Config) -> None:
    ckpt_dir = run_dir / "ckpt" / str(step)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model = QuoridorNet(cfg)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": cfg.asdict(),
            "counters": {"games": 0, "steps": step, "positions": 0},
        },
        ckpt_dir / "state.pt",
    )


def test_probe_yaml_parses_and_positions_are_legal() -> None:
    suite_path = ROOT / "probes" / "quoridor_probes.yaml"
    suite = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    positions = suite["positions"]

    assert len(positions) >= 10
    themes = {item["theme"] for item in positions}
    assert "wall-parity endgame race" in themes
    assert "follow-shadow tactic" in themes
    assert "hoard-vs-early-aggression" in themes
    assert "must-block-mate-in-1" in themes

    for item in positions:
        state = qe.GameState.from_string(item["position"])
        assert state.to_string().startswith("Q1 ")
        assert np.asarray(state.legal_mask()).any(), item["id"]
        assert item.get("assertions"), item["id"]


def test_elo_fit_orders_synthetic_results() -> None:
    elos = fit_elo_ladder(
        [
            {
                "player_a": "strong",
                "player_b": "middle",
                "games": 100,
                "wins_a": 78,
                "wins_b": 18,
                "draws": 4,
            },
            {
                "player_a": "middle",
                "player_b": "weak",
                "games": 100,
                "wins_a": 70,
                "wins_b": 24,
                "draws": 6,
            },
            {
                "player_a": "strong",
                "player_b": "weak",
                "games": 100,
                "wins_a": 88,
                "wins_b": 8,
                "draws": 4,
            },
        ]
    )

    assert elos["strong"] > elos["middle"] > elos["weak"]


def test_arena_smoke_uses_stub_when_checkpoint_missing(tmp_path: Path) -> None:
    cfg = tiny_config()
    summary = run_arena(tmp_path, 1, cfg)

    assert summary["current_kind"] == "stub"
    assert summary["previous"] is None
    assert set(summary["baselines"]) == {"random", "greedy", "minimax-2"}
    assert (tmp_path / "elo.json").exists()
    payload = json.loads((tmp_path / "elo.json").read_text(encoding="utf-8"))
    assert "ckpt_1:stub" in payload["players"]


def test_probe_suite_smoke_with_synthetic_checkpoint(tmp_path: Path) -> None:
    cfg = tiny_config()
    cfg.mcts.sims_per_move = 1
    cfg.selfplay.leaf_batch_size = 2
    write_synthetic_checkpoint(tmp_path, 5, cfg)

    result = run_probe_suite(tmp_path, 5, cfg)

    assert result["summary"]["positions"] >= 10
    assert result["evaluator"]["kind"] == "checkpoint"
    assert (tmp_path / "analysis" / "ckpt_5" / "probes.json").exists()

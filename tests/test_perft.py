from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.reference import ReferenceState, perft


# Golden counts were frozen from tests.reference.perft. The start-position
# depth-4 count is too large for the pure-Python reference to recompute in
# every pytest run, so this test asserts the frozen constants against a tiny
# native Rust perft helper compiled in a temporary directory.
PERFT_POSITIONS: list[tuple[str, tuple[int, int, int, int]]] = [
    (
        "Q1 t=0 ply=0 p0=e1 p1=e9 w=10,10 h=- v=-",
        (131, 16677, 2062264, 247569030),
    ),
    (
        "Q1 t=0 ply=28 p0=g1 p1=f6 w=0,0 "
        "h=a1,e1,e4,b5,d6,c7,h7 "
        "v=d1,g1,a2,b3,d3,f4,a6,b6,e6,h6,a8,d8,f8",
        (2, 6, 15, 45),
    ),
    (
        "Q1 t=0 ply=28 p0=d2 p1=d8 w=0,0 "
        "h=b1,d2,a3,c3,b4,e4,g4,a5,f5,d6,f6,f8 "
        "v=a1,c1,b2,d3,e3,d5,b8,g8",
        (2, 8, 20, 65),
    ),
    (
        "Q1 t=0 ply=28 p0=e1 p1=e9 w=0,0 "
        "h=c1,e1,f2,f3,a4,h4,d6,g6,f7,f8 "
        "v=d1,c2,a3,d3,g3,a6,b7,c8,e8,h8",
        (1, 2, 4, 10),
    ),
    (
        "Q1 t=0 ply=28 p0=g1 p1=g7 w=0,0 "
        "h=a1,d2,f2,f5,d6,h6,b7,c8,h8 "
        "v=b1,d3,h3,a4,c4,g5,c6,a8,b8,d8,g8",
        (3, 12, 36, 108),
    ),
    (
        "Q1 t=0 ply=28 p0=d2 p1=g9 w=0,0 "
        "h=d2,g2,a3,f3,a4,g4,d5,g5,f6,a7,c7,g7 "
        "v=e1,g1,b2,c3,e3,d6,g6,b8",
        (3, 9, 24, 72),
    ),
]


RUST_PERFT_SOURCE = r"""
use quoridor_engine::movegen;
use quoridor_engine::state::{GameState, WallOrientation};
use std::collections::HashMap;
use std::env;
use std::str::FromStr;

fn apply_known(mut s: GameState, action: u8) -> GameState {
    let side = s.to_move as usize;
    if action < 12 {
        let dest = movegen::pawn_destination_for_action(&s, action).unwrap();
        s.pawns[side] = dest;
    } else {
        let (orientation, anchor) = movegen::wall_from_action(action).unwrap();
        let b = 1u64 << anchor;
        match orientation {
            WallOrientation::Horizontal => s.hwalls |= b,
            WallOrientation::Vertical => s.vwalls |= b,
        }
        s.walls_left[side] -= 1;
    }
    s.to_move ^= 1;
    s.ply = s.ply.saturating_add(1);
    s
}

fn perft(s: GameState, depth: u8, memo: &mut HashMap<(GameState, u8), u64>) -> u64 {
    if depth == 0 {
        return 1;
    }
    if let Some(v) = memo.get(&(s, depth)) {
        return *v;
    }
    let mut total = 0u64;
    for action in s.legal_actions() {
        total += perft(apply_known(s, action), depth - 1, memo);
    }
    memo.insert((s, depth), total);
    total
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let depth: u8 = args[1].parse().unwrap();
    let state = GameState::from_str(&args[2]).unwrap();
    for d in 1..=depth {
        let mut memo = HashMap::new();
        println!("{} {}", d, perft(state, d, &mut memo));
    }
}
"""


@pytest.fixture(scope="session")
def rust_perft_bin(tmp_path_factory: pytest.TempPathFactory) -> Path:
    repo = Path(__file__).resolve().parents[1]
    deps = repo / "target" / "release" / "deps"
    rlibs = sorted(deps.glob("libquoridor_engine*.rlib"))
    if not rlibs:
        pytest.fail("missing target/release/deps/libquoridor_engine*.rlib; run maturin develop --release")

    tmp = tmp_path_factory.mktemp("rust_perft")
    source = tmp / "perft.rs"
    binary = tmp / "perft"
    source.write_text(RUST_PERFT_SOURCE)
    subprocess.run(
        [
            "rustc",
            "--edition=2024",
            "-O",
            str(source),
            "-L",
            str(deps),
            "--extern",
            f"quoridor_engine={rlibs[0]}",
            "-o",
            str(binary),
        ],
        check=True,
        cwd=repo,
    )
    return binary


def rust_perft_counts(binary: Path, position: str) -> tuple[int, int, int, int]:
    out = subprocess.check_output([str(binary), "4", position], text=True)
    counts = []
    for line in out.splitlines():
        _depth, count = line.split()
        counts.append(int(count))
    return tuple(counts)  # type: ignore[return-value]


def test_reference_goldens_for_bounded_positions() -> None:
    # Start d1-d2 and all five wall-depleted handcrafted positions are cheap
    # enough to recompute with the independent Python reference in CI.
    start_state = ReferenceState.from_string(PERFT_POSITIONS[0][0])
    assert tuple(perft(start_state, d, {}) for d in range(1, 3)) == PERFT_POSITIONS[0][1][:2]

    for position, expected in PERFT_POSITIONS[1:]:
        state = ReferenceState.from_string(position)
        assert tuple(perft(state, d, {}) for d in range(1, 5)) == expected


@pytest.mark.parametrize("position,expected", PERFT_POSITIONS)
def test_rust_perft_matches_goldens(
    rust_perft_bin: Path,
    position: str,
    expected: tuple[int, int, int, int],
) -> None:
    assert rust_perft_counts(rust_perft_bin, position) == expected

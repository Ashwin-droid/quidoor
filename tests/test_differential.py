from __future__ import annotations

import os
import random

import numpy as np
import quoridor_engine as qe

from tests.reference import ReferenceState


def rust_legal_actions(state: qe.GameState) -> set[int]:
    return {int(a) for a in np.flatnonzero(state.legal_mask())}


def test_random_games_match_reference() -> None:
    games = int(os.environ.get("QUIDOOR_DIFF_GAMES", "500"))
    seed = int(os.environ.get("QUIDOOR_DIFF_SEED", "1729"))
    max_plies = int(os.environ.get("QUIDOOR_DIFF_MAX_PLIES", "200"))
    rng = random.Random(seed)

    for game_idx in range(games):
        ref = ReferenceState.start()
        rust = qe.GameState()

        for ply in range(max_plies):
            assert ref.to_string() == rust.to_string()
            assert ref.winner() == rust.winner()

            ref_actions = set(ref.legal_actions())
            rust_actions = rust_legal_actions(rust)
            assert ref_actions == rust_actions, (
                f"legal action mismatch game={game_idx} ply={ply} "
                f"pos={ref.to_string()} "
                f"ref_only={sorted(ref_actions - rust_actions)} "
                f"rust_only={sorted(rust_actions - ref_actions)}"
            )

            if not ref_actions:
                break

            action = rng.choice(sorted(ref_actions))
            ref.apply(action)
            rust.apply(action)
            assert ref.to_string() == rust.to_string(), (
                f"apply mismatch game={game_idx} ply={ply} action={action}"
            )
            assert ref.winner() == rust.winner()

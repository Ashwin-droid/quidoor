# Quidoor

AlphaZero-style Quoridor training scaffold.

This first wave provides the Rust engine core, PyO3 bindings exposed as
`quoridor_engine`, and the Python configuration source of truth.

Useful checks:

```bash
cargo test --release
cargo bench --no-run
.venv/bin/maturin develop --release
.venv/bin/python -c "import quoridor_engine as qe; s=qe.GameState(); print(s.to_string(), int(s.legal_mask().sum()))"
```

## Playing and analysis

Both play tools show a Stockfish-style evaluation from player 0's
perspective: integer centi-moves (`+100` = one move of shortest-path
advantage, exactly `0` at the start) that switches to `+M7`/`-M4` once the
position is a provably decided race.

The network also learns its own eval: a score head predicts a continuous
moves-ahead number where `|score| <= 20` is a normal eval and anything
beyond encodes a mate claim (`+/-(40 - distance)`). Training targets use
the exact race-mate scores where provable and otherwise blend the
shortest-path heuristic with the game outcome, so the net is penalised for
claiming wins for the side that went on to lose. Checkpoints and replay
buffers from before this head load unchanged (the head starts fresh and its
display is hidden until it has been trained). Tune with
`training.eval_score_weight`.

```bash
# terminal play against a checkpoint (or --opponent stub/random/greedy/minimax-2)
.venv/bin/python tools/play.py --run runs/run1

# web UI against the latest checkpoint: board, wall placement, eval bar
.venv/bin/python tools/serve.py --run runs/run1   # then open http://127.0.0.1:8000/
```

## Regression gate

Play the latest checkpoint against the previous one to catch training
regressions (exit code 1 on regression, report saved under
`<run>/regression/`):

```bash
.venv/bin/python tools/regression.py --run runs/run1 --games 50
```

## Kaggle / T4 inference

NVIDIA T4 GPUs have fast fp16 tensor cores but no useful bf16 path. Run
self-play inference with fp16 while leaving training precision unchanged:

```bash
.venv/bin/python -m trainer.train --inference-precision fp16
```

`--no-bf16` remains available as a deprecated alias for
`--inference-precision fp32` when resuming older commands.

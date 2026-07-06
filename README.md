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

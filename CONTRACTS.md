# Cross-module contracts (binding)

These interfaces are frozen so independently-built modules integrate. If reality forces a
deviation, note it prominently in your final report — do not silently rename.

## Python extension module

- The PyO3 crate builds as importable module **`quoridor_engine`** (maturin, `module-name = "quoridor_engine"` in pyproject.toml).
- `import quoridor_engine as qe`

### Core API (wave 1)

```python
qe.NUM_ACTIONS == 140
qe.BOARD == 9

s = qe.GameState()                    # start position
s = qe.GameState.from_string(txt)     # position notation (see engine docs)
s.to_string() -> str
s.legal_mask() -> np.ndarray          # bool or uint8, shape [140]
s.legal_actions() -> list[int]
s.apply(action: int) -> None          # raises ValueError on illegal
s.winner() -> int | None              # 0, 1, or None
s.to_move -> int
s.ply -> int
s.pawn(player: int) -> tuple[int,int] # (row, col)
s.walls_left(player: int) -> int
s.shortest_path_len(player: int) -> int
s.encode(extra_planes: bool) -> np.ndarray  # f32 [C,9,9], canonical form
s.clone() -> GameState
qe.mirror_action(a: int) -> int
qe.canonical_action(a: int, to_move: int) -> int   # maps absolute->canonical and back (involution)
```

### MCTS / self-play API (wave 2, added to the same module)

```python
cfg = qe.EngineConfig(**kwargs)   # mirrors trainer/config.py fields (mcts + selfplay sections)
sp = qe.SelfPlayManager(cfg, num_games=..., seed=...)
sp.done() -> bool
sp.collect_leaves(max_batch=1024, timeout_ms=2) -> (planes: f32[B,C,9,9], leaf_ids: i64[B], masks: u8[B,140])
sp.submit(leaf_ids, policy_logits: f32[B,140], values: f32[B]) -> None
sp.drain_finished_games() -> list of finished-game records
   # each record: (states: f32[T,C,9,9], policies: f32[T,140], masks: u8[T,140], to_move: u8[T], outcome: int)
   # outcome from player-0 perspective: +1 / -1 / 0(ply-cap draw); z per position derived in Python.

# Single-game batched MCTS for arena/analysis (same pull-based pattern, one game, no noise unless asked):
m = qe.BatchedMcts(cfg, state: GameState, seed=...)
m.collect_leaves(max_batch, timeout_ms) -> same shapes as above
m.submit(...)
m.ready() -> bool                 # sims_per_move reached
m.root_visits() -> f32[140]
m.best_action() -> int
m.advance(action: int) -> None    # re-root (tree reuse); also used to apply opponent's move
qe.stub_eval(planes) -> (uniform_logits, sp_diff_values)  # M3 stub evaluator, pure Rust helper
```

## trainer/config.py

Single `Config` dataclass (nested dataclasses fine) — the source of truth listed in SPEC §8.
`Config.to_engine_kwargs()` returns the dict for `qe.EngineConfig(**...)`.

## Checkpoint layout (frozen)

```
runs/<run_id>/
  ckpt/<step>/state.pt        # torch.save dict: model_state, optimizer_state, scheduler_state,
                              #   counters {games, steps, positions}, config (asdict), rng
  ckpt/<step>/buffer/         # shard files + manifest.json
  ckpt/latest -> <step>       # symlink, updated last
  elo.json                    # Elo ladder + arena history (arena.py owns writing)
  status.json                 # see SPEC §6.1/§7.3
  log.jsonl, tb/, viz/ckpt_<n>/
```

## Cross-module function signatures (wave 3, parallel agents — frozen)

- `trainer/arena.py`: `run_arena(run_dir: Path, ckpt_step: int, config: Config) -> dict`
  Plays vs previous checkpoint + baselines, updates `elo.json`, returns summary dict.
- `trainer/viz.py`: `generate_checkpoint_artifacts(run_dir: Path, ckpt_step: int, config: Config) -> None`
  Loads model from the checkpoint dir, writes `runs/<run_id>/viz/ckpt_<step>/` + refreshes `index.html`.
- `trainer/analyze.py`: `run_probe_suite(run_dir: Path, ckpt_step: int, config: Config) -> dict`
  Runs the YAML concept-probe suite, returns structured results (viz renders them; soft-fail OK).
- `train.py` calls all three wrapped in try/except: they are derived data, failures are logged and skipped.

## Model loading helper (wave 2)

`trainer/model.py` must expose:
- `QuoridorNet(config)` (nn.Module)
- `load_model(ckpt_dir: Path, config: Config, device) -> QuoridorNet` (reads `state.pt`)
- `Inferencer(model, config)` with `infer(planes: np.ndarray, masks: np.ndarray) -> (policy_logits f32[B,140] np, values f32[B] np)` — handles device transfer, bf16 autocast, fixed-shape padding.

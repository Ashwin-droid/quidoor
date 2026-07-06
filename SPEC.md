# Project Spec: AlphaZero for Quoridor (barricade.gg rules)

Build a self-play reinforcement learning system that learns Quoridor from scratch, AlphaZero-style. Hybrid architecture: **Rust** for the game engine + MCTS (CPU-bound hot path), **Python/PyTorch** for the neural network and training loop, bridged with **PyO3/maturin**. Target hardware: single machine, RTX 5070 (Blackwell), Linux (Pop!_OS).

Design goal: maximum self-play throughput on one consumer GPU. Every architectural decision below serves that.

---

## 1. Game Rules (implement exactly, no variants)

Standard Quoridor:

- 9×9 board of cells. Player 0 starts at (row 0, col 4), goal = any cell in row 8. Player 1 starts at (row 8, col 4), goal = any cell in row 0.
- Each player has 10 walls in reserve.
- On your turn: move your pawn one orthogonal step (N/S/E/W), OR place a wall (if you have any left). Passing is illegal. Note the move need not be "forward" — retreating and sideways moves are legal.
- Walls are 2 cells long, placed in the grooves between cells, horizontal or vertical. There are 8×8 = 64 valid anchor slots per orientation (anchor = the intersection point the wall is centered on).
- Wall constraints: (a) may not overlap another wall's segments, (b) may not cross through the middle of a perpendicular wall, (c) may not reduce EITHER player's set of paths to their goal row to zero. Both players must always retain ≥1 route.
- Jumps: if the two pawns are orthogonally adjacent, the mover may jump straight over the opponent to the cell directly behind them, IF that cell exists and the wall-edge between opponent and that cell is open. If the straight jump is blocked (wall or board edge behind opponent), the mover may instead move diagonally to either cell orthogonally adjacent to the opponent (perpendicular to the jump axis), subject to walls. Diagonal moves are ONLY legal in this blocked-jump situation.
- Win: first pawn to reach any cell in its goal row. There are no draws by rule, but see §5 (move cap) for training pragmatics.

**Correctness is sacred.** A rules bug silently poisons the entire training run. See §9 testing requirements.

---

## 2. Repository Layout

```
quoridor-az/            # == this repo root
├── engine/                 # Rust crate (cdylib + rlib)
│   ├── src/
│   │   ├── state.rs        # GameState, move application
│   │   ├── movegen.rs      # legal move generation
│   │   ├── path.rs         # connectivity / shortest path (hot path)
│   │   ├── mcts.rs         # batched PUCT search
│   │   ├── selfplay.rs     # multi-game orchestrator
│   │   ├── encode.rs       # state -> NN input planes, action mapping
│   │   └── lib.rs          # PyO3 bindings
│   └── Cargo.toml
├── trainer/                # Python package
│   ├── model.py            # ResNet definition
│   ├── train.py            # training loop
│   ├── selfplay_driver.py  # drives rust selfplay + GPU inference loop
│   ├── replay_buffer.py
│   ├── arena.py            # model-vs-model & baseline evaluation
│   ├── analyze.py          # opening book extraction, policy dumps
│   └── config.py           # single source of truth for all hyperparams
├── baselines/              # python: random, greedy-path, shallow minimax
├── tests/                  # cross-language golden tests
└── pyproject.toml          # maturin build
```

Build with `maturin develop --release`. All Rust release builds use `lto = true`, `codegen-units = 1`, `opt-level = 3`.

---

## 3. Rust Engine

### 3.1 State representation (compact, copy-cheap)

```rust
pub struct GameState {
    pawns: [u8; 2],        // cell index 0..81 (row*9+col)
    hwalls: u64,           // bitboard over 64 anchor slots (anchor = row*8+col, rows 0..8 exclusive)
    vwalls: u64,
    walls_left: [u8; 2],
    to_move: u8,           // 0 or 1
    ply: u16,
}
```

Total ~20 bytes, `Copy`. MCTS clones states freely instead of make/unmake.

Precompute at startup: for every cell, its neighbor cells; for every (cell, direction) edge, the set of wall placements that would block it (as bitmasks). Edge-blocked check is then `(hwalls & mask_h) != 0 || (vwalls & mask_v) != 0` — no branching on geometry at runtime.

### 3.2 Move generation

Action space is fixed at **140 actions**:
- 0–11: pawn moves, relative encoding: [N, S, E, W, NN, SS, EE, WW, NE, NW, SE, SW] (straight jumps + diagonals per jump rules)
- 12–75: horizontal wall at anchor (action − 12)
- 76–139: vertical wall at anchor (action − 76)

`legal_moves(&self) -> ArrayVec<u8, 140>` (or a 140-bit mask, prefer the mask — the NN needs it anyway).

Wall legality pipeline (in order, cheapest first):
1. `walls_left[to_move] > 0`
2. overlap/cross check via precomputed bitmask AND (a h-wall at anchor a conflicts with h-walls at a−1, a, a+1 same row, and v-wall at a)
3. connectivity check (only for walls passing 1–2) — see §3.3

### 3.3 Pathfinding (the hot path — optimize hard)

Requirement: after any candidate wall, both players must reach their goal rows.

- Base algorithm: A* from pawn to goal row, heuristic = vertical distance. Early-exit BFS is acceptable fallback; benchmark both.
- **Key optimization — cached-path invalidation:** maintain each player's current shortest path (recomputed lazily). A candidate wall requires a full connectivity re-check for player p ONLY if it severs an edge on p's cached path. If it doesn't touch the cached path, the path still exists, wall is legal for p. This eliminates ~90%+ of searches.
- Board graph is implicit (cells + edge-blocked function); do NOT build an explicit adjacency structure per state.
- Provide `shortest_path_len(player) -> u8` publicly — needed for baselines, optional NN feature planes, and analysis.

Performance budget: engine alone should sustain **≥2M legal-move-gen + apply operations/sec/core** on random playouts. Add a criterion benchmark to enforce this.

### 3.4 Encoding (in Rust, so self-play never touches Python per-position)

`encode(state) -> [f32; C*9*9]` in **canonical form**: board rotated 180° when player 1 is to move, so the current player ALWAYS moves toward row 8. Action indices are remapped consistently (N↔S, E↔W, wall anchors rotated). This halves what the net must learn.

Input planes (C = 6):
1. current player pawn (one-hot)
2. opponent pawn (one-hot)
3. h-walls (paint each wall's blocked edges onto the 9×9 cell grid — define convention precisely and document it; simplest: plane value 1 at both cells north of each wall segment)
4. v-walls (same convention, cells west of segments)
5. current player walls_left / 10 (constant plane)
6. opponent walls_left / 10 (constant plane)

Config flag `extra_planes: bool` adds planes 7–8: normalized shortest-path-distance-to-goal maps for both players (cheap to compute given §3.3, likely accelerates early training; run ablation later).

**Symmetry augmentation:** Quoridor has exact left-right mirror symmetry. Implement `mirror(state, policy)` and apply with p=0.5 when writing training samples.

### 3.5 MCTS (batched PUCT, the AlphaZero variant)

Per-node: `N` (visits, u32), `W` (total value, f32), `P` (prior, f16 ok), children indexed by action. Arena-allocate nodes in a `Vec` per tree; no `Rc<RefCell<>>`.

Selection: PUCT — `argmax_a Q(a) + c_puct * P(a) * sqrt(N_parent) / (1 + N(a))` with `c_puct ≈ 1.5` (configurable). Q of unvisited children uses FPU reduction: `Q_parent − fpu_reduction` (default 0.2) rather than 0.

Root exploration (self-play only): Dirichlet noise mixed into root priors, `P' = 0.75·P + 0.25·Dir(α)`, with **α = 10 / avg_legal_moves ≈ 0.08** for Quoridor (do not copy chess's 0.3 — branching factor here is ~130).

Leaf evaluation is **batched and asynchronous** (see §4). While a leaf is pending GPU evaluation, apply **virtual loss** (add `n_vloss=3` to N and −n_vloss to W along the path) so parallel simulations diverge; revert on backprop.

Terminal handling: game-over leaves backprop the true result (±1) without NN. Apply the ply-cap draw rule from §5 as value 0.

Tree reuse: after a move is chosen, re-root the tree at the played child and discard siblings.

Move selection during self-play: sample from `visits^(1/τ)` with τ=1 for the first 16 plies, τ→0 (argmax) afterward. Evaluation/arena play: always argmax, no noise.

### 3.6 Self-play orchestrator + PyO3 boundary (inversion of control)

Do NOT call Python from Rust threads (GIL pain). Instead the Rust side exposes a pull-based API and Python drives the loop:

```python
sp = engine.SelfPlayManager(config)     # runs G concurrent games internally
while not sp.done():
    batch = sp.collect_leaves(max_batch=1024, timeout_ms=2)   # ndarray [B, C, 9, 9] + leaf ids
    if batch.size > 0:
        policy_logits, values = model.infer(batch)             # GPU, fp16/bf16
        sp.submit(leaf_ids, policy_logits, values)             # rust masks illegals, softmaxes, backprops
    finished = sp.drain_finished_games()                        # [(states, policies, outcome), ...]
    buffer.add(finished)
```

Internally `SelfPlayManager` runs G games (default 256) as cooperative state machines on a rayon pool: each game steps its MCTS until it needs a leaf eval, parks the request in a queue, moves on. `collect_leaves` drains the queue; `submit` resumes parked simulations. Zero-copy across the boundary: hand numpy arrays over via `pyo3-numpy`, write encoded planes directly into a preallocated pinned buffer if feasible.

Policy targets stored per position = normalized root visit counts over 140 actions (sparse storage fine, densify in Python).

---

## 4. Python / GPU Side

### 4.1 Network

ResNet, deliberately modest (Quoridor << Go):
- Input: [C, 9, 9]
- Stem: 3×3 conv → 128 channels, BN, ReLU
- Body: **8 residual blocks**, 128 channels (config: `blocks`, `channels`)
- Policy head: 1×1 conv → 2ch → BN → ReLU → FC → 140 logits (mask illegal moves with −inf before softmax; mask comes from engine)
- Value head: 1×1 conv → 1ch → BN → ReLU → FC 64 → ReLU → FC 1 → tanh

Inference config: `bf16` autocast, `torch.compile(mode="max-autotune")`, fixed batch shapes (pad partial batches) so compile caches hold; optionally CUDA graphs. Target ≥20k positions/sec inference on the 5070 at this size — verify with a standalone benchmark script before wiring anything else.

### 4.2 Training loop

Continuous asynchronous style (simpler than AZ's gated generations, works as well):
- Replay buffer: ring buffer of most recent **1,000,000 positions** (config). Persist to disk in shards (`.npz` or arrow) so runs are resumable.
- Alternate phases in one process (simplest): generate X games → train Y steps → checkpoint. Defaults: X=500 games, Y=1000 steps, batch 512.
- Loss: `CE(policy_target, policy_logits) + MSE(z, v) + 1e-4 · L2` (use weight_decay in optimizer, exclude BN/bias).
- Optimizer: SGD momentum 0.9, LR 0.05 → cosine decay over planned run length (config), or AdamW 2e-4 as fallback preset. Grad clip 1.0.
- Log to tensorboard: loss components, policy entropy, value MSE vs actual outcomes, buffer stats, games/hour, GPU utilization, average game length.

### 4.3 Draw pragmatics

Quoridor self-play between weak/confused nets can shuffle forever. Hard cap games at **200 plies** → score as draw (z = 0). Log the draw rate; it should collapse toward ~0 as the net strengthens. No resignation logic in v1 (games are short enough).

### 4.4 Evaluation (arena.py)

Every checkpoint:
1. 200 games vs previous checkpoint (100 as each color, argmax play, small opening temperature for diversity). Track Elo via BayesElo-style logistic fit; maintain a ladder across all checkpoints.
2. 100 games vs fixed baselines: `random`, `greedy` (always step along own shortest path), `minimax-2` (2-ply alpha-beta with shortest-path-difference heuristic). These are floor sanity checks — greedy should be crushed within a few hours of training or something is broken.

No promotion gating: always train from latest, use the Elo ladder purely as monitoring.

---

## 5. Analysis Tooling (the actual point of the project)

`analyze.py` must support:
- **Opening book evolution:** for checkpoints across the run, dump the root policy (post-MCTS, 800 sims, no noise) for the start position and for the top-k most common positions at plies 2–8. Render as heatmaps over the board (walls on anchor grid, moves on cell grid). This shows the opening meta forming over training time.
- **Concept probes:** scripted positions testing known human theory — does the net value wall parity in endgame races, does it discover the follow/shadow tactic, when does it choose wall-hoarding vs early aggression. Store as a YAML suite of FEN-like position strings + expected-tendency assertions (soft checks, logged not enforced).
- **Game viewer:** render any recorded game as SVG frames or a ratatui TUI board with policy/value overlays per move.
- Position string format: define a compact text notation (pawn cells, wall lists, counts, to-move) with `to_string`/`from_string` in Rust, exposed to Python. Used everywhere (tests, probes, viewer).

---

## 6. Observability (first-class, not an afterthought)

The run must be inspectable at all times without stopping it.

### 6.1 Metrics & logging
- Structured logging (JSON lines) to `runs/<run_id>/log.jsonl` + human-readable console. Every phase transition, checkpoint, arena result, and anomaly (NaN loss, empty batches, GPU idle stretches) gets a log event.
- Tensorboard (in `runs/<run_id>/tb/`), all series keyed by both wall-clock and training step:
  - training: total/policy/value loss, policy entropy, value-target correlation, LR, grad norm
  - self-play: games/hour, positions/sec, avg game length, draw rate, avg root policy entropy, MCTS sims/sec, leaf-batch fill ratio (batching health)
  - system: GPU utilization & VRAM, per-core CPU, buffer size/age distribution
  - strength: Elo ladder over checkpoints, win rates vs each baseline
- A `status.json` in the run dir, atomically rewritten every ~30s: current phase, games completed, buffer fill, last checkpoint, ETA of current phase, shutdown state (§7.3). Cheap to poll remotely.

### 6.2 Visual artifacts (auto-generated per checkpoint)
Every checkpoint automatically produces, under `runs/<run_id>/viz/ckpt_<n>/` (dark background, light foreground on ALL rendered output):
- **Opening heatmaps:** root policy (800 sims, no noise) for the start position + top-8 most frequent ply-2/ply-4 positions from recent self-play. Two overlaid layers per image: pawn-move probs on the cell grid, wall probs on the anchor grid.
- **Value calibration plot:** predicted value vs eventual game outcome, bucketed.
- **Sample games:** 5 self-play games rendered as SVG strips (board per move, top-3 policy arrows/wall ghosts, value bar).
- **Concept probe report:** the §5 YAML suite run against this checkpoint → one markdown/HTML summary with pass/drift indicators vs previous checkpoint.
- An `index.html` per run linking all checkpoints' artifacts chronologically — the "watch the meta evolve" page. Static files only.

### 6.3 Live inspection
- `tools/peek.py`: reads `status.json` + latest tensorboard scalars, prints a one-screen run summary in the terminal.
- `tools/play.py`: load any checkpoint, play against it interactively (TUI), with live display of the net's top-5 policy moves and value after each of your moves. Also serves as the manual smell test.

---

## 7. Run Lifecycle: Checkpointing, Graceful Stop, Resume

### 7.1 Checkpoint contents (a checkpoint = everything needed to resume)
Written atomically (write to temp dir, fsync, rename) under `runs/<run_id>/ckpt/<step>/`:
- model weights + optimizer state + LR-scheduler state (`state.pt`)
- replay buffer (sharded; a manifest referencing shard files — shards are append-only so unchanged ones aren't rewritten)
- full config snapshot (frozen copy; resume validates compatibility and refuses on breaking changes like net shape or action-space edits, warns on benign ones like LR)
- counters: total games, total training steps, total positions generated
- Elo ladder state and arena history
- RNG states (torch, numpy, Rust engine seed counters) — best-effort reproducibility, bit-exact replay is a non-goal
- `latest` symlink updated last; keep last K=5 checkpoints + every Nth (default every 10th) permanently

### 7.2 Resume
`python -m trainer.train --resume runs/<run_id>` (or `--resume runs/<run_id>/ckpt/<step>` for an older point). Resume restores buffer, counters, ladder, and continues as if uninterrupted. Also support `--init-from <ckpt>` (weights only, fresh buffer/counters) — the "fork the run with new hyperparams" workflow.

### 7.3 Graceful stop (hard requirement)
Two triggers, identical behavior:
- **SIGINT/SIGTERM** (first signal), or
- `touch runs/<run_id>/STOP` (checked between scheduling units — usable over SSH without hunting PIDs)

On trigger, enter `draining` state (reflected in `status.json` and console):
1. immediately stop starting new self-play games and new training phases
2. let all in-flight games run to completion (keep serving their leaf batches on GPU); bounded by the ply cap this is minutes, not hours — if mid-training-phase instead, finish only the current optimizer step, not the phase
3. flush finished games to the buffer, write a full checkpoint (§7.1)
4. write final `status.json` with `state: stopped_clean`, exit 0

A **second** SIGINT = hard abort: skip draining, still attempt a fast checkpoint of model/optimizer/counters (buffer may lose the in-flight tail — acceptable, logged), exit 130. A third kills immediately, no checkpoint.

`draining` must be reachable from any phase (self-play, training, arena, viz generation). Arena/viz work is abandonable — it's derived data, log and skip.

### 7.4 Crash resilience
Independent of graceful stop: autosave a full checkpoint every N minutes (default 30, config). After a crash/power-loss, `--resume` from the last autosave must work; at most N minutes of work is lost. Buffer shard writes are append-and-manifest so a torn write can't corrupt earlier shards.

---

## 8. Configuration

One `config.py` dataclass (mirrored into a Rust config struct via the PyO3 constructor), covering: board/rule constants, MCTS (sims_per_move=800 selfplay / 400 fast preset, c_puct, fpu, dirichlet α & ε, vloss, τ schedule), self-play (concurrent games, batch size, ply cap), net (blocks, channels, extra_planes), training (buffer size, batch, LR schedule, phase sizes), eval cadence. Everything tunable without recompiling except board size.

---

## 9. Performance Targets (acceptance criteria)

On the target machine:
- Engine: ≥2M random-playout moves/sec/core (criterion bench)
- Self-play throughput: **≥1,500 complete games/hour** at 800 sims/move with the 8×128 net (this is the number that determines whether training is feasible; if unmet, profile: GPU idle % → batching problem; CPU pegged in path.rs → pathfinding problem)
- GPU inference utilization during self-play ≥60%

---

## 10. Milestones (build in this order)

1. **M1 — Engine core:** state, movegen, wall legality, pathfinding. Pass full test suite (§9). Perft-style validation.
2. **M2 — Bindings + baselines:** PyO3 layer, position notation, random/greedy/minimax baselines playing full games. A human can play against greedy via a dead-simple CLI.
3. **M3 — MCTS + fake net:** batched MCTS working end-to-end against a stub evaluator (uniform policy, shortest-path-difference value). MCTS-with-stub should already beat minimax-2 — this validates search before ML enters the picture.
4. **M4 — Training loop + lifecycle:** full self-play → buffer → train → checkpoint → arena cycle running, WITH §7 built in from the start (graceful stop, resume, autosave) — retrofitting resumability is misery. Acceptance test: SIGINT mid-run, resume, counters/buffer/ladder intact; kill -9 mid-run, resume from autosave. Overfit smoke test: train on 1k fixed positions, loss must go to ~0.
5. **M5 — Observability:** §6 dashboards, status.json, per-checkpoint viz artifacts, peek/play tools. Must exist BEFORE the long run, or the long run is a black box.
6. **M6 — First real run:** 24–48h run, monitor Elo ladder, confirm baselines crushed and draw rate collapsing. Exercise graceful stop/resume at least once mid-run on purpose.
7. **M7 — Analysis suite:** opening evolution dumps, concept probes, viewer polish.

---

## 11. Testing Requirements (non-negotiable)

- **Rules unit tests:** exhaustive jump cases (straight jump, blocked-behind → both diagonals, diagonal blocked by wall, opponent on board edge, corner cases), wall overlap/cross matrices, path-blocking rejection (including walls that trap the PLACER — also illegal), no-walls-left behavior.
- **Perft:** node counts of the move tree to depth 4 from the start position and from 5+ handcrafted midgame positions. Compute once with a naive obviously-correct reference implementation (can be slow Python), freeze as golden values, run against the optimized engine in CI.
- **Differential testing:** naive Python rules implementation in `tests/reference.py`; property test plays 10k random games comparing legal-move sets and terminal detection move-by-move against Rust.
- **Encode/decode round-trips**, canonical-form involution (rotating twice = identity, action remap consistency), mirror-symmetry consistency (encode(mirror(s)) == mirror_planes(encode(s))).
- **MCTS sanity:** on a position with a forced mate-in-1, root policy must concentrate on the winning move with ≥95% visits at 200 sims.

---

## 12. Explicit Non-Goals (v1)

- No distributed/multi-GPU training, no ONNX/TensorRT export, no online play against barricade.gg servers, no GUI beyond the TUI/SVG viewer, no board sizes other than 9×9 (but don't hardcode 9 in ways that make 5×5 debug boards impossible — keep it a compile-time const generic if cheap).

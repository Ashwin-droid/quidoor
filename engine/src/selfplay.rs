use crate::EngineConfig;
use crate::encode;
use crate::mcts::{BatchedMcts, EvalRequest, bool_mask_to_u8};
use crate::state::{BOARD, CELLS, GameState, NUM_ACTIONS};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct LeafBatch {
    pub planes: Vec<f32>,
    pub leaf_ids: Vec<i64>,
    pub masks: Vec<u8>,
    pub batch: usize,
    pub channels: usize,
}

impl LeafBatch {
    pub fn empty(channels: usize) -> Self {
        Self {
            planes: Vec::new(),
            leaf_ids: Vec::new(),
            masks: Vec::new(),
            batch: 0,
            channels,
        }
    }
}

#[derive(Clone, Debug)]
pub struct FinishedGameRecord {
    pub states: Vec<f32>,
    pub policies: Vec<f32>,
    pub masks: Vec<u8>,
    pub to_move: Vec<u8>,
    pub position_strings: Vec<String>,
    pub outcome: i8,
    pub len: usize,
    pub channels: usize,
}

#[derive(Clone, Copy, Debug)]
struct PendingOwner {
    game_idx: usize,
    local_leaf_id: i64,
}

#[derive(Clone, Debug)]
struct PositionRecord {
    planes: Vec<f32>,
    policy: [f32; NUM_ACTIONS],
    mask: [u8; NUM_ACTIONS],
    to_move: u8,
    position: String,
}

#[derive(Clone, Debug)]
struct GameRunner {
    mcts: BatchedMcts,
    history: Vec<PositionRecord>,
}

pub struct SelfPlayManager {
    config: EngineConfig,
    total_games: usize,
    started_games: usize,
    finished_games: usize,
    games: Vec<Option<GameRunner>>,
    pending: HashMap<i64, PendingOwner>,
    next_leaf_id: i64,
    finished_records: Vec<FinishedGameRecord>,
    spawning_stopped: bool,
    rng: ChaCha8Rng,
    cursor: usize,
    pool: Option<rayon::ThreadPool>,
}

impl SelfPlayManager {
    pub fn new(config: EngineConfig, num_games: usize, seed: u64) -> Self {
        let pool_threads = std::thread::available_parallelism()
            .map(|threads| threads.get())
            .unwrap_or(1)
            .min(config.concurrent_games.max(1) as usize)
            .max(1);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(pool_threads)
            .thread_name(|idx| format!("quoridor-selfplay-{idx}"))
            .build()
            .expect("rayon self-play pool should build");

        let mut manager = Self {
            config,
            total_games: num_games,
            started_games: 0,
            finished_games: 0,
            games: Vec::new(),
            pending: HashMap::new(),
            next_leaf_id: 1,
            finished_records: Vec::new(),
            spawning_stopped: false,
            rng: ChaCha8Rng::seed_from_u64(seed),
            cursor: 0,
            pool: Some(pool),
        };

        let active = manager.config.concurrent_games.max(1).min(num_games as u32) as usize;
        manager.games.resize_with(active, || None);
        for idx in 0..active {
            manager.start_game_in_slot(idx);
        }
        manager
    }

    pub fn done(&self) -> bool {
        self.finished_games >= self.total_games && self.pending.is_empty()
    }

    pub fn stop_spawning(&mut self) {
        self.spawning_stopped = true;
        self.total_games = self.started_games;
    }

    pub fn collect_leaves(&mut self, max_batch: usize, timeout_ms: u64) -> LeafBatch {
        let pool = self.pool.take().expect("self-play pool must be present");
        let batch = pool.install(|| self.collect_leaves_inner(max_batch, timeout_ms));
        self.pool = Some(pool);
        batch
    }

    fn collect_leaves_inner(&mut self, max_batch: usize, timeout_ms: u64) -> LeafBatch {
        let channels = self.channels();
        if max_batch == 0 || self.done() {
            return LeafBatch::empty(channels);
        }

        self.progress_ready_games();

        let start = Instant::now();
        let timeout = Duration::from_millis(timeout_ms);
        let mut batch = LeafBatch {
            planes: Vec::with_capacity(max_batch * channels * CELLS),
            leaf_ids: Vec::with_capacity(max_batch),
            masks: Vec::with_capacity(max_batch * NUM_ACTIONS),
            batch: 0,
            channels,
        };

        while batch.batch < max_batch {
            let mut made_progress = false;
            let slots = self.games.len();
            if slots == 0 {
                break;
            }

            for offset in 0..slots {
                if batch.batch >= max_batch {
                    break;
                }
                let idx = (self.cursor + offset) % slots;
                self.progress_slot(idx);
                if self.games[idx].is_none() {
                    continue;
                }

                let leaves = self.games[idx]
                    .as_mut()
                    .expect("checked slot")
                    .mcts
                    .collect_leaves(1, 0);
                if leaves.is_empty() {
                    continue;
                }
                made_progress = true;
                for leaf in leaves {
                    self.push_leaf(idx, leaf, &mut batch);
                }
            }

            if slots > 0 {
                self.cursor = (self.cursor + 1) % slots;
            }
            if !made_progress {
                break;
            }
            if timeout_ms > 0 && start.elapsed() >= timeout {
                break;
            }
        }

        batch
    }

    pub fn submit(
        &mut self,
        leaf_ids: &[i64],
        policy_logits: &[f32],
        values: &[f32],
    ) -> Result<(), String> {
        if values.len() != leaf_ids.len() {
            return Err(format!(
                "values length {} does not match leaf_ids length {}",
                values.len(),
                leaf_ids.len()
            ));
        }
        if policy_logits.len() != leaf_ids.len() * NUM_ACTIONS {
            return Err(format!(
                "policy logits length {} does not match batch {} x {NUM_ACTIONS}",
                policy_logits.len(),
                leaf_ids.len()
            ));
        }

        for (row, &leaf_id) in leaf_ids.iter().enumerate() {
            let owner = self
                .pending
                .remove(&leaf_id)
                .ok_or_else(|| format!("unknown leaf id {leaf_id}"))?;
            let Some(game) = self.games.get_mut(owner.game_idx).and_then(Option::as_mut) else {
                return Err(format!(
                    "leaf id {leaf_id} belongs to inactive game slot {}",
                    owner.game_idx
                ));
            };
            let start = row * NUM_ACTIONS;
            game.mcts.submit_one(
                owner.local_leaf_id,
                &policy_logits[start..start + NUM_ACTIONS],
                values[row],
            )?;
        }

        self.progress_ready_games();
        Ok(())
    }

    pub fn drain_finished_games(&mut self) -> Vec<FinishedGameRecord> {
        std::mem::take(&mut self.finished_records)
    }

    pub fn finished_count(&self) -> usize {
        self.finished_games
    }

    pub fn active_count(&self) -> usize {
        self.games.iter().filter(|slot| slot.is_some()).count()
    }

    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    fn push_leaf(&mut self, game_idx: usize, leaf: EvalRequest, batch: &mut LeafBatch) {
        let external_id = self.next_leaf_id;
        self.next_leaf_id += 1;
        self.pending.insert(
            external_id,
            PendingOwner {
                game_idx,
                local_leaf_id: leaf.leaf_id,
            },
        );
        batch.planes.extend_from_slice(&leaf.planes);
        batch.leaf_ids.push(external_id);
        batch.masks.extend_from_slice(&leaf.mask);
        batch.batch += 1;
    }

    fn progress_ready_games(&mut self) {
        for idx in 0..self.games.len() {
            self.progress_slot(idx);
        }
    }

    fn progress_slot(&mut self, idx: usize) {
        loop {
            let Some(game) = self.games.get(idx).and_then(Option::as_ref) else {
                return;
            };

            if let Some(outcome) = game.mcts.terminal_outcome_player0() {
                self.finish_slot(idx, outcome);
                continue;
            }

            if !game.mcts.ready() {
                return;
            }

            self.play_ready_move(idx);
        }
    }

    fn play_ready_move(&mut self, idx: usize) {
        let config = self.config.clone();
        let game = self.games[idx].as_mut().expect("ready slot must be active");
        let state = game.mcts.root_state();
        let policy = game.mcts.root_policy_target();
        let mask = game.mcts.root_mask_u8();
        let planes = encode::encode(&state, config.extra_planes);
        let temperature = if state.ply < config.temperature_moves {
            config.temperature
        } else {
            config.temperature_final
        };

        let Some(action) = game.mcts.select_action_with_temperature(temperature) else {
            self.finish_slot(idx, 0);
            return;
        };

        game.history.push(PositionRecord {
            planes,
            policy,
            mask,
            to_move: state.to_move,
            position: state.to_position_string(),
        });
        game.mcts
            .advance(action)
            .expect("selected root action should advance");
    }

    fn finish_slot(&mut self, idx: usize, outcome: i8) {
        let Some(game) = self.games[idx].take() else {
            return;
        };

        let channels = self.channels();
        let mut states = Vec::with_capacity(game.history.len() * channels * CELLS);
        let mut policies = Vec::with_capacity(game.history.len() * NUM_ACTIONS);
        let mut masks = Vec::with_capacity(game.history.len() * NUM_ACTIONS);
        let mut to_move = Vec::with_capacity(game.history.len());
        let mut position_strings = Vec::with_capacity(game.history.len());
        for record in game.history {
            states.extend_from_slice(&record.planes);
            policies.extend_from_slice(&record.policy);
            masks.extend_from_slice(&record.mask);
            to_move.push(record.to_move);
            position_strings.push(record.position);
        }
        let len = policies_len_to_positions(&policies);

        self.finished_records.push(FinishedGameRecord {
            states,
            policies,
            masks,
            to_move,
            position_strings,
            outcome,
            len,
            channels,
        });
        self.finished_games += 1;

        if !self.spawning_stopped && self.started_games < self.total_games {
            self.start_game_in_slot(idx);
        }
    }

    fn start_game_in_slot(&mut self, idx: usize) {
        if self.started_games >= self.total_games {
            return;
        }
        let seed = self.rng.random::<u64>();
        let mcts = BatchedMcts::new(&self.config, GameState::new(), seed, true);
        self.games[idx] = Some(GameRunner {
            mcts,
            history: Vec::new(),
        });
        self.started_games += 1;
    }

    fn channels(&self) -> usize {
        if self.config.extra_planes {
            encode::EXTRA_PLANES
        } else {
            encode::BASE_PLANES
        }
    }
}

fn policies_len_to_positions(policies: &[f32]) -> usize {
    policies.len() / NUM_ACTIONS
}

pub fn empty_masks_for_states(states: &[GameState]) -> Vec<u8> {
    let mut masks = Vec::with_capacity(states.len() * NUM_ACTIONS);
    for state in states {
        masks.extend_from_slice(&bool_mask_to_u8(state.legal_mask()));
    }
    masks
}

pub fn leaf_batch_shapes(batch: &LeafBatch) -> ((usize, usize, usize, usize), (usize, usize)) {
    (
        (batch.batch, batch.channels, BOARD, BOARD),
        (batch.batch, NUM_ACTIONS),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcts::stub_eval_planes;

    fn tiny_config() -> EngineConfig {
        EngineConfig {
            sims_per_move: 12,
            fast_sims_per_move: 12,
            c_puct: 1.5,
            fpu_reduction: 0.2,
            dirichlet_alpha: 0.08,
            dirichlet_epsilon: 0.25,
            virtual_loss: 3,
            temperature_moves: 4,
            temperature: 1.0,
            temperature_final: 0.0,
            concurrent_games: 2,
            leaf_batch_size: 8,
            collect_timeout_ms: 1,
            ply_cap: 24,
            extra_planes: false,
        }
    }

    fn run_manager(seed: u64) -> Vec<(i8, usize)> {
        let cfg = tiny_config();
        let mut manager = SelfPlayManager::new(cfg, 3, seed);
        let mut finished = Vec::new();
        for _ in 0..20_000 {
            if manager.done() {
                finished.extend(
                    manager
                        .drain_finished_games()
                        .into_iter()
                        .map(|record| (record.outcome, record.len)),
                );
                return finished;
            }
            let batch = manager.collect_leaves(8, 0);
            if batch.batch > 0 {
                let (logits, values) =
                    stub_eval_planes(&batch.planes, batch.batch, batch.channels).unwrap();
                manager.submit(&batch.leaf_ids, &logits, &values).unwrap();
            }
            finished.extend(
                manager
                    .drain_finished_games()
                    .into_iter()
                    .map(|record| (record.outcome, record.len)),
            );
        }
        panic!("self-play manager did not finish");
    }

    #[test]
    fn selfplay_manager_runs_to_completion_with_stub_eval() {
        let finished = run_manager(123);
        assert_eq!(finished.len(), 3);
        assert!(finished.iter().all(|(_, len)| *len > 0));
    }

    #[test]
    fn selfplay_is_deterministic_for_seed() {
        let a = run_manager(9876);
        let b = run_manager(9876);
        assert_eq!(a, b);
    }

    #[test]
    fn finished_records_include_position_strings() {
        let cfg = tiny_config();
        let mut manager = SelfPlayManager::new(cfg, 1, 444);
        for _ in 0..20_000 {
            let batch = manager.collect_leaves(8, 0);
            if batch.batch > 0 {
                let (logits, values) =
                    stub_eval_planes(&batch.planes, batch.batch, batch.channels).unwrap();
                manager.submit(&batch.leaf_ids, &logits, &values).unwrap();
            }
            let records = manager.drain_finished_games();
            if let Some(record) = records.into_iter().next() {
                assert_eq!(record.position_strings.len(), record.len);
                assert!(
                    record
                        .position_strings
                        .iter()
                        .all(|text| text.starts_with("Q1 "))
                );
                assert!(
                    record
                        .position_strings
                        .iter()
                        .all(|text| GameState::from_position_string(text).is_ok())
                );
                return;
            }
        }
        panic!("self-play manager did not finish");
    }

    #[test]
    fn stop_spawning_drains_only_started_games() {
        let cfg = tiny_config();
        let mut manager = SelfPlayManager::new(cfg, 5, 555);
        assert_eq!(manager.active_count(), 2);
        manager.stop_spawning();
        let mut finished = 0;
        for _ in 0..20_000 {
            if manager.done() {
                finished += manager.drain_finished_games().len();
                assert_eq!(finished, 2);
                return;
            }
            let batch = manager.collect_leaves(8, 0);
            if batch.batch > 0 {
                let (logits, values) =
                    stub_eval_planes(&batch.planes, batch.batch, batch.channels).unwrap();
                manager.submit(&batch.leaf_ids, &logits, &values).unwrap();
            }
            finished += manager.drain_finished_games().len();
        }
        panic!("self-play manager did not finish");
    }
}

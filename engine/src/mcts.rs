use crate::EngineConfig;
use crate::encode;
use crate::path;
use crate::state::{BOARD, CELLS, GameState, NUM_ACTIONS, WALL_ANCHORS, cell};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Gamma};
use std::collections::HashMap;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug)]
pub struct MctsConfig {
    pub sims_per_move: u32,
    pub c_puct: f32,
    pub fpu_reduction: f32,
    pub dirichlet_alpha: f32,
    pub dirichlet_epsilon: f32,
    pub virtual_loss: u32,
    pub ply_cap: u16,
    pub extra_planes: bool,
}

impl From<&EngineConfig> for MctsConfig {
    fn from(config: &EngineConfig) -> Self {
        Self {
            sims_per_move: config.sims_per_move,
            c_puct: config.c_puct,
            fpu_reduction: config.fpu_reduction,
            dirichlet_alpha: config.dirichlet_alpha,
            dirichlet_epsilon: config.dirichlet_epsilon,
            virtual_loss: config.virtual_loss,
            ply_cap: config.ply_cap,
            extra_planes: config.extra_planes,
        }
    }
}

#[derive(Clone, Debug)]
pub struct EvalRequest {
    pub leaf_id: i64,
    pub planes: Vec<f32>,
    pub mask: [u8; NUM_ACTIONS],
}

#[derive(Clone, Debug)]
struct Node {
    state: GameState,
    visits: u32,
    value_sum: f32,
    edges: Vec<Edge>,
    expanded: bool,
    pending_evals: u32,
}

impl Node {
    fn new(state: GameState) -> Self {
        Self {
            state,
            visits: 0,
            value_sum: 0.0,
            edges: Vec::new(),
            expanded: false,
            pending_evals: 0,
        }
    }

    fn q(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.value_sum / self.visits as f32
        }
    }
}

#[derive(Clone, Debug)]
struct Edge {
    action: u8,
    prior: f32,
    child: Option<usize>,
    visits: u32,
    value_sum: f32,
}

impl Edge {
    fn q(&self) -> Option<f32> {
        (self.visits > 0).then_some(self.value_sum / self.visits as f32)
    }
}

#[derive(Clone, Copy, Debug)]
struct PathStep {
    node: usize,
    edge: usize,
}

#[derive(Clone, Debug)]
struct PendingEval {
    leaf_node: usize,
    nodes: Vec<usize>,
    steps: Vec<PathStep>,
    virtual_loss: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SimulationStep {
    Scheduled,
    Completed,
    Stalled,
}

#[derive(Clone, Debug)]
pub struct BatchedMcts {
    config: MctsConfig,
    arena: Vec<Node>,
    root: usize,
    rng: ChaCha8Rng,
    root_noise: bool,
    root_noise_applied: bool,
    next_leaf_id: i64,
    pending: HashMap<i64, PendingEval>,
    completed_sims: u32,
}

impl BatchedMcts {
    pub fn new(config: &EngineConfig, state: GameState, seed: u64, root_noise: bool) -> Self {
        Self::with_mcts_config(MctsConfig::from(config), state, seed, root_noise)
    }

    pub fn with_mcts_config(
        config: MctsConfig,
        state: GameState,
        seed: u64,
        root_noise: bool,
    ) -> Self {
        Self {
            config,
            arena: vec![Node::new(state)],
            root: 0,
            rng: ChaCha8Rng::seed_from_u64(seed),
            root_noise,
            root_noise_applied: false,
            next_leaf_id: 1,
            pending: HashMap::new(),
            completed_sims: 0,
        }
    }

    pub fn collect_leaves(&mut self, max_batch: usize, timeout_ms: u64) -> Vec<EvalRequest> {
        if max_batch == 0 {
            return Vec::new();
        }

        let start = Instant::now();
        let timeout = Duration::from_millis(timeout_ms);
        let mut out = Vec::with_capacity(max_batch);

        while out.len() < max_batch && self.can_schedule_more() {
            let before = out.len();
            match self.start_simulation(&mut out) {
                SimulationStep::Scheduled | SimulationStep::Completed => {}
                SimulationStep::Stalled => break,
            }

            if timeout_ms > 0 && start.elapsed() >= timeout {
                break;
            }
            if out.len() == before && !self.can_schedule_more() {
                break;
            }
        }

        out
    }

    pub fn submit_one(
        &mut self,
        leaf_id: i64,
        policy_logits: &[f32],
        value: f32,
    ) -> Result<(), String> {
        if policy_logits.len() != NUM_ACTIONS {
            return Err(format!(
                "policy logits must have length {NUM_ACTIONS}, got {}",
                policy_logits.len()
            ));
        }

        let pending = self
            .pending
            .remove(&leaf_id)
            .ok_or_else(|| format!("unknown leaf id {leaf_id}"))?;
        self.revert_virtual_loss(&pending.nodes, &pending.steps, pending.virtual_loss);

        let leaf = &mut self.arena[pending.leaf_node];
        leaf.pending_evals = leaf.pending_evals.saturating_sub(1);
        if !leaf.expanded && terminal_value(&leaf.state, self.config.ply_cap).is_none() {
            self.expand_with_logits(pending.leaf_node, policy_logits);
        }

        self.backup(&pending.nodes, &pending.steps, value.clamp(-1.0, 1.0));
        self.completed_sims = self.completed_sims.saturating_add(1);
        Ok(())
    }

    pub fn ready(&self) -> bool {
        self.completed_sims >= self.config.sims_per_move && self.pending.is_empty()
    }

    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    pub fn completed_sims(&self) -> u32 {
        self.completed_sims
    }

    pub fn channels(&self) -> usize {
        if self.config.extra_planes {
            encode::EXTRA_PLANES
        } else {
            encode::BASE_PLANES
        }
    }

    pub fn root_state(&self) -> GameState {
        self.arena[self.root].state
    }

    pub fn root_visits(&self) -> [f32; NUM_ACTIONS] {
        let mut visits = [0.0f32; NUM_ACTIONS];
        for edge in &self.arena[self.root].edges {
            visits[edge.action as usize] = edge.visits as f32;
        }
        visits
    }

    pub fn root_policy_target(&self) -> [f32; NUM_ACTIONS] {
        let visits = self.root_visits();
        normalize_visits(&visits)
    }

    pub fn root_mask_u8(&self) -> [u8; NUM_ACTIONS] {
        bool_mask_to_u8(self.arena[self.root].state.legal_mask())
    }

    pub fn best_action(&self) -> Option<u8> {
        self.arena[self.root]
            .edges
            .iter()
            .max_by(|a, b| {
                a.visits
                    .cmp(&b.visits)
                    .then_with(|| a.prior.total_cmp(&b.prior))
                    .then_with(|| b.action.cmp(&a.action))
            })
            .map(|edge| edge.action)
            .or_else(|| {
                self.arena[self.root]
                    .state
                    .legal_actions()
                    .into_iter()
                    .next()
            })
    }

    pub fn select_action_with_temperature(&mut self, temperature: f32) -> Option<u8> {
        if temperature <= 1.0e-6 {
            return self.best_action();
        }

        let root = &self.arena[self.root];
        if root.edges.is_empty() {
            let legal = root.state.legal_actions();
            if legal.is_empty() {
                return None;
            }
            return Some(legal[self.rng.random_range(0..legal.len())]);
        }

        let inv_temp = 1.0 / temperature.max(1.0e-6);
        let mut weights = Vec::with_capacity(root.edges.len());
        let mut total = 0.0f64;
        for edge in &root.edges {
            let weight = if edge.visits == 0 {
                0.0
            } else {
                (edge.visits as f64).powf(inv_temp as f64)
            };
            weights.push((edge.action, weight));
            total += weight;
        }

        if total <= f64::EPSILON || !total.is_finite() {
            let idx = self.rng.random_range(0..root.edges.len());
            return Some(root.edges[idx].action);
        }

        let mut sample = self.rng.random::<f64>() * total;
        for (action, weight) in weights {
            if sample <= weight {
                return Some(action);
            }
            sample -= weight;
        }
        self.best_action()
    }

    pub fn advance(&mut self, action: u8) -> Result<(), String> {
        if !self.pending.is_empty() {
            return Err("cannot advance while leaf evaluations are pending".to_string());
        }
        if action as usize >= NUM_ACTIONS {
            return Err(format!("bad action {action}"));
        }

        let child = self.arena[self.root]
            .edges
            .iter()
            .find(|edge| edge.action == action)
            .and_then(|edge| edge.child);

        if let Some(child_idx) = child {
            self.keep_subtree(child_idx);
        } else {
            let mut state = self.arena[self.root].state;
            state.apply_action(action).map_err(|err| err.to_string())?;
            self.arena.clear();
            self.arena.push(Node::new(state));
            self.root = 0;
        }

        self.completed_sims = 0;
        self.root_noise_applied = false;
        Ok(())
    }

    pub fn terminal_outcome_player0(&self) -> Option<i8> {
        outcome_player0(&self.arena[self.root].state, self.config.ply_cap)
    }

    fn can_schedule_more(&self) -> bool {
        self.completed_sims + self.pending.len() as u32 + 1 <= self.config.sims_per_move
    }

    fn start_simulation(&mut self, out: &mut Vec<EvalRequest>) -> SimulationStep {
        let mut node_idx = self.root;
        let mut nodes = vec![self.root];
        let mut steps = Vec::new();

        loop {
            if let Some(value) = terminal_value(&self.arena[node_idx].state, self.config.ply_cap) {
                self.backup(&nodes, &steps, value);
                self.completed_sims = self.completed_sims.saturating_add(1);
                return SimulationStep::Completed;
            }

            if !self.arena[node_idx].expanded {
                if self.arena[node_idx].pending_evals > 0 {
                    return SimulationStep::Stalled;
                }
                let leaf_id = self.next_leaf_id;
                self.next_leaf_id += 1;

                self.arena[node_idx].pending_evals += 1;
                let virtual_loss = self.config.virtual_loss;
                if virtual_loss > 0 {
                    self.apply_virtual_loss(&nodes, &steps, virtual_loss);
                }

                let request = EvalRequest {
                    leaf_id,
                    planes: encode::encode(&self.arena[node_idx].state, self.config.extra_planes),
                    mask: bool_mask_to_u8(self.arena[node_idx].state.legal_mask()),
                };
                self.pending.insert(
                    leaf_id,
                    PendingEval {
                        leaf_node: node_idx,
                        nodes,
                        steps,
                        virtual_loss,
                    },
                );
                out.push(request);
                return SimulationStep::Scheduled;
            }

            if node_idx == self.root {
                self.apply_root_noise_if_needed();
            }

            if self.arena[node_idx].edges.is_empty() {
                self.backup(&nodes, &steps, 0.0);
                self.completed_sims = self.completed_sims.saturating_add(1);
                return SimulationStep::Completed;
            }

            let edge_idx = self.select_edge(node_idx);
            let child_idx = self.ensure_child(node_idx, edge_idx);
            steps.push(PathStep {
                node: node_idx,
                edge: edge_idx,
            });
            nodes.push(child_idx);
            node_idx = child_idx;
        }
    }

    fn ensure_child(&mut self, node_idx: usize, edge_idx: usize) -> usize {
        if let Some(child) = self.arena[node_idx].edges[edge_idx].child {
            return child;
        }

        let action = self.arena[node_idx].edges[edge_idx].action;
        let mut child_state = self.arena[node_idx].state;
        child_state
            .apply_action(action)
            .expect("expanded edge action must remain legal");
        let child_idx = self.arena.len();
        self.arena.push(Node::new(child_state));
        self.arena[node_idx].edges[edge_idx].child = Some(child_idx);
        child_idx
    }

    fn select_edge(&self, node_idx: usize) -> usize {
        let node = &self.arena[node_idx];
        let parent_visits = node.visits.max(1) as f32;
        let sqrt_parent = parent_visits.sqrt();
        let fpu_q = node.q() - self.config.fpu_reduction;

        let mut best_idx = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for (idx, edge) in node.edges.iter().enumerate() {
            let q = edge.q().unwrap_or(fpu_q);
            let u = self.config.c_puct * edge.prior * sqrt_parent / (1.0 + edge.visits as f32);
            let score = q + u;
            if score > best_score {
                best_score = score;
                best_idx = idx;
            }
        }
        best_idx
    }

    fn expand_with_logits(&mut self, node_idx: usize, policy_logits: &[f32]) {
        let state = self.arena[node_idx].state;
        let mask = state.legal_mask();
        let mut logits = [0.0f32; NUM_ACTIONS];
        logits.copy_from_slice(policy_logits);
        let priors = masked_softmax(&logits, &mask);

        let mut edges = Vec::new();
        for action in 0..NUM_ACTIONS {
            if mask[action] {
                edges.push(Edge {
                    action: action as u8,
                    prior: priors[action],
                    child: None,
                    visits: 0,
                    value_sum: 0.0,
                });
            }
        }
        self.arena[node_idx].edges = edges;
        self.arena[node_idx].expanded = true;

        if node_idx == self.root {
            self.apply_root_noise_if_needed();
        }
    }

    fn apply_root_noise_if_needed(&mut self) {
        if !self.root_noise || self.root_noise_applied {
            return;
        }
        self.root_noise_applied = true;

        let root = &mut self.arena[self.root];
        if root.edges.len() <= 1 {
            return;
        }
        let alpha = self.config.dirichlet_alpha;
        let epsilon = self.config.dirichlet_epsilon.clamp(0.0, 1.0);
        if alpha <= 0.0 || epsilon <= 0.0 {
            return;
        }

        let Ok(gamma) = Gamma::<f32>::new(alpha, 1.0) else {
            return;
        };
        let mut noise = Vec::with_capacity(root.edges.len());
        let mut sum = 0.0f32;
        for _ in 0..root.edges.len() {
            let sample = gamma.sample(&mut self.rng).max(0.0);
            noise.push(sample);
            sum += sample;
        }
        if sum <= f32::EPSILON || !sum.is_finite() {
            return;
        }

        for (edge, sample) in root.edges.iter_mut().zip(noise) {
            let eta = sample / sum;
            edge.prior = (1.0 - epsilon) * edge.prior + epsilon * eta;
        }
    }

    fn apply_virtual_loss(&mut self, nodes: &[usize], steps: &[PathStep], virtual_loss: u32) {
        let loss = virtual_loss as f32;
        for &node_idx in nodes {
            let node = &mut self.arena[node_idx];
            node.visits = node.visits.saturating_add(virtual_loss);
            node.value_sum -= loss;
        }
        for step in steps {
            let edge = &mut self.arena[step.node].edges[step.edge];
            edge.visits = edge.visits.saturating_add(virtual_loss);
            edge.value_sum -= loss;
        }
    }

    fn revert_virtual_loss(&mut self, nodes: &[usize], steps: &[PathStep], virtual_loss: u32) {
        let loss = virtual_loss as f32;
        for &node_idx in nodes {
            let node = &mut self.arena[node_idx];
            node.visits = node.visits.saturating_sub(virtual_loss);
            node.value_sum += loss;
        }
        for step in steps {
            let edge = &mut self.arena[step.node].edges[step.edge];
            edge.visits = edge.visits.saturating_sub(virtual_loss);
            edge.value_sum += loss;
        }
    }

    fn backup(&mut self, nodes: &[usize], steps: &[PathStep], leaf_value: f32) {
        let leaf_depth = steps.len();

        for (depth, &node_idx) in nodes.iter().enumerate() {
            let value = signed_value_for_depth(leaf_value, leaf_depth - depth);
            let node = &mut self.arena[node_idx];
            node.visits = node.visits.saturating_add(1);
            node.value_sum += value;
        }

        for (depth, step) in steps.iter().enumerate() {
            let value = signed_value_for_depth(leaf_value, leaf_depth - depth);
            let edge = &mut self.arena[step.node].edges[step.edge];
            edge.visits = edge.visits.saturating_add(1);
            edge.value_sum += value;
        }
    }

    fn keep_subtree(&mut self, old_root: usize) {
        let mut new_arena = Vec::new();
        self.clone_subtree(old_root, &mut new_arena);
        self.arena = new_arena;
        self.root = 0;
    }

    fn clone_subtree(&self, old_idx: usize, new_arena: &mut Vec<Node>) -> usize {
        let new_idx = new_arena.len();
        let old = &self.arena[old_idx];
        new_arena.push(Node {
            state: old.state,
            visits: old.visits,
            value_sum: old.value_sum,
            edges: Vec::new(),
            expanded: old.expanded,
            pending_evals: 0,
        });

        let mut edges = old.edges.clone();
        for edge in &mut edges {
            edge.child = edge
                .child
                .map(|child_idx| self.clone_subtree(child_idx, new_arena));
        }
        new_arena[new_idx].edges = edges;
        new_idx
    }
}

pub fn terminal_value(state: &GameState, ply_cap: u16) -> Option<f32> {
    if let Some(winner) = state.winner() {
        Some(if winner == state.to_move { 1.0 } else { -1.0 })
    } else if state.ply >= ply_cap {
        Some(0.0)
    } else {
        None
    }
}

pub fn outcome_player0(state: &GameState, ply_cap: u16) -> Option<i8> {
    if let Some(winner) = state.winner() {
        Some(if winner == 0 { 1 } else { -1 })
    } else if state.ply >= ply_cap {
        Some(0)
    } else {
        None
    }
}

pub fn bool_mask_to_u8(mask: [bool; NUM_ACTIONS]) -> [u8; NUM_ACTIONS] {
    let mut out = [0u8; NUM_ACTIONS];
    for (idx, legal) in mask.into_iter().enumerate() {
        out[idx] = u8::from(legal);
    }
    out
}

pub fn normalize_visits(visits: &[f32; NUM_ACTIONS]) -> [f32; NUM_ACTIONS] {
    let sum: f32 = visits.iter().copied().sum();
    if sum <= f32::EPSILON {
        return [0.0; NUM_ACTIONS];
    }
    let mut out = [0.0f32; NUM_ACTIONS];
    for action in 0..NUM_ACTIONS {
        out[action] = visits[action] / sum;
    }
    out
}

pub fn stub_eval_planes(
    planes: &[f32],
    batch: usize,
    channels: usize,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    if channels < encode::BASE_PLANES {
        return Err(format!(
            "stub_eval needs at least {} channels, got {channels}",
            encode::BASE_PLANES
        ));
    }
    let sample_len = channels * CELLS;
    if planes.len() != batch * sample_len {
        return Err(format!(
            "bad planes length {}, expected {} for batch={batch}, channels={channels}",
            planes.len(),
            batch * sample_len
        ));
    }

    let logits = vec![0.0f32; batch * NUM_ACTIONS];
    let mut values = Vec::with_capacity(batch);
    for sample in planes.chunks_exact(sample_len) {
        values.push(stub_value_from_encoded(sample, channels));
    }
    Ok((logits, values))
}

pub fn stub_eval_requests(requests: &[EvalRequest], channels: usize) -> (Vec<f32>, Vec<f32>) {
    let mut planes = Vec::new();
    for request in requests {
        planes.extend_from_slice(&request.planes);
    }
    stub_eval_planes(&planes, requests.len(), channels).expect("requests are internally shaped")
}

fn stub_value_from_encoded(sample: &[f32], channels: usize) -> f32 {
    debug_assert!(channels >= encode::BASE_PLANES);
    let current = one_hot_cell(&sample[0..CELLS]).unwrap_or(cell(0, 4));
    let opponent = one_hot_cell(&sample[CELLS..2 * CELLS]).unwrap_or(cell(BOARD as u8 - 1, 4));
    let hwalls = wall_bits_from_h_plane(&sample[2 * CELLS..3 * CELLS]);
    let vwalls = wall_bits_from_v_plane(&sample[3 * CELLS..4 * CELLS]);

    let state = GameState {
        pawns: [current, opponent],
        hwalls,
        vwalls,
        walls_left: [10, 10],
        to_move: 0,
        ply: 0,
    };
    let current_len = path::shortest_path(&state, 0)
        .map(|trace| trace.len as f32)
        .unwrap_or(80.0);
    let opponent_len = path::shortest_path(&state, 1)
        .map(|trace| trace.len as f32)
        .unwrap_or(80.0);
    ((opponent_len - current_len) / 4.0).tanh()
}

fn one_hot_cell(plane: &[f32]) -> Option<u8> {
    plane
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .and_then(|(idx, &value)| (value > 0.0).then_some(idx as u8))
}

fn wall_bits_from_h_plane(plane: &[f32]) -> u64 {
    let mut bits = 0u64;
    for anchor in 0..WALL_ANCHORS {
        let r = anchor / (BOARD - 1);
        let c = anchor % (BOARD - 1);
        if plane[r * BOARD + c] > 0.5 && plane[r * BOARD + c + 1] > 0.5 {
            bits |= 1u64 << anchor;
        }
    }
    bits
}

fn wall_bits_from_v_plane(plane: &[f32]) -> u64 {
    let mut bits = 0u64;
    for anchor in 0..WALL_ANCHORS {
        let r = anchor / (BOARD - 1);
        let c = anchor % (BOARD - 1);
        if plane[r * BOARD + c] > 0.5 && plane[(r + 1) * BOARD + c] > 0.5 {
            bits |= 1u64 << anchor;
        }
    }
    bits
}

fn masked_softmax(logits: &[f32; NUM_ACTIONS], mask: &[bool; NUM_ACTIONS]) -> [f32; NUM_ACTIONS] {
    let legal_count = mask.iter().filter(|&&legal| legal).count();
    if legal_count == 0 {
        return [0.0; NUM_ACTIONS];
    }

    let mut max_logit = f32::NEG_INFINITY;
    for action in 0..NUM_ACTIONS {
        if mask[action] && logits[action].is_finite() {
            max_logit = max_logit.max(logits[action]);
        }
    }

    if !max_logit.is_finite() {
        let uniform = 1.0 / legal_count as f32;
        let mut out = [0.0f32; NUM_ACTIONS];
        for action in 0..NUM_ACTIONS {
            if mask[action] {
                out[action] = uniform;
            }
        }
        return out;
    }

    let mut out = [0.0f32; NUM_ACTIONS];
    let mut sum = 0.0f32;
    for action in 0..NUM_ACTIONS {
        if mask[action] {
            let value = if logits[action].is_finite() {
                (logits[action] - max_logit).exp()
            } else {
                0.0
            };
            out[action] = value;
            sum += value;
        }
    }

    if sum <= f32::EPSILON || !sum.is_finite() {
        let uniform = 1.0 / legal_count as f32;
        for action in 0..NUM_ACTIONS {
            out[action] = if mask[action] { uniform } else { 0.0 };
        }
        return out;
    }

    for prior in &mut out {
        *prior /= sum;
    }
    out
}

fn signed_value_for_depth(value: f32, plies_from_node_to_leaf: usize) -> f32 {
    if plies_from_node_to_leaf % 2 == 0 {
        value
    } else {
        -value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::cell;

    fn small_config(sims: u32) -> EngineConfig {
        EngineConfig {
            sims_per_move: sims,
            fast_sims_per_move: sims,
            c_puct: 1.5,
            fpu_reduction: 0.2,
            dirichlet_alpha: 0.08,
            dirichlet_epsilon: 0.25,
            virtual_loss: 3,
            temperature_moves: 16,
            temperature: 1.0,
            temperature_final: 0.0,
            concurrent_games: 4,
            leaf_batch_size: 32,
            collect_timeout_ms: 1,
            ply_cap: 200,
            extra_planes: false,
        }
    }

    fn run_with_stub(mcts: &mut BatchedMcts) {
        let channels = if mcts.config.extra_planes {
            encode::EXTRA_PLANES
        } else {
            encode::BASE_PLANES
        };
        while !mcts.ready() {
            let leaves = mcts.collect_leaves(32, 0);
            assert!(!leaves.is_empty(), "MCTS stalled before ready");
            let (logits, values) = stub_eval_requests(&leaves, channels);
            for (idx, leaf) in leaves.iter().enumerate() {
                let start = idx * NUM_ACTIONS;
                mcts.submit_one(
                    leaf.leaf_id,
                    &logits[start..start + NUM_ACTIONS],
                    values[idx],
                )
                .unwrap();
            }
        }
    }

    #[test]
    fn mate_in_one_gets_nearly_all_visits() {
        let state = GameState {
            pawns: [cell(7, 4), cell(1, 0)],
            hwalls: 0,
            vwalls: 0,
            walls_left: [10, 10],
            to_move: 0,
            ply: 20,
        };
        let cfg = small_config(200);
        let mut mcts = BatchedMcts::new(&cfg, state, 7, false);
        run_with_stub(&mut mcts);

        let visits = mcts.root_visits();
        let total: f32 = visits.iter().sum();
        assert!(total > 0.0);
        assert!(
            visits[1] / total >= 0.95,
            "winning south move should dominate, got {:.3}",
            visits[1] / total
        );
    }

    #[test]
    fn virtual_loss_is_reverted_on_submit() {
        let cfg = small_config(4);
        let mut mcts = BatchedMcts::new(&cfg, GameState::new(), 11, false);
        let leaves = mcts.collect_leaves(1, 0);
        assert_eq!(leaves.len(), 1);
        assert_eq!(mcts.pending_count(), 1);
        assert_eq!(mcts.arena[mcts.root].visits, cfg.virtual_loss);
        assert!((mcts.arena[mcts.root].value_sum + cfg.virtual_loss as f32).abs() < 1.0e-6);

        let logits = [0.0f32; NUM_ACTIONS];
        mcts.submit_one(leaves[0].leaf_id, &logits, 0.25).unwrap();
        assert_eq!(mcts.pending_count(), 0);
        assert_eq!(mcts.arena[mcts.root].visits, 1);
        assert!((mcts.arena[mcts.root].value_sum - 0.25).abs() < 1.0e-6);
    }
}

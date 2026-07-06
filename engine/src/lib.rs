pub mod encode;
pub mod mcts;
pub mod movegen;
pub mod path;
pub mod selfplay;
pub mod state;

use crate::state::{BOARD, NUM_ACTIONS};
use numpy::{IntoPyArray, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::borrow::Cow;
use std::fmt::Display;

#[pyclass(name = "GameState")]
#[derive(Clone, Debug)]
pub struct PyGameState {
    pub(crate) inner: state::GameState,
}

#[pymethods]
impl PyGameState {
    #[new]
    fn new() -> Self {
        Self {
            inner: state::GameState::new(),
        }
    }

    #[staticmethod]
    fn from_string(text: &str) -> PyResult<Self> {
        let inner = state::GameState::from_position_string(text)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(Self { inner })
    }

    fn to_string(&self) -> String {
        self.inner.to_position_string()
    }

    fn legal_mask(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let data: Vec<bool> = self.inner.legal_mask().into_iter().collect();
        numpy_array(py, data, None, None)
    }

    fn legal_actions(&self) -> Vec<u16> {
        // Vec<u8> converts to Python bytes; widen so this is a real list[int] per CONTRACTS.md.
        self.inner
            .legal_actions()
            .into_iter()
            .map(u16::from)
            .collect()
    }

    fn apply(&mut self, action: usize) -> PyResult<()> {
        if action >= NUM_ACTIONS {
            return Err(PyValueError::new_err(format!("illegal action {action}")));
        }
        self.inner
            .apply_action(action as u8)
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }

    fn winner(&self) -> Option<u8> {
        self.inner.winner()
    }

    #[getter]
    fn to_move(&self) -> u8 {
        self.inner.to_move
    }

    #[getter]
    fn ply(&self) -> u16 {
        self.inner.ply
    }

    fn pawn(&self, player: usize) -> PyResult<(u8, u8)> {
        if player >= 2 {
            return Err(PyValueError::new_err("player must be 0 or 1"));
        }
        Ok(self.inner.pawn(player as u8).expect("validated player"))
    }

    fn walls_left(&self, player: usize) -> PyResult<u8> {
        if player >= 2 {
            return Err(PyValueError::new_err("player must be 0 or 1"));
        }
        Ok(self
            .inner
            .walls_left(player as u8)
            .expect("validated player"))
    }

    fn shortest_path_len(&self, player: usize) -> PyResult<u8> {
        if player >= 2 {
            return Err(PyValueError::new_err("player must be 0 or 1"));
        }
        self.inner
            .shortest_path_len(player as u8)
            .ok_or_else(|| PyValueError::new_err("player has no path to goal"))
    }

    fn encode(&self, py: Python<'_>, extra_planes: bool) -> PyResult<Py<PyAny>> {
        let data = encode::encode(&self.inner, extra_planes);
        let planes = if extra_planes {
            encode::EXTRA_PLANES
        } else {
            encode::BASE_PLANES
        };
        numpy_array(py, data, Some("float32"), Some((planes, BOARD, BOARD)))
    }

    fn clone(&self) -> Self {
        Self { inner: self.inner }
    }

    fn __repr__(&self) -> String {
        format!("GameState('{}')", self.inner.to_position_string())
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct EngineConfig {
    #[pyo3(get, set)]
    pub sims_per_move: u32,
    #[pyo3(get, set)]
    pub fast_sims_per_move: u32,
    #[pyo3(get, set)]
    pub c_puct: f32,
    #[pyo3(get, set)]
    pub fpu_reduction: f32,
    #[pyo3(get, set)]
    pub dirichlet_alpha: f32,
    #[pyo3(get, set)]
    pub dirichlet_epsilon: f32,
    #[pyo3(get, set)]
    pub virtual_loss: u32,
    #[pyo3(get, set)]
    pub temperature_moves: u16,
    #[pyo3(get, set)]
    pub temperature: f32,
    #[pyo3(get, set)]
    pub temperature_final: f32,
    #[pyo3(get, set)]
    pub concurrent_games: u32,
    #[pyo3(get, set)]
    pub leaf_batch_size: usize,
    #[pyo3(get, set)]
    pub collect_timeout_ms: u64,
    #[pyo3(get, set)]
    pub ply_cap: u16,
    #[pyo3(get, set)]
    pub extra_planes: bool,
}

#[pymethods]
impl EngineConfig {
    #[new]
    #[pyo3(signature = (
        sims_per_move = 800,
        fast_sims_per_move = 400,
        c_puct = 1.5,
        fpu_reduction = 0.2,
        dirichlet_alpha = 0.08,
        dirichlet_epsilon = 0.25,
        virtual_loss = 3,
        temperature_moves = 16,
        temperature = 1.0,
        temperature_final = 0.0,
        concurrent_games = 256,
        leaf_batch_size = 1024,
        collect_timeout_ms = 2,
        ply_cap = 200,
        extra_planes = false
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        sims_per_move: u32,
        fast_sims_per_move: u32,
        c_puct: f32,
        fpu_reduction: f32,
        dirichlet_alpha: f32,
        dirichlet_epsilon: f32,
        virtual_loss: u32,
        temperature_moves: u16,
        temperature: f32,
        temperature_final: f32,
        concurrent_games: u32,
        leaf_batch_size: usize,
        collect_timeout_ms: u64,
        ply_cap: u16,
        extra_planes: bool,
    ) -> Self {
        Self {
            sims_per_move,
            fast_sims_per_move,
            c_puct,
            fpu_reduction,
            dirichlet_alpha,
            dirichlet_epsilon,
            virtual_loss,
            temperature_moves,
            temperature,
            temperature_final,
            concurrent_games,
            leaf_batch_size,
            collect_timeout_ms,
            ply_cap,
            extra_planes,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "EngineConfig(sims_per_move={}, concurrent_games={}, ply_cap={}, extra_planes={})",
            self.sims_per_move, self.concurrent_games, self.ply_cap, self.extra_planes
        )
    }
}

#[pyclass(name = "BatchedMcts")]
#[derive(Debug)]
pub struct PyBatchedMcts {
    inner: mcts::BatchedMcts,
}

#[pymethods]
impl PyBatchedMcts {
    #[new]
    #[pyo3(signature = (config, state, seed = 0))]
    fn new(config: PyRef<'_, EngineConfig>, state: PyRef<'_, PyGameState>, seed: u64) -> Self {
        Self {
            inner: mcts::BatchedMcts::new(&config, state.inner, seed, false),
        }
    }

    #[pyo3(signature = (max_batch = 1024, timeout_ms = 2))]
    fn collect_leaves(
        &mut self,
        py: Python<'_>,
        max_batch: usize,
        timeout_ms: u64,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
        let channels = self.inner.channels();
        let leaves = self.inner.collect_leaves(max_batch, timeout_ms);
        eval_requests_to_numpy(py, leaves, channels)
    }

    fn submit(
        &mut self,
        leaf_ids: PyReadonlyArray1<'_, i64>,
        policy_logits: PyReadonlyArray2<'_, f32>,
        values: PyReadonlyArray1<'_, f32>,
    ) -> PyResult<()> {
        let ids = leaf_ids.as_slice().map_err(contiguous_array_err)?;
        let logits_shape = policy_logits.as_array().shape().to_vec();
        if logits_shape != [ids.len(), NUM_ACTIONS] {
            return Err(PyValueError::new_err(format!(
                "policy_logits must have shape ({}, {NUM_ACTIONS}), got {:?}",
                ids.len(),
                logits_shape
            )));
        }
        let logits = policy_logits.as_slice().map_err(contiguous_array_err)?;
        let values = values.as_slice().map_err(contiguous_array_err)?;
        if values.len() != ids.len() {
            return Err(PyValueError::new_err(format!(
                "values length {} does not match leaf_ids length {}",
                values.len(),
                ids.len()
            )));
        }

        for (idx, &leaf_id) in ids.iter().enumerate() {
            let start = idx * NUM_ACTIONS;
            self.inner
                .submit_one(leaf_id, &logits[start..start + NUM_ACTIONS], values[idx])
                .map_err(PyValueError::new_err)?;
        }
        Ok(())
    }

    fn ready(&self) -> bool {
        self.inner.ready()
    }

    fn root_visits(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        array1_f32(py, self.inner.root_visits().to_vec())
    }

    fn best_action(&self) -> PyResult<u8> {
        self.inner
            .best_action()
            .ok_or_else(|| PyValueError::new_err("root has no legal actions"))
    }

    fn advance(&mut self, action: usize) -> PyResult<()> {
        if action >= NUM_ACTIONS {
            return Err(PyValueError::new_err(format!("bad action {action}")));
        }
        self.inner
            .advance(action as u8)
            .map_err(PyValueError::new_err)
    }

    fn __repr__(&self) -> String {
        format!(
            "BatchedMcts(ready={}, completed_sims={}, pending={})",
            self.inner.ready(),
            self.inner.completed_sims(),
            self.inner.pending_count()
        )
    }
}

#[pyclass(name = "SelfPlayManager")]
pub struct PySelfPlayManager {
    inner: selfplay::SelfPlayManager,
}

type PyFinishedGame = Py<PyFinishedGameRecord>;

#[pyclass(name = "FinishedGameRecord")]
pub struct PyFinishedGameRecord {
    states: Py<PyAny>,
    policies: Py<PyAny>,
    masks: Py<PyAny>,
    to_move: Py<PyAny>,
    outcome: i8,
    position_strings: Vec<String>,
}

#[pymethods]
impl PyFinishedGameRecord {
    #[getter]
    fn states(&self, py: Python<'_>) -> Py<PyAny> {
        self.states.clone_ref(py)
    }

    #[getter]
    fn policies(&self, py: Python<'_>) -> Py<PyAny> {
        self.policies.clone_ref(py)
    }

    #[getter]
    fn masks(&self, py: Python<'_>) -> Py<PyAny> {
        self.masks.clone_ref(py)
    }

    #[getter]
    fn to_move(&self, py: Python<'_>) -> Py<PyAny> {
        self.to_move.clone_ref(py)
    }

    #[getter]
    fn outcome(&self) -> i8 {
        self.outcome
    }

    #[getter]
    fn position_strings(&self) -> Vec<String> {
        self.position_strings.clone()
    }

    fn __len__(&self) -> usize {
        5
    }

    fn __getitem__(&self, index: isize, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let normalized = if index < 0 { index + 5 } else { index };
        match normalized {
            0 => Ok(self.states.clone_ref(py)),
            1 => Ok(self.policies.clone_ref(py)),
            2 => Ok(self.masks.clone_ref(py)),
            3 => Ok(self.to_move.clone_ref(py)),
            4 => Ok(self.outcome.into_pyobject(py)?.into_any().unbind()),
            _ => Err(PyIndexError::new_err("finished-game index out of range")),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "FinishedGameRecord(len={}, outcome={})",
            self.position_strings.len(),
            self.outcome
        )
    }
}

#[pymethods]
impl PySelfPlayManager {
    #[new]
    #[pyo3(signature = (config, num_games = None, seed = 0))]
    fn new(config: PyRef<'_, EngineConfig>, num_games: Option<usize>, seed: u64) -> Self {
        let cfg = config.clone();
        let total_games = num_games.unwrap_or(cfg.concurrent_games.max(1) as usize);
        Self {
            inner: selfplay::SelfPlayManager::new(cfg, total_games, seed),
        }
    }

    fn done(&self) -> bool {
        self.inner.done()
    }

    fn stop_spawning(&mut self) {
        self.inner.stop_spawning()
    }

    #[pyo3(signature = (max_batch = 1024, timeout_ms = 2))]
    fn collect_leaves(
        &mut self,
        py: Python<'_>,
        max_batch: usize,
        timeout_ms: u64,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
        let batch = self.inner.collect_leaves(max_batch, timeout_ms);
        leaf_batch_to_numpy(py, batch)
    }

    fn submit(
        &mut self,
        leaf_ids: PyReadonlyArray1<'_, i64>,
        policy_logits: PyReadonlyArray2<'_, f32>,
        values: PyReadonlyArray1<'_, f32>,
    ) -> PyResult<()> {
        let ids = leaf_ids.as_slice().map_err(contiguous_array_err)?;
        let logits_shape = policy_logits.as_array().shape().to_vec();
        if logits_shape != [ids.len(), NUM_ACTIONS] {
            return Err(PyValueError::new_err(format!(
                "policy_logits must have shape ({}, {NUM_ACTIONS}), got {:?}",
                ids.len(),
                logits_shape
            )));
        }
        let logits = policy_logits.as_slice().map_err(contiguous_array_err)?;
        let values = values.as_slice().map_err(contiguous_array_err)?;
        self.inner
            .submit(ids, logits, values)
            .map_err(PyValueError::new_err)
    }

    fn drain_finished_games(&mut self, py: Python<'_>) -> PyResult<Vec<PyFinishedGame>> {
        self.inner
            .drain_finished_games()
            .into_iter()
            .map(|record| {
                let states = array4_f32(
                    py,
                    record.states,
                    (record.len, record.channels, BOARD, BOARD),
                )?;
                let policies = array2_f32(py, record.policies, (record.len, NUM_ACTIONS))?;
                let masks = array2_u8(py, record.masks, (record.len, NUM_ACTIONS))?;
                let to_move = array1_u8(py, record.to_move)?;
                Py::new(
                    py,
                    PyFinishedGameRecord {
                        states,
                        policies,
                        masks,
                        to_move,
                        outcome: record.outcome,
                        position_strings: record.position_strings,
                    },
                )
            })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "SelfPlayManager(done={}, active_games={}, finished={}, pending={})",
            self.inner.done(),
            self.inner.active_count(),
            self.inner.finished_count(),
            self.inner.pending_count()
        )
    }
}

#[pyfunction]
fn mirror_action(action: usize) -> PyResult<u8> {
    if action >= NUM_ACTIONS {
        return Err(PyValueError::new_err(format!("bad action {action}")));
    }
    encode::mirror_action(action as u8).ok_or_else(|| PyValueError::new_err("bad action"))
}

#[pyfunction]
fn canonical_action(action: usize, to_move: usize) -> PyResult<u8> {
    if action >= NUM_ACTIONS {
        return Err(PyValueError::new_err(format!("bad action {action}")));
    }
    if to_move > 1 {
        return Err(PyValueError::new_err("to_move must be 0 or 1"));
    }
    encode::canonical_action(action as u8, to_move as u8)
        .ok_or_else(|| PyValueError::new_err("bad canonical action request"))
}

#[pyfunction]
fn stub_eval(
    py: Python<'_>,
    planes: PyReadonlyArrayDyn<'_, f32>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let view = planes.as_array();
    let shape = view.shape();
    if shape.len() != 4 || shape[2] != BOARD || shape[3] != BOARD {
        return Err(PyValueError::new_err(format!(
            "planes must have shape [B, C, {BOARD}, {BOARD}], got {:?}",
            shape
        )));
    }
    let batch = shape[0];
    let channels = shape[1];
    let data: Cow<'_, [f32]> = match planes.as_slice() {
        Ok(slice) => Cow::Borrowed(slice),
        Err(_) => Cow::Owned(view.iter().copied().collect()),
    };
    let (logits, values) =
        mcts::stub_eval_planes(&data, batch, channels).map_err(PyValueError::new_err)?;
    Ok((
        array2_f32(py, logits, (batch, NUM_ACTIONS))?,
        array1_f32(py, values)?,
    ))
}

#[pymodule]
fn quoridor_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("NUM_ACTIONS", NUM_ACTIONS)?;
    m.add("BOARD", BOARD)?;
    m.add_class::<PyGameState>()?;
    m.add_class::<EngineConfig>()?;
    m.add_class::<PyBatchedMcts>()?;
    m.add_class::<PySelfPlayManager>()?;
    m.add_class::<PyFinishedGameRecord>()?;
    m.add_function(wrap_pyfunction!(mirror_action, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_action, m)?)?;
    m.add_function(wrap_pyfunction!(stub_eval, m)?)?;
    Ok(())
}

fn eval_requests_to_numpy(
    py: Python<'_>,
    leaves: Vec<mcts::EvalRequest>,
    channels: usize,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    let batch = leaves.len();
    let mut planes = Vec::with_capacity(batch * channels * BOARD * BOARD);
    let mut ids = Vec::with_capacity(batch);
    let mut masks = Vec::with_capacity(batch * NUM_ACTIONS);
    for leaf in leaves {
        planes.extend_from_slice(&leaf.planes);
        ids.push(leaf.leaf_id);
        masks.extend_from_slice(&leaf.mask);
    }
    Ok((
        array4_f32(py, planes, (batch, channels, BOARD, BOARD))?,
        array1_i64(py, ids)?,
        array2_u8(py, masks, (batch, NUM_ACTIONS))?,
    ))
}

fn leaf_batch_to_numpy(
    py: Python<'_>,
    batch: selfplay::LeafBatch,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    Ok((
        array4_f32(
            py,
            batch.planes,
            (batch.batch, batch.channels, BOARD, BOARD),
        )?,
        array1_i64(py, batch.leaf_ids)?,
        array2_u8(py, batch.masks, (batch.batch, NUM_ACTIONS))?,
    ))
}

fn array1_f32(py: Python<'_>, data: Vec<f32>) -> PyResult<Py<PyAny>> {
    Ok(data.into_pyarray(py).into_any().unbind())
}

fn array1_i64(py: Python<'_>, data: Vec<i64>) -> PyResult<Py<PyAny>> {
    Ok(data.into_pyarray(py).into_any().unbind())
}

fn array1_u8(py: Python<'_>, data: Vec<u8>) -> PyResult<Py<PyAny>> {
    Ok(data.into_pyarray(py).into_any().unbind())
}

fn array2_f32(py: Python<'_>, data: Vec<f32>, shape: (usize, usize)) -> PyResult<Py<PyAny>> {
    Ok(data
        .into_pyarray(py)
        .reshape([shape.0, shape.1])?
        .into_any()
        .unbind())
}

fn array2_u8(py: Python<'_>, data: Vec<u8>, shape: (usize, usize)) -> PyResult<Py<PyAny>> {
    Ok(data
        .into_pyarray(py)
        .reshape([shape.0, shape.1])?
        .into_any()
        .unbind())
}

fn array4_f32(
    py: Python<'_>,
    data: Vec<f32>,
    shape: (usize, usize, usize, usize),
) -> PyResult<Py<PyAny>> {
    Ok(data
        .into_pyarray(py)
        .reshape([shape.0, shape.1, shape.2, shape.3])?
        .into_any()
        .unbind())
}

fn contiguous_array_err(err: impl Display) -> PyErr {
    PyValueError::new_err(format!("expected a contiguous numpy array: {err}"))
}

fn numpy_array<T>(
    py: Python<'_>,
    data: Vec<T>,
    dtype_name: Option<&str>,
    shape: Option<(usize, usize, usize)>,
) -> PyResult<Py<PyAny>>
where
    T: for<'py> IntoPyObject<'py>,
{
    let np = py.import("numpy")?;
    let list = PyList::new(py, data)?;
    let arr = if let Some(dtype_name) = dtype_name {
        let kwargs = PyDict::new(py);
        kwargs.set_item("dtype", np.getattr(dtype_name)?)?;
        np.getattr("array")?.call((list,), Some(&kwargs))?
    } else {
        np.getattr("array")?.call1((list,))?
    };
    let arr = if let Some((a, b, c)) = shape {
        arr.call_method1("reshape", (a, b, c))?
    } else {
        arr
    };
    Ok(arr.unbind())
}

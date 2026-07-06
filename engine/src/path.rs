use crate::state::{BOARD, CELLS, GameState, WALL_GRID, WallOrientation, cell, col, row};
use std::sync::OnceLock;

pub const DIR_N: usize = 0;
pub const DIR_S: usize = 1;
pub const DIR_E: usize = 2;
pub const DIR_W: usize = 3;

#[derive(Clone, Copy, Debug, Default)]
pub struct EdgeMask {
    pub neighbor: u8,
    pub valid: bool,
    pub hmask: u64,
    pub vmask: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PathTrace {
    pub len: u8,
    pub cells: [u8; CELLS],
}

static EDGE_MASKS: OnceLock<[[EdgeMask; 4]; CELLS]> = OnceLock::new();

pub fn shortest_path(state: &GameState, player: u8) -> Option<PathTrace> {
    shortest_path_bits(
        state.hwalls,
        state.vwalls,
        state.pawns[player as usize],
        player,
    )
}

pub fn shortest_path_bits(hwalls: u64, vwalls: u64, start: u8, player: u8) -> Option<PathTrace> {
    let goal_row = if player == 0 { BOARD as u8 - 1 } else { 0 };
    shortest_path_to_goal_row(hwalls, vwalls, start, goal_row)
}

pub fn has_path_bits(hwalls: u64, vwalls: u64, start: u8, player: u8) -> bool {
    shortest_path_bits(hwalls, vwalls, start, player).is_some()
}

pub fn shortest_path_to_goal_row(
    hwalls: u64,
    vwalls: u64,
    start: u8,
    goal_row: u8,
) -> Option<PathTrace> {
    let mut parent = [u8::MAX; CELLS];
    let mut visited = [false; CELLS];
    let mut queue = [0u8; CELLS];
    let mut head = 0usize;
    let mut tail = 0usize;

    visited[start as usize] = true;
    parent[start as usize] = start;
    queue[tail] = start;
    tail += 1;

    let mut found = None;
    while head < tail {
        let current = queue[head];
        head += 1;
        if row(current) == goal_row {
            found = Some(current);
            break;
        }
        for dir in 0..4 {
            if let Some(next) = neighbor_if_open_bits(hwalls, vwalls, current, dir) {
                let idx = next as usize;
                if !visited[idx] {
                    visited[idx] = true;
                    parent[idx] = current;
                    queue[tail] = next;
                    tail += 1;
                }
            }
        }
    }

    let goal = found?;
    let mut rev = [0u8; CELLS];
    let mut count = 0usize;
    let mut cur = goal;
    loop {
        rev[count] = cur;
        count += 1;
        if cur == start {
            break;
        }
        cur = parent[cur as usize];
    }

    let mut cells = [0u8; CELLS];
    for i in 0..count {
        cells[i] = rev[count - 1 - i];
    }

    Some(PathTrace {
        len: (count - 1) as u8,
        cells,
    })
}

pub fn distance_map_to_goal_row(hwalls: u64, vwalls: u64, goal_row: u8) -> [u8; CELLS] {
    let mut dist = [u8::MAX; CELLS];
    let mut queue = [0u8; CELLS];
    let mut head = 0usize;
    let mut tail = 0usize;

    for c in 0..BOARD as u8 {
        let start = cell(goal_row, c);
        dist[start as usize] = 0;
        queue[tail] = start;
        tail += 1;
    }

    while head < tail {
        let current = queue[head];
        head += 1;
        let next_dist = dist[current as usize].saturating_add(1);
        for dir in 0..4 {
            if let Some(next) = neighbor_if_open_bits(hwalls, vwalls, current, dir) {
                let idx = next as usize;
                if dist[idx] == u8::MAX {
                    dist[idx] = next_dist;
                    queue[tail] = next;
                    tail += 1;
                }
            }
        }
    }

    dist
}

pub fn neighbor_if_open_bits(hwalls: u64, vwalls: u64, from: u8, dir: usize) -> Option<u8> {
    let edge = edge_mask(from, dir);
    if !edge.valid || (hwalls & edge.hmask) != 0 || (vwalls & edge.vmask) != 0 {
        None
    } else {
        Some(edge.neighbor)
    }
}

pub fn edge_blocked_bits(hwalls: u64, vwalls: u64, from: u8, to: u8) -> bool {
    if let Some(dir) = dir_between(from, to) {
        let edge = edge_mask(from, dir);
        (hwalls & edge.hmask) != 0 || (vwalls & edge.vmask) != 0
    } else {
        true
    }
}

pub fn wall_blocks_edge(orientation: WallOrientation, anchor: u8, from: u8, to: u8) -> bool {
    let Some(dir) = dir_between(from, to) else {
        return false;
    };
    let edge = edge_mask(from, dir);
    let bit = 1u64 << anchor;
    match orientation {
        WallOrientation::Horizontal => edge.hmask & bit != 0,
        WallOrientation::Vertical => edge.vmask & bit != 0,
    }
}

pub fn wall_touches_path(orientation: WallOrientation, anchor: u8, path: &PathTrace) -> bool {
    for i in 0..path.len as usize {
        if wall_blocks_edge(orientation, anchor, path.cells[i], path.cells[i + 1]) {
            return true;
        }
    }
    false
}

pub fn edge_mask(cell_idx: u8, dir: usize) -> EdgeMask {
    edge_masks()[cell_idx as usize][dir]
}

fn edge_masks() -> &'static [[EdgeMask; 4]; CELLS] {
    EDGE_MASKS.get_or_init(build_edge_masks)
}

fn build_edge_masks() -> [[EdgeMask; 4]; CELLS] {
    let invalid = EdgeMask {
        neighbor: 0,
        valid: false,
        hmask: 0,
        vmask: 0,
    };
    let mut table = [[invalid; 4]; CELLS];
    for r in 0..BOARD as u8 {
        for c in 0..BOARD as u8 {
            let idx = cell(r, c);
            for dir in 0..4 {
                if let Some((nr, nc)) = neighbor_coord(r, c, dir) {
                    let (hmask, vmask) = edge_wall_masks(r, c, dir);
                    table[idx as usize][dir] = EdgeMask {
                        neighbor: cell(nr, nc),
                        valid: true,
                        hmask,
                        vmask,
                    };
                }
            }
        }
    }
    table
}

fn neighbor_coord(r: u8, c: u8, dir: usize) -> Option<(u8, u8)> {
    match dir {
        DIR_N if r > 0 => Some((r - 1, c)),
        DIR_S if r + 1 < BOARD as u8 => Some((r + 1, c)),
        DIR_E if c + 1 < BOARD as u8 => Some((r, c + 1)),
        DIR_W if c > 0 => Some((r, c - 1)),
        _ => None,
    }
}

fn edge_wall_masks(r: u8, c: u8, dir: usize) -> (u64, u64) {
    let mut hmask = 0u64;
    let mut vmask = 0u64;
    match dir {
        DIR_N => {
            add_h_anchor(&mut hmask, r as i8 - 1, c as i8 - 1);
            add_h_anchor(&mut hmask, r as i8 - 1, c as i8);
        }
        DIR_S => {
            add_h_anchor(&mut hmask, r as i8, c as i8 - 1);
            add_h_anchor(&mut hmask, r as i8, c as i8);
        }
        DIR_E => {
            add_v_anchor(&mut vmask, r as i8 - 1, c as i8);
            add_v_anchor(&mut vmask, r as i8, c as i8);
        }
        DIR_W => {
            add_v_anchor(&mut vmask, r as i8 - 1, c as i8 - 1);
            add_v_anchor(&mut vmask, r as i8, c as i8 - 1);
        }
        _ => {}
    }
    (hmask, vmask)
}

fn add_h_anchor(mask: &mut u64, r: i8, c: i8) {
    if (0..WALL_GRID as i8).contains(&r) && (0..WALL_GRID as i8).contains(&c) {
        *mask |= 1u64 << (r as u8 * WALL_GRID as u8 + c as u8);
    }
}

fn add_v_anchor(mask: &mut u64, r: i8, c: i8) {
    add_h_anchor(mask, r, c);
}

fn dir_between(from: u8, to: u8) -> Option<usize> {
    let fr = row(from);
    let fc = col(from);
    let tr = row(to);
    let tc = col(to);
    match (tr as i16 - fr as i16, tc as i16 - fc as i16) {
        (-1, 0) => Some(DIR_N),
        (1, 0) => Some(DIR_S),
        (0, 1) => Some(DIR_E),
        (0, -1) => Some(DIR_W),
        _ => None,
    }
}

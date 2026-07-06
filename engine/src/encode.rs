use crate::path;
use crate::state::{
    BOARD, CELLS, GameState, NUM_ACTIONS, WALL_ANCHORS, WALLS_PER_PLAYER, anchor, anchor_col,
    anchor_row, cell, col, row,
};

pub const BASE_PLANES: usize = 6;
pub const EXTRA_PLANES: usize = 8;

#[derive(Clone, Copy, Debug)]
struct CanonicalView {
    current_pawn: u8,
    opponent_pawn: u8,
    hwalls: u64,
    vwalls: u64,
    current_walls: u8,
    opponent_walls: u8,
}

pub fn encode(state: &GameState, extra_planes: bool) -> Vec<f32> {
    let planes = if extra_planes {
        EXTRA_PLANES
    } else {
        BASE_PLANES
    };
    let mut out = vec![0.0f32; planes * CELLS];
    let view = canonical_view(state);

    set_cell(&mut out, 0, view.current_pawn, 1.0);
    set_cell(&mut out, 1, view.opponent_pawn, 1.0);

    for anchor_idx in 0..WALL_ANCHORS as u8 {
        let bit = 1u64 << anchor_idx;
        let r = anchor_row(anchor_idx) as usize;
        let c = anchor_col(anchor_idx) as usize;
        if view.hwalls & bit != 0 {
            out[2 * CELLS + r * BOARD + c] = 1.0;
            out[2 * CELLS + r * BOARD + c + 1] = 1.0;
        }
        if view.vwalls & bit != 0 {
            out[3 * CELLS + r * BOARD + c] = 1.0;
            out[3 * CELLS + (r + 1) * BOARD + c] = 1.0;
        }
    }

    let current_wall_value = view.current_walls as f32 / WALLS_PER_PLAYER as f32;
    let opponent_wall_value = view.opponent_walls as f32 / WALLS_PER_PLAYER as f32;
    for idx in 0..CELLS {
        out[4 * CELLS + idx] = current_wall_value;
        out[5 * CELLS + idx] = opponent_wall_value;
    }

    if extra_planes {
        let current_dist =
            path::distance_map_to_goal_row(view.hwalls, view.vwalls, (BOARD - 1) as u8);
        let opponent_dist = path::distance_map_to_goal_row(view.hwalls, view.vwalls, 0);
        for idx in 0..CELLS {
            out[6 * CELLS + idx] = normalize_distance(current_dist[idx]);
            out[7 * CELLS + idx] = normalize_distance(opponent_dist[idx]);
        }
    }

    out
}

pub fn canonical_action(action: u8, to_move: u8) -> Option<u8> {
    if action as usize >= NUM_ACTIONS || to_move > 1 {
        return None;
    }
    if to_move == 0 {
        Some(action)
    } else {
        Some(rotate_action_180(action))
    }
}

pub fn mirror_action(action: u8) -> Option<u8> {
    if action as usize >= NUM_ACTIONS {
        return None;
    }
    Some(mirror_action_lr(action))
}

pub fn mirror_state(state: &GameState) -> GameState {
    GameState {
        pawns: [
            mirror_cell_lr(state.pawns[0]),
            mirror_cell_lr(state.pawns[1]),
        ],
        hwalls: mirror_wall_bits_lr(state.hwalls),
        vwalls: mirror_wall_bits_lr(state.vwalls),
        walls_left: state.walls_left,
        to_move: state.to_move,
        ply: state.ply,
    }
}

pub fn mirror_policy(policy: &[f32; NUM_ACTIONS]) -> [f32; NUM_ACTIONS] {
    let mut out = [0.0f32; NUM_ACTIONS];
    for action in 0..NUM_ACTIONS as u8 {
        out[mirror_action_lr(action) as usize] = policy[action as usize];
    }
    out
}

pub fn mirror_encoded_lr(encoded: &[f32], planes: usize) -> Vec<f32> {
    assert_eq!(encoded.len(), planes * CELLS);
    let mut out = vec![0.0f32; encoded.len()];
    for p in 0..planes {
        for r in 0..BOARD {
            for c in 0..BOARD {
                if p == 3 {
                    if c < BOARD - 1 {
                        out[p * CELLS + r * BOARD + (BOARD - 2 - c)] =
                            encoded[p * CELLS + r * BOARD + c];
                    }
                } else {
                    out[p * CELLS + r * BOARD + (BOARD - 1 - c)] =
                        encoded[p * CELLS + r * BOARD + c];
                }
            }
        }
    }
    out
}

fn canonical_view(state: &GameState) -> CanonicalView {
    if state.to_move == 0 {
        CanonicalView {
            current_pawn: state.pawns[0],
            opponent_pawn: state.pawns[1],
            hwalls: state.hwalls,
            vwalls: state.vwalls,
            current_walls: state.walls_left[0],
            opponent_walls: state.walls_left[1],
        }
    } else {
        CanonicalView {
            current_pawn: rotate_cell_180(state.pawns[1]),
            opponent_pawn: rotate_cell_180(state.pawns[0]),
            hwalls: rotate_wall_bits_180(state.hwalls),
            vwalls: rotate_wall_bits_180(state.vwalls),
            current_walls: state.walls_left[1],
            opponent_walls: state.walls_left[0],
        }
    }
}

fn set_cell(out: &mut [f32], plane: usize, cell_idx: u8, value: f32) {
    out[plane * CELLS + cell_idx as usize] = value;
}

fn normalize_distance(distance: u8) -> f32 {
    if distance == u8::MAX {
        1.0
    } else {
        distance as f32 / 80.0
    }
}

pub fn rotate_cell_180(cell_idx: u8) -> u8 {
    cell(
        (BOARD - 1) as u8 - row(cell_idx),
        (BOARD - 1) as u8 - col(cell_idx),
    )
}

pub fn mirror_cell_lr(cell_idx: u8) -> u8 {
    cell(row(cell_idx), (BOARD - 1) as u8 - col(cell_idx))
}

pub fn rotate_anchor_180(anchor_idx: u8) -> u8 {
    anchor(
        (BOARD - 2) as u8 - anchor_row(anchor_idx),
        (BOARD - 2) as u8 - anchor_col(anchor_idx),
    )
}

pub fn mirror_anchor_lr(anchor_idx: u8) -> u8 {
    anchor(
        anchor_row(anchor_idx),
        (BOARD - 2) as u8 - anchor_col(anchor_idx),
    )
}

pub fn rotate_wall_bits_180(bits: u64) -> u64 {
    transform_wall_bits(bits, rotate_anchor_180)
}

pub fn mirror_wall_bits_lr(bits: u64) -> u64 {
    transform_wall_bits(bits, mirror_anchor_lr)
}

fn transform_wall_bits(bits: u64, f: fn(u8) -> u8) -> u64 {
    let mut out = 0u64;
    for anchor_idx in 0..WALL_ANCHORS as u8 {
        if bits & (1u64 << anchor_idx) != 0 {
            out |= 1u64 << f(anchor_idx);
        }
    }
    out
}

fn rotate_action_180(action: u8) -> u8 {
    const PAWN: [u8; 12] = [1, 0, 3, 2, 5, 4, 7, 6, 11, 10, 9, 8];
    match action {
        0..=11 => PAWN[action as usize],
        12..=75 => 12 + rotate_anchor_180(action - 12),
        76..=139 => 76 + rotate_anchor_180(action - 76),
        _ => unreachable!("validated action"),
    }
}

fn mirror_action_lr(action: u8) -> u8 {
    const PAWN: [u8; 12] = [0, 1, 3, 2, 4, 5, 7, 6, 9, 8, 11, 10];
    match action {
        0..=11 => PAWN[action as usize],
        12..=75 => 12 + mirror_anchor_lr(action - 12),
        76..=139 => 76 + mirror_anchor_lr(action - 76),
        _ => unreachable!("validated action"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{GameState, anchor, cell};

    fn bit(anchor_idx: u8) -> u64 {
        1u64 << anchor_idx
    }

    #[test]
    fn canonical_action_is_involution() {
        for action in 0..NUM_ACTIONS as u8 {
            assert_eq!(canonical_action(action, 0), Some(action));
            let mapped = canonical_action(action, 1).unwrap();
            assert_eq!(canonical_action(mapped, 1), Some(action));
        }
    }

    #[test]
    fn rotate_180_helpers_are_involutions() {
        for cell_idx in 0..CELLS as u8 {
            assert_eq!(rotate_cell_180(rotate_cell_180(cell_idx)), cell_idx);
        }
        for anchor_idx in 0..WALL_ANCHORS as u8 {
            assert_eq!(rotate_anchor_180(rotate_anchor_180(anchor_idx)), anchor_idx);
        }
    }

    #[test]
    fn mirror_action_and_policy_are_consistent() {
        let mut policy = [0.0f32; NUM_ACTIONS];
        for action in 0..NUM_ACTIONS as u8 {
            let mirrored = mirror_action(action).unwrap();
            assert_eq!(mirror_action(mirrored), Some(action));
            policy[action as usize] = action as f32;
        }
        let mirrored = mirror_policy(&policy);
        for action in 0..NUM_ACTIONS as u8 {
            assert_eq!(
                mirrored[mirror_action(action).unwrap() as usize],
                action as f32
            );
        }
    }

    #[test]
    fn mirror_state_matches_mirrored_encoding_and_legal_mask() {
        let state = GameState {
            pawns: [cell(2, 3), cell(6, 5)],
            hwalls: bit(anchor(1, 2)) | bit(anchor(4, 5)),
            vwalls: bit(anchor(0, 6)) | bit(anchor(5, 1)),
            walls_left: [8, 8],
            to_move: 0,
            ply: 8,
        };
        let mirrored = mirror_state(&state);

        let encoded = encode(&state, false);
        let mirrored_encoded = encode(&mirrored, false);
        assert_eq!(
            mirrored_encoded,
            mirror_encoded_lr(&encoded, BASE_PLANES),
            "mirroring a state should mirror all encoded planes"
        );

        let mask = state.legal_mask();
        let mirrored_mask = mirrored.legal_mask();
        for action in 0..NUM_ACTIONS as u8 {
            assert_eq!(
                mask[action as usize],
                mirrored_mask[mirror_action(action).unwrap() as usize],
                "legal mask should remap under mirror for action {action}"
            );
        }
    }

    #[test]
    fn canonical_encoding_rotates_player_one_to_move() {
        let state = GameState {
            pawns: [cell(2, 3), cell(6, 5)],
            hwalls: bit(anchor(1, 2)),
            vwalls: bit(anchor(5, 6)),
            walls_left: [9, 8],
            to_move: 1,
            ply: 3,
        };
        let encoded = encode(&state, false);
        let current = rotate_cell_180(state.pawns[1]) as usize;
        let opponent = rotate_cell_180(state.pawns[0]) as usize;
        assert_eq!(encoded[current], 1.0);
        assert_eq!(encoded[CELLS + opponent], 1.0);
        assert_eq!(encoded[4 * CELLS], 0.8);
        assert_eq!(encoded[5 * CELLS], 0.9);
    }
}

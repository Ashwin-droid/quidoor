use crate::path::{self, DIR_E, DIR_N, DIR_S, DIR_W, PathTrace};
use crate::state::{
    BOARD, GameState, NUM_ACTIONS, WALL_ANCHORS, WallOrientation, anchor_col, anchor_row, cell,
    col, row,
};

const PAWN_DELTAS: [(i8, i8); 12] = [
    (-1, 0),
    (1, 0),
    (0, 1),
    (0, -1),
    (-2, 0),
    (2, 0),
    (0, 2),
    (0, -2),
    (-1, 1),
    (-1, -1),
    (1, 1),
    (1, -1),
];

pub fn legal_mask(state: &GameState) -> [bool; NUM_ACTIONS] {
    let mut mask = [false; NUM_ACTIONS];
    if state.winner().is_some() {
        return mask;
    }

    add_pawn_moves(state, &mut mask);

    if state.walls_left[state.to_move as usize] > 0 {
        let paths = [path::shortest_path(state, 0), path::shortest_path(state, 1)];
        for anchor in 0..WALL_ANCHORS as u8 {
            if can_place_wall_cached(state, WallOrientation::Horizontal, anchor, &paths) {
                mask[12 + anchor as usize] = true;
            }
            if can_place_wall_cached(state, WallOrientation::Vertical, anchor, &paths) {
                mask[76 + anchor as usize] = true;
            }
        }
    }

    mask
}

pub fn legal_actions(state: &GameState) -> Vec<u8> {
    legal_mask(state)
        .iter()
        .enumerate()
        .filter_map(|(idx, legal)| legal.then_some(idx as u8))
        .collect()
}

pub fn pawn_destination_for_action(state: &GameState, action: u8) -> Option<u8> {
    if action >= 12 {
        return None;
    }
    let (dr, dc) = PAWN_DELTAS[action as usize];
    let pawn = state.pawns[state.to_move as usize];
    let nr = row(pawn) as i8 + dr;
    let nc = col(pawn) as i8 + dc;
    if (0..BOARD as i8).contains(&nr) && (0..BOARD as i8).contains(&nc) {
        Some(cell(nr as u8, nc as u8))
    } else {
        None
    }
}

pub fn wall_from_action(action: u8) -> Option<(WallOrientation, u8)> {
    match action {
        12..=75 => Some((WallOrientation::Horizontal, action - 12)),
        76..=139 => Some((WallOrientation::Vertical, action - 76)),
        _ => None,
    }
}

pub fn can_place_wall(state: &GameState, orientation: WallOrientation, anchor: u8) -> bool {
    let paths = [path::shortest_path(state, 0), path::shortest_path(state, 1)];
    can_place_wall_cached(state, orientation, anchor, &paths)
}

pub fn can_place_wall_cached(
    state: &GameState,
    orientation: WallOrientation,
    anchor: u8,
    paths: &[Option<PathTrace>; 2],
) -> bool {
    if anchor as usize >= WALL_ANCHORS || state.walls_left[state.to_move as usize] == 0 {
        return false;
    }
    if wall_conflicts_bits(state.hwalls, state.vwalls, orientation, anchor) {
        return false;
    }

    let bit = 1u64 << anchor;
    let (new_hwalls, new_vwalls) = match orientation {
        WallOrientation::Horizontal => (state.hwalls | bit, state.vwalls),
        WallOrientation::Vertical => (state.hwalls, state.vwalls | bit),
    };

    for player in 0..2 {
        let Some(trace) = paths[player] else {
            return false;
        };
        if path::wall_touches_path(orientation, anchor, &trace)
            && !path::has_path_bits(new_hwalls, new_vwalls, state.pawns[player], player as u8)
        {
            return false;
        }
    }
    true
}

pub fn wall_conflicts_bits(
    hwalls: u64,
    vwalls: u64,
    orientation: WallOrientation,
    anchor: u8,
) -> bool {
    if anchor as usize >= WALL_ANCHORS {
        return true;
    }
    let bit = 1u64 << anchor;
    match orientation {
        WallOrientation::Horizontal => {
            let mut hmask = bit;
            let c = anchor_col(anchor);
            if c > 0 {
                hmask |= 1u64 << (anchor - 1);
            }
            if c + 1 < (BOARD - 1) as u8 {
                hmask |= 1u64 << (anchor + 1);
            }
            (hwalls & hmask) != 0 || (vwalls & bit) != 0
        }
        WallOrientation::Vertical => {
            let mut vmask = bit;
            let r = anchor_row(anchor);
            if r > 0 {
                vmask |= 1u64 << (anchor - (BOARD - 1) as u8);
            }
            if r + 1 < (BOARD - 1) as u8 {
                vmask |= 1u64 << (anchor + (BOARD - 1) as u8);
            }
            (vwalls & vmask) != 0 || (hwalls & bit) != 0
        }
    }
}

pub fn all_walls_non_conflicting(hwalls: u64, vwalls: u64) -> bool {
    for anchor in 0..WALL_ANCHORS as u8 {
        let bit = 1u64 << anchor;
        if hwalls & bit != 0
            && wall_conflicts_bits(hwalls & !bit, vwalls, WallOrientation::Horizontal, anchor)
        {
            return false;
        }
        if vwalls & bit != 0
            && wall_conflicts_bits(hwalls, vwalls & !bit, WallOrientation::Vertical, anchor)
        {
            return false;
        }
    }
    true
}

fn add_pawn_moves(state: &GameState, mask: &mut [bool; NUM_ACTIONS]) {
    let pawn = state.pawns[state.to_move as usize];
    let opponent = state.pawns[(state.to_move ^ 1) as usize];

    for (dir, simple_action, jump_action) in [
        (DIR_N, 0usize, 4usize),
        (DIR_S, 1usize, 5usize),
        (DIR_E, 2usize, 6usize),
        (DIR_W, 3usize, 7usize),
    ] {
        let Some(adjacent) = path::neighbor_if_open_bits(state.hwalls, state.vwalls, pawn, dir)
        else {
            continue;
        };

        if adjacent != opponent {
            mask[simple_action] = true;
            continue;
        }

        if path::neighbor_if_open_bits(state.hwalls, state.vwalls, opponent, dir).is_some() {
            mask[jump_action] = true;
        } else {
            for &(diag_dir, diag_action) in diagonal_options(dir) {
                if path::neighbor_if_open_bits(state.hwalls, state.vwalls, opponent, diag_dir)
                    .is_some()
                {
                    mask[diag_action] = true;
                }
            }
        }
    }
}

fn diagonal_options(dir: usize) -> &'static [(usize, usize); 2] {
    match dir {
        DIR_N => &[(DIR_E, 8), (DIR_W, 9)],
        DIR_S => &[(DIR_E, 10), (DIR_W, 11)],
        DIR_E => &[(DIR_N, 8), (DIR_S, 10)],
        DIR_W => &[(DIR_N, 9), (DIR_S, 11)],
        _ => unreachable!("direction must be cardinal"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::path;
    use crate::state::{anchor, cell};

    fn bit(anchor_idx: u8) -> u64 {
        1u64 << anchor_idx
    }

    fn test_state(p0: u8, p1: u8) -> GameState {
        GameState {
            pawns: [p0, p1],
            hwalls: 0,
            vwalls: 0,
            walls_left: [10, 10],
            to_move: 0,
            ply: 0,
        }
    }

    #[test]
    fn jump_matrix_straight_blocked_diagonal_and_edges() {
        let state = test_state(cell(4, 4), cell(5, 4));
        let mask = legal_mask(&state);
        assert!(mask[5], "south straight jump should be legal");
        assert!(!mask[1], "simple move into occupied pawn is illegal");
        assert!(
            !mask[10] && !mask[11],
            "diagonals need blocked straight jump"
        );

        let mut blocked = test_state(cell(4, 4), cell(5, 4));
        blocked.hwalls = bit(anchor(5, 4));
        let mask = legal_mask(&blocked);
        assert!(!mask[5], "wall behind opponent blocks straight jump");
        assert!(mask[10] && mask[11], "both south diagonals open");

        let mut one_diag_blocked = test_state(cell(4, 4), cell(5, 4));
        one_diag_blocked.hwalls = bit(anchor(5, 3));
        one_diag_blocked.vwalls = bit(anchor(4, 4));
        let mask = legal_mask(&one_diag_blocked);
        assert!(!mask[5]);
        assert!(!mask[10], "east diagonal is blocked by a vertical wall");
        assert!(mask[11], "west diagonal remains open");

        let edge = test_state(cell(7, 4), cell(8, 4));
        let mask = legal_mask(&edge);
        assert!(!mask[5], "cannot jump off the board");
        assert!(
            mask[10] && mask[11],
            "board edge behind opponent enables diagonals"
        );

        let horizontal = test_state(cell(4, 4), cell(4, 5));
        let mask = legal_mask(&horizontal);
        assert!(mask[6], "east straight jump should be legal");
        assert!(!mask[2]);

        let mut horizontal_blocked = test_state(cell(4, 4), cell(4, 5));
        horizontal_blocked.vwalls = bit(anchor(4, 5));
        let mask = legal_mask(&horizontal_blocked);
        assert!(!mask[6]);
        assert!(mask[8] && mask[10], "east blocked jump exposes NE and SE");
    }

    #[test]
    fn wall_overlap_and_cross_matrices() {
        let mut h = GameState::new();
        h.hwalls = bit(anchor(3, 3));
        assert!(!can_place_wall(
            &h,
            WallOrientation::Horizontal,
            anchor(3, 3)
        ));
        assert!(!can_place_wall(
            &h,
            WallOrientation::Horizontal,
            anchor(3, 2)
        ));
        assert!(!can_place_wall(
            &h,
            WallOrientation::Horizontal,
            anchor(3, 4)
        ));
        assert!(can_place_wall(
            &h,
            WallOrientation::Horizontal,
            anchor(3, 5)
        ));
        assert!(!can_place_wall(&h, WallOrientation::Vertical, anchor(3, 3)));
        assert!(can_place_wall(&h, WallOrientation::Vertical, anchor(3, 4)));

        let mut v = GameState::new();
        v.vwalls = bit(anchor(3, 3));
        assert!(!can_place_wall(&v, WallOrientation::Vertical, anchor(3, 3)));
        assert!(!can_place_wall(&v, WallOrientation::Vertical, anchor(2, 3)));
        assert!(!can_place_wall(&v, WallOrientation::Vertical, anchor(4, 3)));
        assert!(can_place_wall(&v, WallOrientation::Vertical, anchor(5, 3)));
        assert!(!can_place_wall(
            &v,
            WallOrientation::Horizontal,
            anchor(3, 3)
        ));
        assert!(can_place_wall(
            &v,
            WallOrientation::Horizontal,
            anchor(4, 3)
        ));
    }

    #[test]
    fn path_blocking_wall_rejected_including_placer_path() {
        let mut state = GameState::new();
        state.hwalls = bit(anchor(3, 0)) | bit(anchor(3, 2)) | bit(anchor(3, 4));
        state.vwalls = bit(anchor(0, 7)) | bit(anchor(2, 7));
        state.to_move = 0;

        assert!(path::has_path_bits(
            state.hwalls,
            state.vwalls,
            state.pawns[0],
            0
        ));
        assert!(path::has_path_bits(
            state.hwalls,
            state.vwalls,
            state.pawns[1],
            1
        ));

        let candidate = anchor(3, 6);
        assert!(
            !can_place_wall(&state, WallOrientation::Horizontal, candidate),
            "closing the only accessible crossing must be illegal"
        );

        let new_hwalls = state.hwalls | bit(candidate);
        assert!(
            !path::has_path_bits(new_hwalls, state.vwalls, state.pawns[0], 0),
            "the rejected wall traps the placing player"
        );
    }

    #[test]
    fn no_walls_left_removes_wall_actions_only() {
        let mut state = GameState::new();
        state.walls_left[0] = 0;
        let mask = legal_mask(&state);
        assert!(mask[..12].iter().any(|&legal| legal));
        assert!(mask[12..].iter().all(|&legal| !legal));
    }
}

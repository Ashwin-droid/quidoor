use crate::movegen;
use crate::path;
use std::fmt;
use std::str::FromStr;

pub const BOARD: usize = 9;
pub const CELLS: usize = BOARD * BOARD;
pub const WALL_GRID: usize = BOARD - 1;
pub const WALL_ANCHORS: usize = WALL_GRID * WALL_GRID;
pub const NUM_ACTIONS: usize = 140;
pub const WALLS_PER_PLAYER: u8 = 10;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WallOrientation {
    Horizontal,
    Vertical,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GameState {
    pub pawns: [u8; 2],
    pub hwalls: u64,
    pub vwalls: u64,
    pub walls_left: [u8; 2],
    pub to_move: u8,
    pub ply: u16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoveError {
    pub action: u8,
}

impl fmt::Display for MoveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "illegal action {}", self.action)
    }
}

impl std::error::Error for MoveError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsePositionError {
    message: String,
}

impl ParsePositionError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for ParsePositionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ParsePositionError {}

impl Default for GameState {
    fn default() -> Self {
        Self::new()
    }
}

impl GameState {
    pub const fn new() -> Self {
        Self {
            pawns: [4, 76],
            hwalls: 0,
            vwalls: 0,
            walls_left: [WALLS_PER_PLAYER, WALLS_PER_PLAYER],
            to_move: 0,
            ply: 0,
        }
    }

    pub fn winner(&self) -> Option<u8> {
        if row(self.pawns[0]) == BOARD as u8 - 1 {
            Some(0)
        } else if row(self.pawns[1]) == 0 {
            Some(1)
        } else {
            None
        }
    }

    pub fn pawn(&self, player: u8) -> Option<(u8, u8)> {
        if player < 2 {
            let p = self.pawns[player as usize];
            Some((row(p), col(p)))
        } else {
            None
        }
    }

    pub fn walls_left(&self, player: u8) -> Option<u8> {
        if player < 2 {
            Some(self.walls_left[player as usize])
        } else {
            None
        }
    }

    pub fn legal_mask(&self) -> [bool; NUM_ACTIONS] {
        movegen::legal_mask(self)
    }

    pub fn legal_actions(&self) -> Vec<u8> {
        movegen::legal_actions(self)
    }

    pub fn apply_action(&mut self, action: u8) -> Result<(), MoveError> {
        let mask = self.legal_mask();
        if action as usize >= NUM_ACTIONS || !mask[action as usize] {
            return Err(MoveError { action });
        }

        let side = self.to_move as usize;
        if action < 12 {
            let dest = movegen::pawn_destination_for_action(self, action)
                .expect("legal pawn action must have an in-board destination");
            self.pawns[side] = dest;
        } else {
            let (orientation, anchor) =
                movegen::wall_from_action(action).expect("legal wall action must decode");
            let bit = 1u64 << anchor;
            match orientation {
                WallOrientation::Horizontal => self.hwalls |= bit,
                WallOrientation::Vertical => self.vwalls |= bit,
            }
            self.walls_left[side] -= 1;
        }

        self.to_move ^= 1;
        self.ply = self.ply.saturating_add(1);
        Ok(())
    }

    pub fn shortest_path_len(&self, player: u8) -> Option<u8> {
        if player < 2 {
            path::shortest_path(self, player).map(|p| p.len)
        } else {
            None
        }
    }

    /// Return the compact text notation used by probes, viewers, and tests.
    ///
    /// Format:
    ///
    /// `Q1 t=<0|1> ply=<n> p0=<cell> p1=<cell> w=<p0>,<p1> h=<anchors|-> v=<anchors|->`
    ///
    /// Cells use files `a` through `i` and ranks `1` through `9`, with `a1`
    /// at row 0, column 0. Wall anchors use files `a` through `h` and ranks
    /// `1` through `8`, naming the top-left point in the 8x8 anchor grid.
    /// Wall lists are comma-separated and row-major sorted; an empty list is
    /// `-`.
    pub fn to_position_string(&self) -> String {
        format!(
            "Q1 t={} ply={} p0={} p1={} w={},{} h={} v={}",
            self.to_move,
            self.ply,
            cell_to_coord(self.pawns[0]),
            cell_to_coord(self.pawns[1]),
            self.walls_left[0],
            self.walls_left[1],
            wall_list_to_string(self.hwalls),
            wall_list_to_string(self.vwalls)
        )
    }

    pub fn from_position_string(text: &str) -> Result<Self, ParsePositionError> {
        let mut saw_version = false;
        let mut to_move = None;
        let mut ply = None;
        let mut p0 = None;
        let mut p1 = None;
        let mut walls_left = None;
        let mut hwalls = None;
        let mut vwalls = None;

        for token in text.split_whitespace() {
            if token == "Q1" {
                saw_version = true;
                continue;
            }
            let (key, value) = token
                .split_once('=')
                .ok_or_else(|| ParsePositionError::new(format!("bad token `{token}`")))?;
            match key {
                "t" => {
                    let parsed = parse_u8(value, "to_move")?;
                    if parsed > 1 {
                        return Err(ParsePositionError::new("to_move must be 0 or 1"));
                    }
                    to_move = Some(parsed);
                }
                "ply" => {
                    ply = Some(
                        value
                            .parse::<u16>()
                            .map_err(|_| ParsePositionError::new("bad ply"))?,
                    );
                }
                "p0" => p0 = Some(coord_to_cell(value)?),
                "p1" => p1 = Some(coord_to_cell(value)?),
                "w" => walls_left = Some(parse_walls_left(value)?),
                "h" => hwalls = Some(parse_wall_list(value)?),
                "v" => vwalls = Some(parse_wall_list(value)?),
                _ => {
                    return Err(ParsePositionError::new(format!("unknown field `{key}`")));
                }
            }
        }

        if !saw_version {
            return Err(ParsePositionError::new("missing Q1 version marker"));
        }
        let state = Self {
            pawns: [
                p0.ok_or_else(|| ParsePositionError::new("missing p0"))?,
                p1.ok_or_else(|| ParsePositionError::new("missing p1"))?,
            ],
            hwalls: hwalls.ok_or_else(|| ParsePositionError::new("missing h"))?,
            vwalls: vwalls.ok_or_else(|| ParsePositionError::new("missing v"))?,
            walls_left: walls_left.ok_or_else(|| ParsePositionError::new("missing w"))?,
            to_move: to_move.ok_or_else(|| ParsePositionError::new("missing t"))?,
            ply: ply.ok_or_else(|| ParsePositionError::new("missing ply"))?,
        };

        validate_position(&state)?;
        Ok(state)
    }
}

impl fmt::Display for GameState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_position_string())
    }
}

impl FromStr for GameState {
    type Err = ParsePositionError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_position_string(s)
    }
}

pub const fn cell(row: u8, col: u8) -> u8 {
    row * BOARD as u8 + col
}

pub const fn row(cell: u8) -> u8 {
    cell / BOARD as u8
}

pub const fn col(cell: u8) -> u8 {
    cell % BOARD as u8
}

pub const fn anchor(row: u8, col: u8) -> u8 {
    row * WALL_GRID as u8 + col
}

pub const fn anchor_row(anchor: u8) -> u8 {
    anchor / WALL_GRID as u8
}

pub const fn anchor_col(anchor: u8) -> u8 {
    anchor % WALL_GRID as u8
}

fn validate_position(state: &GameState) -> Result<(), ParsePositionError> {
    if state.pawns[0] >= CELLS as u8 || state.pawns[1] >= CELLS as u8 {
        return Err(ParsePositionError::new("pawn out of board"));
    }
    if state.pawns[0] == state.pawns[1] {
        return Err(ParsePositionError::new(
            "pawns may not occupy the same cell",
        ));
    }
    if state.walls_left[0] > WALLS_PER_PLAYER || state.walls_left[1] > WALLS_PER_PLAYER {
        return Err(ParsePositionError::new(
            "wall counts must be between 0 and 10",
        ));
    }
    let placed = state.hwalls.count_ones() + state.vwalls.count_ones();
    let spent = (WALLS_PER_PLAYER - state.walls_left[0]) as u32
        + (WALLS_PER_PLAYER - state.walls_left[1]) as u32;
    if placed != spent {
        return Err(ParsePositionError::new(format!(
            "placed wall count {placed} does not match spent wall count {spent}"
        )));
    }
    if !movegen::all_walls_non_conflicting(state.hwalls, state.vwalls) {
        return Err(ParsePositionError::new(
            "wall layout has overlap or crossing",
        ));
    }
    if path::shortest_path(state, 0).is_none() || path::shortest_path(state, 1).is_none() {
        return Err(ParsePositionError::new("wall layout blocks a player path"));
    }
    Ok(())
}

fn parse_u8(value: &str, name: &str) -> Result<u8, ParsePositionError> {
    value
        .parse::<u8>()
        .map_err(|_| ParsePositionError::new(format!("bad {name}")))
}

fn parse_walls_left(value: &str) -> Result<[u8; 2], ParsePositionError> {
    let (a, b) = value
        .split_once(',')
        .ok_or_else(|| ParsePositionError::new("wall counts must be p0,p1"))?;
    let w0 = parse_u8(a, "wall count")?;
    let w1 = parse_u8(b, "wall count")?;
    Ok([w0, w1])
}

fn parse_wall_list(value: &str) -> Result<u64, ParsePositionError> {
    if value == "-" {
        return Ok(0);
    }
    let mut bits = 0u64;
    for item in value.split(',') {
        let anchor = coord_to_anchor(item)?;
        let bit = 1u64 << anchor;
        if bits & bit != 0 {
            return Err(ParsePositionError::new(format!(
                "duplicate wall anchor `{item}`"
            )));
        }
        bits |= bit;
    }
    Ok(bits)
}

fn coord_to_cell(value: &str) -> Result<u8, ParsePositionError> {
    let bytes = value.as_bytes();
    if bytes.len() != 2 {
        return Err(ParsePositionError::new(format!("bad cell `{value}`")));
    }
    let file = bytes[0];
    let rank = bytes[1];
    if !(b'a'..=b'i').contains(&file) || !(b'1'..=b'9').contains(&rank) {
        return Err(ParsePositionError::new(format!("bad cell `{value}`")));
    }
    Ok(cell(rank - b'1', file - b'a'))
}

fn coord_to_anchor(value: &str) -> Result<u8, ParsePositionError> {
    let bytes = value.as_bytes();
    if bytes.len() != 2 {
        return Err(ParsePositionError::new(format!(
            "bad wall anchor `{value}`"
        )));
    }
    let file = bytes[0];
    let rank = bytes[1];
    if !(b'a'..=b'h').contains(&file) || !(b'1'..=b'8').contains(&rank) {
        return Err(ParsePositionError::new(format!(
            "bad wall anchor `{value}`"
        )));
    }
    Ok(anchor(rank - b'1', file - b'a'))
}

fn cell_to_coord(cell_idx: u8) -> String {
    let file = (b'a' + col(cell_idx)) as char;
    let rank = (b'1' + row(cell_idx)) as char;
    format!("{file}{rank}")
}

fn anchor_to_coord(anchor_idx: u8) -> String {
    let file = (b'a' + anchor_col(anchor_idx)) as char;
    let rank = (b'1' + anchor_row(anchor_idx)) as char;
    format!("{file}{rank}")
}

fn wall_list_to_string(bits: u64) -> String {
    if bits == 0 {
        return "-".to_string();
    }
    let mut out = Vec::new();
    for anchor_idx in 0..WALL_ANCHORS {
        if bits & (1u64 << anchor_idx) != 0 {
            out.push(anchor_to_coord(anchor_idx as u8));
        }
    }
    out.join(",")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bit(anchor_idx: u8) -> u64 {
        1u64 << anchor_idx
    }

    #[test]
    fn notation_round_trip() {
        let state = GameState {
            pawns: [cell(2, 4), cell(6, 4)],
            hwalls: bit(anchor(1, 2)),
            vwalls: bit(anchor(5, 6)),
            walls_left: [9, 9],
            to_move: 1,
            ply: 4,
        };

        let text = state.to_position_string();
        assert_eq!(text, "Q1 t=1 ply=4 p0=e3 p1=e7 w=9,9 h=c2 v=g6");
        let parsed = GameState::from_position_string(&text).unwrap();
        assert_eq!(parsed, state);
        assert_eq!(parsed.to_position_string(), text);
    }

    #[test]
    fn notation_rejects_inconsistent_wall_counts() {
        let err = GameState::from_position_string("Q1 t=0 ply=0 p0=e1 p1=e9 w=10,10 h=a1 v=-")
            .unwrap_err();
        assert!(err.to_string().contains("placed wall count"));
    }
}

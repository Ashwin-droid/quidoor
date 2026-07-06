# Quoridor Position Notation

The engine serializes positions as one whitespace-separated line:

```text
Q1 t=<0|1> ply=<n> p0=<cell> p1=<cell> w=<p0>,<p1> h=<anchors|-> v=<anchors|->
```

Example start position:

```text
Q1 t=0 ply=0 p0=e1 p1=e9 w=10,10 h=- v=-
```

- `Q1` is the format/version marker.
- `t` is the side to move.
- `ply` is the half-move count.
- `p0` and `p1` are pawn cells.
- `w` is the remaining wall count for player 0 and player 1.
- `h` lists horizontal wall anchors and `v` lists vertical wall anchors.
- Empty wall lists are written as `-`.

Coordinates use files `a` through `i` and ranks `1` through `9` for cells,
with `a1` at row 0, column 0. Player 0 starts on `e1` and races toward rank
`9`; player 1 starts on `e9` and races toward rank `1`.

Wall anchors use files `a` through `h` and ranks `1` through `8`, naming the
top-left anchor in the 8x8 wall-anchor grid. Wall lists are sorted row-major
by anchor index.

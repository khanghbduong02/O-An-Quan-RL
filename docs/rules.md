# Ô ăn quan — the ruleset used in this project

Everything under `engine/` implements exactly this document, and every worked
example below is a test in `tests/test_engine.py`. Where regional variants
disagree, section 6 records the choice we made and why.

Sources are listed at the end.

## 1. Board

The board is a **ring of twelve squares**:

```
                 +-----+-----+-----+-----+-----+
                 |  1  |  2  |  3  |  4  |  5  |     player 0 (top)
    (0) QUAN     +-----+-----+-----+-----+-----+     QUAN (6)
                 | 11  | 10  |  9  |  8  |  7  |     player 1 (bottom)
                 +-----+-----+-----+-----+-----+
```

| index | what it is |
|---|---|
| `0` | left mandarin square (*ô quan*) |
| `1`–`5` | player 0's five citizen squares (*ô dân*), left to right |
| `6` | right mandarin square (*ô quan*) |
| `7`–`11` | player 1's five citizen squares, right to left |

Walking `6 → 7 → 8 → … → 11 → 0 → 1 → …` follows the ring. In code,
**clockwise** (`+1`) means "next index", which is left to right along the top
row and right to left along the bottom row; **counter-clockwise** (`-1`) is the
opposite way.

The two mandarin squares are neutral: they belong to neither player, but they
are part of the ring and stones can be sown into them.

## 2. Setup

* one **mandarin piece** (*quân quan*) on each mandarin square — the big stone;
* five **citizen stones** (*quân dân*) in each of the ten citizen squares
  (50 stones in total);
* player 0 moves first (decided by *oẳn tù tì* in real play).

The engine counts only citizen stones in `cells`; the mandarin pieces live in
the separate `quan` flags. A mandarin piece is worth
`Rules.mandarin_value` points (default **10**).

## 3. A turn

### 3.1 Forced refill (checked first)

If all five of your citizen squares are empty when it becomes your turn, you
must place **one captured stone in each of them** (five stones in total) before
moving. Those stones come out of your captured pile. You still move this turn.

If you have fewer than five captured stones, you cannot refill and **the game
ends** (see §5). Missing stones are never borrowed from the opponent in this
project.

### 3.2 Sowing (*rải quân*)

Choose one of your **own non-empty citizen squares** and a direction
(clockwise or counter-clockwise). Lift every stone from that square, leaving it
empty, then drop one stone per square along the ring in the chosen direction —
**including the mandarin squares**. The last stone falls on some square `L`; the
rest of the move is decided by what follows `L`:

| square after `L` | what happens |
|---|---|
| a citizen square holding stones | lift all of them and **keep sowing** from the next square, same direction |
| a mandarin square (piece still there) | **turn ends** — a mandarin square is never lifted |
| empty | try to capture (below); the turn ends afterwards either way |

Captures (only checked when the square after `L` is empty): let `E` be that
empty square and `T` the square after `E`.

* If `T` is empty, **two empty squares in a row: the turn ends**.
* If `T` holds anything (stones, or a mandarin piece), **you capture it**: all
  citizen stones there go to your score, and if `T` is a mandarin square you
  also capture the mandarin piece; `T` becomes an ordinary empty citizen square.
* Captures **chain**: after capturing `T`, look at the square after `T`. If it
  is empty and the next one holds something, capture that too, and so on. The
  chain stops at the first square that is not empty (you do **not** resume
  sowing after a capture), or when two empty squares follow.

That is the whole move; then the turn passes to the other player (after the
next player's refill check, §3.1).

### 3.3 Capturing a mandarin

A mandarin is captured by the normal capture rule with the mandarin square as
the target: your last stone lands one square before an empty square, and the
square after that empty square is a mandarin square holding **at least five**
citizen stones (*quan non*: a young mandarin is not rich enough to take). You
then take the mandarin piece (10 points) plus those stones, and the square
becomes an ordinary citizen square (it can be sown into and captured like any
other from then on).

Three consequences worth being explicit about:

* A **bare** mandarin square (only the mandarin piece) is not a capture target,
  and neither is one holding one to four stones.
* A bare mandarin square counts as *empty* when it sits between your last stone
  and a rich square, so a capture chain can run through it.
* A mandarin square that *follows* your last stone and holds citizen stones
  ends your turn: you can neither lift a mandarin nor take it from there.

The threshold is `Rules.mandarin_capture_min` (default 5). Lowering it to 1
makes any sown-into mandarin capturable, which sounds harmless but is not: the
first sowing of a game puts a stone into a mandarin square, and with a
threshold of 1 self-play games ended after three or four moves. That
measurement is why 5 is the default; `mandarin_capture_min=1` remains available
as the lenient variant.

## 4. What a player may do

* Actions are `(square, direction)`: at most **ten** per turn (five squares ×
  two directions).
* Only your own five squares may be chosen, and only when they hold stones.
* There is no passing.

## 5. End of the game and scoring

The game ends when any of the following happens:

1. **both mandarin pieces have been captured** (*hết quan, tàn dân*);
2. a player must refill but has fewer than five captured stones;
3. no citizen stone has been captured for 60 plies (`Rules.no_progress_plies`).
   Stones can circulate around the ring indefinitely — a player who keeps
   sowing without ever leaving a capture open produces a stable loop — so this
   works like chess's fifty-move rule: play stops, the position is scored as it
   stands, and equal scores are a draw. `Rules.ply_limit` (1000) is only a
   final safety net.

Final score:

* captured citizen stones: 1 point each;
* captured mandarins: `mandarin_value` points each (default 10);
* stones left in your own five squares at the end belong to you;
* stones left in a square whose mandarin you captured belong to you;
* stones still sitting in a mandarin square that was never captured are not
  scored.

Highest score wins; equal scores are a draw. (`Rules.ply_limit`, default 1000,
is a safety net that ends pathological repeating games as a draw; normal games
never reach it.)

## 6. Decisions taken where sources disagree

| Question | Sources disagree | This project |
|---|---|---|
| Value of a mandarin | 5 or 10 citizen stones | **10** (`Rules(mandarin_value=5)` reproduces the other variant) |
| Stones sown into mandarin squares | yes in most descriptions | **yes** — this is how a mandarin square becomes "rich" (*quan non*) |
| Can a turn continue from a mandarin square? | some local rules lift a stone out of it | **no** — landing in front of a mandarin ends the turn |
| When can a mandarin be captured? | some rules forbid taking a "young" mandarin (*quan non*, fewer than five stones) | **only when the square holds at least five citizen stones** (`mandarin_capture_min=5`). A bare or young mandarin is never a target; with a threshold of 1 games ended in three or four moves |
| Games that might never end | not covered by the folk rules | **no-progress draw** after 60 plies without a capture (`no_progress_plies`) |
| Cannot refill | traditional play borrows stones from the opponent | **the game ends** — per the project brief; borrowing is left as an optional variant |
| Captures after a capture | "the turn ends after eating" | **only chained captures continue**; sowing never resumes after a capture |
| Leftovers in a conquered mandarin square | not specified | they go to **the player who captured that mandarin** |
| Number of players | 2–4 | **2** (the 3- and 4-player boards are out of scope) |

## 7. Worked examples

Stone counts are written as `[0, 1, 2, … , 11]`. `Q` marks a mandarin square
that still holds its piece.

### Example 1 — the standard opening move (chains once, captures six)

Position `[0, 5, 5, 5, 5, 5, 0, 5, 5, 5, 5, 5]`, both mandarins present,
player 0 plays **square 1, clockwise**.

1. Lift 5 stones from square 1; they fall on 2, 3, 4, 5, 6, so the mandarin
   square 6 now holds one stone:
   `[0, 0, 6, 6, 6, 6, 1, 5, 5, 5, 5, 5]`.
2. The last stone fell on 6, and square 7 is a citizen square holding 5 stones:
   lift them and sow from 8 — stones fall on 8, 9, 10, 11, 0:
   `[1, 0, 6, 6, 6, 6, 1, 0, 6, 6, 6, 6]`.
3. The last stone fell on 0 (left mandarin square). Square 1 is empty and
   square 2 holds 6 stones, so player 0 captures those 6 stones:
   `[1, 0, 0, 6, 6, 6, 1, 0, 6, 6, 6, 6]`.
4. The square after 2 (square 3) holds stones, so the capture chain stops;
   the turn ends.

Player 0 has captured 6 stones; player 1 is to move. 50 stones still on the
board or in hands, as always.

### Example 2 — sowing continues once, then stops at a mandarin

Position `[0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0]`, player 0 plays **1, clockwise**.

1. Lift 1 stone from square 1, it falls on 2: `[0, 0, 1, 2, 0, …]`.
2. Square 3 holds 2 stones: lift them and sow from 4, they fall on 4 and 5:
   `[0, 0, 1, 0, 1, 1, 0, …]`.
3. The last stone fell on 5 and the square after it is the mandarin square 6:
   a mandarin square is never lifted, so the turn ends. No capture.

### Example 3 — capturing the right mandarin

Position `[0, 3, 0, 0, 2, 0, 5, 0, 0, 0, 0, 0]` (mandarin square 6 holds five
stones), player 0 plays **1, clockwise**.

1. Lift 3 from square 1; stones fall on 2, 3, 4 → `[0, 0, 1, 1, 3, 0, 5, …]`.
2. The last stone fell on 4; square 5 is empty and square 6 is a rich mandarin
   square, so player 0 captures **5 stones and the mandarin piece**: 15 points.
   Square 6 becomes an ordinary empty citizen square.
3. The square after 6 (square 7) is empty and square 8 is empty too, so the
   move ends.

Result `[0, 0, 1, 1, 3, 0, 0, 0, 0, 0, 0, 0]`, one mandarin left (square 0),
player 0 leads 15–0.

### Example 4 — a capture chain

Position `[0, 2, 0, 0, 0, 1, 0, 2, 0, 3, 0, 1]` with both mandarins already
captured, player 0 plays **5, clockwise**.

1. Lift 1 stone from square 5; it falls on 6 → `[…, 1, 2, 0, 3, 0, 1]`.
2. Square 7 holds 2 stones: lift and sow from 8 → stones fall on 8 and 9
   (square 9 now holds 4).
3. The last stone fell on 9; square 10 is empty and square 11 holds 1 stone →
   capture 1.
4. Chain: the square after 11 (square 0) is empty and square 1 holds 2 stones →
   capture 2.
5. Chain: the square after 1 (square 2) is empty and square 3 is empty →
   stop.

Result `[0, 0, 0, 0, 0, 0, 1, 0, 1, 4, 0, 0]`; player 0 captured 3 stones.

### Example 5 — refill after the side is emptied

Position `[0, 4, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]`, mandarin 6 already captured by
player 0, player 1 has 5 captured stones, player 0 plays **1, clockwise**.

1. Lift 4 stones from square 1; they fall on 2, 3, 4, 5.
2. The last stone fell on 5; square 6 (conquered mandarin square, now an
   ordinary square) is empty, and square 7 holds 1 stone → capture 1.
3. Chain: square 8 is empty, square 9 is empty → stop.
4. Player 1's five squares are now all empty, so player 1 places one captured
   stone in each of 7–11 (spending all five).

Result `[0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1]`, player 1 to move, player 1 with
0 captured stones. If player 1 had held fewer than five stones, the game would
have ended here instead.

### Example 6 — both mandarins gone, and the final score

Position `[5, 9, 0, 0, 0, 0, 2, 3, 0, 0, 0, 0]`: mandarin 0 still there with 5
stones in its square, mandarin 6 already captured by player 1 (whose 2 stones
remain in that square), player 1 already has one mandarin (10 points), player 0
plays **1, clockwise**.

1. Lift 9 stones from square 1; they fall on 2 … 10 (one each). Square 6 now
   holds 3 and square 7 holds 4.
2. The last stone fell on 10; square 11 is empty and square 0 is a rich
   mandarin square → player 0 captures its 5 stones **and the mandarin**: 15
   points.
3. Chain: square 1 is empty and square 2 holds 1 stone → capture 1 more stone.
4. Chain: square 3 holds a stone → stop. Both mandarins are gone, so the game
   ends immediately.

Result `[0, 0, 0, 1, 1, 1, 3, 4, 1, 1, 1, 0]`.

Final score:

* player 0: 6 captured stones + 10 (mandarin 0) + 3 stones in squares 3–5 +
  0 stones in its conquered mandarin square = **19**;
* player 1: 0 captured stones + 10 (mandarin 6) + 7 stones in squares 7–11 +
  3 stones in its conquered mandarin square 6 = **20**.

Player 1 wins 20–19.

## 8. Sources

* Vietnamese Wikipedia, *Ô ăn quan* — setup, sowing, capture chain, *quan non*,
  refill (*thả cá*), and the "hết quan, tàn dân" ending:
  <https://vi.wikipedia.org/wiki/Ô_ăn_quan>
* Mancala World (Wikimanqala), *Ô Ăn Quan* — board, sowing through the mandarin
  squares, capture rule, refill, mandarin worth 10:
  <https://mancala.fandom.com/wiki/Ô_Ăn_Quan>
* English Wikipedia, *Ô ăn quan* — board layout, objective, mandarin value
  10 (sometimes 5): <https://en.wikipedia.org/wiki/Ô_ăn_quan>

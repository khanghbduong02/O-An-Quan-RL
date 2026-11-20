# Results and analysis

Everything here was measured on this machine with the code in this repository.
Numbers are reproducible with the commands at the end of the page.

## Method

* 40 games per pairing unless stated, alternating seats, seed 1 unless stated.
* Two random opening plies per game (`opening_plies=2`) so that two
  deterministic agents do not replay the same game 40 times.
* Score = (wins + 0.5·draws) / games.
* Elo is fitted from the full tournament (`eval/elo.py`), not from single
  match-ups. Per-matchup Elo is `-400·log10(1/score - 1)`.

At 40 games the standard error of a score near 0.5 is ≈ 0.08, so a 95 %
interval is roughly ±0.15. Treat differences smaller than that as directions,
not measurements.

Ruleset: `Rules(stones_per_square=5, mandarin_value=10, mandarin_capture_min=5,
no_progress_plies=60)` — see `docs/rules.md`. The long-form explanation of every
design decision, with the maths, is in `docs/learning-guide.md`.

## 0. A search bug, and what it changed

Every number in §1–§5 of this document dated before the fix was measured with a
sign error in both search implementations: terminal values were stored from the
wrong perspective.

`State.returns()` reports the value for `state.player`, and a finished game
never hands the turn over, so at a terminal state `state.player` is the player
who **moved last** — the opponent of the node's mover. Node values are
conventionally stored for the player to move at the node, so a terminal value
must be negated once. It was not:

* `alphazero/mcts.py` evaluated finished leaves with `state.returns()` directly
  (and `_expand()` with it), so a move that wins immediately looked, to the
  search, like a move that loses immediately;
* `baselines/mcts.py` scored its random playouts for `state.player` after the
  game had ended, which is the same flip.

Measured directly with a neutral network (uniform policy, zero value): the
winning terminal move received **0.08** of the root visits while losing moves
received 0.85–0.92. After the fix the winning move receives 0.90–0.92 and the
losing moves are avoided. Two regression tests now pin this down
(`tests/test_agents.py`).

What it changed, all at seed 1, 40 games:

| Measurement | With the bug | Fixed |
|---|---|---|
| plain MCTS 100 vs greedy | 0.000 | 0.800 / 0.7625 / 0.8375 (seeds 1–3) |
| base model, 400 sims vs greedy | 0.100 | 0.950 |
| base model, 100 sims vs greedy | — | 0.938 |
| random network, 64-sim self-play | 60–100 ply shuffles | 14–21 plies, all ended by both mandarins captured |

The archived pre-fix numbers are kept in §7 so the earlier analysis can still be
read for what it got right and wrong.

## 1. Classical baselines

| Player | vs random | vs greedy | vs alpha-beta 4 | vs MCTS 100 |
|---|---|---|---|---|
| random | — | 0.125 | — | — |
| greedy | 0.875 | — | 0.85 | 1.000 |
| alpha-beta 1 | 0.838 | 0.275 | — | — |
| alpha-beta 2 | 0.750 | 0.35–0.42 | — | — |
| alpha-beta 4 | 0.525 | 0.15 | — | — |
| alpha-beta 3 vs 2 / 4 vs 3 | — | — | 0.30 / 0.417 | — |
| plain MCTS 100 (random rollouts) | — | 0.800 / 0.7625 / 0.8375 | — | — |

**Greedy (take the biggest immediate capture) is the strongest classical
player.** Alpha-beta is not buggy — it reproduces an exhaustive minimax exactly
on 263/263 random positions — but it still loses to greedy, and *more depth
makes it worse* (depth 1 → 0.275, depth 2 → 0.40, depth 4 → 0.15, depth 5 →
0.13 against greedy).

That is a statement about the evaluation function, not the search: a static
"score difference + 0.25 · stones on my side" leaf evaluation is wrong in a game
where a move is a chain of captures and where leaving stones in front of the
opponent hands them over. Deeper search optimises that wrong objective harder.
Weight sweeps (material weight −0.25…1.0, an opponent-threat term, random
tie-breaking) moved the depth-2 result between 0.13 and 0.45 and never beat
greedy.

Plain MCTS with random rollouts is a weak but real player: ~0.80 against greedy
at 100 playouts, reproducible across seeds. Random playouts model capture chains
poorly, so it needs many playouts to edge ahead of a one-ply heuristic.

## 2. AlphaZero, cold start (measured before the fix)

Config: 128×2 network, 96 simulations, 40 games per iteration, 19 input
features, value target = game outcome, 20 iterations (800 games, ~11 min).

| Opponent | Score |
|---|---|
| random | 0.450 |
| greedy | 0.150 |
| alpha-beta 4 | 0.113 |
| plain MCTS 100 | 0.875 |

Policy-only (no search) was *worse* still: 0.062 against greedy.

The agent beats plain MCTS — the only opponent that is weaker than it — and
loses to everything else. Diagnosing it at the time:

* self-play games were 60–100 plies long (random play averages 18, greedy 8), so
  the policy had learned to shuffle stones rather than capture;
* measured directly, the trained agent captured **3.62 points per own move**,
  versus **3.48 for random** and **8.73 for greedy** — no capture instinct at
  all, despite a policy loss that had fallen from 2.3 (uniform) to 1.10.

More of the same did not help: 6300 further games with 12 worker processes (66
iterations, `runs/base_az`) left it at 0.40 against random, and switching the
value target to dense `score_diff` (`runs/base_az_dense`) improved value fit
from 0.75 to 0.19 but kept the policy at random strength — 0.417 against random
with 96 simulations, still ~3.6 points captured per move.

**Caveat added after finding the sign bug.** These self-play targets were
produced by a search that preferred losing lines, so part of this failure was
the bug, not the value function. The value-error argument in §3 is still the
best explanation for why the *cold start* could not get traction, but it has not
been re-measured with the fix, and re-running this configuration is the first
open experiment (§6).

## 3. The two other obstacles in the cold start

**The value error is bigger than a capture.** The value head's error was
~0.19–0.25 in units where 1.0 = 50 points, i.e. a root-mean-square error of
**roughly 20 points**. A good capture is worth 5–20 points. With 64–96
simulations over ~6 legal moves, each child collects about 10–15 value samples,
so the search's Q-values are dominated by value noise: it cannot distinguish
"I take 15 points now" from "I take nothing". Increasing simulations at that
point converges *harder* onto the wrong values.

**The 19 input features were raw board counts.** The network had to infer, from
counts alone, **which** of its ten sowing moves lands on a rich square and
captures. That mapping is exactly the game, and it was not representable or
learnable from a policy head fed only board state.

## 4. Two fixes that worked

1. **Per-action capture features.** `alphazero/features.py` now appends ten
   values — "points this action captures right now" — so the input is 29 wide.
   This hands the network the greedy signal directly.
2. **Supervised warm start** (`pretrain.py`): 12 000 greedy-vs-greedy games,
   98 952 positions, 6 epochs, 256×2 network. Policy cross-entropy fell to 0.60
   against the teacher, value MSE to 0.21.

Result (`runs/base_model.npz`, **no search**):

| Opponent | Seed 0 | Seed 1 |
|---|---|---|
| random | 0.812 | 0.925 |
| greedy | 0.425 | 0.388 |
| alpha-beta 4 | 0.725 | 0.675 |
| plain MCTS 100 | 1.000 | 0.238 |

This is the shipped base model and, without search, it is roughly greedy's
equal (greedy scores 0.875 against random; the base model 0.81–0.93) and
clearly ahead of alpha-beta 4. The spread between the two seeds — especially
the MCTS 100 result, 1.000 vs 0.238 — is a warning about 40-game samples, not a
difference in the checkpoint.

## 5. Search on top of the network now helps

Same checkpoint, search budget on the x-axis (seed 1, 40 games per cell):

| Setup | vs random | vs greedy | vs alpha-beta 4 | vs MCTS 100 |
|---|---|---|---|---|
| raw policy (0 sims) | 0.925 | 0.388 | 0.675 | 0.238 |
| + 32 sims | 1.000 | — | — | — |
| + 100 sims | 0.988 | 0.938 | 0.863 | 0.750 |
| + 400 sims | — | 0.950 | — | — |

This is the monotone improvement that AlphaZero is supposed to produce: a prior
of roughly greedy strength becomes a player that beats greedy convincingly once
PUCT spends 100 simulations per move. The earlier conclusion in this repository
— "search makes it worse, the value head must be the bottleneck" — was an
artefact of the sign bug: with winning terminal moves scored as losses, more
search genuinely did make the agent worse.

## 6. Warm-started AlphaZero with self-play

`runs/base_az2/final.npz`: resumed from the warm start, 64 simulations, 96 games
per iteration, 12 workers, dense `score_diff` values, ~700 s budget. Evaluated
at 64 sims (seed 1):

| Opponent | Score |
|---|---|
| random | 0.975 |
| greedy | 0.613 |
| alpha-beta 4 | 0.700 |
| plain MCTS 100 | 0.350 |

Self-play on top of the warm start helps against random and holds its own
against greedy and alpha-beta 4, but it is behind the plain warm-started model
at 100 simulations (0.938 vs greedy). Two readings, both consistent with what
we know: 64 vs 100 simulations is a real handicap, and 24 iterations of
self-play is not enough to improve on a greedy-imitating teacher. Running it
longer is the obvious next experiment.

## 7. Archive: numbers measured with the search bug

Kept so the earlier analysis is still readable. These are **not** the current
results.

| Setup | vs random | vs greedy | vs alpha-beta 4 | vs MCTS 100 |
|---|---|---|---|---|
| base model, raw policy (bug-free; no search involved) | 0.812 | 0.425 | 0.725 | 1.000 |
| base model + 32 sims vs random | 0.575 | — | — | — |
| base model + 400 sims vs greedy | — | 0.100 | — | — |
| base_az2 + 64 sims, before re-measurement | 0.425 | — | — | — |
| plain MCTS 100 vs greedy | — | 0.000 | — | — |

The policy-only rows are unaffected by the bug (no search is involved); they
differ from §4 only because they were run at seed 0.

## 8. Actor-critic baseline

`ppo/a2c.py` (REINFORCE with a value baseline, entropy bonus, 200 iterations ×
24 games, 128×2, policy-only evaluation):

| Opponent | Score |
|---|---|
| random | 0.588 |
| greedy | 0.087 |
| alpha-beta 4 | 0.700 |

It lands where cold-start AlphaZero landed: better than random, better than
alpha-beta 4, well behind greedy. Same disease — no search, a value function too
coarse for a capture-chain game, and no teacher to bootstrap the prior.

## 9. What to try next

* **Re-run cold-start AlphaZero with the fixed search.** The original failure
  was measured with corrupted self-play targets; this is the cheapest way to
  find out how much of it was the bug and how much was the value function.
* Train the value head to convergence on fixed data before any search
  (supervised `score_diff` regression), then re-run PUCT and check whether the
  score keeps rising past 400 simulations.
* Replace Monte-Carlo outcome targets with TD(λ) targets so the value learns
  from its own bootstrap instead of a noisy final margin.
* Self-play against a league (greedy + past checkpoints), so passivity gets
  punished instead of mirrored.
* Give alpha-beta a quiescence search over capture chains and a capture-aware
  evaluation; the current one is provably correct but structurally blind.
* Scale: 32 cores are available; `train.py --workers 12` gave ~640 games/minute.
  A serious run is 10⁵–10⁶ games, i.e. hours, not minutes.

## Reproducing

```bash
# baselines
python evaluate.py --ckpt runs/base_model.npz --sims 0 --games 40 \
    --opponents random,greedy,alphabeta4,mcts100

# base model with search
python evaluate.py --ckpt runs/base_model.npz --sims 100 --games 40 --seed 1 \
    --opponents random,greedy,alphabeta4,mcts100

# the base model
python pretrain.py --run runs/pretrain_greedy --games 12000 --epochs 6 --hidden 256

# AlphaZero from cold start
python train.py --run runs/base_az --iters 20 --games 40 --sims 96

# AlphaZero with 12 workers, dense values
python train.py --run runs/base_az --iters 500 --games 96 --sims 96 --workers 12 \
    --value-target score_diff --time-budget 900

# AlphaZero from the warm start
python train.py --run runs/base_az2 --resume runs/pretrain_greedy/final.npz \
    --hidden 256 --sims 64 --games 96 --workers 12 --value-target score_diff \
    --time-budget 700

# actor-critic
python train_a2c.py --run runs/base_a2c --iters 200 --games 24
```

Raw logs and per-run JSON: `runs/az_train.log`, `runs/az_dense.log`,
`runs/az2_train.log`, `runs/a2c_train.log`, `runs/base_*/eval*.json`,
`runs/base_model_eval_*.json`, `runs/remeasure_*.json` (the post-fix
re-measurements).

**Checkpoint compatibility.** Checkpoints store their feature width and action
count. `runs/base_model.npz`, `runs/base_az2/final.npz` and
`runs/pretrain_greedy/final.npz` are 29-feature and load; `runs/base_az/*` and
`runs/base_a2c/*` are from the 19-feature era and raise
`ValueError: checkpoint was built for a different encoding`. They are kept only
as records of that experiment.

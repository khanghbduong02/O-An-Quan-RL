# Results and analysis

Everything here was measured on this machine with the code in this repository.
Numbers are reproducible with the commands at the end of the page.

## Method

* 40 games per pairing unless stated, alternating seats, seed 1.
* Two random opening plies per game (`opening_plies=2`) so that two
  deterministic agents do not replay the same game 40 times.
* Score = (wins + 0.5·draws) / games.
* Elo is fitted from the full tournament (`eval/elo.py`), not from single
  match-ups.

Ruleset: `Rules(mandarin_capture_min=5, no_progress_plies=60)` — see
`docs/rules.md`.

## 1. Classical baselines

| Player | vs random | vs greedy | vs alpha-beta 4 | vs MCTS 100 |
|---|---|---|---|---|
| random | — | 0.125 | — | — |
| greedy | 0.875 | — | 0.85 | 1.000 |
| alpha-beta 1 | 0.838 | 0.275 | — | — |
| alpha-beta 2 | 0.750 | 0.35–0.42 | — | — |
| alpha-beta 4 | 0.525 | 0.15 | — | — |
| alpha-beta 3 vs 2 / 4 vs 3 | — | — | 0.30 / 0.417 | — |
| plain MCTS 100 (random rollouts) | — | 0.000 | — | — |

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

Plain MCTS with random rollouts is hopeless here (0–40 against greedy): random
playouts do not respect capture chains at all.

## 2. AlphaZero, cold start

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
loses to everything else. Diagnosing it:

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

## 3. Root cause: the value error is bigger than a capture

The value head's error was ~0.19–0.25 in units where 1.0 = 50 points, i.e. a
root-mean-square error of **roughly 20 points**. A good capture is worth 5–20
points. With 64–96 simulations over ~6 legal moves, each child collects about
10–15 value samples, so the search's Q-values are dominated by value noise: it
cannot distinguish "I take 15 points now" from "I take nothing". The visit
distribution stays near-uniform, the policy target carries no signal, and the
policy never learns to capture. Increasing simulations at that point converges
*harder* onto the wrong values, and self-play never punishes the passivity
because both sides are passive (draws, 10–20 % of games).

There was a second, independent defect: the 19 input features were raw board
counts. The network had to infer, from counts alone, **which** of its ten sowing
moves lands on a rich square and captures. That mapping is exactly the game, and
it was not representable or learnable from a policy head fed only board state.

## 4. Two fixes that worked

1. **Per-action capture features.** `alphazero/features.py` now appends ten
   values — "points this action captures right now" — so the input is 29 wide.
   This hands the network the greedy signal directly.
2. **Supervised warm start** (`pretrain.py`): 12 000 greedy-vs-greedy games,
   98 952 positions, 6 epochs, 256×2 network. Policy cross-entropy fell to 0.60
   against the teacher.

Result (`runs/base_model.npz`, **no search**):

| Opponent | Score |
|---|---|
| random | 0.812 |
| greedy | 0.425 |
| alpha-beta 4 | 0.725 |
| plain MCTS 100 | 1.000 |

This is the strongest learned player in the repository and the shipped base
model. It is close to greedy's own strength (0.875 vs random) without copying
its tie-breaks, and it beats alpha-beta 4 by a wide margin — the opposite
result to the cold-start runs.

## 5. Search on top of the network makes it worse

| Setup | vs random | vs greedy |
|---|---|---|
| base model, raw policy | 0.812 | 0.425 |
| + 32 sims | 0.575 | — |
| + 400 sims | — | 0.100 |

PUCT with an inaccurate value head overrides a good prior with noise, and more
simulations mean trusting that head more. AlphaZero fine-tuning from the warm
start reproduced the same effect: after 24 iterations of 96-game self-play with
64 simulations the checkpoint fell to 0.425 against random with search and 0.450
policy-only, and its self-play games went straight back to 60–90 plies.

So the bottleneck in this repository is the **value function's accuracy**, not
search breadth. A value head that is right to ±3 points instead of ±20 would
make the same search work; that needs either far more self-play (the classic
AlphaZero answer, orders of magnitude beyond what fits in one session here) or a
different value target.

## 6. Actor-critic baseline

`ppo/a2c.py` (REINFORCE with a value baseline, entropy bonus, 200 iterations ×
24 games, 128×2, policy-only evaluation):

| Opponent | Score |
|---|---|
| random | 0.588 |
| greedy | 0.087 |
| alpha-beta 4 | 0.700 |

It lands in the same place as cold-start AlphaZero: better than random, better
than alpha-beta 4, well behind greedy. Same disease — a value function too
coarse for a capture-chain game, and no search to compensate.

## 7. What to try next

* Train the value head to convergence on fixed data before any search
  (supervised `score_diff` regression), then re-run PUCT: it is the cheapest way
  to test the "search is fine, values are not" hypothesis.
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

# base model
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
`runs/base_model_eval_*.json`.

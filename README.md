# Ô Ăn Quan RL

Self-play reinforcement learning for **Ô ăn quan**, the Vietnamese
mancala-style game: an AlphaZero-style policy/value network with PUCT MCTS, an
A2C baseline, and classical opponents, all measured against each other.

The project brief is in [Ô Ăn Quan RL Agent.md](Ô Ăn Quan RL Agent.md); the measured
results and the reasons behind them are in [docs/results.md](docs/results.md);
and [docs/learning-guide.md](docs/learning-guide.md) explains the environment
design, every algorithm and the maths behind it, for readers learning the
material.

## Status

| Phase | Goal | State |
|---|---|---|
| 1 | Pin rules, engine, tests, CLI to play | **done** |
| 2 | Random, greedy, alpha-beta, plain MCTS baselines + tournaments | **done** |
| 3 | Self-play actor-critic (A2C) agent | **done** |
| 4 | Policy/value network and PUCT MCTS | **done** |
| 5 | AlphaZero self-play loop (multiprocess) | **done** |
| 6–8 | Tournaments, Elo, analysis, report | **done** (see `docs/results.md`) |

## Quick start

Python 3.10+ and NumPy (no PyTorch: the network, its backprop and AdamW are
written against NumPy directly).

```bash
pip install -r requirements.txt

python play.py                                        # you (P0) vs the base model
python play.py --p1 random                            # you vs random
python play.py --p0 random --p1 net --sims 0          # watch the base model play
python play.py --p0 human --p1 human                  # hot seat
```

Move syntax is `<square> <direction>`, e.g. `3 cw` or `5 ccw`; commands are
`help`, `moves`, `board`, `quit`.

```bash
python -m pytest tests -q          # 40 tests
```

## Training and evaluation

```bash
# AlphaZero self-play (use --workers for parallel games; 96 games/iter is a
# reasonable default with 12 workers)
python train.py --run runs/base_az --iters 200 --games 96 --sims 96 \
    --workers 12 --value-target score_diff --time-budget 900

# supervised warm start from the greedy teacher (see docs/results.md for why)
python pretrain.py --run runs/pretrain_greedy --games 12000 --epochs 6 --hidden 256

# actor-critic baseline
python train_a2c.py --run runs/base_a2c --iters 200 --games 24

# measure a checkpoint
python evaluate.py --ckpt runs/base_model.npz --sims 0 --games 40 \
    --opponents random,greedy,alphabeta4,mcts100

# compare two trained models head to head
python evaluate.py --ckpt runs/base_model.npz --sims 0 --games 100 \
    --opponents net:runs/your_run/final.npz
```

`--opponents` accepts `random`, `greedy`, `alphabeta<N>`, `mcts<N>` and
`net:<path>`. `--sims 0` plays the raw policy head; a positive value runs PUCT
first.

## Results (40 games per pairing, seed 1)

Score = (wins + 0.5·draws) / games; both sides alternate; two random opening
plies per game so deterministic agents do not replay one identical game.

| Player | vs random | vs greedy | vs alpha-beta 4 | vs MCTS 100 |
|---|---|---|---|---|
| greedy | 0.875 | — | — | — |
| alpha-beta 2 | 0.750 | 0.42 | — | — |
| alpha-beta 4 | 0.525 | 0.15 | 0.42 (vs depth 3) | — |
| plain MCTS 100 | — | 0.800 | — | — |
| **`runs/base_model.npz`** (learned, no search) | **0.925** | **0.388** | **0.675** | **0.238** |
| base_model + 32 sims | 1.000 | — | — | — |
| **base_model + 100 sims** | **0.988** | **0.938** | **0.863** | **0.750** |
| base_model + 400 sims | — | 0.950 | — | — |
| `runs/base_az2/final.npz` (warm start + self-play, 64 sims) | 0.975 | 0.613 | 0.700 | 0.350 |
| `runs/base_a2c/final.npz` (policy only) | 0.588 | 0.087 | 0.700 | — |

Three things stand out, and they are discussed in
[docs/results.md](docs/results.md):

1. **Greedy is the strongest classical opponent.** Alpha-beta is verified
   correct (it matches an exhaustive minimax on 263/263 random positions) yet
   loses to greedy at every depth and gets *weaker* as depth grows: the static
   "score difference + material" evaluation is systematically misleading in a
   game where captures are chains.
2. **The fix was the encoding, not more search.** Feeding the network "points
   this action captures right now" (10 extra input features) plus a supervised
   warm start from greedy produces a policy that is roughly greedy's equal
   without any search (0.925 against random, 0.388 against greedy).
3. **With a decent prior, search helps monotonically.** The same checkpoint
   scores 0.938 against greedy at 100 simulations and 0.950 at 400, versus
   0.388 raw. Getting there required fixing a terminal-value sign error in both
   search implementations: finished positions kept `player` on the player who
   moved *last*, so winning terminal moves were scored as losses and search
   looked harmful. See §0 of `docs/results.md` for the measurements.

Caveat worth keeping in mind: with 40 games per pairing, the 95 % interval on
a score near 0.5 is about ±0.15, so small differences are directions, not
measurements.

## Base model for comparison

`runs/base_model.npz` is the shipped base model (256×2 network, 99k positions
from 12k greedy games, 6 supervised epochs). Play it with `python play.py`, or
score your own run against it:

```bash
python evaluate.py --ckpt runs/your_run/final.npz --sims 0 --games 100 \
    --opponents random,greedy,alphabeta4,net:runs/base_model.npz
```

Checkpoints are plain `.npz` files (`net.save` / `PolicyValueNet.load`), so a
run directory can be moved or deleted freely.

## Ruleset

[`docs/rules.md`](docs/rules.md) is the single source of truth: coordinates,
turn procedure, capture chains, mandarin captures, forced refill, end
conditions and scoring, with six hand-traced examples that are also tests.
Decisions worth knowing:

* sowing drops stones into the mandarin squares too, but a mandarin square is
  never lifted from — landing just in front of one ends the turn;
* a mandarin is captured only when it is *rich* (`mandarin_capture_min = 5`
  stones around it), otherwise a captured square is worth 10 points but leaves
  the mandarin in place;
* captured stones are the only way to refill an empty side, and a player who
  cannot refill loses;
* a game ends when both mandarins are gone, when a player cannot refill, or
  after `no_progress_plies = 60` plies without a capture (a draw rule — stones
  otherwise circulate forever).

## Layout

```
engine/            rules, immutable state, move generation, ASCII board
baselines/         random, greedy, alpha-beta, plain MCTS, network agent
alphazero/         features, NumPy policy/value net, PUCT MCTS, self-play,
                   trainer, multiprocess self-play pool
ppo/               A2C (REINFORCE with a value baseline)
eval/              arena (games/matches, alternating seats) and Elo fitting
docs/rules.md      the fixed ruleset with worked examples
docs/results.md    measurements, failure analysis, reproduction commands
train.py           AlphaZero self-play training
pretrain.py        supervised warm start from the greedy teacher
train_a2c.py       actor-critic training
evaluate.py        checkpoint evaluation against baselines or another net
play.py            terminal UI
docs/learning-guide.md  how and why it is built: env design, algorithms, maths
tests/             rule, network and agent tests (40)
```

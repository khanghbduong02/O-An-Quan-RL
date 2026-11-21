# Learning guide: building an Ô ăn quan RL agent from scratch

This is the long-form companion to the code. `docs/rules.md` says what the game
is, `docs/results.md` says what the agents score, and this file explains *why
each piece is built the way it is*, derives the maths that the NumPy code
implements, and records the failure modes that cost the most time.

The file has two parts.

* **Part I (§0–§16)** is about *this* project: the rules as an MDP, the feature
  encoding, the hand-written network and its gradients, alpha-beta, plain MCTS,
  PUCT, the self-play loop, A2C, the warm start, evaluation, and the experiment
  story including the sign bug. Every section ends with the file that
  implements it.
* **Part II (§17–§28)** is the general handbook: the MDP formalism and the
  Bellman equations, the full algorithm map (tabular, DQN, policy gradient,
  PPO, SAC, model-based, search), the maths behind each objective, reward and
  environment design, exploration, engineering, statistics for evaluating
  agents, a debugging playbook keyed by symptom, a compute ladder, and a
  step-by-step blueprint for starting the next project. It uses Part I as its
  worked example but does not depend on the details of this game.

Read Part I in order the first time. Read Part II when you are about to start
something new — §26 is the checklist, §28 is the reading order.

A map, if you are looking for one thing:

| Question | Section |
|---|---|
| How do I write down my problem as an MDP? | §2, §17 |
| Which algorithm should I use? | §18 (table), §26 |
| Why does the policy gradient have that form? | §17.4, §18.3 |
| How does AlphaZero actually improve a policy? | §7, §8, §19 |
| How should I reward / shape / observe? | §2.4, §20 |
| How do I debug a run that is not learning? | §24 |
| How many games do I need to believe a result? | §11, §23 |
| What do I build first in a new project? | §26 |
| What should I read next? | §28 |

---

## 0. What was built

A complete, dependency-light reinforcement-learning stack for one specific
board game:

| Layer | File(s) | What it does |
|---|---|---|
| Rules + state | `engine/` | immutable position, legal moves, capture chains, endings |
| Feature encoding | `alphazero/features.py` | position → 29 floats, action mask |
| Policy/value net | `alphazero/network.py` | MLP with hand-written backprop and AdamW |
| PUCT search | `alphazero/mcts.py` | AlphaZero-style tree search using the net |
| Self-play + training | `alphazero/self_play.py`, `trainer.py`, `parallel.py` | generate games, learn, checkpoint |
| Class baselines | `baselines/` | random, greedy, alpha-beta, plain MCTS |
| Actor-critic | `ppo/a2c.py` | REINFORCE with a value baseline |
| Warm start | `pretrain.py` | imitate greedy, then let search improve it |
| Evaluation | `eval/`, `evaluate.py` | matches, Elo, per-opponent tables |
| Playable UI | `play.py` | human vs any agent in a terminal |

No PyTorch. The network is 29 → 256 → 256 → {10 logits, 1 value}; forward and
backward passes are ~120 lines of NumPy. That is deliberate: the point of the
project is to see every gradient, not to hide it behind autograd.

**The one-paragraph summary of what the experiments found.** Greedy (take the
biggest immediate capture) is the strongest classical player; alpha-beta is
provably correct but loses to greedy and gets weaker with depth because its
leaf evaluation misunderstands capture chains. A cold-start AlphaZero deadlocks
— it learns to shuffle stones instead of capturing. The two fixes that worked
were (a) feeding the network, as input, *how many points each move captures
right now*, and (b) a supervised warm start that imitates greedy. With those,
the learned agent beats greedy 0.94–0.95 with search and 0.39–0.43 without it,
and search now helps monotonically (0.39 → 0.94 → 0.95 as simulations grow).
A search bug — values stored from the wrong perspective at terminal nodes —
was silently poisoning all the early search results; §2.5 and §12 tell that
story.

---

## 1. The game as a computational object

Ô ăn quan is a mancala-family game for two players on a ring of twelve squares.
Two of the squares hold **mandarin** pieces (big stones, worth 10 points); the
other ten hold five **citizen** stones each. You pick one of your five squares
and a direction, lift all the stones there, and drop them one per square around
the ring.

What makes it interesting — and what makes it different from, say, tic-tac-toe
— is the end of a turn:

1. If the square after your last stone holds stones, you **lift them and keep
   sowing** (a chain of sowing steps in one move).
2. If the square after your last stone is a mandarin square, your turn just
   ends; you never lift a mandarin.
3. If the square after your last stone is **empty**, you capture what comes
   after it — and the capture **chains** through alternating empty / non-empty
   squares until two empty squares in a row.

So a single move can be a chain of sowing steps followed by a chain of
captures. A "move" is not one stone placement; it is a small deterministic
program whose length depends on the board.

Numbers worth remembering, because they drive every design decision:

| Quantity | Value |
|---|---|
| cells | 12 (2 mandarin + 10 citizen) |
| legal moves per turn | at most 10 (5 squares × 2 directions) |
| stones on the board | 50 citizen + 2 mandarins |
| typical game length | 8 plies (greedy) to 20 plies (random) |
| branching factor | ~5 (only non-empty squares count) |
| typical capture | 5–15 points |
| score range | roughly 0–70 |

That last pair is the crux of the whole project: **the value of a single good
decision (≈10 points) is comparable to the error of a small neural network
trained for a few minutes**, and search is only useful if it can resolve
differences of that size.

*Code:* `engine/layout.py` (geometry), `engine/state.py` (move resolution),
`docs/rules.md` (the full ruleset with six worked examples).

---

## 2. Environment design: the MDP

### 2.1 State

The state is a frozen dataclass (`engine/state.py`):

```python
cells:        12 stone counts (citizens only)
quan:         2 booleans — is each mandarin piece still there?
quan_captor:  2 ints — who captured each mandarin square (-1 = nobody)
captured:     2 ints — citizen stones captured so far
mandarins_taken: 2 ints
player:       who is to move
ply, last_capture_ply
done, end_reason
```

Two choices matter here.

**Immutability.** `apply()` returns a *new* state; nothing is mutated. Search
duplicates positions thousands of times per second, and immutable states make
that trivially safe: an MCTS node can hold a reference to a position forever,
in any process, without defensive copying. It also makes the engine easy to
test — you can compare two states with `==` and mean it.

**Separating stones from the mandarin piece.** `cells[i]` counts only citizen
stones. A mandarin square with its piece is `quan[i] == True` and *may* also
hold citizen stones (that is exactly what makes it capturable: the *quan non*
rule says only a square holding ≥ 5 citizen stones can be taken). The mandarin
piece itself is not in `cells`, so the invariant

```python
sum(state.cells) + sum(state.captured) == 50
```

holds at every point of every game, and is asserted in tests. If you ever
"simplify" this by putting the mandarin in the stone count, you lose that
invariant and every capture rule becomes ambiguous.

Also note `quan_captor`: when a mandarin is captured the square becomes an
ordinary citizen square, but the stones that later sit there belong to whoever
took the mandarin. That is only knowable if you remember the captor, which is
why the field exists and why `final_points()` has a special case for it.

### 2.2 Action space

An action is `(cell, direction)` with `cell ∈ your five squares`, `direction ∈
{+1, -1}`. `action_index(cell, direction) = local_square_index * 2 +
(direction == ccw)` maps to a flat `0..9`, **independent of which player is
acting**: index 0 always means "my leftmost square, clockwise". That choice is
what makes a single policy head usable by both players — more on this in §3.

Legality is dynamic (a square with zero stones cannot be chosen), so the
network is always masked (see §4.2). A subtle but important point: the mask is
*not* part of the state and not learned. Illegality is a hard constraint, and
the code enforces it in three places — `legal_actions()` in the engine, the
mask in the network's softmax, and an assertion in `baselines/mcts.py`.

### 2.3 Transition

`apply()` is a deterministic function of `(state, action)`. It also returns a
`MoveResult` describing what happened (lifted stones, sowing order, captures,
mandarins taken, points). The result is used by UIs, tests and the greedy
baseline; the search only needs the next state.

The transition has three phases worth understanding, because they are where
most rule bugs live:

1. **Sow.** Lift `cells[cell]`, drop one stone per square in the direction
   until the stones run out. Record the last square `L`.
2. **Resolve.** Look at the square after `L`. Stones and it is a citizen
   square → lift and keep sowing (loop back to 1). Stones and it is a mandarin
   square → stop. Empty → go to 3.
3. **Capture chain.** While the square after the empty one holds something,
   take it (and the mandarin, if it is a rich mandarin square), then look one
   further. Two empty squares in a row stops the chain. Sowing never resumes
   after a capture.

A loop counter (`MAX_TURN_STEPS = 2000`) turns a rules bug into an exception
instead of a hang. Always put such a guard in an engine you are learning from.

### 2.4 Terminal conditions and reward

The game ends when both mandarins are gone, when a player must refill but has
fewer than five captured stones, when 60 plies pass with no capture, or (safety
net) at 1000 plies.

The 60-ply no-progress rule is not a formality. Stones circulate forever: a
player who keeps sowing along the ring without ever leaving a capture open
creates a stable loop, and two such players draw forever. Without the rule, a
random-vs-random game can run for thousands of plies, which destroys both the
training data distribution and the wall-clock budget. This is the mancala
analogue of chess's fifty-move rule, and it is why average game length is a
*diagnostic*: a jump from ~18 plies to 60–100 plies means an agent has learned
to shuffle, and that is precisely the symptom that diagnosed the cold-start
failure (§12).

The reward is **terminal and sparse**: `+1 / 0 / -1` from the perspective of
the player who moved. Both self-play value targets and A2C use it (with an
optional dense variant, `score_diff = (mine - theirs)/50`, clipped to
[-1, 1]). Note that 50 is the total stone count, so the dense target is
approximately a normalised final margin; the `/50` keeps it in the same range
as `tanh` output.

`points()` (captures only) and `final_points()` (captures + stones left on the
board + stones in conquered mandarin squares) are different functions on
purpose. Only `final_points()` decides the winner, and it raises if the game is
still running — a cheap guard against scoring a live position by accident.

### 2.5 The terminal-perspective trap (a real bug)

This one is worth memorising, because it is invisible in casual testing.

`State.player` means "player to move". When a move ends the game, `_end_move()`
returns early *without* handing the turn over — there is nobody to hand it to.
So at a terminal state, `state.player` is the player who **moved last**, not
the player "to move".

Now look at how search stores values. Every MCTS node stores the value of its
position **from the point of view of the player to move at that node**, and
`_select()` reads a child's value from the parent's side by negating it:

```python
q = -(child.value / child.visits)
```

That convention is consistent for network-evaluated nodes (the value head is
trained on "value for the player to move"). But a *terminal* node was being
evaluated with `state.returns(rules)`, which returns the value for
`state.player` — the player who just moved, i.e. the **opponent** of the node's
mover. The sign was flipped exactly at the leaves where the truth is known.

Consequences:

* A move that wins the game immediately looked, to the search, like a move that
  loses immediately. With a neutral network, the winning terminal move got ~8 %
  of the visits while losing moves got 85–92 %.
* Every AlphaZero self-play target in the early runs was computed by a search
  that disliked winning. The policy learned from those visit counts.
* The same bug was in `baselines/mcts.py`: `_rollout()` scored the playout for
  `state.player` *after* the game had finished.

The fix is one line in each place:

```python
# alphazero/mcts.py
def _leaf_value(self, node):
    return -node.state.returns(self.rules)   # flip back to the node's mover

# baselines/mcts.py
perspective = (1 - state.player) if state.done else state.player
```

Why did 38 tests miss it? Because the tests checked that moves were legal and
that games terminated, not that search *preferred a win*. The two regression
tests added with the fix do exactly that: build a position where one legal move
captures the last mandarin and ends the game, run search with a network that
has no opinion (uniform policy, zero value), and assert the winning move gets
most of the visits. Any future refactor that breaks the sign convention fails
those tests immediately.

**Lesson.** In any zero-sum search, write down the perspective convention once
("node value = value for the player to move at the node") and add one test per
place where a terminal value is produced. A sign error is silent: the search
still returns legal moves, the games still end, the losses still go down.

---

## 3. Features: what the network is allowed to see

`alphazero/features.py` produces 29 floats:

| indices | meaning | scaling |
|---|---|---|
| 0–11 | citizen stones per square | `min(x, 40) / 20` |
| 12–13 | mandarin piece present | 0/1 |
| 14–15 | citizen stones captured | `x / 50` |
| 16–17 | mandarins taken | `x / 2` |
| 18 | player to move | 0/1 |
| 19–28 | **points each of the 10 actions captures right now** | `min(x, 40) / 20` |

Three design notes.

**Absolute, not rotated.** A tempting simplification is to encode the board
from the mover's point of view (rotate the ring so "my squares" are always
1–5). Here the action space is already player-independent (index 0 = my
leftmost square), so an absolute encoding plus a player-to-move bit is
sufficient — and it means one network, one feature layout, and no rotation
logic to get wrong when the mandarin squares are captured and stop being
special.

**The capture feature is the single most important line in the repository.**
Indices 19–28 are computed by actually calling `state.apply()` on each legal
move and recording `result.points`. That is 10 extra `apply()` calls per
position (~90 µs), which is expensive relative to a 256-wide MLP forward pass,
and it is worth every microsecond.

Without it, the network had to infer *from raw counts alone* which of its ten
sowing moves lands next to a rich square — a chain-of-sowing-and-capture
computation that is, in effect, the entire game. A small MLP trained for a few
minutes cannot represent it. With it, the greedy signal is an input, and the
policy head only has to learn to *prefer* good captures rather than to
*discover* them.

**Scaling.** Everything is divided by a rough maximum so that inputs are O(1)
and the initial weights (He-normal) produce sane activations. Features that
exceed their nominal maximum (a square can hold far more than 40 stones after
a chain) are clipped, which is fine: 40 stones in one square is already
"everything", and clipping prevents one outlier position from dominating
gradients.

---

## 4. The network, and the maths behind it

`alphazero/network.py` is a small two-headed MLP:

```
x (29) → W1,b1 → ReLU → W2,b2 → ReLU → h (hidden)
                                     ├─ Wp,bp → logits (10)
                                     └─ Wv,bv → tanh → value (1)
```

with `hidden = 256`, `layers = 2` for the shipped model (128 is the default in
the code; the shipped one is wider).

### 4.1 Forward pass

For layer `l`: `pre_l = post_{l-1} W_l + b_l`, `post_l = max(pre_l, 0)`.
Then `z = h Wp + bp` and `v = tanh(h Wv + bv)`.

`tanh` bounds the value to (-1, 1), matching the ±1 outcome target and keeping
the value gradient well-scaled (`d tanh = 1 - v²`, which is largest near 0 and
vanishes at the extremes — desirable, since a confident value should move
slowly).

### 4.2 Masked softmax

Illegal actions must get exactly zero probability and must not affect the
gradient. The code sets illegal logits to `-1e9` before the softmax:

```python
masked = np.where(mask > 0, logits, -1e9)
masked -= masked.max(axis=1, keepdims=True)
w = np.exp(masked) * (mask > 0)
p = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
```

Because `p_illegal` is exactly 0, the gradient `p - π` is 0 there and the
policy head receives no signal for illegal moves. Note that the mask is applied
at *every* place a distribution is produced — training, `predict()`, and A2C —
so the network can never emit an illegal move even before search.

### 4.3 Losses and gradients

Policy loss (cross-entropy against a target distribution `π`, which is either
the one-hot teacher move or the normalised search visit counts):

```
L_pi = -(1/N) Σ_i Σ_a π_ia log p_ia
```

Value loss (mean squared error against `z ∈ {-1, 0, 1}` or the dense target):

```
L_v = (1/N) Σ_i (v_i - z_i)²
```

Total: `L = L_pi + L_v` (equal weights; AlphaZero uses the same).

Backprop, step by step:

1. **Policy head.** With softmax + cross-entropy, `∂L_pi/∂z_a = (p_a - π_a)/N`.
   This is the whole reason softmax and cross-entropy are paired; it is the
   cleanest gradient in the network.
2. **Value head.** `∂L_v/∂v = 2(v - z)/N`, and through `tanh`,
   `∂L_v/∂pre_v = 2(v - z)(1 - v²)/N`.
3. **Trunk.** `∂L/∂h = (∂L/∂z) Wpᵀ + (∂L/∂pre_v) Wvᵀ`.
4. **Each hidden layer, backwards.** `∂L/∂pre_l = ∂L/∂post_l ⊙ [pre_l > 0]`
   (the ReLU mask), then `∂L/∂W_l = post_{l-1}ᵀ ∂L/∂pre_l` and
   `∂L/∂b_l = Σ rows ∂L/∂pre_l`.

Every one of those lines appears in `_backprop()`. If you want to practice
backprop, re-derive them on paper and compare — this network is small enough to
check by hand, and finite-difference gradient checking on a 2-position batch is
a 10-line exercise (see §16).

### 4.4 AdamW

`_adam()` is standard Adam with bias correction plus decoupled weight decay:

```
m ← 0.9 m + 0.1 g
v ← 0.999 v + 0.001 g²
θ ← θ - lr · (m/(1-0.9^t)) / (sqrt(v/(1-0.999^t)) + 1e-8)
θ ← θ - lr · wd · θ          # decoupled, applied as a separate step
```

with `lr = 1e-3`, `wd = 1e-4`. Decoupled decay (AdamW) keeps the regularisation
from being scaled by the adaptive denominator, which matters when gradients
vary by orders of magnitude between the policy head (tiny at first) and the
trunk.

**Why hand-written?** Not to be clever. Writing `_backprop` forces you to
decide exactly what the value head predicts and in whose frame, which is where
the project's worst bug lived. Autograd would have hidden that decision, not
fixed it.

---

## 5. Classical search I: alpha-beta

`baselines/alphabeta.py` is negamax with alpha-beta pruning:

```
value(state, depth, α, β):
    if terminal: return final margin for the player to move
    if depth == 0: return evaluate(state)
    best = -∞
    for action in ordered(state):
        best = max(best, -value(apply(action), depth-1, -β, -α))
        α = max(α, best); if α ≥ β: break
    return best
```

The evaluation is `score_diff + 0.25 · (my stones − their stones)`, minus an
optional threat term. Move ordering is by immediate points, which is enough at
branching factor 5 (and it makes the "greedy" choice the first line searched,
so shallow searches return greedy's move).

This agent is **verified correct**: on 263/263 random positions it reproduces
the value of an exhaustive minimax. And yet it is *weak*, and gets weaker with
depth (0.28 vs greedy at depth 1, 0.40 at depth 2, 0.15 at depth 4, 0.13 at
depth 5).

That is not a contradiction. Alpha-beta is correct **with respect to its leaf
evaluation**, and the leaf evaluation is wrong for this game in a structural
way:

* The game's value is created by capture chains that start several sowing steps
  deep inside a single move. A depth-4 search sees four *moves*, but each move
  is itself a multi-step program; the static evaluation at the horizon cannot
  see a chain that is two sowing steps away.
* "Stones on my side" is not a proxy for safety. Stones sitting in front of a
  rich square are *ammunition for the opponent*: they are exactly what a capture
  chain eats. The evaluation rewards the player for leaving stones there.
* More depth means optimising that wrong objective more precisely, which is why
  the score *decreases* with depth. This is the classic horizon-effect failure
  and a good demonstration that "more search" is not a universal fix.

The honest conclusion: the search is fine; the heuristic is blind. What would
fix it is a quiescence search over capture chains and a capture-aware
evaluation — a real project in itself, and the reason the RL approach is
interesting here.

---

## 6. Classical search II: plain MCTS

`baselines/mcts.py` is the textbook UCB1 tree search:

* selection: `a = argmax[ Q(s,a) + c·sqrt(ln N(s) / N(s,a)) ]`, `c = 1.4`;
* expansion: pop an untried action, create the child;
* simulation: play random moves to the end;
* backup: add `±1` up the path, flipping sign each level.

Two implementation details that are easy to get wrong:

**Values are per-node-perspective.** A child's stored value is from the child's
side; the parent reads `-(child.value / child.visits)` for exploitation. Every
backup step negates before adding. Get this wrong and the tree prefers moves
that lose — see §2.5, which is where this codebase had exactly this bug.

**Rollout results must be captured before the playout.** The score has to be
computed for the player to move at the *start* of the rollout, and stored with
that player. Because `State.player` stays on the last mover when a game ends,
`_rollout()` snapshots `perspective` first and uses it for both the finished
and truncated cases.

Plain MCTS with random rollouts is *weak but not useless* here: 0.80 / 0.76 /
0.84 against greedy over three seeds (40 games each). The old result — 0.000 —
was the sign bug, not a property of the algorithm. Random playouts are still a
poor model of a capture-chain game, so MCTS needs ~100 playouts per move to
reach "slightly better than greedy", while a learned prior with 100 simulations
reaches 0.94.

Why keep it at all? Because it is the clean baseline that isolates the value of
(a) a learned prior and (b) a learned value. When you compare PUCT-with-network
against UCB1-with-random-rollouts at the same simulation budget, you are
measuring exactly what learning bought you.

---

## 7. PUCT: AlphaZero's search

`alphazero/mcts.py` replaces UCB1's exploration term with a prior-weighted one:

```
a = argmax[ Q(s,a) + c_puct · P(s,a) · sqrt(N(s)) / (1 + N(s,a)) ]
```

* `Q(s,a) = -child.value / child.visits` (negated: child's view is the
  opponent's),
* `P(s,a)` is the network's policy prior for that action,
* `c_puct = 1.5`,
* `N(s)` is the parent's visit count.

The shape of the bonus does two things: it shrinks like `1/N` as a child is
visited (exploration decays per action), and it grows like `sqrt(N)` as the
parent is visited (the tree keeps exploring elsewhere). Crucially, the bonus is
*proportional to the prior*, so a move the network likes gets tried earlier.
With random rollouts there is no prior, which is why UCB1 needs `sqrt(ln N/N)`
instead — a prior-free bonus has to be estimated from visits alone.

Search procedure per simulation:

```
node = root
while node has children:
    node = select(node)          # PUCT
value = expand(node) if not terminal else leaf_value(node)
for visited in reversed(path):
    visited.visits += 1
    visited.value  += value
    value = -value               # flip at every level
```

Expansion calls `net.predict(state)` once, creates one child per legal action
with its prior, and returns the network's value. The root distribution returned
to the self-play loop is the normalised visit count.

Two training-time details:

* **Dirichlet noise at the root** (`α = 0.3`, `ε = 0.25`):
  `P ← (1-ε)P + ε·Dir(α)`. This guarantees that every legal move gets tried at
  least occasionally, so the search cannot entrench the prior's blind spots.
  It is only used in self-play, never in evaluation.
* **Temperature.** Moves are sampled from `π^(1/τ)`; `τ = 1` for the first 8
  plies (opening variety) and `τ → 0` (argmax) afterwards.

The number of simulations is the accuracy/CPU dial. With ~5 legal moves, 100
simulations gives ~20 visits per child, which is enough to resolve a 10-point
difference if the value head is accurate to a few points — and not enough if
it is accurate to ±20 (§12).

---

## 8. The AlphaZero loop

`alphazero/trainer.py` + `self_play.py` + `parallel.py`:

```
for iteration:
    records = self_play(net, games=G, sims=S)     # multiprocess
    buffer.extend(records)                        # deque, last 400 games
    for step in range(40):
        batch = sample(buffer, 128)
        net.train_batch(x, π, z, mask)            # policy CE + value MSE
    save checkpoint
```

Self-play details:

* The target `π` is the **normalised visit distribution**, not the sampled
  move. That is the whole point: the search is a (much) better policy than the
  raw network, so the network is trained to imitate its own search. This is
  policy improvement by distillation, iterated.
* `z` is the game result for the player at that position (`+1/-1/0`, or the
  dense margin). The *same* game result is used for both players' positions,
  with the sign flipped — that is correct for a zero-sum game, and it is why a
  single self-play game yields twice as many useful samples as it has plies.
* `buffer_games = 400` keeps recent games only. Too large a buffer and the
  network keeps learning from a policy it has outgrown; too small and the
  gradients are noisy and correlated.

Parallelism: `SelfPlayPool` sends `(weights, seed, hidden, layers, sims, ...)`
to worker processes; each rebuilds a network from the arrays, plays one game,
and returns the record. Self-play is embarrassingly parallel because games are
independent and the weights are read-only. With 12 workers this reached ~640
games/minute, and gradient steps stay in the parent process.

What to log, and why: `plies_mean` (a jump means shuffling), score-difference
mean (is anyone actually capturing?), `policy_loss`, `value_loss`, and buffer
size. A falling policy loss with rising plies is the signature of learning to
imitate a search that is itself playing badly — which is exactly what the
buggy runs looked like.

---

## 9. The actor-critic baseline (A2C)

`ppo/a2c.py` is REINFORCE with a learned baseline, no search:

```
play a game by sampling from π(·|s)
for each position:  A = z - V(s)        # z = ±1/0 outcome
loss = -A log π(a|s) + c_v (V(s) - z)² - c_e H(π)
```

Because the only reward is terminal, no bootstrapping is needed: `z` *is* the
return. Advantages are normalised (zero mean, unit variance) per iteration to
keep the policy gradient's scale stable, and an entropy bonus (`c_e = 0.01`)
delays collapse to a deterministic policy.

The entropy gradient, since it is the part people get wrong:

```
H = -Σ_a p_a log p_a
∂H/∂z_b = -p_b (log p_b + H)
```

The objective *maximises* entropy, so the loss contains `-c_e H`, and the head
gradient contributes `+c_e · p_b (log p_b + H)` — the two minus signs cancel,
which is why the code adds a positive term. Sign-check it on a 2-action case
by hand if you ever touch it.

Measured result: 0.59 vs random, 0.09 vs greedy, 0.70 vs alpha-beta 4
(policy-only). Same story as cold-start AlphaZero — better than random, better
than alpha-beta 4, far behind greedy — and the same disease: no search, and a
value function too coarse for a capture-chain game. It is here to show that
"RL without search on this game" and "RL with search on this game" are
different problems, and that the second one needs a prior worth searching with.

---

## 10. The supervised warm start

`pretrain.py` is the cheapest effective intervention in the project:

1. Play 12 000 greedy-vs-greedy games (with 0–2 random opening plies for
   variety), recording `(features, one-hot greedy move, mask, player)`.
2. Label each position with the final margin, `z = clip((mine-theirs)/50)`.
3. Train the same network with the same loss for 6 epochs (~99 000 positions).

Policy cross-entropy fell to 0.60 against the teacher and value MSE to 0.21.
The resulting `runs/base_model.npz` scores 0.93 vs random and 0.39 vs greedy
*policy-only*, and 0.94 vs greedy with 100 simulations.

Is this "cheating"? It is imitation learning, and it is standard practice when
you have a decent heuristic and limited compute (AlphaGo's supervised policy
network did exactly this before RL). What it buys is not strength — greedy is
still slightly stronger policy-only — but a *prior with the right shape*: it
puts probability on capturing moves and gives the value head a signal that
tracks the score. Search can improve a good prior; it cannot improve a uniform
one within a realistic compute budget.

The honest caveat: the teacher's ceiling is close. `base_model` policy-only is
roughly greedy-strength, and beating greedy decisively (0.94) required search.
A bigger teacher (or a search-based teacher, i.e. properly working AlphaZero
self-play) would raise the ceiling.

---

## 11. Evaluation: how to not fool yourself

`eval/arena.py` + `eval/elo.py` + `evaluate.py`.

* **Score** = `(wins + 0.5·draws) / games`.
* **Alternate seats.** `first_player` flips every game; player 0 moves first,
  which is a real advantage in this game.
* **Random opening plies.** Two random plies before the agents take over.
  Deterministic agents would otherwise replay one identical game 40 times and
  you would be reporting a single sample with `n = 1` in disguise.
* **Elo from the tournament**, not from single match-ups: `fit_elo` does
  gradient ascent on the Bradley-Terry likelihood over all pairings, centring
  the mean at 0.
* **Per-matchup Elo difference** is `-400·log10(1/score - 1)`.

The number to keep in your head: **with 40 games, the standard error of a score
near 0.5 is ≈ sqrt(0.25/40) ≈ 0.08**, so a 95 % interval is roughly ±0.15. Most
of the differences in `docs/results.md` are therefore *directions*, not
measurements. Two consequences:

1. Never report a single 40-game match as proof of a 0.05 improvement.
2. When you see a large, reproducible swing (0.00 → 0.80, or 0.10 → 0.95), it
   is almost certainly a bug or a real qualitative change, not noise. That
   heuristic is what caught the sign error.

Seeds matter too: `runs/base_model.npz` policy-only scored 0.812/0.425/0.725/
1.000 at seed 0 and 0.925/0.388/0.675/0.238 at seed 1. The honest way to read
that is "marginally behind greedy, clearly ahead of alpha-beta 4, and roughly
even with plain MCTS" — with error bars straddling all of it.

---

## 12. The experiment story, in the order it happened

1. **Cold-start AlphaZero.** 20 iterations × 40 games × 96 sims, 19 features.
   0.45 vs random, 0.15 vs greedy, 0.06 policy-only vs greedy. Self-play games
   were 60–100 plies and the agent captured 3.6 points per move — the same as
   random. Diagnosis: "the value error (~20 points RMS) is bigger than a
   capture, so search cannot see the difference; and the raw-count encoding
   cannot express which sowing move captures."
2. **More of the same.** 6 300 more games with 12 workers changed nothing (0.40
   vs random). Dense value targets improved value fit (0.75 → 0.19) but not
   policy strength. That is the classic sign that the bottleneck is not the
   optimiser.
3. **Two fixes.** Per-action capture features (29 inputs) + supervised warm
   start. Result: a policy-only agent at roughly greedy strength.
4. **Search on top made it worse.** 400 sims scored 0.10 vs greedy, down from
   0.43 policy-only. Diagnosis at the time: PUCT trusts a noisy value head over
   a good prior.
5. **The sign bug.** A code review of the search path — not an experiment —
   found that terminal values were stored from the mover's side, so the search
   preferred moves that lost on the spot. Both `alphazero/mcts.py` and
   `baselines/mcts.py` were affected; two regression tests were added.
6. **Re-measurement.** Everything involving search was re-run. Search now helps
   monotonically, plain MCTS beats greedy, and random-net self-play stopped
   shuffling. The cold-start failures were *partly* caused by corrupted
   self-play targets, so the earlier "value error dominates" explanation is
   now only a partial explanation.
7. **Self-play, second attempt.** With the fixed search, a lower learning rate
   (3e-4) and more search per self-play move (128 sims), eight iterations on
   top of a fresh warm start did improve the agent — 0.844 vs plain MCTS 100
   against 0.766 for the warm start itself — while alpha-beta 4 stayed flat
   (0.859 vs 0.881, inside the error bars). The lesson is narrower than
   "self-play works": at this budget it needs a prior that is already good and
   a learning rate that does not erase it. Tuning `c_puct` and enlarging the
   warm start changed nothing beyond noise.

The meta-lesson is the important one: the story in step 4 was plausible,
well-argued, and wrong. It survived because it was consistent with a symptom
(search hurt) that had a simpler cause. Whenever a result is surprising in a
way that implicates your method rather than your opponent, check the sign
conventions first.

---

## 13. Corrected results

Ruleset `Rules(mandarin_capture_min=5, no_progress_plies=60)`; 40 games per
pairing, alternating seats, two random opening plies, seed 1 unless stated.

### Classical

| Player | vs random | vs greedy | vs alpha-beta 4 | vs MCTS 100 |
|---|---|---|---|---|
| greedy | 0.875 | — | 0.85 | 1.000 |
| alpha-beta 4 | 0.525 | 0.15 | — | — |
| plain MCTS 100 | — | 0.80 / 0.76 / 0.84 (3 seeds) | — | — |

### `runs/base_model.npz` (warm start, 256×2)

| Simulations | vs random | vs greedy | vs alpha-beta 4 | vs MCTS 100 |
|---|---|---|---|---|
| 0 (policy only) | 0.925 | 0.388 | 0.675 | 0.238 |
| 32 | 1.000 | — | — | — |
| 100 | 0.988 | 0.938 | 0.863 | 0.750 |
| 400 | — | 0.950 | — | — |

### `runs/base_az2/final.npz` (warm start + self-play, 64 sims)

| Opponent | Score |
|---|---|
| random | 0.975 |
| greedy | 0.613 |
| alpha-beta 4 | 0.700 |
| plain MCTS 100 | 0.350 |

### `runs/az_big/final.npz` (warm start v2 + 8 self-play iterations)

160 games per pairing, seed 3:

| Simulations | vs greedy | vs alpha-beta 4 | vs MCTS 100 |
|---|---|---|---|
| 100 | 0.966 | 0.859 | 0.844 |
| 400 | 0.978 | 0.891 | 0.884 |

### `runs/base_a2c/final.npz` (A2C, policy only)

| Opponent | Score |
|---|---|
| random | 0.588 |
| greedy | 0.087 |
| alpha-beta 4 | 0.700 |

The headline: **a decent prior plus PUCT beats every classical baseline**
(0.94–0.98 vs greedy at 100–400 simulations), and the same network without
search does not (0.39). Alpha-beta remains verified correct and remains weaker
than greedy — a fact about evaluation functions, not about pruning.

Full tables, the archived pre-fix numbers and the reproduction commands are in
`docs/results.md`.

---

## 14. Lessons that generalise

1. **A terminal-perspective sign error is the most expensive bug in game RL.**
   It is invisible in win rates against random opponents (both sides suffer
   equally), it makes search look harmful, and it poisons self-play targets.
   One test per terminal-value site.
2. **Your representation is part of your algorithm.** The same optimiser, same
   network, same search went from "learns nothing" to "beats greedy" when ten
   features describing the immediate consequence of each action were added.
   Before scaling compute, ask whether the function is representable at all.
3. **A correct search with a wrong evaluation gets worse with depth.** "More
   search" is a hypothesis, not a fix. Test it: if depth hurts, the leaf
   evaluation or the value function is the problem.
4. **Sparse terminal rewards make self-play slow to discover tactics.** A good
   teacher, an action-consequence feature, or a dense target is often cheaper
   than 10× the games.
5. **Imitation is a legitimate bootstrap.** Warm-starting from a heuristic is
   the difference between a prior that search can refine and one it cannot.
6. **Report error bars, or say what you did not measure.** With 40 games,
   ±0.15 is the resolution. Nine of ten "improvements" in a hobby project are
   inside the noise.
7. **Log behaviour, not just losses.** Mean game length, points per move and
   draw rate diagnosed the cold-start failure long before any Elo table did.

---

## 15. Where to go next, ranked by information per hour

1. **Re-run cold-start AlphaZero with the bug fixed**, and see how much of the
   original failure was the bug rather than the value function. This is the
   single most interesting open question in the repository, and it is one
   command (`train.py --iters 20 --games 40 --sims 96`).
2. **Give alpha-beta a quiescence search** over capture chains. If it then
   beats greedy, the "evaluation is blind" story is confirmed and you have a
   much stronger classical baseline for free.
3. **Train the value head to convergence** on fixed data (supervised
   `score_diff` regression, many epochs). Measured on 461 positions its error is
   ≈ 20 points (RMSE 0.40 where 1.0 = 50 points) and it is close to unbiased, so
   this is a noise problem, not a bias problem: shrink the noise and the search
   should keep scaling past 400 simulations.
4. **A league instead of pure self-play**: mix in greedy and past checkpoints
   so passivity is punished instead of mirrored. Standard fix for
   self-play collapsing into draws.
5. **TD(λ) value targets** instead of the final margin, so the value learns
   from its own bootstraps and a single game yields many graded targets.
6. **Scale.** 10⁵–10⁶ games is hours on this machine (32 cores, ~640
   games/minute with 12 workers). Everything above is worth doing before
   spending those hours.

---

## 16. Cheat sheet, glossary, exercises

### Numbers

```
features 29   hidden 256×2   actions 10   lr 1e-3   wd 1e-4
c_puct 1.5    Dirichlet α 0.3, ε 0.25       UCB1 c 1.4
warm start: 12k games, 99k positions, 6 epochs, CE 0.60, value MSE 0.21
evaluation: 40 games, 2 opening plies, alternating seats, seed 1
noise floor at 40 games: ±0.15
```

### Glossary

* **PUCT** — polynomial upper confidence trees; UCB with a prior-weighted
  exploration bonus.
* **Prior** — the network's policy output for a position, used as `P(s,a)`.
* **Visit distribution** — normalised child visit counts at the root; the
  self-play policy target.
* **Value head** — scalar output in (-1, 1), the expected result for the player
  to move.
* **Quan non** — a "young mandarin": a mandarin square with fewer than five
  citizen stones, not capturable.
* **Refill** — when your side is empty you must spend five captured stones to
  put one in each square; if you cannot, you lose.
* **No-progress draw** — 60 plies without a capture ends the game, because
  stones otherwise circulate forever.
* **Elo difference** — `-400 log10(1/score - 1)`; a 0.75 score is ≈ +190.
* **Warm start** — supervised pre-training on a heuristic teacher before RL.

### Exercises

1. **Gradient check.** Take a 2-position batch, compute the analytic gradients
   from `_backprop`, and compare with finite differences
   (`(L(θ+ε) - L(θ-ε))/2ε`) for a few entries of `W1`. Everything should agree
   to ~1e-6.
2. **Sign test.** Write a position where one move wins immediately, run search
   with a zero-value, uniform-policy network, and assert that the winning move
   gets the most visits. (This is exactly the regression test added in §2.5;
   write it before looking at `tests/test_agents.py`.)
3. **Depth curve.** Measure alpha-beta at depths 1–6 against greedy. Explain
   the shape of the curve using §5. Then add a one-ply quiescence search and
   measure again.
4. **Feature ablation.** Train `pretrain.py` twice, once with the ten
   action-capture features zeroed out, and compare policy-only strength. This
   reproduces the project's central claim in about ten minutes.
5. **Simulation sweep.** Evaluate `base_model` at 0, 10, 32, 100, 400, 1000
   simulations against greedy. Plot score vs sims and relate the knee to the
   value head's error.
6. **Error bars.** Re-run one pairing at seeds 1–5 and compute the spread. How
   many of the differences in §13 survive?

### Reading

* Silver et al., *Mastering the game of Go without human knowledge* (2017) —
  the PUCT + self-play loop this repository imitates.
* Sutton & Barto, *Reinforcement Learning: An Introduction*, ch. 13 (policy
  gradient) and ch. 5 (Monte Carlo control).
* Browne et al., *A Survey of Monte Carlo Tree Search Methods* (2012) — UCT,
  and why the exploration bonus has the shape it has.
* `docs/rules.md` in this repository — the ruleset is the specification; read
  it before changing the engine.

---

# Part II — the general handbook

Part I is one worked example. Part II is the transferable material: what is true
of reinforcement learning in general, what the standard algorithms are, how they
are derived, how they fail, and how to run a project so that your measurements
mean something. It is organised so you can read it cold, but the examples are
from Part I.

---

## 17. The formalism, properly

Everything in this repository is an instance of one object. Get the object
right and most algorithm questions answer themselves.

### 17.1 The Markov decision process

An MDP is a tuple `(S, A, P, R, γ)`:

```
S      set of states
A      set of actions (may depend on the state: A(s))
P(s'|s,a)  transition probability
R(s,a)     expected immediate reward (sometimes R(s,a,s'))
γ ∈ [0,1]  discount factor
```

with the **Markov property**: the future depends on the past only through the
current state. That single assumption is what makes dynamic programming
possible. If your state is missing information that affects the future, the
process is a POMDP and you either add memory to the state or you accept that no
value function can be exact.

Two details that cause real bugs:

* **Episodic vs continuing.** Games are episodic: a terminal state exists, and
the return is finite without discounting. Control tasks are often continuing
and need `γ < 1` to keep the return finite. In this repository the reward is
terminal and `γ = 1` is fine — there is always an end.
* **Who is in `A`.** In a two-player game, "action" at your turn is your move;
at the opponent's turn it is theirs. The clean way to write it down is a
single-agent MDP with alternating turns, where the opponent is part of `P`.
That is exactly what `self_play.py` does: it hands the wheel to the opponent's
policy and records the position from the mover's point of view.

### 17.2 Return, value, and action-value

```
G_t = r_{t+1} + γ r_{t+2} + γ² r_{t+3} + ...
V^π(s) = E_π[ G_t | S_t = s ]
Q^π(s,a) = E_π[ G_t | S_t = s, A_t = a ]
A^π(s,a) = Q^π(s,a) - V^π(s)          # advantage
```

The advantage is the quantity every policy-gradient method is really about:
*how much better is this action than what I would have done on average?* Positive
advantage → increase its probability. In this repo `z - V(s)` in `ppo/a2c.py`
is exactly an advantage estimate with a one-sample return.

**Discounting** does two jobs: it keeps infinite sums finite, and it encodes
urgency. `γ = 0.99` has an effective horizon of about `1/(1-γ) = 100` steps. If
your episode is 8 plies (greedy Ô ăn quan) then `γ = 1` and `γ = 0.99` are
indistinguishable; if it is 5 000 steps, they are different problems.

### 17.3 The Bellman equations

The expectation form (for a fixed policy):

```
V^π(s)  = Σ_a π(a|s) [ R(s,a) + γ Σ_s' P(s'|s,a) V^π(s') ]
Q^π(s,a) = R(s,a) + γ Σ_s' P(s'|s,a) V^π(s')
```

The optimality form:

```
V*(s)  = max_a [ R(s,a) + γ Σ_s' P(s'|s,a) V*(s') ]
Q*(s,a) = R(s,a) + γ Σ_s' P(s'|s,a) max_a' Q*(s',a')
π*(s)  = argmax_a Q*(s,a)
```

These are fixed-point equations. Define the Bellman operator `T` by the
right-hand side; then `T` is a **γ-contraction** in the sup norm:

```
|| T V1 - T V2 ||∞ ≤ γ || V1 - V2 ||∞      (because max and Σ are non-expansive)
```

Banach's fixed-point theorem then gives: a unique fixed point `V*`, and
value iteration `V ← T V` converges at rate `γ^k`, with error bound
`||V_k - V*||∞ ≤ γ^k ||V_0 - V*||∞`. Two consequences worth internalising:

1. **Why bootstrapping works.** You can improve an estimate using the estimate
   itself, because the operator pulls every guess toward the truth.
2. **Why `γ` near 1 is hard.** Convergence gets slow exactly when the horizon
   is long. This is the same difficulty that makes sparse-reward RL hard, seen
   from the dynamic-programming side.

Policy iteration alternates **evaluate** (`V^π`, by solving or iterating) and
**improve** (`π'(s) = argmax_a Q^π(s,a)`), and the policy improvement theorem
says `V^{π'} ≥ V^π` — improvement never hurts, and it is strict unless `π` is
already optimal. AlphaZero is a stochastic, approximate policy-iteration
scheme: search evaluates, the improved visit distribution becomes the target,
the network is fitted to it (§19).

### 17.4 The policy gradient theorem

When you cannot enumerate states, you parametrise `π_θ(a|s)` and maximise
`J(θ) = E_{s~d^π}[V^π(s)]`. The gradient is:

```
∇J(θ) = Σ_s d^π(s) Σ_a ∇π_θ(a|s) Q^π(s,a)
      = E_{s~d^π, a~π_θ}[ ∇log π_θ(a|s) · Q^π(s,a) ]
```

Derivation sketch (worth doing once on paper):

1. `∇V^π(s) = ∇ Σ_a π(a|s) Q^π(s,a)`.
2. `Q^π` depends on `θ` only through the *next* state's value, so expanding
   recursively gives `∇V^π(s) = Σ_a [∇π(a|s)] Q^π(s,a) + Σ_a π(a|s) γ Σ_s' P ∇V^π(s')`.
3. Unrolling that recursion turns the second term into a sum over the
   discounted state-visitation distribution `d^π`, and the first term survives
   as the expression above.
4. Use `∇π = π ∇log π` to get the sampleable form. That identity is the whole
   trick: you never differentiate through the environment, only through your
   own log-probabilities.

**Baselines.** For any function `b(s)` independent of `a`,
`E[∇log π(a|s) b(s)] = b(s) ∇Σ_a π(a|s) = 0`. So you may subtract a baseline
for free:

```
∇J = E[ ∇log π(a|s) (Q^π(s,a) - b(s)) ]
```

Choosing `b = V^π` leaves the advantage, which has much lower variance and
unchanged expectation. That is the actor-critic idea in one line, and it is why
`ppo/a2c.py` subtracts `V(s)` and normalises the result. In a zero-sum game
the other useful baseline is the *opponent's* value — any quantity that does
not depend on your action is legal.

**Variance.** The gradient estimate is unbiased but noisy: with `±1` terminal
rewards only, one game gives one bit of information per position. Four standard
reductions, all used somewhere in this repo:

| Technique | Effect |
|---|---|
| baseline / advantage | removes the constant offset; usually 10–100× variance cut |
| entropy bonus | keeps the policy stochastic, prevents premature collapse |
| reward-to-go instead of full return | removes irrelevant future randomness |
| many parallel games per update | averaging, and decorrelated samples |

### 17.5 Monte Carlo vs temporal difference, and the bias–variance dial

Two ways to estimate `V^π(s)`: play to the end and use `G_t` (**Monte Carlo**,
unbiased, high variance, must wait for termination), or use one step and
bootstrap (**TD**, biased by the current estimate, low variance, online).

```
TD(0):   V(s) ← V(s) + α [ r + γ V(s') - V(s) ]
                         └────── δ, the TD error ──────┘
```

`n`-step returns interpolate: `G_t^{(n)} = r_{t+1} + ... + γ^{n-1} r_{t+n} +
γ^n V(s_{t+n})`. `n = 1` is TD, `n = ∞` is Monte Carlo. **GAE(λ)** (§18.3) is
the exponentially weighted average of all `n`, with a single knob.

In this project the choice is concrete and was measured: outcome-only (`±1`)
versus dense `score_diff` value targets. Dense targets fit the value better
(MSE 0.75 → 0.19) without making the policy stronger — the *bottleneck was not
the optimiser* (§12). That is the general lesson: a better-fitting value
function only helps if search or the policy actually consumes it accurately
enough to change a decision.

### 17.6 Function approximation and the deadly triad

With a table, `Q(s,a)` has an entry per state-action pair and everything above
is exact. With a network, `Q_φ(s,a)` generalises — and generalisation is where
the guarantees stop.

Sutton and Barto's **deadly triad**: function approximation + bootstrapping +
off-policy data can diverge, because the update targets move with the
parameters being updated. All three modern families are ways of taming it:

* **DQN** — freeze the target with a lagged network, and keep data in a replay
  buffer (breaks the correlation, still off-policy).
* **PPO/A2C** — stay on-policy and bootstrap only one step (or GAE with
  moderate λ), so targets are close to the data that produced them.
* **AlphaZero** — bootstrap from a search value that is *itself* grounded in
  real terminal outcomes within the tree, and train on freshly generated games.

If a run diverges, the triad is the first thing to check: reduce bootstrapping
(smaller `λ`, shorter `n`), reduce off-policyness, or slow the target down
(lower learning rate, slower target updates). `NaN` losses are usually this,
plus an unnormalised input or a log of zero.

---

## 18. The algorithm map

You do not need every algorithm. You need to recognise which *family* your
problem belongs to, because the families fail differently.

| Family | Examples | Data | Model? | Action space | Strength | Typical failure |
|---|---|---|---|---|---|---|
| Bandits | UCB1, Thompson | online | no | any | explore/exploit with no state | non-stationarity |
| Tabular TD | Q-learning, SARSA, Dyna | online | no (Dyna: learned) | discrete | exact, tiny, debuggable | state explosion |
| Value-based deep | DQN, Double/Dueling, Rainbow | off-policy replay | no | discrete | sample-efficient reuse | overestimation, instability |
| Policy gradient | REINFORCE, A2C, PPO | on-policy | no | discrete + continuous | simple, stable-ish, scales | sample hunger, variance |
| Max-entropy AC | SAC, TD3 (deterministic) | off-policy | no | continuous | best sample efficiency in control | tuning, overestimation |
| Model-based | Dyna, MBPO, Dreamer, MuZero | either | yes / learned | any | sample efficiency, planning | model bias |
| Search + learned eval | AlphaZero, Expert Iteration | self-generated | implicit | discrete | very strong play | needs a simulator, compute |
| Black-box | CMA-ES, ES, CEM | population | no | low-dim | trivially parallel, robust | sample cost, no credit assignment |

### 18.1 Bandits: the degenerate MDP

`γ = 0`, no state, `K` arms. UCB1 picks
`argmax_i [ Q_i + c sqrt(2 ln N / N_i) ]`; Thompson sampling samples from each
arm's posterior and picks the best draw. Everything about explore/exploit is
visible here in its cleanest form, and the same UCB bonus reappears in MCTS
(§6, §19) with the state folded in. If your problem has context but no
long-horizon consequences, it is a **contextual bandit**: use it, do not build
an MDP.

### 18.2 Value-based: Q-learning and DQN

```
Q-learning (off-policy, target uses max):
    Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') - Q(s,a) ]

SARSA (on-policy, target uses the action actually taken):
    Q(s,a) ← Q(s,a) + α [ r + γ Q(s',a') - Q(s,a) ]
```

The `max` makes Q-learning optimistic: with function approximation and noise,
`E[max] ≥ max E`, so Q-values drift upward — **maximisation bias**. DQN's
standard fixes, in increasing order of sophistication:

1. **Replay buffer** — decorrelate samples, reuse experience (off-policy only).
2. **Target network** — update `Q_target` every `C` steps; stops the target
   chasing the prediction.
3. **Double DQN** — select with the online net, evaluate with the target net:
   `r + γ Q_target(s', argmax_a Q_online(s',a))`. Removes most of the bias.
4. **Huber loss** — clip the gradient of large TD errors.
5. **Dueling heads** — split `Q = V + A - mean(A)`; useful when most actions
   are equivalent, which is common.
6. **Prioritised replay** — sample by `|δ|`; faster, but changes the
distribution (importance weights needed).
7. **n-step returns** — propagate reward faster; standard in Rainbow.

When to use it: **discrete actions, a cheap simulator, and you want to squeeze
every sample**. Do not use it for continuous control (the `max` is a hard
optimisation inside the update) and do not expect it to be stable without the
tricks above.

### 18.3 Policy gradient, actor-critic, GAE, PPO

**REINFORCE**: `θ ← θ + α ∇log π(a|s) G_t`. Unbiased, high variance, on-policy.

**Actor-critic**: replace `G_t` with `δ = r + γV(s') - V(s)`, train `V` by
regression, train `π` with `δ`. Bias goes up (the critic is wrong), variance
drops a lot. `ppo/a2c.py` is the terminal-reward special case where `δ = z - V(s)`.

**GAE(λ)** — the standard advantage estimator:

```
δ_t = r_t + γ V(s_{t+1}) - V(s_t)
A_t = δ_t + γλ δ_{t+1} + (γλ)² δ_{t+2} + ...
```

`λ = 0` → one-step TD (low variance, high bias). `λ = 1` → Monte Carlo (zero
bias, high variance). `λ ≈ 0.95` is the default because it is usually the
sweet spot on the curve, not because it is principled.

**PPO** replaces the fragile trust region of TRPO with a clipped ratio:

```
r_t(θ) = π_θ(a_t|s_t) / π_θold(a_t|s_t)
L^CLIP = E_t [ min( r_t Â_t , clip(r_t, 1-ε, 1+ε) Â_t ) ]
```

Read it as: take the normal policy-gradient step, but if the new policy has
already moved `ε` (typically 0.2) in the improving direction, stop paying for
moving further. The `min` means the objective is flat (gradient zero) once the
ratio is clipped *and* the advantage is positive; if the advantage is negative
the clip acts on the other side. The full loss adds
`-c_v (V - V_target)²` and `-c_e H(π)`, `c_v ≈ 0.5`, `c_e ≈ 0.01`.

PPO's practical details matter more than its theory: normalise advantages,
clip gradients, run several epochs over each batch with a *fixed* `θ_old`,
and monitor the approximate KL — if it runs away, your `ε`, learning rate or
epochs are wrong. Note this repository's `ppo/` directory is A2C, not PPO;
the name is historical, and the difference is the clip and the multi-epoch
batch.

### 18.4 Continuous control: DDPG, TD3, SAC

**DDPG** = DQN with a deterministic actor `μ(s)` and `Q(s, μ(s))`; explore with
added noise. **TD3** = DDPG plus three fixes: twin critics (take the min),
delayed actor updates, and target-policy smoothing. **SAC** = TD3's machinery
with a stochastic actor and a changed objective:

```
J(π) = Σ_t E[ r_t + α H(π(·|s_t)) ]
```

Maximising entropy means "be as random as you can while still getting reward",
which produces excellent exploration and a policy that is robust to perturbation.
`α` can be learned automatically against a target entropy. On continuous control
benchmarks SAC is the default recommendation; on discrete actions it also works
(soft Q-learning) but DQN/PPO are simpler.

### 18.5 Model-based

The cheapest form is **Dyna**: do a real update, then `k` imagined updates from
a learned or tabulated model. The modern forms learn a latent model:

* **MuZero** — learns `(representation, dynamics, prediction)` and runs MCTS in
  the *latent* space, trained only to predict reward, policy and value. This is
  the natural next step from this repository: it removes the need for a
  perfect simulator, at the cost of a much harder training problem.
* **Dreamer** — learns a recurrent latent model and trains the actor entirely
  on imagined rollouts; very sample-efficient on pixels.
* **MBPO** — short branched rollouts from a probabilistic ensemble; the
  ensemble disagreement is the safety valve against model exploitation.

Model-based RL is where sample efficiency comes from, and where silent failure
comes from too: the policy finds the model's errors. Always keep a real
environment in the loop for evaluation.

### 18.6 Search, with and without learning

| Method | Needs | Value source | Best when |
|---|---|---|---|
| Minimax / alpha-beta | perfect simulator, small-ish tree | exact + heuristic | short games, strong heuristic, need certainty |
| MCTS / UCT | simulator | random rollouts | large branching, no heuristic, anytime |
| PUCT / AlphaZero | simulator + network | learned value, learned prior | large branching, expensive eval, lots of compute |
| MuZero | simulator only for training | learned latent model | no simulator at deployment |

Alpha-beta is exact given enough depth and a correct leaf value, and it prunes
aggressively (best case `O(b^{d/2})`). MCTS is anytime and handles huge
branching, but its strength depends entirely on the quality of its simulations.
PUCT is strictly better than UCT *when the prior and value are good*, and
strictly worse when they are not — which is exactly what §12 step 4 looked like
before the bug was found.

### 18.7 Choosing, in six questions

1. **Do you have a simulator?** If no, you need model-based RL or offline RL
   (or a real system you are allowed to poke, which is a different risk profile).
2. **Is the state small enough to tabulate (`< 10⁶`)?** Tabulate first. It is
   the only setting where you can prove your code correct, and it takes an hour.
3. **Discrete or continuous actions?** Discrete → DQN or PPO. Continuous →
   SAC (or PPO if on-policy simplicity matters more than samples).
4. **Is the horizon long and the reward sparse?** Then representation, reward
   shaping and exploration will decide the project, not the algorithm (§20, §21).
5. **Is there a heuristic teacher?** Imitation plus RL (warm start) is usually
   the highest return per hour of work — §10 is the proof.
6. **Do you need strength more than novelty?** Add search on top of a learned
   model (AlphaZero-style), and budget for the simulator throughput it requires.

When in doubt, start with the simplest algorithm that can represent the value
function you need, and add machinery only when a measurement demands it.

---

## 19. Search and learning together

### 19.1 Why AlphaZero is policy iteration

Write the loop without the neural network:

```
π_0 = anything
t = 0
repeat:
    for each visited state, run search using π_t (as prior) and V_t (as value)
    π_{t+1}(a|s) ∝ N(s,a)          # normalised visit counts
    V_{t+1}(s)  = value backed up through the tree
```

That is **policy iteration with an approximate evaluation**. The policy
improvement theorem is what licenses it: if `π'` puts more mass on actions with
higher `Q^π`, then `V^{π'} ≥ V^π`. For the greedy policy this is unconditional.
For a stochastic improvement such as a softmax or the visit distribution, the
result holds under the mild assumption that `Q^π(s,a) ≥ V^π(s)` for actions
`π` already plays — i.e. you do not degrade the actions you were already
choosing.

The theorem is per-state and exact; the implementation is neither. Search
estimates `Q` from a finite number of simulations, the network approximates
both `π` and `V`, and the target is fitted with SGD. Every failure mode of
AlphaZero is one of those three approximations breaking (§24).

**Expert iteration** is the same idea when the search is a plain lookahead
instead of MCTS: generate targets with search, fit a policy, repeat. It is
often stronger per unit compute than MCTS self-play on small games, because
search and training are decoupled.

### 19.2 PUCT, derived from what it is imitating

UCB1's bonus `c sqrt(ln N / N_a)` comes from a confidence interval on a mean
estimated from `N_a` samples: the width shrinks like `1/sqrt(N_a)`, and `ln N`
keeps exploration alive as the tree grows. That bound assumes each action's
value is unknown but exchangeable — no reason to try one before another.

PUCT breaks that symmetry with the prior:

```
U(s,a) = c_puct · P(s,a) · sqrt(N(s)) / (1 + N(s,a))
```

* When `N(s,a) = 0`, the bonus is proportional to `P(s,a)` — try what the
  network likes first.
* As `N(s,a)` grows, the bonus decays like `1/N`, independent of the prior, so
  a bad prior cannot permanently suppress an action that keeps looking good.
* `sqrt(N(s))` makes the whole tree keep exploring as it gets bigger.

It is a heuristic, not a theorem: the prior is not a confidence bound. But it
has the right two limits — *prior-dominated at the frontier, evidence-dominated
afterwards* — and that is what makes 100 simulations outperform 10 000 random
playouts on games where a forward model is expensive.

### 19.3 The three things search does, and what each costs

| Search produces | Used as | Cost | Fails when |
|---|---|---|---|
| a better policy (visit counts) | self-play target `π` | `sims × eval` per move | value noise dominates the Q difference |
| a better value (backed-up mean) | bootstrap target (MuZero) or diagnostic | same | tree too shallow to reach a decisive line |
| a decision now | the move actually played | same | you needed strength, not learning |

A useful diagnostic split: evaluate the *same* checkpoint at 0, 32, 100, 400
simulations (§13). If the curve is flat, search cannot extract anything — the
prior or the value is the bottleneck. If it rises and then falls, you have a
search bug or a value that is confidently wrong (§12 step 4). Monotone rise with
diminishing returns is the healthy shape.

### 19.4 Value targets: what the network is actually fitted to

| Target | Formula | Bias | Variance | Note |
|---|---|---|---|---|
| game outcome | `z ∈ {-1,0,1}` | none for the true value | high | what AlphaZero does; slow with long games |
| dense margin | `clip((mine-theirs)/50)` | none (same expectation) | lower | better fit, not automatically better play (§12) |
| n-step | `Σ γ^k r + γ^n V(s_n)` | bootstrap bias | lower | needs a value estimate already |
| TD(λ) / GAE | weighted sum | tunable | tunable | the standard dial in actor-critic |
| search value | backed-up tree mean | tree-depth bias | lowest | MuZero; the tree must be deep enough |
| distributional | histogram over `z` | none | lower | C51/QR-DQN; better gradients, more machinery |

Two practical rules. First, **targets must be for the player to move at the
state** — the sign convention of §2.5, which is the single most common silent
bug in this family. Second, **do not mix target definitions across a run**: a
checkpoint trained on `±1` and resumed on `score_diff` is a new experiment, and
the value head's calibration changes (§19.5).

### 19.5 Calibration, and when to stop adding simulations

A value head can be *unbiased* and still useless: if its RMSE is 20 points and a
decision is worth 10, the search is averaging noise. Measure it directly —
collect `(prediction, true outcome)` pairs on positions from real games, then
report RMSE, mean error (bias) and correlation, plus a reliability curve
(predicted decile vs realised mean). This repository does exactly that in
`docs/results.md` §6b: RMSE ≈ 0.40 (≈ 20 points), correlation 0.60–0.64,
roughly unbiased.

There is a clean stopping rule hiding in those numbers. With `N` evaluations of
a child's value, the standard error is `σ/sqrt(N)`. Search can resolve a
difference `Δ` only when `σ/sqrt(N) < Δ`, i.e.

```
N > (σ / Δ)²        # σ = value RMSE in points, Δ = decision size in points
```

With `σ ≈ 20` and `Δ ≈ 10`, that is `N > 4` evaluations *per child* — which 100
simulations over 5 moves already provides, and which is why the curve bends
there. To go further you must shrink `σ` (better value targets, bigger net,
converged training), not add simulations. That is the general form of "search
cannot fix a bad value function".

### 19.6 Parallel search, and why it is not free

`alphazero/parallel.py` parallelises *games*, not simulations, because that is
trivially correct: independent games, read-only weights. Parallelising one
tree (leaf parallelisation, virtual loss, batched evaluation) gives better GPU
utilisation per game but introduces:

* **virtual loss** — temporarily decrement a leaf's value while it is being
evaluated so other threads do not pile onto the same branch; it makes the
search slightly wrong in exchange for throughput;
* **non-determinism** — results depend on thread interleaving, so a seed no
  longer reproduces a game exactly;
* **batching** — the batch dimension must be padded (action masks differ), and
  padding must not leak into gradients.

Rule of thumb: parallelise games when you are CPU-bound; parallelise the tree
when a single forward pass is expensive (GPU, big net, pixel input).

---

## 20. Exploration

The hardest part of RL in practice, and the part least covered by the
algorithms.

### 20.1 Why it is a data problem, not a noise problem

Policy gradient with a deterministic policy has no gradient for an action it
never takes: `log π(a|s) → -∞` means zero probability, hence zero contribution.
Value-based methods can in principle discover a better action from `max_a Q`,
but only if `Q` generalises to it. So **the policy can only improve on actions
it samples**, and sampling is controlled by the policy plus whatever noise you
add. That is why exploration is really about the distribution of the data you
train on, which is why off-policy replay (retaining old, more random data) is
such a large practical advantage for DQN and SAC.

### 20.2 The menu

| Mechanism | Where | Cost | Good for |
|---|---|---|---|
| ε-greedy | action selection | none | discrete, tabular, DQN |
| Boltzmann / temperature | action selection | none | graded exploration, matches a prior |
| optimistic initialisation | tabular `Q` | none | small MDPs, provably explores |
| UCB / count bonus | action selection | needs counts | bandits, small state spaces |
| entropy bonus | loss | one hyperparameter | policy gradient, prevents collapse |
| root Dirichlet noise | MCTS | none | self-play, breaks prior blind spots |
| action temperature schedule | self-play | none | opening variety |
| intrinsic reward (RND, ICM, curiosity) | reward | a second network | sparse reward, pixel input |
| Go-Explore / archive | outer loop | memory | hard-exploration games |
| population / league | training loop | × opponents | multi-agent, exploits, cyclic strategies |
| domain randomisation | environment | simulator control | sim-to-real robustness |

Two that matter most for games:

**Root Dirichlet noise.** `P ← (1-ε)P + ε Dir(α)` with `α ≈ 0.3`, `ε ≈ 0.25`
for a ~10-action game. `α` small makes the noise sparse (a few moves get most
of the mass), `α` large makes it uniform. It is applied *only at the root and
only in self-play* — evaluation uses the clean prior. Its purpose is precisely
to prevent the search from entrenching the prior's blind spots, which is what
made the cold-start runs shuffle (§12).

**Temperature.** Sampling from `π^{1/τ}` with `τ = 1` early and `τ → 0` later
gives opening diversity and late precision. If you never drop the temperature,
the agent blunders at the end of games; if you drop it immediately, self-play
produces one game repeated.

### 20.3 Exploration and exploitation are the same trade-off as bias and variance

Every exploration knob trades *information now* for *reward now*. There is no
setting that is right for all problems, and the right diagnostic is behavioural:
look at the entropy of the policy, the diversity of openings, the fraction of
states visited more than once, and the score variance across seeds. If entropy
collapses in the first few hundred updates the run is over, whatever the loss
says.

---

## 21. Reward and environment design

This is where most projects are actually won or lost. The algorithm is usually
the easy part.

### 21.1 The reward is the specification

An agent optimises exactly what you wrote, not what you meant. The classic
failures — a boat racer that circles a lagoon harvesting point pickups, a
simulated robot that grows a third leg because the reward counts foot contact —
are not bugs in the agent. They are correct solutions to a mis-specified
problem. Assume this will happen to you, and design against it:

* write the reward in terms of the **outcome you care about**, not a proxy;
* keep proxies additive and small, and test the optimal policy under the
  shaped reward against the true objective;
* watch for degenerate strategies after every reward change — `docs/results.md`
  is full of them (60–100 ply shuffles, 3.6 points captured per move);
* prefer a **terminal** reward plus a strong heuristic teacher over a clever
  dense reward. This project's entire fix was a feature and a teacher, not a
  reward change.

### 21.2 Potential-based shaping, and why it is safe

If you must add a dense term, the only form that provably preserves the optimal
policy is potential-based:

```
F(s, s') = γ Φ(s') - Φ(s)          for any Φ
```

Sketch: over a trajectory the telescopes sum to `γ^T Φ(s_T) - Φ(s_0)`, a
constant independent of the actions taken, so every policy's return shifts by
the same amount and the ranking is unchanged. Setting `Φ(s) = -distance to goal`,
or `Φ = V_heuristic`, gives a dense gradient for free. Any shaping that is not
the difference of a potential can and will change what the optimal policy is —
which is fine if you know it, and fatal if you do not.

### 21.3 Time, termination, and truncation

Three distinct events, often conflated in libraries:

* **Terminal** — the episode ended because of the task. The value of the last
  state is 0, so `V(s_T) = r_T`.
* **Truncated** — you stopped it (time limit, step budget). The state is not
  terminal; bootstrapping must use `V(s_T)`, not zero. Treating truncation as
  termination teaches the agent that running out the clock is worth nothing,
  or worse, that it is worth winning.
* **Draw** — a legitimate outcome here (`no_progress_plies = 60`). It is
  terminal, with reward 0, and it *must* be distinguishable from truncation.

In this repository the draw rule exists because stones circulate forever: an
environment without it does not terminate. That is a general lesson — *any*
environment you build needs a proof (or a guard) that every episode ends.
`MAX_TURN_STEPS = 2000` and the 1 000-ply cap are exactly that guard.

### 21.4 Observation design

Rules that generalise far beyond this game:

1. **Encode consequences, not just state.** The 10 per-action capture features
   (§3) turned an unlearnable function into a learnable one. If computing "what
   would happen if I did X" is cheap and deterministic, hand it to the network.
   This is legal because it is information about the *current* state, not the
   future.
2. **Normalise to O(1) and clip outliers.** Networks trained with Adam are
   robust but not immune; unnormalised inputs are the most common cause of a
   first-night `NaN`.
3. **Choose a consistent frame.** Either absolute plus a side-to-move bit (this
   repo) or canonical (mirror/rotate so the mover is always player 0). Do not
   mix frames; do not flip signs in one place and not another.
4. **Never leak the future.** Features computed after the move, or an opponent's
   hidden information, make offline scores meaningless.
5. **Keep rendering out of the state.** `play.py` and `viz.py` read the state;
   the state knows nothing about them. Otherwise every UI change becomes a
   distribution shift.
6. **Augment symmetries only if they are real.** If the board is symmetric under
   rotation, augment; if not (as here, where the mandarin squares and the
   clockwise order break it), augmenting teaches the agent a false equivalence.

### 21.5 Action space design

* Keep it **player-independent** if you can: index the same way for both sides
  and let features carry whose turn it is. One policy head beats two.
* **Mask illegality at every distribution.** A hard constraint should never be
  a learned preference.
* Prefer a **small, meaningful** factorisation (`(cell, direction)` = 10) over a
  flat one-hot of every sequence. The action should be the decision, not the
  implementation of the decision.
* Watch for **action aliasing**: two indices that always have the same effect
  split the probability mass and waste capacity.

### 21.6 Version the environment

A rule change invalidates every checkpoint and every number in your results
table. Write the ruleset into the config, the checkpoint and the results header
(this repository's `Rules(...)` line in `docs/results.md` does exactly that),
and keep old results with their ruleset, archived rather than deleted.

---

## 22. Engineering an RL project

### 22.1 The simulator is the engine of the project

Throughput is the unit of progress. This repository plays ~640 games/minute on
12 workers with a 5-move branching factor; a 160-game evaluation at 400
simulations takes ~145 s. Every algorithmic choice should be evaluated against "how
many games per minute does this buy or cost?" — which is why the 10 extra
`apply()` calls in the feature encoder looked expensive (≈ 90 µs) and were
still worth it: they changed *learnability*, which no amount of throughput
fixes.

Measure it explicitly: `games/s`, `sims/s`, `positions/s`, `train steps/s`.
If you cannot state those four numbers for your project, you do not know where
the time goes.

### 22.2 Things that save days

* **A working tabular version first.** Same rules, tiny state, Q-table. It
  proves the engine, the reward and the loop, and it is the debugging oracle
  for the neural version.
* **One interface for every agent** (`act(state) -> action`). Evaluation, the
  arena and the UI should not know whether they are talking to a table, a net
  or a human. This repository's `make_agent` in `viz.py`/`play.py` is the whole
  contract.
* **Checkpoints carry their config.** `net.save` stores weights; the run should
  also write `config.json` (features version, ruleset, hyperparameters). A
  weights file without a config is unloadable in three months.
* **Seeds everywhere, from one integer.** Derive worker seeds as
  `(seed, game_index)`; never from global state. On Windows, multiprocessing
  uses `spawn`, so worker code must be importable and guarded by
  `if __name__ == "__main__":` — a missing guard is the classic infinite
  process spawn.
* **Fail loudly on invariants.** `assert sum(cells) + sum(captured) == 50`
  after every move costs nothing and catches every silent rule bug.
* **Time-boxed runs.** `--time-budget` stops training on wall clock rather than
  iterations, so a slow experiment does not eat the evening.

### 22.3 Tests that are specific to RL

Ordinary unit tests do not catch the failures that matter. Add these:

| Test | Catches |
|---|---|
| engine invariants over 10k random games | rule bugs, non-termination |
| hand-traced rule examples | interpretation bugs (`docs/rules.md` has six) |
| search prefers an immediate win with a neutral net | perspective/sign bugs (§2.5) |
| exhaustive minimax agreement on small positions | pruning / backup bugs |
| same seed → identical result | hidden global randomness |
| overfit one batch to ~zero loss | broken backprop, dead gradients, mask bugs |
| finite-difference gradient check | analytic gradient errors |
| "learns a tiny task" (3×3 grid, 200 updates) | loop wiring, target construction |
| policy entropy stays > 0 for N updates | premature collapse |

The two cheapest and highest-value are the invariant fuzz test and the
"prefers an immediate win" test. Both are ten lines.

### 22.4 What to log

Losses are the least informative signals in RL. Log, every iteration:

* **behaviour**: mean episode length, points per move, draw rate, capture rate,
  opening diversity (entropy of first moves);
* **learning**: policy loss, value loss, value calibration (RMSE / bias against
  realised outcomes), gradient norm, entropy;
* **data**: buffer size, unique states seen, fraction of positions where the
  target disagrees with the policy's argmax;
* **evaluation**: win rate on a *fixed* set of seeds against a *fixed* set of
  opponents, every `k` iterations — this is the only number that matters, and
  it must be computed identically every time.

A falling policy loss with rising episode length is the signature of imitating
a search that is itself shuffling (§8, §12). No loss curve would have told you
that.

Everything above is cheap; everything below is about not fooling yourself.

---

## 23. Evaluation, statistics, and self-deception

RL results are unusually easy to fake by accident, because every run is a
random variable and every benchmark is small.

### 23.1 What one number means

For a score `p̂` over `n` games, the standard error is `sqrt(p̂(1-p̂)/n)`.
At `p̂ = 0.5` that is `0.5/sqrt(n)`. The sample sizes follow directly:

| Want to resolve | Games needed (95 %, p ≈ 0.5) |
|---|---|
| ±0.20 | 25 |
| ±0.15 | 44 |
| ±0.10 | 100 |
| ±0.05 | 400 |
| ±0.03 | 1 100 |
| ±0.02 | 2 500 |

This repository's 40-game cells have SE ≈ 0.08, a 95 % interval of ±0.15, and
several of its published differences are exactly that size (§11, §13). Near
0 or 1 the normal approximation breaks; use the **Wilson interval**, or just
report 20 wins of 20 as "≥ 0.83 at 95 %" instead of "1.000".

### 23.2 Cheap variance reduction

* **Paired openings.** Play the same opening twice, once per seat, and score the
  pair. This removes opening luck, typically cutting variance by 2–3× — the
  standard trick in computer-chess testing. This repo's `opening_plies` and
  alternating seats are a weaker version of the same idea.* **Common random numbers.** Give every agent under comparison the same seeds
  and opening set. Comparing on different seeds adds noise for nothing.
* **Deterministic agents, deterministic games.** If a policy is deterministic
  and the game has no randomness, 40 games collapse to 1 distinct game. This is
  why random openings exist at all.
* **Sequential tests.** SPRT (likelihood ratio between "no difference" and
  "this much better") stops after ~30–50 % of the games an equivalent
  fixed-size test needs. It has a false-positive rate you choose, and it is what
  chess testing frameworks use.

### 23.3 The traps

1. **Tuning on the test set.** Every time you adjust a hyperparameter and re-run
   the same 40 games, you overfit them. Hold out seeds: tune on seeds 1–3,
   report on seeds 4–6.
2. **Multiple comparisons.** Ten variants tested at 95 % gives a ≈ 40 % chance
   of at least one false "win". Pre-register the comparison, or replicate the
   winner on fresh seeds before believing it.
3. **Opponent-pool gaming.** "Beats the previous checkpoint 70 %" can mean the
   new policy is a specialist that loses to greedy. Always report against a
   fixed pool that includes a **random** and a **weak heuristic** opponent, not
   just against your own past selves. In adversarial settings, single-agent
   win rates are not a strength measure at all — the honest measure is
   **exploitability** (loss against a best response), or a Nash average over a
   pool.
4. **Train/eval mismatch.** If self-play uses 128 simulations, root noise and
   temperature, but evaluation uses 100 simulations and a clean prior, then
   your training curve and your eval curve are measuring different agents. Make
   the eval loop a fixed, separate function, and never change it mid-project
   without re-measuring everything.
5. **Silent checkpoint drift.** "final.npz" is not "best.npz". Keep both, and
   select on evaluation, not on the last iteration.
6. **Reporting the best seed.** A result that only appears at seed 0 is not a
   result. This repo's own `base_model` swings between 1.000 and 0.238 against
   MCTS 100 across two seeds — a vivid reminder.

### 23.4 A practical evaluation harness

Build this before you build the second algorithm:

```
evaluate.py --ckpt X --sims S --games N --opponents random,greedy,ab4,mcts100
```

with: alternating seats, fixed opening seeds, per-opponent score + Wilson
interval, Elo fitted from the whole matrix, and output written as JSON so runs
can be compared later. Elo is only meaningful *within* one pool and one time
window; `-400 log10(1/score - 1)` is the per-matchup conversion, and it is
unbounded as score → 1, so quote the score too.

Add a **regression suite**: 20–50 hand-picked positions with a known best move
(or a known win), evaluated every time you touch the search or the encoder. A
change that keeps the score but breaks a tactic is a regression you would
otherwise ship.

---

## 24. Debugging playbook

RL bugs are quiet. The loop runs, the losses fall, the games end. Use symptom →
cause, and always start with the cheapest oracle.

### 24.1 The four-question triage

In this order, because each answer invalidates the later ones:

1. **Is the environment right?** Invariants, hand-traced examples, an exhaustive
   minimax on small positions. If the rules are wrong, nothing downstream means
   anything.
2. **Can the network learn at all?** Overfit eight positions to ~zero loss, and
   run a finite-difference gradient check. If not, the bug is in the maths.
3. **Is the target right?** Signs, perspective, truncation vs termination, the
   mask. This is where the expensive bug in this project lived.
4. **Is the evaluation right?** Same seeds, same sims, same openings for every
   arm; enough games to see the effect you claim.

### 24.2 Symptom table

| Symptom | Most likely cause | First check |
|---|---|---|
| Loss falls, strength flat | learning a target that is not better than the policy (bad search, bad value) | is `π` target sharper than the policy? does eval sims = train sims? |
| Games get long, agent shuffles | no capture incentive; terminal reward too sparse; entropy too high | points per move vs random; mean plies |
| Search makes it worse | terminal-perspective sign; value overrides a good prior; too many sims on a noisy value | neutral-net win-preference test; calibration RMSE |
| 0.0 vs one opponent, fine vs others | a specific exploit; opponent pool too narrow | replay losses, look for a repeated pattern |
| Value loss plateaus high | targets inconsistent (mixed definitions), LR too low, capacity too small | target histogram; can it overfit one batch? |
| Policy collapses to one action | entropy coefficient too low, advantage mis-scaled, no exploration | policy entropy over time; action histogram |
| NaN / Inf | unnormalised input, `log(0)`, exploding gradient, learning rate | clip grads, check feature ranges, lower LR |
| Improves then degrades | catastrophic forgetting; replay too small; LR too high for fine-tuning | buffer size; LR schedule; eval every iteration |
| Self-play plateau / cycles | pure self-play mirrors its own weaknesses | add greedy and old checkpoints to the pool (§15 item 4) |
| Results not reproducible | global RNG, thread interleaving, non-deterministic eval | same-seed test; set worker seeds from `(seed, i)` |
| Reads better than it plays | eval noise, cherry-picked seed, train/eval mismatch | Wilson interval; fresh seeds |

### 24.3 Bisect by substituting oracles

When a learned system underperforms, replace one component at a time with a
perfect one. The first oracle that fixes it localises the bug:

* **Uniform policy, zero value** → does search still play legally and prefer
  immediate wins? (Tests the tree, not the net.)
* **True outcome as the value** → does search improve? (Tests the value head.)
* **Greedy as the policy** → does the loop still learn? (Tests the trainer.)
* **Exhaustive minimax as the target** → does the net fit it? (Tests capacity
  and features.)

This is how §12 step 5 was found: a code review, guided by "search hurts" and
a neutral-network experiment, not a hyperparameter sweep.

### 24.4 Beware plausible stories

The step-4 explanation ("PUCT trusts a noisy value over a good prior") was
coherent, matched the symptom, and was wrong. Rules that keep you honest:

* a surprising result that implicates your *method* should trigger a
  correctness check before a tuning sweep;
* if a change makes things dramatically better or dramatically worse (0.10 →
  0.95), suspect a bug in both directions;
* write down the prediction *before* the experiment, then check it. If the
  prediction is vague, the experiment cannot fail, and therefore teaches
  nothing.

---

## 25. The compute ladder

Ordered by information per hour, using this project's measured costs. Climb only
when the rung below is exhausted.

| # | Rung | Cost here | What it bought |
|---|---|---|---|
| 1 | Correct engine + invariants + hand examples | hours | everything else is meaningful |
| 2 | Evaluation harness with intervals | 1 h | the ability to tell signal from noise |
| 3 | Strong classical baselines | 2 h | greedy turned out to be the bar |
| 4 | Tabular / tiny-task sanity run | 1 h | proof the loop learns |
| 5 | Representation (action-consequence features) | 2 h | 0.06 → 0.39 policy-only vs greedy |
| 6 | Imitation warm start | 10 min | prior good enough for search to improve |
| 7 | Search on top (PUCT) | 1 h | 0.39 → 0.94 at 100 sims |
| 8 | Self-play fine-tune at low LR | ~4 min for 8 iters | 0.766 → 0.844 vs MCTS 100 |
| 9 | More simulations | minutes | 0.966 → 0.978 vs greedy |
| 10 | Bigger net / more data / longer training | hours–days | the remaining few points |
| 11 | Architecture, auxiliary losses, MuZero-style model | days | a different project |

Two general observations. First, **rungs 1–7 are almost all engineering, and they
produced nearly all of the strength**; rungs 9–11 buy single points. Second, in
most games performance is roughly linear in `log(compute)` once the
representation is right — which means the interesting decision is not "train
longer" but "is my function representable, and is my target honest?"

---

## 26. Blueprint for the next project

A checklist you can follow cold. Each step has a deliverable and a gate; do not
skip a gate because the step feels obvious.

### 26.1 Step 0 — write the specification

* Rules / dynamics, **exactly**, including the edge cases and the termination
  proof. If an episode might not end, add a guard and a rule.
* Observation, action space, reward, and what "good" means.
* The ruleset string that will appear in every result header.

**Gate:** you can hand the spec to someone else and they implement the same game.

### 26.2 Step 1 — engine and tests

* Immutable state, `apply(state, action) -> (state', info)`, `legal_actions`,
  `is_terminal`, `returns`.
* ASCII render, move parser, terminal UI (`play.py` equivalent) so you can see it.
* Invariant fuzz test (10k random games), hand-traced examples as tests, loop
  guard, and a test that every game ends.

**Gate:** 10k random games complete, invariants hold, you have played it by hand.

### 26.3 Step 2 — baselines and the arena

* random, greedy (or your best heuristic), alpha-beta or plain MCTS.
* `evaluate.py` with alternating seats, random openings, fixed seeds, JSON out.

**Gate:** you know which classical baseline is strongest, and its score against
random. That number is the bar your learner must clear.

### 26.4 Step 3 — a loop that provably learns

* Tabular Q-learning on a tiny version (3×3, or a reduced action set): does it
  beat random? 
* Then a tiny network on the same task: overfit-one-batch, gradient check.

**Gate:** both beat random, and the network can overfit eight samples.

### 26.5 Step 4 — features before architecture

* Write the observation as explicit tensors. Add "consequences of each action"
  if they are cheap and deterministic.
* Ablate: train with and without them. If the ablation changes nothing, they
  are not doing work.

**Gate:** a policy-only network clearly beats random, and is in the same league
as your heuristic (here: 0.39 vs greedy — close, not ahead).

### 26.6 Step 5 — search, then verify it

* Add PUCT (or the appropriate planner) on top.
* **Before measuring strength**, run the neutral-network win-preference test and
  the same-seed determinism test.
* Measure score vs simulations: 0, 32, 100, 400.

**Gate:** search improves monotonically with simulations, and the 0-sim baseline
is worse than 100 sims. If search hurts, stop and debug (§24.2) — do not tune.

### 26.7 Step 6 — learning signal

* Warm start from the heuristic if one exists (cheapest big win).
* Then self-play / policy gradient at a low enough learning rate that the prior
  survives; evaluate every iteration on fixed seeds; keep the best checkpoint.
* Log behaviour, not just losses.

**Gate:** the training curve and the eval curve agree, and eval is computed
identically at every checkpoint.

### 26.8 Step 7 — only now, scale and tune

Order: more simulations → more data → bigger net → architecture → auxiliary
losses. Tune one knob at a time, on held-out seeds, with the interval in hand
(§23). Write every rejected experiment into your results file with the number
that rejected it — `docs/results.md` §6b is that habit, and it is why this
guide can say which ideas are dead.

### 26.9 A skeleton that has worked

```
<game>/
  engine/        rules + state + render        (no RL dependencies)
  features.py    state -> tensor + mask
  net.py         policy/value model            (PyTorch for a real project)
  search.py      PUCT / MCTS / planner
  selfplay.py    generate games -> records
  trainer.py     loss, optimiser, checkpoints
  baselines/     random, heuristic, classical search
  eval/          arena, intervals, Elo
  play.py        human vs agent
  viz.py         recorded game viewer
  tests/         engine, search, net, loop
  docs/rules.md  spec; docs/results.md measurements
```

Rules of thumb for defaults, if you have no better information:

| Setting | Default | Why |
|---|---|---|
| discount `γ` | 1.0 episodic, 0.99 continuing | horizon |
| Adam LR | 1e-3 small net, 3e-4 fine-tune / PPO | stability |
| weight decay | 1e-4 (AdamW) | mild |
| batch | 64–256 (CPU), 1k–4k (PPO on GPU) | gradient noise |
| value loss weight | 0.5–1.0 | balance with policy CE |
| entropy coefficient | 0.01 (A2C/PPO), root noise in self-play | exploration |
| PPO clip `ε` | 0.2, 3–10 epochs, GAE λ 0.95 | standard |
| PUCT `c` | 1.5 (1.0–2.5 all plausible) | tune last, if at all |
| Dirichlet | α 0.3, ε 0.25 | ~10 actions |
| temperature | 1 for the first ~10 % of plies, then 0 | diversity, then precision |
| simulations | 100 to start, 400 to confirm | cost |
| eval | ≥ 100 games, paired openings, fixed seeds | resolution |
| buffer | 10⁴–10⁶ transitions, recent games for self-play | forgetting vs staleness |

### 26.10 A first week, if you are doing this alone

| Session | Work | Done when |
|---|---|---|
| 1 | spec, engine, invariants, ASCII board | you can play it in the terminal |
| 2 | greedy + random + arena + intervals | you have a baseline table |
| 3 | tabular Q-learning, tiny task | beats random |
| 4 | features + net + overfit test | policy-only ≈ heuristic |
| 5 | PUCT + sign tests + sim sweep | search helps monotonically |
| 6 | warm start + self-play + logging | eval improves over the warm start |
| 7 | statistics, write-up, rejected experiments | numbers have intervals, claims are modest |

---

## 27. What transfers to other domains

| Domain | Algorithm | What this project already taught you |
|---|---|---|
| Board / card games | PUCT + self-play, or expert iteration | perspective conventions, visit targets, search budget |
| Continuous control / robotics | SAC, PPO, TD3 | reward shaping, truncation, entropy, sim-to-real gap |
| Recommenders, ads, pricing | contextual bandits, off-policy evaluation (IPS) | exploration, non-stationarity, why win rate ≠ value |
| Scheduling, routing, games with constraints | MCTS + heuristics, model-based | action masking, rollout quality, anytime planning |
| LLM post-training (RLHF/GRPO/DPO) | PPO-style policy gradient with a reward model + KL penalty | advantage estimation, entropy collapse, reward hacking, eval methodology |
| Anything with a simulator | model-based RL, MuZero | the simulator is your throughput ceiling |

The recurring transferable ideas, in order of how often they are the actual
answer:

1. **The representation is part of the algorithm** (§3, §14.2).
2. **A heuristic teacher is the cheapest strong prior** (§10).
3. **A hard constraint is a mask, not a penalty** (§2.2).
4. **Perspective/sign conventions deserve a test each** (§2.5).
5. **Evaluate with intervals and fixed seeds, or you are guessing** (§23).
6. **A correct search over a wrong evaluation gets worse with depth** (§5).
7. **Look at behaviour, not losses** (§22.4).

For LLM post-training specifically, the mapping is close enough to be useful:
the policy is the model, the "search" is sampling several completions (or a
tree), the "value" is the reward model, and the KL penalty is what stops the
policy from wandering into the reward model's blind spots — exactly the
"value noise vs decision size" problem of §19.5, with the reward model playing
the role of the noisy critic.

---

## 28. Reading and practice order

Read one thing at a time, and implement something after each. The order below
assumes you have finished Part I.

**Foundations (2–3 weeks)**

1. Sutton & Barto, *Reinforcement Learning: An Introduction* — ch. 1–6 (MDP,
   dynamic programming, Monte Carlo, TD), then ch. 13 (policy gradient). Ch. 11
   (off-policy approximation) explains why the deadly triad is real.
2. Karpathy's "Pong from Pixels" / any REINFORCE-from-scratch post — for the
   feel of variance reduction.
3. OpenAI Spinning Up, part 1–3 — the cleanest short survey of the algorithm map
   in §18, with code.

**The methods used here (1 week)**

4. Silver et al. 2017, *Mastering the game of Go without human knowledge* —
   PUCT, self-play, visit targets.
5. Schulman et al. 2015, *Trust Region Policy Optimization*, and 2017 *PPO* —
   read PPO for the clip objective, TRPO for why it exists.
6. Schulman et al. 2015, *High-Dimensional Continuous Control Using Generalized
   Advantage Estimation* — GAE, and a clean bias/variance discussion.
7. Ng, Harada & Russell 1999, *Policy Invariance Under Reward Transformations* —
   potential-based shaping.

**Going further (when you need it)**

8. Mnih et al. 2015 (DQN) and Hessel et al. 2017 (Rainbow) — the value-based
   family and its tricks.
9. Haarnoja et al. 2018 (SAC) — max-entropy control.
10. Schrittwieser et al. 2020 (MuZero) — search without a known model.
11. Silver et al. 2018 (AlphaZero) — the cleaner statement of the loop.
12. "Deep Reinforcement Learning that Matters" (Henderson et al. 2018) — the
    statistical hygiene of §23, with evidence.
13. "Implementation Matters in Deep RL" (Engstrom et al. 2020) — why the
    details beat the algorithm name.

**Codebases worth reading end-to-end (pick one, not ten)**

* CleanRL — single-file PPO/DQN, closest in spirit to this repo.
* Stable-Baselines3 — what production-quality RL code looks like.
* OpenSpiel — game environments and search, a good reference for abstractions.
* LightZero — MCTS + self-play done at scale, a natural successor to this code.

**Practice, in this order**

1. Re-run this project's §15 experiment 1 (cold start with the fixed search).
2. Do the five exercises at the end of §16.
3. Re-implement one algorithm from §18 from scratch in NumPy, then again in
   PyTorch, and compare the gradient check.
4. Then start the next project with §26 open beside you.

---

## Appendix A — formula sheet

```
return              G_t = Σ_k γ^k r_{t+k}
value               V^π(s) = E_π[G_t | s]
advantage           A^π(s,a) = Q^π(s,a) - V^π(s)
bellman (policy)    V^π(s) = Σ_a π(a|s)[R + γ Σ_s' P V^π(s')]
bellman (optimal)   V*(s)  = max_a [R + γ Σ_s' P V*(s')]
contraction         ||TV1 - TV2||∞ ≤ γ||V1 - V2||∞
TD error            δ_t = r_t + γV(s_{t+1}) - V(s_t)
GAE                 A_t = Σ_k (γλ)^k δ_{t+k}
policy gradient     ∇J = E[∇log π(a|s) A^π(s,a)]
entropy             H(π) = -Σ_a π_a log π_a
PPO ratio           r_t = π_θ(a|s) / π_old(a|s)
PPO objective       E[min(r_t Â, clip(r_t, 1±ε) Â)]
UCT                 Q_a + c sqrt(ln N / N_a)
PUCT                Q_a + c P(s,a) sqrt(N) / (1 + N_a)
softmax+CE grad     ∂L/∂z_a = (p_a - π_a)/N
value head grad     ∂L/∂pre_v = 2(v - z)(1 - v²)/N
shaping (safe)      F = γΦ(s') - Φ(s)
score SE            sqrt(p(1-p)/n)
Elo diff            -400 log10(1/score - 1)
Elo expected        1 / (1 + 10^(-Δ/400))
sample size         n ≈ 4p(1-p)/Δ²   (95 %)
search stopping     N per child > (value RMSE / decision size)²
```

## Appendix B — glossary additions

Terms from Part II that Part I's glossary does not define:

* **on-policy / off-policy** — whether the data used for an update came from the
  current policy. PPO is on-policy; DQN and SAC are off-policy.
* **bootstrapping** — using the current value estimate as part of the target.
* **deadly triad** — function approximation + bootstrapping + off-policy data;
  the combination that can diverge.
* **replay buffer** — stored transitions, resampled for updates; only valid
  off-policy.
* **target network** — a lagged copy of the value network used to compute
  targets, so the target does not chase the prediction.
* **maximisation bias** — `E[max] ≥ max E`; the systematic overestimate that
  Double DQN corrects.
* **GAE(λ)** — exponentially weighted multi-step advantage; λ trades bias for
  variance.
* **PPO clip** — the ratio cap that keeps an update inside a trust region.
* **expert iteration** — generate targets with search, fit a policy, repeat.
* **truncation** — an artificial episode end; must be bootstrapped, unlike a
  terminal state.
* **potential-based shaping** — reward added as `γΦ(s') - Φ(s)`; provably
  policy-preserving.
* **exploitability** — how much a best-response opponent beats a policy by; the
  honest strength measure in multi-agent settings.
* **SPRT** — sequential probability ratio test; stops a match as soon as the
  evidence is decisive.
* **Wilson interval** — a binomial confidence interval that behaves near 0 and 1.
* **IPS** — inverse propensity scoring; off-policy evaluation for bandits and
  recommenders.
* **KL penalty** — in LLM post-training, a term that keeps the trained policy
  near the reference model; the analogue of trusting your reward model less the
  further you move.
* **temperature** — exponent applied to a policy before normalising; `τ < 1`
  sharpens, `τ > 1` flattens.
* **virtual loss** — a temporary decrement used to spread parallel MCTS workers
  across branches.

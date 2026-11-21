# Learning guide: building an Ô ăn quan RL agent from scratch

This is the long-form companion to the code. `docs/rules.md` says what the game
is, `docs/results.md` says what the agents score, and this file explains *why
each piece is built the way it is*, derives the maths that the NumPy code
implements, and records the failure modes that cost the most time.

Read it in order the first time. Every section ends with the file that
implements it, so you can jump from an idea to the code that runs it.

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

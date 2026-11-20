# Ô Ăn Quan RL Agent (AlphaZero-style Self-Play)

Build the Vietnamese mancala-style game **Ô ăn quan** from scratch and train an agent through self-play with a policy-value network and MCTS.

**Difficulty: 2/3** | **Compute: single consumer GPU or CPU is enough**

---

## 1. Motivation

Chess, Go and Xiangqi are heavily studied, and strong open-source engines exist. Ô ăn quan has very little RL work, a small board, a modest branching factor, and interesting mechanics: chained sowing, captures after empty squares, and a rule for refilling when a player runs out of stones. It is small enough for MCTS to run cheaply, so a full AlphaZero-style loop is feasible solo.

## 2. Research Questions

1. How much does **search** (MCTS) add over a pure policy/value agent with no search on this game?
2. How does strength scale with **MCTS simulations per move** vs. **network size**, under a fixed compute budget?
3. Does self-play rediscover known human heuristics (e.g. protecting mandarin squares, avoiding moves that leave captures open), or find unexpected ones?
4. Does strength depend on which rule variant is used? (Optional.)

## 3. Game Rules (fix one ruleset, document it)

Board: 10 small squares (5 per player) and 2 large "mandarin" squares at the ends, arranged in a loop.

- Each small square starts with 5 citizen stones; each mandarin square starts with 1 mandarin stone (worth 10 citizens, but check your chosen variant).
- On your turn, pick one non-empty small square on your side and a direction (clockwise / counter-clockwise). Pick up all stones and sow them one per square.
- After the last stone: if the next square is non-empty (and not a mandarin square), pick it up and keep sowing. If the next square is empty and the one after is non-empty, capture that one. If two empties follow, or the next square is a mandarin square, the turn ends.
- If you have no stones on your side at your turn, you place 5 captured stones (one in each of your squares) and continue; if you cannot, the game ends.
- The game ends when both mandarin squares are captured. Remaining stones go to the player on whose side they lie. Highest total wins.

> Rules vary by region and source. Choose one, write it as `docs/rules.md` with worked examples, and use those examples as unit tests. This matters more than anything else in the first phase.

## 4. RL Formulation

| Component | Design |
|---|---|
| Agents | Two players, trained by self-play (zero-sum) |
| State | 12 square counts, mandarin-present flags, both players' captured scores, player to move, borrow state |
| Actions | `(square, direction)` -> at most 10 discrete actions, with legal-move masking |
| Reward | +1 / -1 / 0 at game end (win/loss/draw). Optional shaped variant using score difference to test shaping |
| Objective | Maximize expected game outcome against the opponent |

The game is deterministic and fully observable, so exact MCTS applies without belief tracking.

## 5. Approach

1. Rules engine in Python, then vectorize (NumPy / Numba) for speed. Verify with unit tests and random-play invariants (stone conservation).
2. Baselines:
   - Random agent
   - Greedy agent (maximizes immediate capture)
   - Minimax / alpha-beta with a score-difference heuristic (a strong classical baseline for this size)
   - Plain MCTS with random rollouts
3. Learning agents:
   - **A:** PPO or A2C self-play, no search
   - **B:** AlphaZero-style: small MLP/ResNet with policy + value heads, MCTS (PUCT), self-play replay buffer
4. Frameworks: PyTorch, Gymnasium-style interface (or PettingZoo AEC), Weights & Biases / TensorBoard.

## 6. Experiments

| # | Experiment | Compares |
|---|---|---|
| E1 | Engine validation | Rule unit tests, random-play invariants |
| E2 | Classical baselines | Random vs greedy vs alpha-beta (depth 4/6/8) vs plain MCTS |
| E3 | Policy-gradient self-play | Agent A vs baselines |
| E4 | AlphaZero self-play | Agent B vs baselines and vs Agent A |
| E5 | Search budget scaling | 25 / 100 / 400 / 1600 simulations per move |
| E6 | Network size | Small vs medium net at fixed simulations |
| E7 | Value target | Win/loss vs score-difference value |
| E8 | First-move advantage | Win rate as first vs second player |

## 7. Evaluation

- **Win rate** vs. each baseline over 200+ games, alternating who moves first.
- **Elo rating** across training checkpoints (round-robin tournament of checkpoints).
- **Compute-matched comparisons:** report strength vs. wall-clock / simulations.
- **Qualitative:** annotate several self-play games, visualize policy priors on key positions, and test hand-built tactical puzzles (capture chains).
- **Success criteria:** Agent B beats alpha-beta and plain MCTS under equal time limits, with analysis of what search contributes. If it does not, an honest analysis of why (e.g. small game where alpha-beta is already near-optimal) is still a valid result.

## 8. Roadmap

| Phase | Goal |
|---|---|
| 1 | Pin down rules, implement engine, tests, simple CLI/GUI to play |
| 2 | Random, greedy, alpha-beta, plain MCTS baselines; tournament harness |
| 3 | Gym/PettingZoo wrapper; PPO self-play (Agent A) |
| 4 | Policy-value network and PUCT MCTS |
| 5 | AlphaZero self-play loop, first training runs |
| 6 | Scaling experiments (E5-E7), Elo tracking |
| 7 | Extra seeds, qualitative analysis, optional rule-variant study |
| 8 | Report, demo, playable web/CLI interface |

## 9. Risks

- **Rule ambiguity** leads to wrong results. Mitigation: documented ruleset and tests.
- **Self-play instability or cycling.** Mitigation: keep a pool of past checkpoints; evaluate against fixed baselines, not only against itself.
- **MCTS too slow in Python.** Mitigation: Numba/C++ for the engine, batched network evaluation.
- Game may be small enough that alpha-beta is near-perfect. Mitigation: treat this as a finding; also measure at low time budgets where learning helps.

## 10. Repository Layout

```
o-an-quan-rl/
  engine/        # rules, state, move generation
  baselines/     # random, greedy, alphabeta, mcts
  alphazero/     # network, mcts, self_play, trainer
  ppo/           # policy-gradient self-play
  eval/          # tournaments, elo
  docs/rules.md
  README.md
```

## 11. Initial Resources

- Silver et al. (2017), *Mastering Chess and Shogi by Self-Play with a General RL Algorithm* (AlphaZero)
- Silver et al. (2016), *Mastering the game of Go with deep neural networks and tree search*
- Browne et al. (2012), *A Survey of Monte Carlo Tree Search Methods*
- OpenSpiel (DeepMind) for reference implementations of mancala-style games and AlphaZero
- Open-source AlphaZero implementations (e.g. alpha-zero-general) for structure reference
- Vietnamese-language sources and videos on Ô ăn quan rules, to fix your ruleset

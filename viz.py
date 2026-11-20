"""Watch a game in the browser: play, pause and step through it.

    python viz.py --p0 net --p1 greedy
    python viz.py --p0 net --p1 net --sims 100 --out runs/selfplay.html
    python viz.py --p0 alphabeta4 --p1 greedy --out runs/classical.html

The game is played once by the chosen agents, recorded position by position,
and written as one self-contained HTML file: no server, no JavaScript
dependencies, no network access.  Open the file and use the controls (space
plays and pauses, the arrow keys step one move, the slider scrubs).

Squares are drawn as the ring they are: the two mandarin squares at the ends,
player 0's five squares along the top, player 1's along the bottom.  Squares
that received stones in the last move glow; capture targets flash red.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from baselines import NetAgent
from baselines.base import Agent
from engine import DEFAULT_RULES, Rules, action_name, initial_state
from evaluate import build_opponent

QUAN_INDEX = {0: 0, 6: 1}


def make_agent(spec: str, *, checkpoint: str, simulations: int, seed: int) -> Agent:
    """``net`` (with ``--sims``) or any name ``evaluate.py`` understands."""
    if spec == "net":
        from alphazero import PolicyValueNet

        if not Path(checkpoint).exists():
            raise SystemExit(
                f"checkpoint not found: {checkpoint!r} "
                "(train one with: python pretrain.py --run runs/pretrain_greedy)"
            )
        return NetAgent(
            PolicyValueNet.load(checkpoint), simulations=simulations, seed=seed
        )
    return build_opponent(spec, seed)


def _frame(state, rules, *, mover=None, action=None, result=None) -> Dict:
    """One recorded position, plus what the move that produced it did."""
    return {
        "cells": list(state.cells),
        "quan": list(state.quan),
        "quan_captor": list(state.quan_captor),
        "captured": list(state.captured),
        "mandarins_taken": list(state.mandarins_taken),
        "points": list(state.points(rules)),
        "player": state.player,
        "ply": state.ply,
        "done": state.done,
        "end_reason": state.end_reason,
        "mover": mover,
        "action": None if action is None else list(action),
        "action_name": None if action is None else action_name(*action),
        "gained": 0 if result is None else result.points,
        "sown": [] if result is None else list(result.sown),
        "captures": [] if result is None else [cell for cell, _ in result.captures],
        "mandarins": [] if result is None else list(result.mandarins_captured),
    }


def record_game(
    p0: Agent,
    p1: Agent,
    *,
    seed: int = 0,
    opening_plies: int = 0,
    rules: Rules = DEFAULT_RULES,
) -> Dict:
    """Play one game and return it as frames for the viewer."""
    rng = random.Random(seed)
    state = initial_state(rules)
    for _ in range(opening_plies):
        if state.done:
            break
        state, _ = state.apply(rng.choice(state.legal_actions()), rules)

    frames: List[Dict] = [_frame(state, rules)]
    while not state.done:
        mover = state.player
        agent = p0 if mover == 0 else p1
        action = agent.choose(state)
        state, result = state.apply(action, rules)
        frames.append(_frame(state, rules, mover=mover, action=action, result=result))

    final_p0, final_p1 = state.final_points(rules)
    return {
        "p0": getattr(p0, "name", "player 0"),
        "p1": getattr(p1, "name", "player 1"),
        "final": [final_p0, final_p1],
        "end_reason": state.end_reason,
        "rules": {
            "mandarin_value": rules.mandarin_value,
            "mandarin_capture_min": rules.mandarin_capture_min,
            "no_progress_plies": rules.no_progress_plies,
            "stones_per_square": rules.stones_per_square,
        },
        "frames": frames,
    }


_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  * { box-sizing: border-box; }
  body {
    margin: 0; min-height: 100vh; color: #f3ece2;
    font-family: ui-sans-serif, system-ui, "Segoe UI", Roboto, sans-serif;
    background: radial-gradient(1200px 600px at 15% -10%, #2a231c 0%, #15120f 55%, #100e0c 100%);
  }
  .wrap { max-width: 1180px; margin: 0 auto; padding: 26px 22px 44px; }
  h1 { font-size: 18px; font-weight: 600; margin: 0 0 4px; letter-spacing: .01em; }
  .sub { color: #9d9184; font-size: 13px; margin: 0 0 20px; }
  .layout { display: grid; grid-template-columns: minmax(0, 1.6fr) minmax(270px, .8fr); gap: 22px; align-items: start; }
  @media (max-width: 880px) { .layout { grid-template-columns: 1fr; } }

  .board {
    display: grid; gap: 9px;
    grid-template-columns: 76px repeat(5, minmax(0, 1fr)) 76px;
    grid-template-rows: repeat(2, minmax(84px, 1fr));
    grid-template-areas:
      "q0 c1 c2 c3 c4 c5 q6"
      "q0 c11 c10 c9 c8 c7 q6";
    padding: 12px; border-radius: 18px;
    background: linear-gradient(160deg, #201b16, #17130f);
    border: 1px solid #352c24;
    box-shadow: 0 24px 60px rgba(0,0,0,.45), inset 0 1px 0 rgba(255,255,255,.04);
  }
  .cell {
    position: relative; border-radius: 12px; padding: 8px 6px 6px;
    background: #1d1814; border: 1px solid #332a22;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    transition: background .18s ease, border-color .18s ease, box-shadow .18s ease;
    min-height: 84px;
  }
  .cell.p0 { border-color: #274a52; }
  .cell.p1 { border-color: #50332b; }
  .cell.quan {
    background: linear-gradient(150deg, #2b2117, #221a12);
    border-color: #6b5527;
  }
  .cell .num { font-size: 22px; font-weight: 650; line-height: 1; color: #f4ead9; }
  .cell.empty .num { color: #6f6357; }
  .cell .dots { display: flex; flex-wrap: wrap; gap: 3px; justify-content: center; max-width: 62px; margin: 5px 0 2px; min-height: 8px; }
  .cell .dots i { width: 6px; height: 6px; border-radius: 50%; background: #cbb99b; opacity: .75; display: block; }
  .cell.quan .dots i { background: #e2b85f; }
  .cell .tag { font-size: 10px; letter-spacing: .09em; text-transform: uppercase; color: #8b7d6d; }
  .cell.quan .tag { color: #b99a52; }
  .cell.p0 .tag { color: #5d99a5; }
  .cell.p1 .tag { color: #b07463; }
  .cell.origin { box-shadow: 0 0 0 2px rgba(255,255,255,.55) inset; }
  .cell.sown { background: #2a2318; box-shadow: 0 0 0 2px rgba(232,196,106,.75) inset; }
  .cell.taken { background: #3a1d17; box-shadow: 0 0 0 2px rgba(226,131,107,.85) inset; }
  .cell.gone { opacity: .55; }

  .scorebar { display: flex; gap: 10px; align-items: center; margin: 16px 0 12px; flex-wrap: wrap; }
  .chip {
    display: flex; align-items: baseline; gap: 8px; padding: 8px 13px; border-radius: 11px;
    background: #1c1713; border: 1px solid #322a22; font-size: 13px; color: #b8a893;
  }
  .chip b { font-size: 20px; color: #f5ecdf; font-variant-numeric: tabular-nums; }
  .chip.p0 { border-color: #2b5560; }
  .chip.p1 { border-color: #5a382e; }
  .chip.p0 b { color: #7fc7d6; }
  .chip.p1 b { color: #e59b82; }
  .status { margin-left: auto; font-size: 13px; color: #cbbdab; }
  .status.win { color: #f0d08a; font-weight: 600; }

  .controls { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; padding: 10px; border-radius: 13px; background: #191410; border: 1px solid #2d2620; }
  button {
    font: inherit; font-size: 13px; color: #ecdfcd; background: #241d17;
    border: 1px solid #3a3129; border-radius: 9px; padding: 7px 11px; cursor: pointer;
    transition: background .15s ease, border-color .15s ease;
  }
  button:hover { background: #2e251d; border-color: #4d4136; }
  button.primary { background: #3b2f1c; border-color: #6d5527; color: #f6e3b6; min-width: 92px; }
  button.primary:hover { background: #4a3a22; }
  input[type=range] { flex: 1 1 140px; accent-color: #d8ac56; }
  select { font: inherit; font-size: 13px; color: #ecdfcd; background: #241d17; border: 1px solid #3a3129; border-radius: 9px; padding: 6px 8px; }
  .mono { font-variant-numeric: tabular-nums; color: #9d9184; font-size: 12px; min-width: 52px; text-align: right; }

  aside { display: flex; flex-direction: column; gap: 14px; }
  .panel { background: #191410; border: 1px solid #2d2620; border-radius: 14px; padding: 13px 14px; }
  .panel h2 { font-size: 11px; letter-spacing: .12em; text-transform: uppercase; color: #8d8072; margin: 0 0 9px; font-weight: 600; }
  ol#log { list-style: none; margin: 0; padding: 0; max-height: 300px; overflow-y: auto; font-size: 12.5px; }
  ol#log li { display: flex; gap: 8px; align-items: baseline; padding: 4px 6px; border-radius: 7px; cursor: pointer; color: #bfb0a0; }
  ol#log li:hover { background: #231c16; }
  ol#log li.active { background: #2c2419; color: #f4ead9; }
  ol#log li .n { color: #7c7064; min-width: 22px; text-align: right; font-variant-numeric: tabular-nums; }
  ol#log li b { color: #eec276; font-weight: 600; }
  .who { font-size: 11px; letter-spacing: .04em; }
  .who.p0 { color: #7fc7d6; }
  .who.p1 { color: #e59b82; }
  .legend { font-size: 12.5px; color: #a69786; line-height: 1.55; }
  .legend b { color: #dccbb4; font-weight: 600; }
  .keys { font-size: 11.5px; color: #7f7265; margin-top: 8px; }
  kbd { background: #241d17; border: 1px solid #3a3129; border-radius: 5px; padding: 1px 5px; font-size: 11px; color: #c9b9a4; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Ô ăn quan — __MATCHUP__</h1>
  <p class="sub">One recorded game. Play it, pause it, step through it move by move.</p>

  <div class="layout">
    <main>
      <div class="board" id="board"></div>

      <div class="scorebar">
        <div class="chip p0"><span id="name0">P0</span><b id="score0">0</b></div>
        <div class="chip p1"><span id="name1">P1</span><b id="score1">0</b></div>
        <div class="status" id="status">ready</div>
      </div>

      <div class="controls">
        <button id="btn-first" title="Back to the start (Home)">⏮</button>
        <button id="btn-back" title="Step back (left arrow)">◀</button>
        <button id="btn-play" class="primary" title="Play / pause (space)">▶ Play</button>
        <button id="btn-next" title="Step forward (right arrow)">▶|</button>
        <input id="scrub" type="range" min="0" max="0" value="0" step="1" title="Scrub through the game">
        <span class="mono" id="ply">0 / 0</span>
        <select id="speed" title="Playback speed">
          <option value="1800">0.4×</option>
          <option value="1100" selected>1×</option>
          <option value="650">1.7×</option>
          <option value="350">3×</option>
        </select>
      </div>
    </main>

    <aside>
      <div class="panel">
        <h2>Moves</h2>
        <ol id="log"></ol>
      </div>
      <div class="panel">
        <h2>Reading the board</h2>
        <div class="legend">
          <b>quan</b> squares (both ends) hold the mandarins: 10 points each, and
          only capturable once they hold at least five citizen stones.<br>
          <b>P0</b> squares are along the top, <b>P1</b> along the bottom. Numbers
          are citizen stones; the dots are the same count drawn out.<br>
          A move lifts every stone from one square and sows them around the ring,
          continuing while the next square holds stones. A capture chains through
          alternating empty and non-empty squares.<br>
          <span style="color:#e8c46a">Gold outline</span> = squares that received a
          stone in this move. <span style="color:#e2836b">Red</span> = captured.
        </div>
        <div class="keys"><kbd>space</kbd> play/pause · <kbd>←</kbd> <kbd>→</kbd> step · <kbd>home</kbd> restart</div>
      </div>
    </aside>
  </div>
</div>

<script>
const GAME = __GAME_JSON__;
const frames = GAME.frames;
const OWNER = {1:0, 2:0, 3:0, 4:0, 5:0, 7:1, 8:1, 9:1, 10:1, 11:1};
const QUAN = {0:0, 6:1};
const NAMES = [GAME.p0, GAME.p1];

const board = document.getElementById('board');
const cellEls = [];
for (let i = 0; i < 12; i++) {
  const el = document.createElement('div');
  el.className = 'cell';
  if (QUAN[i] !== undefined) {
    el.classList.add('quan');
    el.style.gridArea = i === 0 ? 'q0' : 'q6';
  } else {
    el.classList.add(OWNER[i] === 0 ? 'p0' : 'p1');
    el.style.gridArea = 'c' + i;
  }
  el.innerHTML = '<div class="num">0</div><div class="dots"></div><div class="tag"></div>';
  board.appendChild(el);
  cellEls.push(el);
}

document.getElementById('name0').textContent = 'P0 · ' + NAMES[0];
document.getElementById('name1').textContent = 'P1 · ' + NAMES[1];

const logEl = document.getElementById('log');
const entries = [];
for (let k = 0; k < frames.length; k++) {
  const f = frames[k];
  if (!f.action) continue;
  const li = document.createElement('li');
  li.innerHTML = '<span class="n">' + f.ply + '</span>' +
    '<span class="who p' + f.mover + '">P' + f.mover + '</span>' +
    '<span>' + f.action_name + (f.gained ? ' <b>+' + f.gained + '</b>' : '') + '</span>';
  li.addEventListener('click', function () { pause(); setFrame(k); });
  logEl.appendChild(li);
  entries.push({el: li, frame: k});
}

const scrub = document.getElementById('scrub');
scrub.max = String(frames.length - 1);
const playBtn = document.getElementById('btn-play');
const statusEl = document.getElementById('status');
const plyEl = document.getElementById('ply');
const speedEl = document.getElementById('speed');
let index = 0;
let timer = null;

function render() {
  const f = frames[index];
  for (let i = 0; i < 12; i++) {
    const el = cellEls[i];
    const isQuan = QUAN[i] !== undefined;
    const alive = isQuan ? !!f.quan[QUAN[i]] : false;
    el.classList.toggle('empty', f.cells[i] === 0 && !alive);
    el.classList.toggle('gone', isQuan && !alive);
    el.classList.toggle('sown', f.sown.indexOf(i) >= 0);
    el.classList.toggle('taken', f.captures.indexOf(i) >= 0);
    el.classList.toggle('origin', !!f.action && f.action[0] === i);

    el.querySelector('.num').textContent = f.cells[i];
    const dots = el.querySelector('.dots');
    let html = '';
    const shown = Math.min(f.cells[i], 12);
    for (let d = 0; d < shown; d++) html += '<i></i>';
    dots.innerHTML = html;

    let tag = '';
    if (isQuan) {
      if (alive) tag = 'quan';
      else if (f.quan_captor[QUAN[i]] >= 0) tag = 'to P' + f.quan_captor[QUAN[i]];
      else tag = 'empty';
    } else {
      tag = 'P' + OWNER[i];
    }
    el.querySelector('.tag').textContent = tag;
  }

  document.getElementById('score0').textContent = f.points[0];
  document.getElementById('score1').textContent = f.points[1];

  if (f.done) {
    const a = GAME.final[0], b = GAME.final[1];
    const verdict = a === b ? 'draw ' + a + '–' + b
                            : 'P' + (a > b ? 0 : 1) + ' wins ' + Math.max(a, b) + '–' + Math.min(a, b);
    statusEl.textContent = verdict + ' (' + GAME.end_reason + ')';
    statusEl.className = 'status win';
  } else {
    statusEl.textContent = 'P' + f.player + ' (' + NAMES[f.player] + ') to move';
    statusEl.className = 'status';
  }

  plyEl.textContent = index + ' / ' + (frames.length - 1);
  scrub.value = String(index);
  for (const entry of entries) {
    entry.el.classList.toggle('active', entry.frame === index);
  }
  const active = entries.filter(function (e) { return e.frame === index; })[0];
  if (active && active.el.scrollIntoView) {
    active.el.scrollIntoView({block: 'nearest'});
  }
}

function setFrame(k) {
  index = Math.max(0, Math.min(frames.length - 1, k));
  render();
}

function play() {
  if (timer !== null) return;
  if (index >= frames.length - 1) index = 0;
  playBtn.textContent = '⏸ Pause';
  timer = window.setInterval(function () {
    if (index >= frames.length - 1) { pause(); return; }
    setFrame(index + 1);
  }, Number(speedEl.value));
}

function pause() {
  if (timer !== null) window.clearInterval(timer);
  timer = null;
  playBtn.textContent = '▶ Play';
}

playBtn.addEventListener('click', function () { timer === null ? play() : pause(); });
document.getElementById('btn-next').addEventListener('click', function () { pause(); setFrame(index + 1); });
document.getElementById('btn-back').addEventListener('click', function () { pause(); setFrame(index - 1); });
document.getElementById('btn-first').addEventListener('click', function () { pause(); setFrame(0); });
scrub.addEventListener('input', function () { pause(); setFrame(Number(scrub.value)); });
speedEl.addEventListener('change', function () { if (timer !== null) { pause(); play(); } });

document.addEventListener('keydown', function (event) {
  if (event.key === ' ' || event.code === 'Space') { event.preventDefault(); timer === null ? play() : pause(); }
  else if (event.key === 'ArrowRight') { event.preventDefault(); pause(); setFrame(index + 1); }
  else if (event.key === 'ArrowLeft') { event.preventDefault(); pause(); setFrame(index - 1); }
  else if (event.key === 'Home') { event.preventDefault(); pause(); setFrame(0); }
});

render();
</script>
</body>
</html>
"""


def build_html(game: Dict) -> str:
    """Render one recorded game as a self-contained page."""
    payload = json.dumps(game, separators=(",", ":")).replace("</", "<\\/")
    matchup = f"{game['p0']} (P0) vs {game['p1']} (P1)"
    title = f"O an quan - {game['p0']} vs {game['p1']}"
    return (
        _HTML.replace("__GAME_JSON__", payload)
        .replace("__MATCHUP__", matchup)
        .replace("__TITLE__", title)
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Record a game and open a viewer for it.")
    parser.add_argument("--p0", default="net", help="player 0: net, random, greedy, alphabetaN, mctsN, net:PATH")
    parser.add_argument("--p1", default="greedy", help="player 1 (same names)")
    parser.add_argument("--ckpt", default="runs/base_model.npz", help="checkpoint for --p0/--p1 net")
    parser.add_argument("--sims", type=int, default=0, help="MCTS simulations for net players")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opening-plies", type=int, default=0, help="random plies played before the agents")
    parser.add_argument("--out", default="runs/game.html", help="where to write the page")
    parser.add_argument("--no-open", action="store_true", help="do not try to open the page")
    args = parser.parse_args(argv)

    p0 = make_agent(args.p0, checkpoint=args.ckpt, simulations=args.sims, seed=args.seed)
    p1 = make_agent(args.p1, checkpoint=args.ckpt, simulations=args.sims, seed=args.seed + 1)
    game = record_game(p0, p1, seed=args.seed, opening_plies=args.opening_plies)

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_html(game), encoding="utf-8")

    final = game["final"]
    print(f"{game['p0']} (P0) vs {game['p1']} (P1): {len(game['frames']) - 1} plies, "
          f"final {final[0]}-{final[1]} ({game['end_reason']})")
    print(f"wrote {output}")

    if not args.no_open:
        import webbrowser
        webbrowser.open(output.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

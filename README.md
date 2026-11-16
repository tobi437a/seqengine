# seqengine

A C++ engine for the Sequence card game (1v1 variant), with a pybind11
binding and a self-contained Python harness.

## Quick start

```bash
make python      # build the C++ engine + pybind11 module
cd python
python3 play.py  # GUI game against the engine; every game is logged to games/
```

Needs a C++20 compiler, GNU Make, and `pip install pybind11`. On Windows
use `g++` from MinGW-w64 and GNU Make (MSYS2 or chocolatey); if `python`
isn't on PATH but `py` is, run `make PY=py python`.

The build targets the CPU it runs on (`-march=native`) — the engine is
popcount-heavy and much slower without a hardware `POPCNT`. For binaries
you can copy to another machine, use `make ARCH=-march=x86-64-v2`.

## What's here

- **Engine** (`src/state.*`, `src/eval.hpp`): bitboard state, move
  generation, sequence detection, dead cards, draw / reshuffle. The
  evaluation (window-based Φ and shaping score) is bit-identical to the
  Python reference in `python/game_engine.py` (`make validate`).
- **Search** (`src/search.*`): Information-Set MCTS with PUCT, one
  lock-free tree shared by all threads, tree reuse across moves,
  truncated rollouts, and move-score / value tables fitted to the
  engine's own searches. Wins 100 % against a random opponent and ~80 %
  against the one-ply heuristic at 2000 iterations.
- **Python** (`python/`): `mcts.py` wraps the binding as `MCTSOpponent`;
  `play.py` is a GUI game that records every move; `play_irl.py` runs
  the engine for a game on a physical board.
- **Tools** (`tools/`): benchmark, profiler, match and SPRT harnesses,
  and the self-play → table-fitting pipeline.

## Build targets

```
make python      # pybind11 module into python/
make             # all C++ tools
make test        # unit tests, incl. a self-check of the search internals
make validate    # cross-check src/eval.hpp against game_engine.py
make bench       # throughput of eval, rollouts and the search
make compete     # MCTS vs random and vs the one-ply heuristic
make sprt        # SPRT A/B between two configs (make sprt SPRT_ARGS="--tc ltc")
make ablation    # N threads vs 1 thread at a fixed budget
make profile     # where the time goes (make profile PROFILE_ARGS="10 2000")
make data        # self-play training data into build/data
make fit         # fit src/learned_tables.hpp to it (numpy + torch), then rebuild
make codegen     # regenerate src/board_data.* from python/board_layout.py
```

## Performance

`make bench` on an 8-core / 16-thread Zen 4:

| Workload                      | Throughput        |
|-------------------------------|-------------------|
| `total_phi` (192-window eval) | ~56M calls/s      |
| Random rollouts (full games)  | ~1.3M games/s     |
| Heuristic decision            | ~17M decisions/s  |
| MCTS iterations, 1 thread     | ~160k iters/s     |
| MCTS iterations, 16 threads   | ~1.2M iters/s     |

With the learned tables off the search runs ~300k iters/s on one thread
and ~2.0M on 16 (see *Learned tables* below). Generating and scoring
moves inside rollouts is about two thirds of the time.

## Using the engine from Python

```python
from game_engine import SequenceGame
from seq_actions import get_legal_action_mask, int_to_action
from mcts import MCTSOpponent

mcts = MCTSOpponent(iterations=20000)   # uses every hardware thread

game = SequenceGame()
game.reset()
while not game.done:
    player = game.current_player
    action = int_to_action(mcts(game, player, get_legal_action_mask(game, player)))
    card = game.hands[player][action[0]]
    game.step(action)
    mcts.advance(action, card)          # optional: lets the next search reuse the tree
```

`MCTSOpponent` has the same `(game, player, legal_mask) → action_int`
signature as `heuristic_action` and `random_action`. Keyword arguments
set `MCTSConfig` fields; `mcts.last_stats` reports iterations, depth,
time and the chosen move's win estimate.

## How the search works

Single-Observer Information-Set MCTS (Cowling, Powley & Whitehouse
2012). `src/search.hpp` documents every config field.

- **Determinization.** Each iteration deals the opponent a fresh random
  hand from the unseen cards and walks the one shared tree using only
  the moves legal in that deal. Cards drawn during an iteration are
  random unseen cards — the deck is never shuffled.
- **Moves are cards, not hand slots.** A tree edge is "two-eyed jack on
  cell 37", not "slot 3", so an opponent move isn't split across seven
  slot-children and two copies of a card are one move.
- **Selection.** PUCT over all legal moves with a softmax prior on the
  move scores; unvisited moves get the parent's value minus
  `fpu_reduction`, and a node is created only when selected. The final
  move is the most-visited root child.
- **Rollouts.** ε-greedy on the move score (ε = 0.30) for
  `rollout_cutoff` = 12 plies, then a static evaluation. Random plies
  choose among non-jack moves — otherwise a two-eyed jack, ~90 of ~105
  legal moves, gets thrown away almost every time (+33 Elo).
- **Scoring.** Each of the 192 five-windows has one byte of state
  (joker?, my chips, their chips) kept up to date as moves are made. A
  move's score sums one table entry per window through its cell; a
  position's value sums one per window on the board. Two departures
  from `eval.hpp`, +26 Elo together: a window with two or more locked
  chips can never complete and scores zero (`dead_windows`), and
  completing a five gets a bonus (`shaping_seq_bonus`).
- **Learned tables** (`learned_policy`, `learned_value`, default on).
  The table entries come from `src/learned_tables.hpp` instead of
  `eval.hpp`'s weights. Move scores are fitted so their softmax matches
  the search's own visit distribution, and know whether the mover holds
  cards for the rest of a window; the value is a logistic regression on
  who won. +23 Elo at 3000 iterations, +9 at 20000, for ~1.8× the time
  per iteration. Regenerate with `make data && make fit`.
- **Threads.** `n_threads` workers share the tree lock-free, with
  virtual loss. 16 threads against 1 at equal iterations shows no
  measurable loss at 8.4× the throughput. C++ defaults to 1 thread
  (deterministic given a seed); the Python wrapper to all of them.
- **Tree reuse.** Report played moves with `advance()` and the next
  search starts from the matching subtree.

### What the measurements say

Paired self-play (same deal, seats swapped), 3000 iterations:

- **Luck dominates.** About three quarters of game pairs split 1–1, and
  an engine allowed to see the opponent's hand gains only +75 Elo.
  Real improvements look small.
- **More search helps, with diminishing returns:** 500 → 2000
  iterations +27 Elo, 2000 → 8000 +23, 8000 → 32000 +9.
- **The tree below the first move barely matters.** Replacing the
  opponent's in-tree replies with samples from the prior changes
  nothing measurable. Strength comes from what the rollouts do.

Tried and rejected, all within noise or worse: a bonus for blocking an
open four, a larger completion bonus, a sharper prior, rollout cutoffs
of 6 and 20, ε = 0.15, c = 1.2 and 3.0, sampling rollout moves from the
softmax, root-parallel trees, discounted step rewards as the rollout
value, and forcing sequence-completing moves in rollouts.

A single 800-pair result is ±12 Elo; re-run with another seed before
believing anything under +20.

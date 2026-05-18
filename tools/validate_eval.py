"""
Cross-validate src/eval.hpp against game_engine.py.

Runs build/eval_dump, then for each position emitted:
  - reconstructs board_chips in a Python SequenceGame
  - recomputes total Φ for both players by summing _window_weight over
    the 192 windows (matching the C++ enumeration order)
  - recomputes the shaping score for every M-line move
  - asserts bit-equality with the C++ values to within 1e-8

Exits 0 if everything matches, 1 otherwise.

The point is not to maintain two implementations; it's to catch any
divergence right now, while the Python reference is still the spec.
"""
import os
import sys
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
# board_layout.py and game_engine.py live in ../python/ (shared with end-user code).
sys.path.insert(0, os.path.normpath(os.path.join(HERE, '..', 'python')))

import game_engine as ge
import board_layout as bl

ROOT = os.path.normpath(os.path.join(HERE, '..'))
BINARY = os.path.join(ROOT, 'build', 'eval_dump')

TOL = 1e-8

# Card type indexing must match tools/gen_board_data.py:
#   suit_idx (s=0,h=1,d=2,c=3) * 12 + rank_idx (2..10,Q,K,A)
#   48 = ONE_EYED_JACK, 49 = TWO_EYED_JACK
SUITS = ['spades', 'hearts', 'diamonds', 'clubs']
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'Q', 'K', 'A']
ONE_EYED_JACK_IDX = 48
TWO_EYED_JACK_IDX = 49


# Enumerate windows in the same order as tools/gen_board_data.py so that
# any indexing-order dependency in the C++ code shows up as a mismatch.
DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]
WINDOWS = []
for dr, dc in DIRS:
    for r in range(10):
        for c in range(10):
            er, ec = r + 4 * dr, c + 4 * dc
            if 0 <= er < 10 and 0 <= ec < 10:
                WINDOWS.append((r, c, dr, dc))
assert len(WINDOWS) == 192


def py_total_phi(game, player):
    total = 0.0
    for r, c, dr, dc in WINDOWS:
        total += game._window_weight(r, c, dr, dc, player)
    return total


def py_shaping_score(game, player, card_type, cell):
    """Mirror of evaluate.py:_shaping_score, but parameterized by
    card_type (an int from the C++ encoding) rather than a hand slot."""
    if cell < 0:
        return 0.0
    GAMMA = 0.97
    row, col = divmod(cell, 10)
    my_chip = ge.PLAYER0 if player == 0 else ge.PLAYER1

    phi_me_before  = game._affected_windows_value(row, col, player)
    phi_opp_before = game._affected_windows_value(row, col, 1 - player)

    saved = game.board_chips[row][col]
    try:
        if card_type == ONE_EYED_JACK_IDX:
            game.board_chips[row][col] = ge.EMPTY
        else:
            game.board_chips[row][col] = my_chip
        phi_me_after  = game._affected_windows_value(row, col, player)
        phi_opp_after = game._affected_windows_value(row, col, 1 - player)
    finally:
        game.board_chips[row][col] = saved

    return ((GAMMA * phi_me_after - phi_me_before)
          + (phi_opp_before - GAMMA * phi_opp_after))


def reconstruct_game(chips_flat):
    g = ge.SequenceGame()
    for r in range(10):
        for c in range(10):
            g.board_chips[r][c] = chips_flat[r * 10 + c]
    return g


def main():
    if not os.path.exists(BINARY):
        print(f"missing {BINARY} — run `make eval_dump` first")
        return 1

    result = subprocess.run([BINARY, '300'], capture_output=True, text=True, cwd=ROOT)
    if result.returncode != 0:
        print(f"eval_dump failed: {result.stderr}")
        return 1

    n_pos = n_pos_ok = 0
    n_mov = n_mov_ok = 0
    worst_phi = 0.0
    worst_score = 0.0
    failures = []  # collect a handful for reporting

    current_game = None  # the most recent P line, for the following M lines

    for line in result.stdout.splitlines():
        parts = line.split()
        if not parts:
            continue

        if parts[0] == 'P':
            # P <100 chip ints> <phi_p0> <phi_p1>
            assert len(parts) == 1 + 100 + 2, f"bad P line: {line}"
            chips = [int(x) for x in parts[1:101]]
            cpp_p0 = float(parts[101])
            cpp_p1 = float(parts[102])

            game = reconstruct_game(chips)
            py_p0 = py_total_phi(game, 0)
            py_p1 = py_total_phi(game, 1)

            e0 = abs(cpp_p0 - py_p0)
            e1 = abs(cpp_p1 - py_p1)
            worst_phi = max(worst_phi, e0, e1)

            n_pos += 1
            if e0 < TOL and e1 < TOL:
                n_pos_ok += 1
                current_game = game
            else:
                current_game = None  # don't validate M lines off a bad P
                if len(failures) < 3:
                    failures.append(
                        f'P #{n_pos}: cpp=({cpp_p0:.10f},{cpp_p1:.10f}) '
                        f'py=({py_p0:.10f},{py_p1:.10f}) err=({e0:.2e},{e1:.2e})'
                    )

        elif parts[0] == 'M':
            # M <player> <card_type> <cell> <shaping_score>
            assert len(parts) == 5, f"bad M line: {line}"
            player    = int(parts[1])
            card_type = int(parts[2])
            cell      = int(parts[3])
            cpp_score = float(parts[4])

            n_mov += 1
            if current_game is None:
                continue
            py_score = py_shaping_score(current_game, player, card_type, cell)
            err = abs(cpp_score - py_score)
            worst_score = max(worst_score, err)
            if err < TOL:
                n_mov_ok += 1
            elif len(failures) < 6:
                failures.append(
                    f'M (P#{n_pos}, p={player} card={card_type} cell={cell}): '
                    f'cpp={cpp_score:.10f} py={py_score:.10f} err={err:.2e}'
                )

    print(f"positions:       {n_pos_ok}/{n_pos} match (max |Δ| = {worst_phi:.2e})")
    print(f"shaping scores:  {n_mov_ok}/{n_mov} match (max |Δ| = {worst_score:.2e})")
    if failures:
        print("first failures:")
        for f in failures:
            print(f"  {f}")
    ok = (n_pos_ok == n_pos) and (n_mov_ok == n_mov)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

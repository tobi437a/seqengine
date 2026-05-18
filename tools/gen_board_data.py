"""
Generate src/board_data.{hpp,cpp} from board_layout.py.

Outputs:
  - CELL_TO_CARD[100]              : cell idx -> card idx (or -1 for joker)
  - CARD_TO_CELLS[48][2]           : regular card idx -> (cell_a, cell_b)
  - JOKER_MASK                     : BB100 of the 4 corner cells
  - N_WINDOWS, WINDOW_MASKS[]      : 5-in-a-row bitmasks
  - WINDOW_START_CELL[], WINDOW_DIR[]
  - CELL_WINDOWS[100][...]         : per-cell list of windows passing through it

Card encoding (50 types):
  suit_idx (0..3) * 12 + rank_idx (0..11)         -> 0..47    regular cards
  48 = ONE_EYED_JACK                              (J of spades, J of hearts)
  49 = TWO_EYED_JACK                              (J of diamonds, J of clubs)
Suit order: spades=0, hearts=1, diamonds=2, clubs=3
Rank order: 2,3,4,5,6,7,8,9,10,Q,K,A    (no J on board)
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
# board_layout.py lives in ../python/ — the dev tools (this codegen and
# validate_eval.py) and end-user code share one copy.
sys.path.insert(0, os.path.normpath(os.path.join(HERE, '..', 'python')))

import board_layout as bl

SUITS = ['spades', 'hearts', 'diamonds', 'clubs']
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'Q', 'K', 'A']
SUIT_IDX = {s: i for i, s in enumerate(SUITS)}
RANK_IDX = {r: i for i, r in enumerate(RANKS)}

N_REGULAR = 48
ONE_EYED_JACK_IDX = 48
TWO_EYED_JACK_IDX = 49
N_CARD_TYPES = 50

def card_idx(card):
    if card == bl.ONE_EYED_JACK:
        return ONE_EYED_JACK_IDX
    if card == bl.TWO_EYED_JACK:
        return TWO_EYED_JACK_IDX
    suit, rank = card
    return SUIT_IDX[suit] * 12 + RANK_IDX[rank]

def card_name(idx):
    if idx == ONE_EYED_JACK_IDX: return 'ONE_EYED_JACK'
    if idx == TWO_EYED_JACK_IDX: return 'TWO_EYED_JACK'
    suit = SUITS[idx // 12]
    rank = RANKS[idx % 12]
    return f'{rank}-{suit[0].upper()}'

# --- cell -> card / joker -------------------------------------------------

cell_to_card = [-1] * 100   # -1 means joker corner
joker_mask_lo = 0
joker_mask_hi = 0
card_to_cells = {i: [] for i in range(N_REGULAR)}

for r in range(10):
    for c in range(10):
        cell = r * 10 + c
        item = bl.BOARD[r][c]
        if item == bl.JOKER:
            if cell < 64:
                joker_mask_lo |= (1 << cell)
            else:
                joker_mask_hi |= (1 << (cell - 64))
        else:
            cidx = card_idx(item)
            cell_to_card[cell] = cidx
            card_to_cells[cidx].append(cell)

# Sanity: each regular card should appear at exactly 2 cells. Surface
# anything weird in board_layout.py rather than silently shipping it.
problems = []
for cidx in range(N_REGULAR):
    n = len(card_to_cells[cidx])
    if n != 2:
        problems.append((card_name(cidx), n, card_to_cells[cidx]))
if problems:
    print("WARNING: cards not appearing exactly twice on the board:", file=sys.stderr)
    for name, n, cells in problems:
        print(f"  {name}: {n} cells -> {cells}", file=sys.stderr)
    # Pad/truncate to exactly 2 entries (-1 sentinel for missing). The C++
    # side filters -1, so a card with only 1 board cell still works.
    for cidx in range(N_REGULAR):
        while len(card_to_cells[cidx]) < 2:
            card_to_cells[cidx].append(-1)
        card_to_cells[cidx] = card_to_cells[cidx][:2]

# --- windows --------------------------------------------------------------

DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]   # right, down, down-right, down-left
windows = []   # list of (mask_lo, mask_hi, start_cell, dir_idx)

for d_idx, (dr, dc) in enumerate(DIRS):
    for r in range(10):
        for c in range(10):
            end_r = r + 4 * dr
            end_c = c + 4 * dc
            if not (0 <= end_r < 10 and 0 <= end_c < 10):
                continue
            mlo = mhi = 0
            for i in range(5):
                nr = r + dr * i
                nc = c + dc * i
                cell = nr * 10 + nc
                if cell < 64:
                    mlo |= (1 << cell)
                else:
                    mhi |= (1 << (cell - 64))
            windows.append((mlo, mhi, r * 10 + c, d_idx))

N_WINDOWS = len(windows)
assert N_WINDOWS == 192, f"expected 192 windows, got {N_WINDOWS}"

# Per-cell list of window indices that include the cell. Used by sequence
# detection: we only inspect the (≤20) windows passing through the cell
# that was just changed.
cell_windows = [[] for _ in range(100)]
for w_idx, (mlo, mhi, _, _) in enumerate(windows):
    for cell in range(100):
        bit = (mlo >> cell) & 1 if cell < 64 else (mhi >> (cell - 64)) & 1
        if bit:
            cell_windows[cell].append(w_idx)

max_cw = max(len(x) for x in cell_windows)
print(f"max windows through any cell: {max_cw}")  # expect 20 (interior cells)

# --- emit -----------------------------------------------------------------

def hex64(x):
    return f"0x{x:016x}ULL"

OUT_DIR = os.path.normpath(os.path.join(HERE, '..', 'src'))
os.makedirs(OUT_DIR, exist_ok=True)

# board_data.hpp
hpp = f"""// AUTOGENERATED by tools/gen_board_data.py — do not edit.
#pragma once
#include <array>
#include <cstdint>
#include "types.hpp"

namespace seq {{

constexpr int N_CARD_TYPES = {N_CARD_TYPES};
constexpr int N_REGULAR_CARDS = {N_REGULAR};
constexpr int ONE_EYED_JACK = {ONE_EYED_JACK_IDX};
constexpr int TWO_EYED_JACK = {TWO_EYED_JACK_IDX};
constexpr int N_WINDOWS = {N_WINDOWS};

// cell index (0..99) -> card type (0..47), or -1 for joker corner
extern const int8_t CELL_TO_CARD[100];

// regular card type (0..47) -> two cell indices on the board (-1 if absent)
extern const int8_t CARD_TO_CELLS[N_REGULAR_CARDS][2];

// joker corners
extern const BB100 JOKER_MASK;

// 5-in-a-row windows
extern const BB100 WINDOW_MASKS[N_WINDOWS];

// per-cell list of windows passing through it (variable length, terminated by -1).
// int16_t because window indices run 0..191 — doesn't fit in int8_t.
constexpr int MAX_WINDOWS_PER_CELL = {max_cw};
extern const int16_t CELL_WINDOWS[100][MAX_WINDOWS_PER_CELL + 1];

const char* card_name(int card_idx);

}} // namespace seq
"""

with open(os.path.join(OUT_DIR, 'board_data.hpp'), 'w') as f:
    f.write(hpp)

# board_data.cpp
lines = []
lines.append('// AUTOGENERATED by tools/gen_board_data.py — do not edit.')
lines.append('#include "board_data.hpp"')
lines.append('')
lines.append('namespace seq {')
lines.append('')

# CELL_TO_CARD
lines.append('const int8_t CELL_TO_CARD[100] = {')
for r in range(10):
    row_vals = [cell_to_card[r * 10 + c] for c in range(10)]
    lines.append('  ' + ','.join(f'{v:3d}' for v in row_vals) + ',')
lines.append('};')
lines.append('')

# CARD_TO_CELLS
lines.append('const int8_t CARD_TO_CELLS[N_REGULAR_CARDS][2] = {')
for cidx in range(N_REGULAR):
    a, b = card_to_cells[cidx]
    lines.append(f'  {{{a:3d}, {b:3d}}},  // {card_name(cidx)}')
lines.append('};')
lines.append('')

# JOKER_MASK
lines.append(f'const BB100 JOKER_MASK = {{{hex64(joker_mask_lo)}, {hex64(joker_mask_hi)}}};')
lines.append('')

# WINDOW_MASKS
lines.append('const BB100 WINDOW_MASKS[N_WINDOWS] = {')
DIR_NAMES = ['→', '↓', '↘', '↙']
for i, (mlo, mhi, start, d) in enumerate(windows):
    sr, sc = start // 10, start % 10
    lines.append(f'  {{{hex64(mlo)}, {hex64(mhi)}}},  // {i:3d}: ({sr},{sc}) {DIR_NAMES[d]}')
lines.append('};')
lines.append('')

# CELL_WINDOWS — fixed-width 2D array, terminated with -1
lines.append('const int16_t CELL_WINDOWS[100][MAX_WINDOWS_PER_CELL + 1] = {')
for cell in range(100):
    ws = cell_windows[cell]
    padded = ws + [-1] * (max_cw + 1 - len(ws))
    lines.append('  {' + ','.join(f'{v:3d}' for v in padded) + '},')
lines.append('};')
lines.append('')

# card_name (debug)
lines.append('static const char* CARD_NAMES[50] = {')
for i in range(N_CARD_TYPES):
    lines.append(f'  "{card_name(i)}",')
lines.append('};')
lines.append('const char* card_name(int card_idx) {')
lines.append('  if (card_idx < 0 || card_idx >= N_CARD_TYPES) return "??";')
lines.append('  return CARD_NAMES[card_idx];')
lines.append('}')
lines.append('')

lines.append('} // namespace seq')

with open(os.path.join(OUT_DIR, 'board_data.cpp'), 'w') as f:
    f.write('\n'.join(lines))

print(f"wrote {OUT_DIR}/board_data.hpp and board_data.cpp")
print(f"  {N_CARD_TYPES} card types, {N_WINDOWS} windows")

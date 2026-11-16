"""
Fit the search's move-score and value tables to self-play data.

    ./build/gen_data DATA_DIR            # plays the games, writes *.bin
    python tools/fit_tables.py DATA_DIR  # writes src/learned_tables.hpp

The search scores a move by summing one table entry per 5-window through
the move's cell, and a position by summing one entry per window on the
board; the entry depends only on the window's state (joker?, my chips,
their chips) and the game stage. By default those entries come from the
hand-set weights in eval.hpp. This script fits them instead:

  * move scores  — so that softmax(score) over a position's legal moves
    matches the visit distribution the search itself produced there
    (cross-entropy). Used as the PUCT prior and the rollout policy.
  * value        — logistic regression of who won the game (blended with
    the search's own root value, which is far less noisy) on the window
    counts, jacks in hand, sequences and side to move.

Both models are linear in the same per-window counts the engine already
maintains, so using them costs nothing at search time.

Needs numpy and torch.
"""
import os
import sys

import numpy as np
import torch

N_STATES = 36
N_STAGES = 4
N_HAND = 3          # hand-support levels of a placement window: 0, 1, 2+

POLICY_DTYPE = np.dtype([
    ('decision', '<i4'), ('visits', '<i4'), ('kind', 'u1'), ('stage', 'u1'),
    ('hist', 'u1', (N_HAND, N_STATES)), ('dead', 'u1'), ('pad', 'u1'),
])
VALUE_DTYPE = np.dtype([
    ('game', '<i4'), ('ply', 'u1'), ('stm', 'u1'), ('seq0', 'u1'), ('seq1', 'u1'),
    ('j2_0', 'u1'), ('j1_0', 'u1'), ('j2_1', 'u1'), ('j1_1', 'u1'),
    ('root_q_p0', '<f4'), ('winner', 'u1'),
    ('hist', 'u1', (N_STATES,)), ('dead', 'u1'), ('pad', 'u1', (2,)),
])
assert POLICY_DTYPE.itemsize == 120 and VALUE_DTYPE.itemsize == 56


def state_table():
    """(j, a, b) for each dense state id — same order as window_state_id."""
    out = []
    for j in range(2):
        for a in range(6):
            for b in range(6):
                if a + b + j <= 5:
                    out.append((j, a, b))
    assert len(out) == N_STATES
    return out


STATES = state_table()
STATE_ID = {s: i for i, s in enumerate(STATES)}
# id of the same window seen from the other side
SWAP = np.array([STATE_ID[(j, b, a)] for (j, a, b) in STATES])


def lbfgs(params, loss_fn, steps=200):
    opt = torch.optim.LBFGS(params, lr=1.0, max_iter=steps, history_size=30,
                            line_search_fn='strong_wolfe',
                            tolerance_grad=1e-9, tolerance_change=1e-12)

    def closure():
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        return loss
    opt.step(closure)
    return float(loss_fn())


# ---------------------------------------------------------------------------
# Move scores
# ---------------------------------------------------------------------------

def fit_policy(path, l2=1e-4, min_visits=200):
    rows = np.fromfile(path, dtype=POLICY_DTYPE)
    print(f'policy: {len(rows):,} move rows')

    # Renumber decisions densely; drop ones with too few visits to trust.
    _, group = np.unique(rows['decision'], return_inverse=True)
    n_groups = group.max() + 1
    total = np.bincount(group, weights=rows['visits'], minlength=n_groups)
    keep = total[group] >= min_visits
    rows, group = rows[keep], group[keep]
    _, group = np.unique(group, return_inverse=True)
    n_groups = group.max() + 1
    total = np.bincount(group, weights=rows['visits'], minlength=n_groups)
    print(f'        {n_groups:,} decisions, {len(rows):,} rows kept')

    n_rows = len(rows)
    target = torch.tensor(rows['visits'] / total[group], dtype=torch.float32)
    kind = torch.tensor(rows['kind'].astype(np.int64))
    stage = torch.tensor(rows['stage'].astype(np.int64))
    group_t = torch.tensor(group, dtype=torch.int64)
    # Row indices and window histograms, split by stage and by which
    # table the move reads (placement or removal).
    place_rows, remove_rows = [], []
    for st in range(N_STAGES):
        sel = np.flatnonzero((rows['stage'] == st) & (rows['kind'] <= 1))
        place_rows.append((torch.tensor(sel, dtype=torch.int64),
                           torch.tensor(rows['hist'][sel].reshape(len(sel), -1),
                                        dtype=torch.float32)))
        sel = np.flatnonzero((rows['stage'] == st) & (rows['kind'] == 2))
        remove_rows.append((torch.tensor(sel, dtype=torch.int64),
                            torch.tensor(rows['hist'][sel][:, 0, :],
                                         dtype=torch.float32)))

    # score = hist · table[stage] + bias[stage, kind]; a table is a shared
    # base plus offsets per hand-support level and per stage (offsets are
    # regularized harder, so rare combinations fall back on the base).
    base_place = torch.zeros(N_STATES, requires_grad=True)
    base_remove = torch.zeros(N_STATES, requires_grad=True)
    h_place = torch.zeros(N_HAND, N_STATES, requires_grad=True)
    d_place = torch.zeros(N_STAGES, N_STATES, requires_grad=True)
    hd_place = torch.zeros(N_STAGES, N_HAND, N_STATES, requires_grad=True)
    d_remove = torch.zeros(N_STAGES, N_STATES, requires_grad=True)
    # Hand support only modifies level 1 and 2+, and only for windows
    # the mover has started and the opponent hasn't blocked — the engine
    # then has to work the support out for just those windows.
    h_mask = torch.tensor([0.0] + [1.0] * (N_HAND - 1)).unsqueeze(1) * torch.tensor(
        [1.0 if (a >= 1 and b == 0) else 0.0 for (_, a, b) in STATES]).unsqueeze(0)

    def place_table():      # stages × hand levels × states
        return (base_place + (h_place * h_mask).unsqueeze(0)
                + d_place.unsqueeze(1) + hd_place * h_mask)
    bias = torch.zeros(N_STAGES, 4, requires_grad=True)   # kind 0 fixed at 0
    kind_mask = torch.tensor([0.0, 1.0, 1.0, 1.0])

    def scores():
        s = (bias * kind_mask)[stage, kind]
        tp = place_table()
        for st in range(N_STAGES):
            idx, h = place_rows[st]
            s = s.index_add(0, idx, h @ tp[st].reshape(-1))
            idx, h = remove_rows[st]
            s = s.index_add(0, idx, h @ (base_remove + d_remove[st]))
        assert s.shape[0] == n_rows
        return s

    def loss_fn():
        s = scores()
        # log-softmax within each decision
        mx = torch.full((n_groups,), -1e30).scatter_reduce(
            0, group_t, s, reduce='amax', include_self=True)
        e = torch.exp(s - mx[group_t])
        lse = torch.log(torch.zeros(n_groups).index_add_(0, group_t, e)) + mx
        ce = -(target * (s - lse[group_t])).sum() / n_groups
        reg = l2 * (base_place.pow(2).sum() + base_remove.pow(2).sum()
                    + 3 * h_place.pow(2).sum()
                    + 10 * (d_place.pow(2).sum() + d_remove.pow(2).sum())
                    + 30 * hd_place.pow(2).sum())
        return ce + reg

    with torch.no_grad():
        before = float(loss_fn())
    after = lbfgs([base_place, base_remove, h_place, d_place, hd_place,
                   d_remove, bias], loss_fn)
    with torch.no_grad():
        ent = -(target * torch.log(target.clamp_min(1e-12))).sum() / n_groups
        s = scores()
        best = torch.full((n_groups,), -1e30).scatter_reduce(
            0, group_t, s, reduce='amax', include_self=True)
        tbest = torch.zeros(n_groups).scatter_reduce(
            0, group_t, target, reduce='amax', include_self=True)
        # how often the model's top move is the search's top move
        hit = ((s >= best[group_t] - 1e-9) & (target >= tbest[group_t] - 1e-9))
        agree = torch.zeros(n_groups).index_add_(0, group_t, hit.float()).clamp_max(1).mean()
    print(f'        cross-entropy {before:.4f} -> {after:.4f} '
          f'(target entropy {float(ent):.4f}), top-move agreement {float(agree):.1%}')

    place = place_table().detach().numpy()
    remove = (base_remove + d_remove).detach().numpy()
    b = (bias * kind_mask).detach().numpy()
    return place, remove, b[:, 1], b[:, 2], b[:, 3]


# ---------------------------------------------------------------------------
# Value
# ---------------------------------------------------------------------------

def fit_value(path, outcome_weight=0.5, l2=1e-4):
    rows = np.fromfile(path, dtype=VALUE_DTYPE)
    rows = rows[rows['winner'] < 2]
    print(f'value:  {len(rows):,} positions from {len(np.unique(rows["game"])):,} games')

    has0 = (rows['seq0'] > 0).astype(np.int64)
    has1 = (rows['seq1'] > 0).astype(np.int64)
    st0 = torch.tensor(2 * has0 + has1)            # player 0's stage
    st1 = torch.tensor(2 * has1 + has0)            # player 1's stage
    h0 = torch.tensor(rows['hist'], dtype=torch.float32)          # (j, n0, n1)
    h1 = torch.tensor(rows['hist'][:, SWAP], dtype=torch.float32)  # (j, n1, n0)
    jacks = {k: torch.tensor(rows[k], dtype=torch.float32)
             for k in ('j2_0', 'j1_0', 'j2_1', 'j1_1')}
    seq_diff = torch.tensor(has0 - has1, dtype=torch.float32)
    tempo_sign = torch.tensor(np.where(rows['stm'] == 0, 1.0, -1.0), dtype=torch.float32)

    z = torch.tensor((rows['winner'] == 0).astype(np.float32))
    q = torch.tensor((rows['root_q_p0'] + 1.0) / 2.0).clamp(0.0, 1.0)
    target = outcome_weight * z + (1.0 - outcome_weight) * q

    base = torch.zeros(N_STATES, requires_grad=True)
    delta = torch.zeros(N_STAGES, N_STATES, requires_grad=True)
    w_j2 = torch.zeros(N_STAGES, requires_grad=True)
    w_j1 = torch.zeros(N_STAGES, requires_grad=True)
    w_seq = torch.zeros(1, requires_grad=True)
    w_tempo = torch.zeros(1, requires_grad=True)

    def logit():
        u = base + delta
        x = (h0 * u[st0]).sum(1) - (h1 * u[st1]).sum(1)
        x = x + w_j2[st0] * jacks['j2_0'] - w_j2[st1] * jacks['j2_1']
        x = x + w_j1[st0] * jacks['j1_0'] - w_j1[st1] * jacks['j1_1']
        return x + w_seq * seq_diff + w_tempo * tempo_sign

    bce = torch.nn.functional.binary_cross_entropy_with_logits

    def loss_fn():
        reg = l2 * (base.pow(2).sum() + 10 * delta.pow(2).sum())
        return bce(logit(), target) + reg

    with torch.no_grad():
        before = float(loss_fn())
    after = lbfgs([base, delta, w_j2, w_j1, w_seq, w_tempo], loss_fn)
    with torch.no_grad():
        x = logit()
        acc = ((x > 0).float() == z).float().mean()
        out_ll = bce(x, z)
    print(f'        loss {before:.4f} -> {after:.4f}; predicts the winner '
          f'{float(acc):.1%} of the time (outcome log-loss {float(out_ll):.4f})')

    return ((base + delta).detach().numpy(), w_j2.detach().numpy(),
            w_j1.detach().numpy(), float(w_seq), float(w_tempo))


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

def fmt_table(name, t):
    lines = [f'inline constexpr double {name}[N_STAGES][N_WINDOW_STATES] = {{']
    for st in range(N_STAGES):
        vals = ', '.join(f'{v:.4f}' for v in t[st])
        lines.append(f'    {{{vals}}},')
    lines.append('};')
    return '\n'.join(lines)


def fmt_table3(name, t):
    lines = [f'inline constexpr double {name}[N_STAGES][N_HAND_LEVELS][N_WINDOW_STATES] = {{']
    for st in range(N_STAGES):
        lines.append('    {')
        for h in range(N_HAND):
            vals = ', '.join(f'{v:.4f}' for v in t[st][h])
            lines.append(f'        {{{vals}}},')
        lines.append('    },')
    lines.append('};')
    return '\n'.join(lines)


def fmt_vec(name, v):
    return (f'inline constexpr double {name}[N_STAGES] = {{'
            + ', '.join(f'{x:.4f}' for x in v) + '};')


def write_header(path, policy, value, note):
    place, remove, mj2, mj1, mdead = policy
    val, vj2, vj1, vseq, vtempo = value
    text = f'''// AUTOGENERATED by tools/fit_tables.py — do not edit.
// {note}
//
// Indexing: [stage][window_state_id(j, mine, theirs)] — see
// window_states.hpp. Stage = 2·(I have a sequence) + (they have one).
// LEARNED_PLACE has one more index after the stage: the window's hand
// support (0, 1, 2+ of its other empty cells fillable from hand).
// Move scores are natural-log weights (softmax them for a policy); the
// value terms are logits of the win probability.
#pragma once
#include "window_states.hpp"

namespace seq {{

// --- move scores -----------------------------------------------------------
// Per window through the target cell: placing a chip / removing theirs.
{fmt_table3('LEARNED_PLACE', place)}
{fmt_table('LEARNED_REMOVE', remove)}
// Added once per move: spending a two-eyed / one-eyed jack, declaring a
// dead card.
{fmt_vec('LEARNED_MOVE_J2', mj2)}
{fmt_vec('LEARNED_MOVE_J1', mj1)}
{fmt_vec('LEARNED_MOVE_DEAD', mdead)}

// --- value -----------------------------------------------------------------
// logit P(I win) = Σ_windows V[my stage][j, mine, theirs] + jacks I hold
//                  − (the same from their side) ± sequence and tempo terms.
{fmt_table('LEARNED_VALUE', val)}
{fmt_vec('LEARNED_VALUE_J2', vj2)}
{fmt_vec('LEARNED_VALUE_J1', vj1)}
inline constexpr double LEARNED_VALUE_SEQ = {vseq:.4f};
inline constexpr double LEARNED_VALUE_TEMPO = {vtempo:.4f};

}} // namespace seq
'''
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(text)
    print(f'wrote {path}')


def describe(policy, value):
    place, remove, mj2, mj1, mdead = policy
    val = value[0]
    print('\nplace score by (mine, theirs), no joker, stage 0, hand support 0 / 1 / 2+:')
    for a in range(5):
        cells = []
        for b in range(5 - a):
            i = STATE_ID[(0, a, b)]
            cells.append('/'.join(f'{place[0][h][i]:+.2f}' for h in range(N_HAND)))
        print(f'  mine={a}: ' + '  '.join(cells))
    print('jack move bias (2-eyed / 1-eyed / dead) per stage:')
    for st in range(N_STAGES):
        print(f'  stage {st}: {mj2[st]:+.2f} / {mj1[st]:+.2f} / {mdead[st]:+.2f}')
    print('value by my chips in an unblocked window (no joker), stage 0 / stage 2:')
    print('  ' + '  '.join(f'{a}: {val[0][STATE_ID[(0, a, 0)]]:+.3f}/{val[2][STATE_ID[(0, a, 0)]]:+.3f}'
                          for a in range(6)))
    print(f'value: jack2 {np.round(value[1], 3)}, jack1 {np.round(value[2], 3)}, '
          f'sequence {value[3]:+.3f}, tempo {value[4]:+.3f}')


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    data = sys.argv[1]
    here = os.path.dirname(os.path.abspath(__file__))
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.normpath(
        os.path.join(here, '..', 'src', 'learned_tables.hpp'))
    torch.manual_seed(0)
    policy = fit_policy(os.path.join(data, 'policy.bin'))
    value = fit_value(os.path.join(data, 'value.bin'))
    describe(policy, value)
    n_games = len(np.unique(np.fromfile(os.path.join(data, 'value.bin'),
                                        dtype=VALUE_DTYPE)['game']))
    write_header(out, policy, value, f'Fitted to {n_games:,} self-play games.')


if __name__ == '__main__':
    main()

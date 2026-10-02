// Position evaluation.
//
// All bitboard-driven; one window is a couple of
// popcounts plus a branch.

#pragma once
#include "types.hpp"
#include "state.hpp"
#include "board_data.hpp"

namespace seq {

// Weight as a function of "my chip count" (counting jokers) in a window.
// The k=5 entry stays at
// 1.0 (rather than dropping to 0) so that completing a sequence doesn't
// produce a negative shaping spike — the +10 sequence bonus is the upside.
inline constexpr double PHI_WINDOW_WEIGHTS[6] = {
    0.00,   // 0
    0.01,   // 1
    0.05,   // 2
    0.20,   // 3
    1.00,   // 4
    1.00,   // 5
};

inline constexpr double SHAPING_GAMMA = 0.97;

namespace detail {
// Value of one window given precomputed player/opponent masks:
//   opp_mask = chips[opp]    | locked[opp]
//   me_mask  = chips[player] | locked[player] | JOKER_MASK
// Hoisting these out of the inner loop is the difference between two
// BB100 ORs per window (192× for total_phi) and two ORs per call.
inline double window_phi_masked(BB100 opp_mask, BB100 me_mask, int win_idx) {
    const BB100 w = WINDOW_MASKS[win_idx];
    if ((w & opp_mask).any()) return 0.0;
    return PHI_WINDOW_WEIGHTS[(w & me_mask).popcount()];
}
} // namespace detail

// Value of a single window from `player`'s perspective. Blocked iff the
// opponent has any chip (locked or unlocked) in the window.
inline double window_phi(const GameState& s, int win_idx, int player) {
    const int opp = 1 - player;
    return detail::window_phi_masked(
        s.chips[opp]    | s.locked[opp],
        s.chips[player] | s.locked[player] | JOKER_MASK,
        win_idx);
}

// Sum over all 192 windows. Static eval at a search-tree leaf.
inline double total_phi(const GameState& s, int player) {
    const int opp = 1 - player;
    const BB100 opp_mask = s.chips[opp]    | s.locked[opp];
    const BB100 me_mask  = s.chips[player] | s.locked[player] | JOKER_MASK;
    double total = 0.0;
    for (int w = 0; w < N_WINDOWS; ++w) {
        total += detail::window_phi_masked(opp_mask, me_mask, w);
    }
    return total;
}

// total_phi(s, 0) - total_phi(s, 1) in one pass over the windows. The
// two sums accumulate separately in window order, so the result is bit-
// identical to calling total_phi twice.
inline double phi_diff_p0(const GameState& s) {
    const BB100 b0 = s.chips[0] | s.locked[0];
    const BB100 b1 = s.chips[1] | s.locked[1];
    const BB100 m0 = b0 | JOKER_MASK;
    const BB100 m1 = b1 | JOKER_MASK;
    double t0 = 0.0, t1 = 0.0;
    for (int i = 0; i < N_WINDOWS; ++i) {
        const BB100 w = WINDOW_MASKS[i];
        const bool has0 = (w & b0).any();
        const bool has1 = (w & b1).any();
        t0 += has1 ? 0.0 : PHI_WINDOW_WEIGHTS[(w & m0).popcount()];
        t1 += has0 ? 0.0 : PHI_WINDOW_WEIGHTS[(w & m1).popcount()];
    }
    return t0 - t1;
}

// Sum over windows passing through `cell` only (≤ 20 windows). Used for
// shaping-reward deltas: only these windows can have their value changed
// by a move at `cell`, so we don't need a full-board scan.
inline double affected_phi(const GameState& s, int cell, int player) {
    const int opp = 1 - player;
    const BB100 opp_mask = s.chips[opp]    | s.locked[opp];
    const BB100 me_mask  = s.chips[player] | s.locked[player] | JOKER_MASK;
    const int16_t* ws = CELL_WINDOWS[cell];
    double total = 0.0;
    for (int i = 0; ws[i] >= 0; ++i) {
        total += detail::window_phi_masked(opp_mask, me_mask, ws[i]);
    }
    return total;
}

// Snapshot of the four bitboards a shaping_score evaluation reads. Build
// once per decision (one mask construction per pick_rollout_move call
// instead of four per candidate move) and pass to the masked
// shaping_score overload.
struct ShapingMasks {
    BB100 p_mask;     // chips[player] | locked[player] | JOKER_MASK
    BB100 o_mask;     // chips[opp]    | locked[opp]    | JOKER_MASK
    BB100 p_blocker;  // chips[player] | locked[player]   (blocks opp windows)
    BB100 o_blocker;  // chips[opp]    | locked[opp]      (blocks me windows)
};

inline ShapingMasks build_shaping_masks(const GameState& s, int player) {
    const int opp = 1 - player;
    const BB100 p_blocker = s.chips[player] | s.locked[player];
    const BB100 o_blocker = s.chips[opp]    | s.locked[opp];
    return ShapingMasks{
        p_blocker | JOKER_MASK,
        o_blocker | JOKER_MASK,
        p_blocker,
        o_blocker,
    };
}

// Net shaping score for `player` playing `card_type` at `cell`. Mirrors
// evaluate.py:_shaping_score. The caller passes the card-type index (not
// the hand slot) because at the eval stage we only care whether it's a
// one-eyed jack (chip-removal) or a chip-placement.
//
// Conceptually each window through `cell` contributes
//     γ·me_after − me_before + opp_before − γ·opp_after
// where me_* / opp_* are window_phi from each side before and after the
// move. Every window in CELL_WINDOWS[cell] contains `cell`, so the four
// terms collapse to two mask tests and at most two popcounts:
//
//   placement (cell empty, not a joker):
//     me:  opp blocking is unchanged; my count goes pc -> pc+1.
//     opp: after the move my chip sits in the window, so opp_after = 0.
//     => the term depends only on the window, not on `cell` — which is
//        what lets ShapingCache memoize it per window.
//   removal (cell holds an unlocked opp chip):
//     me:  before, opp's chip at `cell` blocks the window -> me_before = 0.
//     opp: my blocking is unchanged; opp's count goes oc -> oc-1.
//
// The accumulation keeps the exact expression shape of the four-term
// form (structurally-zero terms written as 0.0), so results are bit-
// identical to it.
namespace detail {
inline double placement_window_term(const ShapingMasks& pre, int win_idx) {
    const BB100 w = WINDOW_MASKS[win_idx];
    double me_before = 0.0, me_after = 0.0, opp_before = 0.0;
    if (!(w & pre.o_blocker).any()) {
        const int pc = (w & pre.p_mask).popcount();
        me_before = PHI_WINDOW_WEIGHTS[pc];
        me_after  = PHI_WINDOW_WEIGHTS[pc + 1];
    }
    if (!(w & pre.p_blocker).any()) {
        opp_before = PHI_WINDOW_WEIGHTS[(w & pre.o_mask).popcount()];
    }
    return SHAPING_GAMMA * me_after - me_before
         + opp_before - SHAPING_GAMMA * 0.0;
}

inline double removal_score(const ShapingMasks& pre, int cell) {
    BB100 cb; cb.set(cell);
    const BB100 o_blocker_post = pre.o_blocker & ~cb;
    const int16_t* ws = CELL_WINDOWS[cell];
    double delta = 0.0;
    for (int i = 0; ws[i] >= 0; ++i) {
        const BB100 w = WINDOW_MASKS[ws[i]];
        double me_after = 0.0, opp_before = 0.0, opp_after = 0.0;
        if (!(w & o_blocker_post).any()) {
            me_after = PHI_WINDOW_WEIGHTS[(w & pre.p_mask).popcount()];
        }
        if (!(w & pre.p_blocker).any()) {
            const int oc = (w & pre.o_mask).popcount();
            opp_before = PHI_WINDOW_WEIGHTS[oc];
            opp_after  = PHI_WINDOW_WEIGHTS[oc - 1];
        }
        delta += SHAPING_GAMMA * me_after - 0.0
               + opp_before - SHAPING_GAMMA * opp_after;
    }
    return delta;
}
} // namespace detail

inline double shaping_score(const ShapingMasks& pre, int card_type, int cell) {
    if (cell < 0) return 0.0;  // dead-card swap
    if (card_type == ONE_EYED_JACK) return detail::removal_score(pre, cell);
    const int16_t* ws = CELL_WINDOWS[cell];
    double delta = 0.0;
    for (int i = 0; ws[i] >= 0; ++i) {
        delta += detail::placement_window_term(pre, ws[i]);
    }
    return delta;
}

// Convenience overload preserving the original (GameState&, player, ...)
// signature. Builds masks per call, so hot callers should hoist
// build_shaping_masks above their per-move loop and use the masked
// overload directly.
inline double shaping_score(const GameState& s, int player, int card_type, int cell) {
    if (cell < 0) return 0.0;
    return shaping_score(build_shaping_masks(s, player), card_type, cell);
}

// Memoizes shaping_score by (flavor, cell) over the lifetime of a single
// move-scoring loop. Two flavors: placement (any non-one-eyed-jack card)
// and removal (one-eyed jack). shaping_score's result depends only on the
// shaping masks plus those two parameters, so within one loop the masks
// are fixed and a hit returns the cached value.
//
// Hot loops collide on (flavor, cell) routinely: a two-eyed-jack slot
// emits a move for every empty non-joker cell, so two two-eyed jacks
// double-count the same 96 cells; every regular card has two board
// positions and a duplicate hand slot revisits both; one-eyed jacks
// likewise collide across duplicate slots. PUCT's prior loop hits the
// exact same set of (flavor, cell) pairs as the rollout loop a few lines
// up, so the second pass is a pure cache replay if both use one cache.
struct ShapingCache {
    const ShapingMasks& masks;
    double  scores[2][N_CELLS];
    uint8_t seen[2][N_CELLS] = {};
    // Placement terms are a function of the window alone (see
    // placement_window_term), and neighbouring cells share most of their
    // windows — memoize per window too. A two-eyed jack in hand otherwise
    // re-evaluates each window ~5x across the ~90 cells it can reach.
    double  wterm[N_WINDOWS];
    uint8_t wseen[N_WINDOWS] = {};

    explicit ShapingCache(const ShapingMasks& m) : masks(m) {}

    // cell < 0 (dead-card declaration) bypasses the cache and returns 0,
    // matching shaping_score's contract.
    double score(int card_type, int cell) {
        if (cell < 0) return 0.0;
        const int f = (card_type == ONE_EYED_JACK) ? 1 : 0;
        if (!seen[f][cell]) {
            scores[f][cell] = f ? detail::removal_score(masks, cell)
                                : placement_score(cell);
            seen[f][cell]   = 1;
        }
        return scores[f][cell];
    }

private:
    // Same summation order as shaping_score -> bit-identical result.
    double placement_score(int cell) {
        const int16_t* ws = CELL_WINDOWS[cell];
        double delta = 0.0;
        for (int i = 0; ws[i] >= 0; ++i) {
            const int w = ws[i];
            if (!wseen[w]) {
                wterm[w] = detail::placement_window_term(masks, w);
                wseen[w] = 1;
            }
            delta += wterm[w];
        }
        return delta;
    }
};

} // namespace seq

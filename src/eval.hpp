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
// One window-loop pass folds all four before/after x me/opp evaluations
// into the running delta. This replaces four separate affected_phi calls
// that each rebuilt their own masks and re-walked the windows through
// `cell` -- about 80 window evaluations per move drops to about 20.
inline double shaping_score(const ShapingMasks& pre, int card_type, int cell) {
    if (cell < 0) return 0.0;  // dead-card swap

    // Post-move masks: exactly one bit differs between before and after.
    //   one-eyed jack: opp's UNLOCKED chip at `cell` is removed -> clear
    //                  the bit in o_mask and o_blocker (locked[opp] does
    //                  not have it by precondition, and `cell` is not a
    //                  joker corner since opp has a chip there).
    //   placement:     player's chip lands on an empty cell -> set the
    //                  bit in p_mask and p_blocker.
    //
    // Built with | / & ~ rather than copy-then-set/reset so all four
    // posts are const for the window loop — leaves the compiler free to
    // keep them in registers across the 20-window pass instead of
    // treating the local as a freshly-written, possibly-aliased object.
    BB100 cb; cb.set(cell);
    const bool is_remove = (card_type == ONE_EYED_JACK);
    const BB100 not_cb         = ~cb;
    const BB100 p_mask_post    = is_remove ? pre.p_mask    : pre.p_mask    | cb;
    const BB100 p_blocker_post = is_remove ? pre.p_blocker : pre.p_blocker | cb;
    const BB100 o_mask_post    = is_remove ? pre.o_mask    & not_cb : pre.o_mask;
    const BB100 o_blocker_post = is_remove ? pre.o_blocker & not_cb : pre.o_blocker;

    const int16_t* ws = CELL_WINDOWS[cell];
    double delta = 0.0;
    for (int i = 0; ws[i] >= 0; ++i) {
        const BB100 w = WINDOW_MASKS[ws[i]];

        // From player's perspective: opp's blocker blocks, count p_mask.
        const double me_before = (w & pre.o_blocker).any() ? 0.0
            : PHI_WINDOW_WEIGHTS[(w & pre.p_mask).popcount()];
        const double me_after  = (w & o_blocker_post).any() ? 0.0
            : PHI_WINDOW_WEIGHTS[(w & p_mask_post).popcount()];

        // From opp's perspective: player's blocker blocks, count o_mask.
        const double opp_before = (w & pre.p_blocker).any() ? 0.0
            : PHI_WINDOW_WEIGHTS[(w & pre.o_mask).popcount()];
        const double opp_after  = (w & p_blocker_post).any() ? 0.0
            : PHI_WINDOW_WEIGHTS[(w & o_mask_post).popcount()];

        delta += SHAPING_GAMMA * me_after - me_before
               + opp_before - SHAPING_GAMMA * opp_after;
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

    explicit ShapingCache(const ShapingMasks& m) : masks(m) {}

    // cell < 0 (dead-card declaration) bypasses the cache and returns 0,
    // matching shaping_score's contract.
    double score(int card_type, int cell) {
        if (cell < 0) return 0.0;
        const int f = (card_type == ONE_EYED_JACK) ? 1 : 0;
        if (!seen[f][cell]) {
            scores[f][cell] = shaping_score(masks, card_type, cell);
            seen[f][cell]   = 1;
        }
        return scores[f][cell];
    }
};

} // namespace seq

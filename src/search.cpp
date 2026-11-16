#include "search.hpp"
#include "learned_tables.hpp"
#include "profile.hpp"
#include "window_states.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <thread>

namespace seq {

namespace {

// ---------------------------------------------------------------------------
// Window-state tables
// ---------------------------------------------------------------------------
//
// The search keeps, next to the bitboards, one byte per 5-window:
//
//     w = 72·dead + 36·j + 6·n0 + n1
//
// with n0/n1 the number of (non-joker) chips each player has in the
// window, j = 1 if the window contains a joker corner, and dead = 1 once
// the window can never become a sequence for anyone (see below).
// Everything the search needs from the evaluation — the shaping score of
// a placement or a removal, Φ, "did this just make five" — is a function
// of that byte, so it is one table lookup per window instead of bitboard
// ANDs and popcounts. A move touches the ≤ 20 windows through its cell.
//
// Scores are fixed-point integers (FP units per Φ point). Every weight
// in eval.hpp is a multiple of 1e-4, so the integers are exact: equal
// scores compare equal instead of differing in the last float bit.
//
// Dead windows: a new sequence may share at most one chip with an
// existing one, so a window holding two or more of a player's locked
// chips can never complete for that player — and it is blocked for the
// opponent by those same chips. eval.hpp still scores such a window (the
// finished sequence itself counts as a "5", the windows one step along
// the line as "4"s), which rewards the opponent for "blocking" a line
// that was never a threat. With cfg.dead_windows the search scores them
// as zero for both sides.

constexpr int FP        = 10000;
constexpr int W_DEAD    = 72;
constexpr int N_WSTATES = 144;

// Board geometry in the compact form the hot loops want.
struct Geometry {
    uint8_t winit[N_WINDOWS];       // 36 if the window holds a joker
    uint8_t cw[N_CELLS][MAX_WINDOWS_PER_CELL];   // windows through a cell
    uint8_t cwn[N_CELLS];

    Geometry() {
        for (int w = 0; w < N_WINDOWS; ++w) {
            winit[w] = (WINDOW_MASKS[w] & JOKER_MASK).any() ? 36 : 0;
        }
        for (int c = 0; c < N_CELLS; ++c) {
            int k = 0;
            for (; CELL_WINDOWS[c][k] >= 0; ++k) cw[c][k] = uint8_t(CELL_WINDOWS[c][k]);
            cwn[c] = uint8_t(k);
        }
    }
};

const Geometry GEO;

struct Tables {
    // Move scores (fixed-point log-weights for the rollout policy and
    // the prior), indexed by the mover's stage and seat.
    // per window: place a chip — also by hand support (window_states.hpp);
    // the hand-tuned table is the same at every support level
    int32_t place[N_STAGES][2][N_HAND_LEVELS][N_WSTATES];
    // 1 where the placement score actually depends on the support level,
    // so the hot loop only works it out for those windows
    uint8_t hand_sensitive[N_STAGES][2][N_WSTATES];
    int32_t remove[N_STAGES][2][N_WSTATES];   // per window: remove opp chip
    int32_t move_j2[N_STAGES];                // added when the move spends
    int32_t move_j1[N_STAGES];                //   a two-/one-eyed jack
    int32_t move_dead[N_STAGES];              // dead-card declaration

    // Static value, player 0's view: tanh(value_scale · Σ), where Σ sums
    // value[stage][w] over the windows plus the scalar terms below.
    // `stage` here is player 0's stage.
    int32_t value[N_STAGES][N_WSTATES];
    int32_t value_j2[N_STAGES];               // per jack held, by holder's stage
    int32_t value_j1[N_STAGES];
    int32_t value_bias[N_STAGES];
    int32_t value_tempo;                      // for the side to move
    double  value_scale;

    uint8_t five[2][N_WSTATES];     // window is five of player's (or joker)

    explicit Tables(const MCTSConfig& cfg) {
        std::memset(this, 0, sizeof(*this));

        for (int idx = 0; idx < W_DEAD; ++idx) {    // dead states stay zero
            const int j = idx / 36;
            const int n[2] = {(idx % 36) / 6, idx % 6};
            for (int p = 0; p < 2; ++p) {
                five[p][idx] = (n[1 - p] == 0 && n[p] + j == 5);
            }
        }
        if (cfg.learned_policy) fill_learned_policy();
        else                    fill_handtuned_policy(cfg);
        if (cfg.learned_value)  fill_learned_value();
        else                    fill_handtuned_value(cfg);
    }

private:
    static int32_t fp(double x) { return int32_t(std::llround(x * FP)); }

    // eval.hpp's shaping score, as a per-window table.
    void fill_handtuned_policy(const MCTSConfig& cfg) {
        int32_t W[7], G[7];         // weight and γ·weight, fixed-point
        for (int k = 0; k < 6; ++k) {
            W[k] = fp(PHI_WINDOW_WEIGHTS[k]);
            G[k] = fp(SHAPING_GAMMA * PHI_WINDOW_WEIGHTS[k]);
        }
        W[6] = W[5]; G[6] = G[5];   // unreachable counts; keep lookups in range
        const int32_t seq_bonus = fp(cfg.shaping_seq_bonus);

        for (int st = 0; st < N_STAGES; ++st) {
            for (int idx = 0; idx < W_DEAD; ++idx) {
                const int j = idx / 36;
                const int n[2] = {(idx % 36) / 6, idx % 6};
                for (int p = 0; p < 2; ++p) {
                    const int me = n[p], opp = n[1 - p];
                    const int pc = std::min(me + j, 5);     // my count incl. joker
                    const int oc = std::min(opp + j, 5);
                    // Placement (mirrors eval.hpp placement_window_term):
                    //   γ·me_after − me_before + opp_before
                    int32_t v = (opp == 0 ? G[pc + 1] - W[pc] : 0)
                              + (me == 0 ? W[oc] : 0);
                    // ... plus a bonus when the placement completes the five.
                    if (opp == 0 && me + j == 4) v += seq_bonus;
                    for (int h = 0; h < N_HAND_LEVELS; ++h) place[st][p][h][idx] = v;
                    // Removal of one opp chip (mirrors removal_score):
                    //   γ·me_after + opp_before − γ·opp_after
                    remove[st][p][idx] = (opp == 1 ? G[pc] : 0)
                        + ((me == 0 && opp >= 1) ? W[oc] - G[oc - 1] : 0);
                }
            }
            move_j2[st]   = -fp(cfg.jack2_value);
            move_j1[st]   = -fp(cfg.jack1_value);
            move_dead[st] = 0;
        }
    }

    // tanh(cutoff_scale · (Φ0 − Φ1 + seq bonus + jacks in hand)).
    void fill_handtuned_value(const MCTSConfig& cfg) {
        for (int st = 0; st < N_STAGES; ++st) {
            for (int idx = 0; idx < W_DEAD; ++idx) {
                const int j = idx / 36;
                const int n0 = (idx % 36) / 6, n1 = idx % 6;
                const int32_t phi0 = n1 == 0 ? fp(PHI_WINDOW_WEIGHTS[std::min(n0 + j, 5)]) : 0;
                const int32_t phi1 = n0 == 0 ? fp(PHI_WINDOW_WEIGHTS[std::min(n1 + j, 5)]) : 0;
                value[st][idx] = phi0 - phi1;
            }
            value_j2[st]   = fp(cfg.jack2_value);
            value_j1[st]   = fp(cfg.jack1_value);
            value_bias[st] = fp(cfg.cutoff_seq_bonus) * ((st >> 1) - (st & 1));
        }
        value_tempo = 0;
        value_scale = cfg.cutoff_scale / FP;
    }

    // Tables fitted to the engine's own searches (tools/fit_tables.py).
    void fill_learned_policy() {
        for (int st = 0; st < N_STAGES; ++st) {
            for (int idx = 0; idx < W_DEAD; ++idx) {
                const int j = idx / 36;
                const int n[2] = {(idx % 36) / 6, idx % 6};
                for (int p = 0; p < 2; ++p) {
                    const int id = window_state_id(j, n[p], n[1 - p]);
                    if (id < 0) continue;
                    for (int h = 0; h < N_HAND_LEVELS; ++h) {
                        place[st][p][h][idx] = fp(LEARNED_PLACE[st][h][id]);
                        if (place[st][p][h][idx] != place[st][p][0][idx]) {
                            hand_sensitive[st][p][idx] = 1;
                        }
                    }
                    remove[st][p][idx] = fp(LEARNED_REMOVE[st][id]);
                }
            }
            move_j2[st]   = fp(LEARNED_MOVE_J2[st]);
            move_j1[st]   = fp(LEARNED_MOVE_J1[st]);
            move_dead[st] = fp(LEARNED_MOVE_DEAD[st]);
        }
    }

    // logit P(player 0 wins) = U(0) − U(1) + tempo, with each player's U
    // summed from their own stage's table; value = tanh(logit / 2).
    void fill_learned_value() {
        for (int st = 0; st < N_STAGES; ++st) {
            const int st1 = 2 * (st & 1) + (st >> 1);      // player 1's stage
            for (int idx = 0; idx < W_DEAD; ++idx) {
                const int j = idx / 36;
                const int n0 = (idx % 36) / 6, n1 = idx % 6;
                const int id0 = window_state_id(j, n0, n1);
                const int id1 = window_state_id(j, n1, n0);
                if (id0 < 0) continue;
                value[st][idx] = fp(LEARNED_VALUE[st][id0]) - fp(LEARNED_VALUE[st1][id1]);
            }
            value_j2[st]   = fp(LEARNED_VALUE_J2[st]);
            value_j1[st]   = fp(LEARNED_VALUE_J1[st]);
            value_bias[st] = fp(LEARNED_VALUE_SEQ) * ((st >> 1) - (st & 1));
        }
        value_tempo = fp(LEARNED_VALUE_TEMPO);
        value_scale = 0.5 / FP;
    }
};

// Per-search constants derived from the config.
struct Params {
    const MCTSConfig& cfg;
    Tables   tb;
    uint64_t eps_threshold;     // rollout ε as a 53-bit integer threshold
    int      max_plies;
    bool     use_vl;

    Params(const MCTSConfig& c, bool multi_threaded)
        : cfg(c),
          tb(c),
          eps_threshold(uint64_t(std::max(0.0, std::min(1.0, c.rollout_epsilon))
                                 * double(uint64_t(1) << 53))),
          max_plies(c.rollout_cutoff > 0
                        ? std::min(c.rollout_cutoff, c.max_rollout_steps)
                        : c.max_rollout_steps),
          use_vl(multi_threaded) {}
};

// ---------------------------------------------------------------------------
// Search state
// ---------------------------------------------------------------------------

struct SState {
    GameState g;
    uint8_t   w[N_WINDOWS];
};

// Flag the windows that hold two or more of `player`'s locked chips.
void mark_dead_windows(SState& s, int player) {
    const BB100 locked = s.g.locked[player];
    for (int i = 0; i < N_WINDOWS; ++i) {
        if (s.w[i] < W_DEAD && (WINDOW_MASKS[i] & locked).popcount() >= 2) {
            s.w[i] = uint8_t(s.w[i] + W_DEAD);
        }
    }
}

void init_windows(SState& s, const Params& P) {
    const BB100 b0 = s.g.chips[0] | s.g.locked[0];
    const BB100 b1 = s.g.chips[1] | s.g.locked[1];
    for (int i = 0; i < N_WINDOWS; ++i) {
        const BB100 m = WINDOW_MASKS[i];
        s.w[i] = uint8_t(GEO.winit[i] + 6 * (m & b0).popcount() + (m & b1).popcount());
    }
    if (P.cfg.dead_windows) {
        mark_dead_windows(s, 0);
        mark_dead_windows(s, 1);
    }
}

inline int32_t window_sum(const SState& s, const int32_t* table, int cell) {
    const uint8_t* ws = GEO.cw[cell];
    int32_t sum = 0;
    for (int i = 0, n = GEO.cwn[cell]; i < n; ++i) sum += table[s.w[ws[i]]];
    return sum;
}

// Placement score with hand support: each window through the cell reads
// the table for how many of its other empty cells `support` covers.
// `cache` (optional, 192 entries preset to -1) memoizes the per-window
// support counts when many cells are scored against the same support —
// a two-eyed jack's ~90 targets share most of their windows.
inline int32_t window_sum_supported(const SState& s,
                                    const int32_t (*table)[N_WSTATES],
                                    const uint8_t* sensitive,
                                    int cell, const BB100& support,
                                    int8_t* cache) {
    const int self = support.test(cell);    // the target itself doesn't count
    const uint8_t* ws = GEO.cw[cell];
    int32_t sum = 0;
    for (int i = 0, n = GEO.cwn[cell]; i < n; ++i) {
        const int w  = ws[i];
        const int st = s.w[w];
        int h = 0;
        if (sensitive[st]) {
            int c;
            if (cache == nullptr) {
                c = (WINDOW_MASKS[w] & support).popcount();
            } else if ((c = cache[w]) < 0) {
                c = (WINDOW_MASKS[w] & support).popcount();
                cache[w] = int8_t(c);
            }
            h = std::min(c - self, N_HAND_LEVELS - 1);
        }
        sum += table[h][st];
    }
    return sum;
}

// Uniform integer in [0, n) — multiply-shift, no division.
inline int rand_below(Xoshiro256pp& rng, int n) {
    return int(((rng() >> 32) * uint64_t(n)) >> 32);
}

// Draw a uniformly random card from the deck. The search never looks at
// deck order — it is hidden information — so instead of shuffling the
// whole deck per determinization we pull a random card whenever one is
// drawn. Same distribution, a fraction of the RNG calls.
inline int8_t draw_random(GameState& g, Xoshiro256pp& rng) {
    CardPile& d = g.deck;
    if (d.count == 0) {
        if (g.discard.count == 0) return -1;
        d = g.discard;              // reshuffle the discard pile
        g.discard.clear();
    }
    const int j = rand_below(rng, d.count);
    const int8_t c = d.data[j];
    d.data[j] = d.data[--d.count];
    return c;
}

// The opponent's hand is hidden: return it to the deck and deal a fresh
// random one. Called once per iteration on the iteration's state copy,
// so the search averages over many guesses of the hidden cards.
void determinize(SState& s, int our_player, Xoshiro256pp& rng) {
    SEQ_PROFILE_SCOPE("MCTS::determinize");
    GameState& g = s.g;
    const int opp = 1 - our_player;
    for (int i = 0; i < HAND_SIZE; ++i) {
        if (g.hands[opp][i] >= 0) {
            g.deck.push_back(g.hands[opp][i]);
            g.hands[opp][i] = -1;
        }
    }
    for (int i = 0; i < HAND_SIZE; ++i) {
        if (g.deck.count > 0) g.hands[opp][i] = draw_random(g, rng);
    }
}

// GameState::make_move for moves that came out of gen_moves: no
// validation, window bytes kept in sync, sequence detection only when a
// window through the cell actually reached five.
inline void apply_move(SState& s, const Params& P, Move m, Xoshiro256pp& rng) {
    SEQ_PROFILE_SCOPE("MCTS::apply_move");
    GameState& g = s.g;
    const int p   = g.current_player;
    const int opp = 1 - p;
    const int8_t card = g.hands[p][m.card_idx];

    if (m.is_dead()) {
        g.discard.push_back(card);
        g.hands[p][m.card_idx] = draw_random(g, rng);
        g.dead_card_used[p] = true;
        return;
    }

    const int cell = m.cell;
    const uint8_t* ws = GEO.cw[cell];
    const int n = GEO.cwn[cell];
    if (card == ONE_EYED_JACK) {
        g.chips[opp].reset(cell);
        const uint8_t d = (p == 0) ? 1 : 6;      // one fewer opp chip
        for (int i = 0; i < n; ++i) s.w[ws[i]] -= d;
    } else {
        g.chips[p].set(cell);
        const uint8_t d = (p == 0) ? 6 : 1;
        const uint8_t* five = P.tb.five[p];
        unsigned made_five = 0;
        for (int i = 0; i < n; ++i) made_five |= five[s.w[ws[i]] += d];
        if (made_five) {
            const int made = g.check_sequences(p, cell);
            g.sequences[p] += made;
            if (made > 0 && P.cfg.dead_windows) mark_dead_windows(s, p);
        }
    }

    g.discard.push_back(card);
    g.hands[p][m.card_idx] = draw_random(g, rng);

    if (g.sequences[p] >= SEQUENCES_TO_WIN) {
        g.done   = true;
        g.winner = p;
        return;
    }
    g.dead_card_used[p] = false;
    g.current_player    = opp;
}

// ---------------------------------------------------------------------------
// Moves, as the search sees them
// ---------------------------------------------------------------------------
//
// A tree edge is identified by WHAT was played, not by the hand slot it
// was played from:
//
//     regular card on cell c   key = c
//     two-eyed jack on cell c  key = 128 + c
//     one-eyed jack on cell c  key = 256 + c
//     dead card of type t      key = 384 + t
//
// (The cell determines a regular card's type, so (kind, cell) is the
// whole move.) Slots are an accident of hand order: the opponent's hand
// is re-dealt every iteration, so slot-keyed edges would split one
// opponent move over up to seven children, and two copies of a card in
// our own hand would split one move over two. Card keys also mean the
// board at a node is a function of the path alone.

constexpr int KEY_J2   = 128;
constexpr int KEY_J1   = 256;
constexpr int KEY_DEAD = 384;
constexpr int N_KEYS   = 512;

inline uint16_t key_of(int card_type, int cell) {
    if (cell < 0)                   return uint16_t(KEY_DEAD + card_type);
    if (card_type == TWO_EYED_JACK) return uint16_t(KEY_J2 + cell);
    if (card_type == ONE_EYED_JACK) return uint16_t(KEY_J1 + cell);
    return uint16_t(cell);
}

// Distinct moves: ≤ 96 jack targets (two-eyed on empty cells, one-eyed
// on opponent chips — disjoint sets of non-joker cells) + 14 + 7.
constexpr int MAX_SMOVES = 128;

struct SMoveList {
    int      n = 0;
    Move     move[MAX_SMOVES];
    uint16_t key[MAX_SMOVES];
    int32_t  score[MAX_SMOVES];
};

// Legal moves for the side to move, one per distinct (card, target).
// Mirrors GameState::legal_moves_for, including the dead-card allowance.
// With `with_scores`, also fills each move's shaping score minus the
// hand value of the jack it spends (dead-card declarations score 0).
void gen_moves(const SState& s, const Params& P, SMoveList& out, bool with_scores) {
    SEQ_PROFILE_SCOPE("MCTS::gen_moves");
    const GameState& g = s.g;
    const int p   = g.current_player;
    const int opp = 1 - p;
    const BB100 occupied = g.chips[0] | g.chips[1] | g.locked[0] | g.locked[1];
    const int stage = stage_of(g.sequences[p], g.sequences[opp]);
    const int32_t (*place_tb)[N_WSTATES] = P.tb.place[stage][p];
    const int32_t* remove_tb = P.tb.remove[stage][p];

    // Hand support (learned scores only): the empty cells each slot's
    // regular card could fill.
    const bool hand_aware = with_scores && P.cfg.learned_policy;
    BB100 slot_cells[HAND_SIZE];
    if (hand_aware) {
        for (int i = 0; i < HAND_SIZE; ++i) {
            const int c = g.hands[p][i];
            if (c < 0 || c >= N_REGULAR_CARDS) continue;
            for (int k = 0; k < 2; ++k) {
                const int cc = CARD_TO_CELLS[c][k];
                if (cc >= 0 && !occupied.test(cc)) slot_cells[i].set(cc);
            }
        }
    }
    const uint8_t* sensitive = P.tb.hand_sensitive[stage][p];
    // Score of placing on `cell` with the card in `slot`, whose support
    // is every other slot's cells.
    BB100  support;
    int    support_slot = -1;
    int8_t support_cache[N_WINDOWS];
    auto place_score = [&](int slot, int cell, bool many_cells) -> int32_t {
        if (!hand_aware) return window_sum(s, place_tb[0], cell);
        if (slot != support_slot) {
            support = BB100{};
            for (int i = 0; i < HAND_SIZE; ++i) {
                if (i != slot) support |= slot_cells[i];
            }
            support_slot = slot;
            if (many_cells) std::memset(support_cache, -1, sizeof(support_cache));
        }
        if (support.empty()) return window_sum(s, place_tb[0], cell);
        return window_sum_supported(s, place_tb, sensitive, cell, support,
                                    many_cells ? support_cache : nullptr);
    };

    int n = 0;
    uint64_t seen = 0;              // card types already handled
    int dead_slot[HAND_SIZE];
    int n_dead = 0;

    for (int slot = 0; slot < HAND_SIZE; ++slot) {
        const int8_t card = g.hands[p][slot];
        if (card < 0) continue;
        const uint64_t bit = uint64_t(1) << card;
        if (seen & bit) continue;
        seen |= bit;

        const int before = n;
        if (card == TWO_EYED_JACK) {
            BB100 targets = ~occupied;
            targets &= ~JOKER_MASK;
            while (targets.any()) {
                const int cell = targets.pop_lsb();
                out.move[n] = Move{int8_t(slot), int8_t(cell)};
                out.key[n]  = uint16_t(KEY_J2 + cell);
                if (with_scores) out.score[n] = place_score(slot, cell, true) + P.tb.move_j2[stage];
                ++n;
            }
        } else if (card == ONE_EYED_JACK) {
            BB100 targets = g.chips[opp];
            while (targets.any()) {
                const int cell = targets.pop_lsb();
                out.move[n] = Move{int8_t(slot), int8_t(cell)};
                out.key[n]  = uint16_t(KEY_J1 + cell);
                if (with_scores) out.score[n] = window_sum(s, remove_tb, cell) + P.tb.move_j1[stage];
                ++n;
            }
        } else {
            for (int k = 0; k < 2; ++k) {
                const int cell = CARD_TO_CELLS[card][k];
                if (cell < 0 || occupied.test(cell)) continue;
                out.move[n] = Move{int8_t(slot), int8_t(cell)};
                out.key[n]  = uint16_t(cell);
                if (with_scores) out.score[n] = place_score(slot, cell, false);
                ++n;
            }
        }
        if (n == before) dead_slot[n_dead++] = slot;
    }

    if (n_dead > 0 && (!g.dead_card_used[p] || n == 0)) {
        for (int i = 0; i < n_dead; ++i) {
            const int slot = dead_slot[i];
            out.move[n]  = Move{int8_t(slot), Move::DEAD};
            out.key[n]   = uint16_t(KEY_DEAD + g.hands[p][slot]);
            out.score[n] = P.tb.move_dead[stage];
            ++n;
        }
    }
    out.n = n;
}

// ---------------------------------------------------------------------------
// Rollout
// ---------------------------------------------------------------------------
//
// ε-greedy on the shaping score: with probability ε a uniformly random
// move, otherwise the best-scoring one (first in generation order on
// ties). Pure greedy would make every rollout from a state identical.
//
// A rollout that is still running after `rollout_cutoff` plies stops and
// scores the position statically; one that reaches the end of the game
// returns ±1 · length_decay^plies.

inline int32_t hand_jack_value(const Tables& tb, const GameState& g, int p) {
    const int stage = stage_of(g.sequences[p], g.sequences[1 - p]);
    int32_t v = 0;
    for (int i = 0; i < HAND_SIZE; ++i) {
        const int c = g.hands[p][i];
        if (c == TWO_EYED_JACK)      v += tb.value_j2[stage];
        else if (c == ONE_EYED_JACK) v += tb.value_j1[stage];
    }
    return v;
}

inline double terminal_value_p0(const GameState& g) {
    if (g.winner == 0) return  1.0;
    if (g.winner == 1) return -1.0;
    return 0.0;
}

// Σ over the windows of the stage's value table (player 0's view).
inline int32_t window_value_fp(const SState& s, const Params& P) {
    const int32_t* tb = P.tb.value[stage_of(s.g.sequences[0], s.g.sequences[1])];
    int32_t sum = 0;
    for (int i = 0; i < N_WINDOWS; ++i) sum += tb[s.w[i]];
    return sum;
}

double static_value_p0(const SState& s, const Params& P) {
    const Tables& tb = P.tb;
    const GameState& g = s.g;
    const int32_t sum = window_value_fp(s, P)
                      + tb.value_bias[stage_of(g.sequences[0], g.sequences[1])]
                      + hand_jack_value(tb, g, 0)
                      - hand_jack_value(tb, g, 1)
                      + (g.current_player == 0 ? tb.value_tempo : -tb.value_tempo);
    return std::tanh(tb.value_scale * double(sum));
}

double rollout(SState& s, Xoshiro256pp& rng, const Params& P) {
    SEQ_PROFILE_SCOPE("MCTS::rollout");
    const MCTSConfig& cfg = P.cfg;
    SMoveList L;
    int plies = 0;
    for (; plies < P.max_plies && !s.g.done; ++plies) {
        int pick = 0;
        // Decide random-vs-greedy before generating, so a random ply
        // skips scoring altogether.
        const bool random_ply = (rng() >> 11) < P.eps_threshold;
        gen_moves(s, P, L, !random_ply);
        if (L.n == 0) break;
        if (L.n > 1) {
            if (random_ply) {
                pick = rand_below(rng, L.n);
                if (!cfg.random_ply_jacks) {
                    // Don't let a random ply throw a jack away: choose
                    // among the non-jack moves when there are any.
                    int plain[MAX_SMOVES];
                    int n_plain = 0;
                    for (int i = 0; i < L.n; ++i) {
                        if (L.key[i] < KEY_J2 || L.key[i] >= KEY_DEAD) plain[n_plain++] = i;
                    }
                    if (n_plain > 0) pick = plain[rand_below(rng, n_plain)];
                }
            } else {
                int32_t best = L.score[0];
                for (int i = 1; i < L.n; ++i) {
                    if (L.score[i] > best) { best = L.score[i]; pick = i; }
                }
            }
        }
        apply_move(s, P, L.move[pick], rng);
    }
    if (!s.g.done && cfg.rollout_cutoff > 0 && plies >= P.max_plies) {
        return static_value_p0(s, P);
    }
    const double terminal = terminal_value_p0(s.g);
    if (terminal == 0.0 || cfg.length_decay == 1.0) return terminal;
    return terminal * std::pow(cfg.length_decay, double(plies));
}

// ---------------------------------------------------------------------------
// Tree
// ---------------------------------------------------------------------------
//
// One tree, shared by every search thread, lock-free.
//
//   * Nodes live in a preallocated arena and are claimed with an atomic
//     bump index; they are never freed or moved during a search.
//   * A node's children form a singly linked list (first_child →
//     next_sibling). A new child is prepended with a compare-and-swap on
//     first_child, after its key and next_sibling are written, so a
//     reader walking the list always sees fully-formed nodes. If two
//     threads race to add the same move, the loser finds the winner's
//     node in the entries that appeared ahead of it and uses that.
//   * Statistics are atomics updated with relaxed fetch_add.
//   * Virtual loss: while a thread is below a node, the node's `vl`
//     counter makes it look like one extra lost visit to the others, so
//     simultaneous descents spread out instead of piling onto one line.
//     Single-threaded searches skip it.
//
// Index 0 is the root, and since the root is nobody's child, 0 doubles
// as "none" for first_child / next_sibling — a zeroed arena is a valid
// tree of one empty root.

constexpr int64_t VAL_ONE = int64_t(1) << 24;   // fixed-point 1.0 for value sums

struct Node {
    std::atomic<int32_t> visits{0};
    std::atomic<int32_t> vl{0};             // in-flight descents below here
    std::atomic<int64_t> value{0};          // Σ value (player 0's view) · VAL_ONE
    std::atomic<int32_t> first_child{0};
    int32_t  next_sibling = 0;
    uint16_t key = 0;
};

} // namespace

struct MCTSEngine::Tree {
    std::unique_ptr<Node[]> nodes;
    int32_t                 capacity = 0;
    std::atomic<int32_t>    used{0};

    void allocate(int32_t cap) {
        nodes.reset(new Node[size_t(cap)]());
        capacity = cap;
        used.store(1, std::memory_order_relaxed);    // the root
    }
};

namespace {

// Deepest path we track. A descent that somehow gets this deep just
// rolls out from there.
constexpr int MAX_PATH = 256;

// One ISMCTS iteration: determinize, descend with PUCT (materializing at
// most one new node), roll out, back up. Returns the depth reached.
int iteration(Node* nodes, int32_t capacity, std::atomic<int32_t>& used,
              SState s, Xoshiro256pp& rng, const Params& P)
{
    SEQ_PROFILE_SCOPE("MCTS::ismcts_iteration");
    const MCTSConfig& cfg = P.cfg;

    if (cfg.determinize) determinize(s, s.g.current_player, rng);

    int32_t path[MAX_PATH];
    int     path_len = 0;
    path[path_len++] = 0;
    int32_t node_idx = 0;

    SMoveList L;
    while (!s.g.done && path_len < MAX_PATH) {
        gen_moves(s, P, L, true);
        if (L.n == 0) break;

        SEQ_PROFILE_SCOPE("MCTS::select_puct");
        const int player = s.g.current_player;
        Node& node = nodes[node_idx];

        // Match existing children to this determinization's legal moves:
        // mark the legal keys, then walk the child list once.
        uint64_t legal_bits[N_KEYS / 64] = {};
        uint8_t  idx_of_key[N_KEYS];
        int32_t  child[MAX_SMOVES];
        double   child_n[MAX_SMOVES];
        double   child_q[MAX_SMOVES];       // mover's perspective
        for (int i = 0; i < L.n; ++i) {
            const uint16_t k = L.key[i];
            legal_bits[k >> 6] |= uint64_t(1) << (k & 63);
            idx_of_key[k] = uint8_t(i);
            child[i] = 0;
        }

        const int32_t head = node.first_child.load(std::memory_order_acquire);
        const double  sign = (player == 0) ? 1.0 : -1.0;
        double sum_visits = 0.0;
        for (int32_t c = head; c != 0; c = nodes[c].next_sibling) {
            const Node& cn = nodes[c];
            const uint16_t k = cn.key;
            if (!((legal_bits[k >> 6] >> (k & 63)) & 1)) continue;
            const int i = idx_of_key[k];
            const int32_t v  = cn.visits.load(std::memory_order_relaxed);
            const int32_t vl = P.use_vl ? cn.vl.load(std::memory_order_relaxed) : 0;
            const double  val = double(cn.value.load(std::memory_order_relaxed))
                              * (1.0 / double(VAL_ONE));
            const double  n = double(v + vl);
            child[i]   = c;
            child_n[i] = n;
            // In-flight descents count as losses for the mover.
            child_q[i] = (sign * val - double(vl)) / std::max(1.0, n);
            sum_visits += n;
        }

        // Softmax prior over the legal moves' scores.
        int32_t max_s = L.score[0];
        for (int i = 1; i < L.n; ++i) max_s = std::max(max_s, L.score[i]);
        double prior[MAX_SMOVES];
        double sum_exp = 0.0;
        for (int i = 0; i < L.n; ++i) {
            prior[i] = std::exp(double(L.score[i] - max_s) * (1.0 / FP));
            sum_exp += prior[i];
        }

        // First-play urgency: an unvisited move is worth the node's own
        // mean (mover's view) minus a small reduction. The +1 under the
        // root lets the very first selection at a node follow the prior.
        const int32_t nv = node.visits.load(std::memory_order_relaxed);
        const double parent_q = nv > 0
            ? sign * double(node.value.load(std::memory_order_relaxed))
                   / (double(VAL_ONE) * double(nv))
            : 0.0;
        const double fpu_q   = parent_q - cfg.fpu_reduction;
        const double explore = cfg.ucb_c * std::sqrt(sum_visits + 1.0) / sum_exp;

        int    best_i = -1;
        double best_score = -std::numeric_limits<double>::infinity();
        int    n_tied = 0;
        for (int i = 0; i < L.n; ++i) {
            double q, n;
            if (child[i] != 0 && child_n[i] > 0.0) { q = child_q[i]; n = child_n[i]; }
            else                                   { q = fpu_q;      n = 0.0; }
            const double score = q + explore * prior[i] / (1.0 + n);
            if (score > best_score + 1e-12) {
                best_score = score;
                best_i     = i;
                n_tied     = 1;
            } else if (score > best_score - 1e-12) {
                // Uniform among ties (reservoir) — without it structural
                // ties resolve by generation order.
                ++n_tied;
                if (rand_below(rng, n_tied) == 0) best_i = i;
            }
        }
        if (best_i < 0) break;

        int32_t next = child[best_i];
        const bool expanded = (next == 0);
        if (expanded) {
            SEQ_PROFILE_SCOPE("MCTS::expand");
            const uint16_t key = L.key[best_i];
            const int32_t fresh = used.fetch_add(1, std::memory_order_relaxed);
            if (fresh >= capacity) {
                // Arena exhausted (cannot happen with the sizing in
                // suggest_move, but stay safe): roll out without a node.
                apply_move(s, P, L.move[best_i], rng);
                break;
            }
            nodes[fresh].key = key;
            next = fresh;
            int32_t scanned = head;      // list entries we've already checked
            int32_t expect  = head;
            for (;;) {
                nodes[fresh].next_sibling = expect;
                if (node.first_child.compare_exchange_weak(
                        expect, fresh,
                        std::memory_order_release, std::memory_order_acquire)) {
                    break;
                }
                // Someone prepended since we looked. If one of the new
                // entries is this same move, use it and abandon ours.
                int32_t found = 0;
                for (int32_t x = expect; x != scanned; x = nodes[x].next_sibling) {
                    if (nodes[x].key == key) { found = x; break; }
                }
                if (found != 0) { next = found; break; }
                scanned = expect;
            }
        }

        apply_move(s, P, L.move[best_i], rng);
        if (P.use_vl) nodes[next].vl.fetch_add(1, std::memory_order_relaxed);
        node_idx = next;
        path[path_len++] = node_idx;
        if (expanded) break;             // roll out from the fresh leaf
    }

    const double value_p0 = s.g.done ? terminal_value_p0(s.g)
                                     : rollout(s, rng, P);

    {
        SEQ_PROFILE_SCOPE("MCTS::backup");
        const int64_t v = std::llround(value_p0 * double(VAL_ONE));
        for (int i = 0; i < path_len; ++i) {
            Node& n = nodes[path[i]];
            n.value.fetch_add(v, std::memory_order_relaxed);
            n.visits.fetch_add(1, std::memory_order_relaxed);
            if (P.use_vl && i > 0) n.vl.fetch_sub(1, std::memory_order_relaxed);
        }
    }
    return path_len - 1;
}

inline int32_t find_child(const Node* nodes, int32_t parent, uint16_t key) {
    for (int32_t c = nodes[parent].first_child.load(std::memory_order_relaxed);
         c != 0; c = nodes[c].next_sibling) {
        if (nodes[c].key == key) return c;
    }
    return 0;
}

} // namespace

// ---------------------------------------------------------------------------
// Self-check
// ---------------------------------------------------------------------------
//
// The search has its own fast copies of things that live elsewhere: move
// generation and make_move (state.cpp) and the shaping score and Φ
// (eval.hpp). This plays random games and checks, at every ply, that
// the fast versions agree with the reference ones.

int search_selfcheck(int n_games, uint64_t seed) {
    int bad = 0;
    Xoshiro256pp rng(seed);
    MCTSConfig cfg;             // reference scoring: eval.hpp, nothing added
    cfg.dead_windows        = false;
    cfg.shaping_seq_bonus   = 0.0;
    cfg.learned_policy      = false;
    cfg.learned_value       = false;
    const Params P(cfg, false);

    for (int game = 0; game < n_games; ++game) {
        GameState ref;
        ref.reset(rng());
        SState inc;                 // window bytes maintained incrementally
        inc.g = ref;
        init_windows(inc, P);

        for (int ply = 0; ply < 2000 && !ref.done; ++ply) {
            // Window bytes: incremental == recomputed from the bitboards.
            SState chk;
            chk.g = ref;
            init_windows(chk, P);
            if (std::memcmp(chk.w, inc.w, sizeof(chk.w)) != 0) ++bad;

            // Move generation: same set of moves as GameState's.
            MoveList ml;
            ref.legal_moves(ml);
            SMoveList sl;
            gen_moves(inc, P, sl, true);
            if (sl.n != ml.count) ++bad;
            const int p = ref.current_player;
            for (int i = 0; i < sl.n && i < ml.count; ++i) {
                bool found = false;
                for (int k = 0; k < ml.count; ++k) found |= (ml.moves[k] == sl.move[i]);
                if (!found) ++bad;

                // Scores: fixed-point == eval.hpp's doubles.
                const Move m = sl.move[i];
                const int card = ref.hands[p][m.card_idx];
                if (sl.key[i] != key_of(card, m.cell)) ++bad;
                double want = 0.0;
                if (!m.is_dead()) {
                    want = shaping_score(ref, p, card, m.cell);
                    if (card == TWO_EYED_JACK) want -= cfg.jack2_value;
                    if (card == ONE_EYED_JACK) want -= cfg.jack1_value;
                }
                if (std::fabs(want - double(sl.score[i]) / FP) > 1e-9) ++bad;
            }

            // Φ difference.
            if (std::fabs(double(window_value_fp(inc, P)) / FP - phi_diff_p0(ref)) > 1e-9) ++bad;

            if (ml.count == 0) break;
            const Move m = ml.moves[rand_below(rng, ml.count)];

            // apply_move vs make_move: same board, sequences and turn.
            // (Draws differ by design — the search draws at random — so
            // hands and piles are re-synced from the reference state.)
            apply_move(inc, P, m, rng);
            if (!ref.make_move(m)) { ++bad; break; }
            for (int q = 0; q < 2; ++q) {
                if (inc.g.chips[q] != ref.chips[q] || inc.g.locked[q] != ref.locked[q]
                    || inc.g.sequences[q] != ref.sequences[q]
                    || inc.g.dead_card_used[q] != ref.dead_card_used[q]) ++bad;
            }
            if (inc.g.current_player != ref.current_player || inc.g.done != ref.done
                || inc.g.winner != ref.winner) ++bad;
            inc.g = ref;
        }
    }
    return bad;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

MCTSEngine::MCTSEngine(MCTSConfig cfg) : cfg_(cfg) {
    if (cfg_.seed == 0) {
        std::random_device rd;
        uint64_t s = (uint64_t(rd()) << 32) ^ rd();
        rng_.seed(s);
    } else {
        rng_.seed(cfg_.seed);
    }
}

// Out-of-line where Tree is complete.
MCTSEngine::~MCTSEngine() = default;
MCTSEngine::MCTSEngine(MCTSEngine&&) noexcept = default;
MCTSEngine& MCTSEngine::operator=(MCTSEngine&&) noexcept = default;

void MCTSEngine::advance(const GameState& before, Move m) {
    if (m.card_idx < 0 || m.card_idx >= HAND_SIZE) return;
    advance_card(before.hands[before.current_player][m.card_idx], m.cell);
}

void MCTSEngine::advance_card(int card_type, int cell) {
    if (!cfg_.tree_reuse) return;
    if (card_type < 0 || card_type >= N_CARD_TYPES || cell >= N_CELLS) {
        // Not a move we can follow: forget the tree.
        tree_.reset();
        pending_.clear();
        return;
    }
    pending_.push_back(key_of(card_type, cell));
}

// Get the tree ready for a search of `budget` iterations: re-root it
// along the moves played since the last search (keeping that subtree's
// statistics), or start from an empty root if there is nothing to keep.
// Returns the number of visits carried over at the new root.
int MCTSEngine::prepare_tree(int budget) {
    const int32_t slack = 64;       // nodes lost to expansion races
    std::vector<int32_t> order;     // kept nodes (old indices), BFS order

    if (cfg_.tree_reuse && tree_ && !pending_.empty()) {
        const Node* old = tree_->nodes.get();
        int32_t idx = 0;
        for (uint16_t key : pending_) {
            idx = find_child(old, idx, key);
            if (idx == 0) break;
        }
        if (idx != 0 && old[idx].first_child.load(std::memory_order_relaxed) != 0) {
            order.push_back(idx);
            for (size_t i = 0; i < order.size(); ++i) {
                for (int32_t c = old[order[i]].first_child.load(std::memory_order_relaxed);
                     c != 0; c = old[c].next_sibling) {
                    order.push_back(c);
                }
            }
        }
    }
    pending_.clear();

    auto fresh = std::make_unique<Tree>();
    fresh->allocate(int32_t(order.size()) + budget + slack);
    int carried = 0;
    if (!order.empty()) {
        // Copy in BFS order: a node's children land contiguously, in
        // their old list order, so the child lists stay intact and
        // siblings end up adjacent in memory.
        const Node* old = tree_->nodes.get();
        Node* nn = fresh->nodes.get();
        int32_t next = 1;
        for (size_t i = 0; i < order.size(); ++i) {
            const Node& o = old[order[i]];
            Node& n = nn[i];
            n.visits.store(o.visits.load(std::memory_order_relaxed), std::memory_order_relaxed);
            n.value.store(o.value.load(std::memory_order_relaxed), std::memory_order_relaxed);
            n.key = (i == 0) ? uint16_t(0) : o.key;
            int32_t first = 0, prev = 0;
            for (int32_t c = o.first_child.load(std::memory_order_relaxed);
                 c != 0; c = old[c].next_sibling) {
                const int32_t ci = next++;
                if (first == 0) first = ci; else nn[prev].next_sibling = ci;
                prev = ci;
            }
            n.first_child.store(first, std::memory_order_relaxed);
        }
        fresh->used.store(int32_t(order.size()), std::memory_order_relaxed);
        carried = nn[0].visits.load(std::memory_order_relaxed);
    }
    tree_ = std::move(fresh);
    return carried;
}

Move MCTSEngine::suggest_move(const GameState& s_in) {
    SEQ_PROFILE_SCOPE("MCTS::suggest_move");
    using clk = std::chrono::high_resolution_clock;
    const auto t0 = clk::now();

    stats_ = {};
    root_moves_.clear();

    // Worker count: explicit, or every hardware thread; never more
    // threads than there is work for.
    int n_threads = cfg_.n_threads;
    if (n_threads <= 0) {
        const unsigned hc = std::thread::hardware_concurrency();
        n_threads = hc == 0 ? 1 : int(hc);
    }
    const int budget = std::max(1, cfg_.iterations);
    n_threads = std::max(1, std::min({n_threads, 64, budget / 64 + 1}));

    const Params P(cfg_, n_threads > 1);

    SState root;
    root.g = s_in;
    init_windows(root, P);

    // Our own legal moves don't depend on the opponent's hand, so read
    // them off the un-determinized state.
    SMoveList root_moves;
    gen_moves(root, P, root_moves, false);
    stats_.root_legal_moves = root_moves.n;
    if (root_moves.n == 0) { stats_.best_move = Move{}; return Move{}; }
    if (root_moves.n == 1) {
        stats_.best_move        = root_moves.move[0];
        stats_.best_move_visits = 1;
        stats_.seconds = std::chrono::duration<double>(clk::now() - t0).count();
        return stats_.best_move;
    }

    stats_.reused_visits = prepare_tree(budget);
    Tree& tree = *tree_;
    Node* nodes = tree.nodes.get();

    std::atomic<int> claimed{0};
    auto work = [&](uint64_t seed, int* max_depth) {
        Xoshiro256pp rng(seed);
        int deepest = 0;
        while (claimed.fetch_add(1, std::memory_order_relaxed) < budget) {
            const int d = iteration(nodes, tree.capacity, tree.used, root, rng, P);
            if (d > deepest) deepest = d;
        }
        *max_depth = deepest;
    };

    std::vector<uint64_t> seeds(static_cast<size_t>(n_threads));
    for (auto& s : seeds) s = rng_();
    std::vector<int> depths(static_cast<size_t>(n_threads), 0);
    if (n_threads == 1) {
        work(seeds[0], &depths[0]);
    } else {
        std::vector<std::thread> threads;
        threads.reserve(size_t(n_threads) - 1);
        for (int t = 1; t < n_threads; ++t) {
            threads.emplace_back(work, seeds[size_t(t)], &depths[size_t(t)]);
        }
        work(seeds[0], &depths[0]);
        for (auto& th : threads) th.join();
    }
    // Racing threads can over-claim arena slots; clamp for the next reroot.
    if (tree.used.load(std::memory_order_relaxed) > tree.capacity) {
        tree.used.store(tree.capacity, std::memory_order_relaxed);
    }

    // Robust child: the legal root move with the most visits, ties
    // broken at random.
    const double sign = (s_in.current_player == 0) ? 1.0 : -1.0;
    int     best_i = 0, n_tied = 0;
    int32_t best_v = -1;
    root_moves_.reserve(size_t(root_moves.n));
    for (int i = 0; i < root_moves.n; ++i) {
        const int32_t c = find_child(nodes, 0, root_moves.key[i]);
        const int32_t v = c ? nodes[c].visits.load(std::memory_order_relaxed) : 0;
        const double mean = v > 0
            ? sign * double(nodes[c].value.load(std::memory_order_relaxed))
                   / (double(VAL_ONE) * double(v))
            : 0.0;
        root_moves_.push_back(RootMoveStat{
            root_moves.move[i], v, 0.5 * (std::max(-1.0, std::min(1.0, mean)) + 1.0)});
        if (v > best_v) {
            best_v = v; best_i = i; n_tied = 1;
        } else if (v == best_v) {
            ++n_tied;
            if (rng_.rand_int(0, n_tied - 1) == 0) best_i = i;
        }
    }

    stats_.iterations        = budget;
    stats_.max_depth         = *std::max_element(depths.begin(), depths.end());
    stats_.best_move         = root_moves.move[best_i];
    stats_.best_move_visits  = best_v;
    stats_.best_move_winrate = root_moves_[size_t(best_i)].winrate;
    {
        const int32_t rv = nodes[0].visits.load(std::memory_order_relaxed);
        const double mean = rv > 0
            ? sign * double(nodes[0].value.load(std::memory_order_relaxed))
                   / (double(VAL_ONE) * double(rv))
            : 0.0;
        stats_.root_winrate = 0.5 * (std::max(-1.0, std::min(1.0, mean)) + 1.0);
    }
    stats_.threads           = n_threads;
    stats_.seconds = std::chrono::duration<double>(clk::now() - t0).count();
    return stats_.best_move;
}

} // namespace seq

#include "state.hpp"
#include "profile.hpp"
#include <algorithm>
#include <cassert>

namespace seq {

// ---------------------------------------------------------------------------
// Deck construction
// ---------------------------------------------------------------------------
//
// TWO full decks combined, plus jacks.
// Each "deck" contributes 4 suits * 12 ranks = 48 regular cards, plus
// 2 one-eyed jacks (spades, hearts) and 2 two-eyed jacks (diamonds, clubs).
// Total per deck = 52, x2 = 104 cards.

static void build_deck(CardPile& d, Xoshiro256pp& rng) {
    d.clear();
    for (int copy = 0; copy < 2; ++copy) {
        for (int card = 0; card < N_REGULAR_CARDS; ++card) {
            d.push_back(int8_t(card));
        }
        // 2 one-eyed jacks (spades + hearts) and 2 two-eyed jacks
        // (diamonds + clubs) per deck.
        d.push_back(int8_t(ONE_EYED_JACK));
        d.push_back(int8_t(ONE_EYED_JACK));
        d.push_back(int8_t(TWO_EYED_JACK));
        d.push_back(int8_t(TWO_EYED_JACK));
    }
    std::shuffle(d.begin(), d.end(), rng);
}

// ---------------------------------------------------------------------------
// Construction / reset
// ---------------------------------------------------------------------------

GameState::GameState() {
    reset();
}

void GameState::reset(uint64_t seed) {
    rng.seed(seed);
    chips[0] = chips[1] = BB100{};
    locked[0] = locked[1] = BB100{};
    for (int p = 0; p < 2; ++p)
        for (int i = 0; i < HAND_SIZE; ++i)
            hands[p][i] = -1;
    discard.clear();
    build_deck(deck, rng);
    for (int p = 0; p < 2; ++p) {
        for (int i = 0; i < HAND_SIZE; ++i) {
            hands[p][i] = draw();
        }
    }
    sequences[0] = sequences[1] = 0;
    current_player = 0;
    done = false;
    winner = -1;
    dead_card_used[0] = dead_card_used[1] = false;
}

int8_t GameState::draw() {
    if (deck.empty()) {
        // Reshuffle the discard pile back into the deck. Mirrors the
        // Python engine; very rare in practice since the 104-card deck
        // is larger than the longest reasonable game.
        if (discard.empty()) return -1;
        deck = discard;          // CardPile is trivially copyable
        discard.clear();
        std::shuffle(deck.begin(), deck.end(), rng);
    }
    int8_t c = deck.back();
    deck.pop_back();
    return c;
}

// ---------------------------------------------------------------------------
// Cell inspection
// ---------------------------------------------------------------------------

Chip GameState::cell_chip(int cell) const {
    if (locked[0].test(cell)) return SEQUENCE0;
    if (locked[1].test(cell)) return SEQUENCE1;
    if (chips[0].test(cell))  return PLAYER0;
    if (chips[1].test(cell))  return PLAYER1;
    return EMPTY;
}

int GameState::board_chip_count() const {
    return chips[0].popcount() + chips[1].popcount()
         + locked[0].popcount() + locked[1].popcount();
}

// ---------------------------------------------------------------------------
// Dead-card check
// ---------------------------------------------------------------------------

bool GameState::is_card_dead(int8_t card, int player) const {
    int opp = 1 - player;
    BB100 occupied = chips[0] | chips[1] | locked[0] | locked[1];

    if (card == ONE_EYED_JACK) {
        // Dead iff no legal target: opponent has no unlocked chips on the board.
        return chips[opp].empty();
    }
    if (card == TWO_EYED_JACK) {
        // Dead iff every non-joker cell is occupied. Joker corners are
        // never legal placements (they're already "everyone's chip"), so
        // exclude them from the "playable" universe.
        BB100 playable = ~JOKER_MASK & ~occupied;
        // Mask off bits above 100.
        playable.hi &= BB100::HI_MASK;
        return playable.empty();
    }
    // Regular card: dead iff both its cells are occupied.
    int8_t a = CARD_TO_CELLS[card][0];
    int8_t b = CARD_TO_CELLS[card][1];
    bool a_dead = (a < 0) || occupied.test(a);
    bool b_dead = (b < 0) || occupied.test(b);
    return a_dead && b_dead;
}

// ---------------------------------------------------------------------------
// Move generation
// ---------------------------------------------------------------------------

void GameState::legal_moves_for(int player, MoveList& out) const {
    SEQ_PROFILE_SCOPE("state::legal_moves");
    out.clear();
    if (done) return;

    int opp = 1 - player;
    BB100 occupied = chips[0] | chips[1] | locked[0] | locked[1];

    // Phase 1 — collect real placements per card. We track which slots
    // contributed nothing (dead cards) for the dead-card pass below.
    bool slot_dead[HAND_SIZE];
    int  real_moves_before = out.count;

    for (int slot = 0; slot < HAND_SIZE; ++slot) {
        int8_t card = hands[player][slot];
        if (card < 0) { slot_dead[slot] = false; continue; }

        int before = out.count;

        if (card == TWO_EYED_JACK) {
            // Every empty non-joker cell.
            BB100 targets = ~occupied;
            targets &= ~JOKER_MASK;
            targets.hi &= BB100::HI_MASK;
            while (targets.any()) {
                int cell = targets.pop_lsb();
                out.push(Move{int8_t(slot), int8_t(cell)});
            }
        } else if (card == ONE_EYED_JACK) {
            // Every opponent unlocked chip (locked chips can't be removed).
            BB100 targets = chips[opp];
            while (targets.any()) {
                int cell = targets.pop_lsb();
                out.push(Move{int8_t(slot), int8_t(cell)});
            }
        } else {
            // Regular card: at most two board cells.
            int8_t a = CARD_TO_CELLS[card][0];
            int8_t b = CARD_TO_CELLS[card][1];
            if (a >= 0 && !occupied.test(a)) out.push(Move{int8_t(slot), a});
            if (b >= 0 && !occupied.test(b)) out.push(Move{int8_t(slot), b});
        }

        slot_dead[slot] = (out.count == before);
    }

    bool has_real_moves = (out.count > real_moves_before);

    // Phase 2 — append dead-card declarations subject to the per-turn
    // swap allowance.
    //   * if the player still has their per-turn swap, every actually-dead
    //     card in hand may be declared (in addition to real moves);
    //   * if they've used it, declarations come back only as a deadlock
    //     fallback when no real moves exist.
    bool allow_dead =
        !dead_card_used[player] || !has_real_moves;

    if (allow_dead) {
        for (int slot = 0; slot < HAND_SIZE; ++slot) {
            int8_t card = hands[player][slot];
            if (card < 0) continue;
            if (slot_dead[slot]) {
                out.push(Move{int8_t(slot), Move::DEAD});
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Sequence detection
// ---------------------------------------------------------------------------

int GameState::check_sequences(int player, int cell) {
    SEQ_PROFILE_SCOPE("state::check_sequences");
    if (cell < 0) return 0;

    // Only valid when the cell I'm checking around belongs to me — either
    // unlocked or already locked. A one-eyed-jack removal that emptied the
    // cell skips here; the Python engine has the same guard.
    if (!chips[player].test(cell) && !locked[player].test(cell)) {
        return 0;
    }

    int opp = 1 - player;
    BB100 my_full     = chips[player] | locked[player];
    BB100 my_or_joker = my_full | JOKER_MASK;
    BB100 opp_full    = chips[opp] | locked[opp];

    int new_sequences = 0;

    // Iterate the (≤20) windows passing through this cell, in the order
    // emitted by the codegen — which mirrors the Python iteration order
    // (direction-major, then offset). Order matters: when a window
    // completes here, we lock its non-joker cells immediately, and a
    // *later* candidate window in the same call can be disqualified by
    // the "share at most one chip with a previous sequence" rule.
    const int16_t* ws = CELL_WINDOWS[cell];
    for (int i = 0; ws[i] >= 0; ++i) {
        BB100 win = WINDOW_MASKS[ws[i]];

        // Window blocked if opponent has any chip in it (the all-mine-or-
        // joker test below also catches this, but checking opp_full first
        // is the natural Python parallel).
        if ((win & opp_full).any()) continue;

        // All 5 cells must be mine-or-joker.
        if (!my_or_joker.contains(win)) continue;

        // "Share at most one chip with a previous sequence": count
        // already-locked non-joker cells in the window.
        BB100 locked_nonjoker_in_win = locked[player] & win;
        // Joker cells aren't in locked[player], so no separate ~joker mask
        // is needed — but be explicit for clarity if maintained:
        // locked_nonjoker_in_win &= ~JOKER_MASK;
        if (locked_nonjoker_in_win.popcount() > 1) continue;

        // New sequence. Lock all non-joker cells in this window.
        BB100 to_lock = win & ~JOKER_MASK;
        to_lock.hi &= BB100::HI_MASK;
        locked[player] |= to_lock;
        chips[player]  &= ~to_lock;
        chips[player].hi &= BB100::HI_MASK;

        ++new_sequences;
    }

    return new_sequences;
}

// ---------------------------------------------------------------------------
// make_move
// ---------------------------------------------------------------------------

bool GameState::make_move(Move m) {
    SEQ_PROFILE_SCOPE("state::make_move");
    if (done) return false;

    int player = current_player;
    int opp = 1 - player;

    if (m.card_idx < 0 || m.card_idx >= HAND_SIZE) return false;
    int8_t card = hands[player][m.card_idx];
    if (card < 0) return false;

    // --- Dead-card declaration (free swap, same player keeps the turn) ---
    if (m.is_dead()) {
        discard.push_back(card);
        hands[player][m.card_idx] = draw();
        dead_card_used[player] = true;
        return true;
    }

    if (m.cell < 0 || m.cell >= N_CELLS) return false;
    int cell = m.cell;

    // Validate placement.
    BB100 occupied = chips[0] | chips[1] | locked[0] | locked[1];

    if (card == TWO_EYED_JACK) {
        if (JOKER_MASK.test(cell)) return false;
        if (occupied.test(cell))   return false;
        chips[player].set(cell);
    } else if (card == ONE_EYED_JACK) {
        // Must be on an opponent *unlocked* chip — locked is immune.
        if (!chips[opp].test(cell)) return false;
        chips[opp].reset(cell);
    } else {
        // Regular card — cell must be one of its two board positions, empty.
        int8_t a = CARD_TO_CELLS[card][0];
        int8_t b = CARD_TO_CELLS[card][1];
        if (cell != a && cell != b) return false;
        if (occupied.test(cell)) return false;
        chips[player].set(cell);
    }

    // Detect (and lock) any new sequences passing through `cell`.
    // One-eyed jacks only remove an opponent chip, so the cell can't
    // belong to `player` afterward — check_sequences' own guard would
    // bail out. Skip the call (and its profile scope) outright.
    if (card != ONE_EYED_JACK) {
        int new_seqs = check_sequences(player, cell);
        sequences[player] += new_seqs;
    }

    // Discard the played card and draw a replacement.
    discard.push_back(card);
    hands[player][m.card_idx] = draw();

    // Win check.
    if (sequences[player] >= SEQUENCES_TO_WIN) {
        done = true;
        winner = player;
        return true;
    }

    // Real move ends the turn — reset the swap allowance for next time.
    dead_card_used[player] = false;
    current_player = opp;
    return true;
}

} // namespace seq

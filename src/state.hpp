// Sequence game state.
//
// The representation uses bitboards
// for chips (unlocked) and locked sequence chips, one per player, so that
// move generation, sequence detection, and (later) window-based eval are
// all expressible as bitwise ops.
//
// Move convention:
//   Move{card_idx, cell}     - play card from hand[card_idx] onto cell 0..99
//   Move{card_idx, DEAD=-1}  - declare card_idx as dead, swap for a new card,
//                              same player acts again

#pragma once
#include "types.hpp"
#include "board_data.hpp"
#include "rng.hpp"
#include <array>
#include <cstdint>
#include <cassert>

namespace seq {

struct Move {
    int8_t card_idx;
    int8_t cell;

    static constexpr int8_t DEAD = -1;

    bool is_dead() const { return cell == DEAD; }
    bool operator==(Move o) const { return card_idx == o.card_idx && cell == o.cell; }
};

// Two full 52-card decks make 104 cards total — the worst-case bound on
// either the draw deck or the discard pile (every card on the board, in
// hands, in deck, or in discard simultaneously sums to 104).
constexpr int DECK_SIZE = 104;

// LIFO stack of card-type indices used for both the draw deck and the
// discard pile. Replaces std::vector so GameState is trivially copyable
// — per-iteration `state = s_in` becomes a memcpy with no heap traffic,
// which is the dominant cost saving on the ISMCTS hot path.
struct CardPile {
    int8_t data[DECK_SIZE] = {};
    int    count = 0;

    void   push_back(int8_t c) { assert(count < DECK_SIZE); data[count++] = c; }
    void   pop_back()          { assert(count > 0); --count; }
    int8_t back() const        { assert(count > 0); return data[count - 1]; }
    bool   empty() const       { return count == 0; }
    int    size()  const       { return count; }
    void   clear()             { count = 0; }
    int8_t* begin()            { return data; }
    int8_t* end()              { return data + count; }
};

// Worst-case bound on legal moves at one state:
//   - 4 two-eyed jacks × 96 empty non-joker cells              = 384
//   - 4 one-eyed jacks × (≤30 opp chips on the board)          = 120
//   - 7 hand slots × 2 board positions for a regular card      =  14
//   - 7 dead-card declarations                                 =   7
// → ~525. 768 leaves comfortable headroom; the array is bytes so the
// over-allocation is cheap (~1.5 KB per MoveList).
constexpr int MAX_MOVES = 768;

struct MoveList {
    Move moves[MAX_MOVES];
    int count = 0;

    void push(Move m) {
        // Cheap belt-and-braces — if move gen ever overshoots the bound
        // we'd rather know loudly than trash the stack.
        assert(count < MAX_MOVES);
        moves[count++] = m;
    }
    void clear()      { count = 0; }
    Move operator[](int i) const { return moves[i]; }
    int  size() const { return count; }
};

struct GameState {
    // Bitboards. `chips[p]` is p's unlocked chips; `locked[p]` is p's chips
    // that are part of a completed sequence (immune to one-eyed jacks).
    // Joker corners never appear in any of these — they're a separate mask
    // that counts as "anyone's chip" for sequence purposes.
    BB100 chips[2]   = {};
    BB100 locked[2]  = {};

    // Hands: card-type indices into the 0..49 space (-1 = empty slot)
    int8_t hands[2][HAND_SIZE] = {};

    // Deck and discard, drawn from the back (deck.pop_back()).
    CardPile deck;
    CardPile discard;

    int sequences[2]      = {0, 0};
    int current_player    = 0;
    bool done             = false;
    int winner            = -1;  // -1 if no winner yet
    bool dead_card_used[2] = {false, false};

    Xoshiro256pp rng;

    GameState();

    // Fresh game with a given seed.
    void reset(uint64_t seed = 0xC0FFEE);

    // Fill `out` with the legal moves available to the side to move.
    void legal_moves(MoveList& out) const { legal_moves_for(current_player, out); }
    void legal_moves_for(int player, MoveList& out) const;

    // Apply a move. Returns false if the move is malformed / illegal —
    // callers should only pass moves emitted by legal_moves().
    bool make_move(Move m);

    // Inspect the chip state at a given cell (for tests / rendering).
    Chip cell_chip(int cell) const;

    // True if `card` (a card-type index, 0..49) has no legal placement
    // given the current board. Used during move gen.
    bool is_card_dead(int8_t card, int player) const;

    // Total chips on the board (for sanity tests).
    int board_chip_count() const;

    // Lock newly-formed sequences passing through `cell`. Returns count.
    // Public because it's a useful engine primitive (search rollouts) and
    // it lets tests drive sequence detection without going through the
    // hand/deck plumbing.
    int check_sequences(int player, int cell);

private:
    // Draw one card from the deck, reshuffling the discard if needed.
    // Returns -1 if both piles are exhausted.
    int8_t draw();
};

} // namespace seq

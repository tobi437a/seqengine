// Test harness for the Sequence engine.
//
// Three layers of confidence-building:
//   1. Sequence detection on hand-built positions — covers the trickiest
//      bit of game logic (windows through a cell, "share at most 1 chip"
//      with prior sequences, joker corners as wildcards).
//   2. Move-gen sanity at game start.
//   3. Random self-play soak: 1000 games at random with a fresh seed each,
//      checking no crashes / illegal-move attempts / runaway games.
//
// Built with bare asserts and a count of passing/failing tests; no
// framework dependency yet. Easy to swap to Catch2 later if it grows.

#include "../src/state.hpp"
#include "../src/board_data.hpp"
#include "../src/search.hpp"
#include "../src/rng.hpp"
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <random>

using namespace seq;

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond) do { \
    if (cond) { ++g_pass; } \
    else { ++g_fail; std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); } \
} while (0)

#define SECTION(name) std::printf("\n== %s ==\n", name)

// ---------------------------------------------------------------------------
// 1. Sequence detection
// ---------------------------------------------------------------------------

// Reset to an empty board (no chips, no hands needed) so we can drive
// sequence detection in isolation.
static void clear_board(GameState& s) {
    s.chips[0] = s.chips[1] = BB100{};
    s.locked[0] = s.locked[1] = BB100{};
    s.sequences[0] = s.sequences[1] = 0;
}

static void test_horizontal_sequence() {
    SECTION("horizontal sequence in row 5, cols 1..5");
    GameState s;
    clear_board(s);
    // Place player-0 chips at (5,1), (5,2), (5,3), (5,4), then the
    // "last chip" at (5,5) completes the run.
    s.chips[0].set(cell_of(5, 1));
    s.chips[0].set(cell_of(5, 2));
    s.chips[0].set(cell_of(5, 3));
    s.chips[0].set(cell_of(5, 4));
    s.chips[0].set(cell_of(5, 5));
    int seqs = s.check_sequences(0, cell_of(5, 5));
    CHECK(seqs == 1);
    // All five non-joker cells should now be locked, not unlocked.
    for (int c = 1; c <= 5; ++c) {
        CHECK(s.cell_chip(cell_of(5, c)) == SEQUENCE0);
    }
    CHECK(s.chips[0].popcount() == 0);
    CHECK(s.locked[0].popcount() == 5);
}

static void test_joker_corner_in_sequence() {
    SECTION("joker corner counts as a chip");
    GameState s;
    clear_board(s);
    // Corner (0,0) is a joker. Place chips at (0,1)..(0,4); they should
    // form a sequence WITH the joker corner contributing as the 5th cell.
    s.chips[0].set(cell_of(0, 1));
    s.chips[0].set(cell_of(0, 2));
    s.chips[0].set(cell_of(0, 3));
    s.chips[0].set(cell_of(0, 4));
    int seqs = s.check_sequences(0, cell_of(0, 4));
    CHECK(seqs == 1);
    // Non-joker cells locked; joker corner remains unset in both maps
    // (it's "everyone's" via JOKER_MASK, not a chip per se).
    CHECK(s.cell_chip(cell_of(0, 0)) == EMPTY);
    for (int c = 1; c <= 4; ++c) {
        CHECK(s.cell_chip(cell_of(0, c)) == SEQUENCE0);
    }
}

static void test_diagonal_sequence() {
    SECTION("down-right diagonal sequence");
    GameState s;
    clear_board(s);
    for (int i = 0; i < 5; ++i) s.chips[0].set(cell_of(2 + i, 3 + i));
    int seqs = s.check_sequences(0, cell_of(4, 5));  // middle of the run
    CHECK(seqs == 1);
}

static void test_blocked_by_opponent() {
    SECTION("opponent chip blocks the window");
    GameState s;
    clear_board(s);
    for (int c = 1; c <= 5; ++c) s.chips[0].set(cell_of(5, c));
    s.chips[1].set(cell_of(5, 3));  // opp drops in the middle
    s.chips[0].reset(cell_of(5, 3));  // (replace, so cell is opp-only)
    int seqs = s.check_sequences(0, cell_of(5, 5));
    CHECK(seqs == 0);
}

static void test_six_in_a_row_two_sequences() {
    SECTION("6-in-a-row → 2 overlapping sequences (share ≤ 1 chip)");
    GameState s;
    clear_board(s);
    // Six cells in a row. The Python engine treats this as one sequence;
    // the second window (cols 2..6) would share 4 cells with the first
    // (cols 1..5), so locked_count = 4 > 1 → rejected.
    for (int c = 1; c <= 6; ++c) s.chips[0].set(cell_of(5, c));
    int seqs = s.check_sequences(0, cell_of(5, 6));
    CHECK(seqs == 1);
}

static void test_nine_in_a_row_two_sequences() {
    SECTION("9-in-a-row → 2 sequences sharing exactly 1 chip");
    GameState s;
    clear_board(s);
    // Cells (5,1)..(5,9). The (5,1..5) and (5,5..9) windows share only
    // cell (5,5), so both should lock as separate sequences when the
    // last chip closes both at once (placing (5,9) won't trigger the
    // first window — it's not in that window — but the (5,5..9) window
    // will, and (5,1..5) was already locked previously).
    //
    // To stress the "share at most 1" rule in a single call, we instead
    // arrange 5 already-locked at (5,1..5) and 4 unlocked at (5,6..9),
    // then place at (5,5) ... but it's already locked. So a more direct
    // setup: place 8 chips and use the 5th-from-each-end placement.
    //
    // Setup: cells (5,1)..(5,4) and (5,6)..(5,9) are placed. We then
    // place (5,5), which sits at the right end of (5,1..5) and the left
    // end of (5,5..9). Both windows complete simultaneously, both
    // include (5,5) which has no prior lock, so both should fire.
    for (int c = 1; c <= 4; ++c) s.chips[0].set(cell_of(5, c));
    for (int c = 6; c <= 9; ++c) s.chips[0].set(cell_of(5, c));
    s.chips[0].set(cell_of(5, 5));  // the closing move
    int seqs = s.check_sequences(0, cell_of(5, 5));
    CHECK(seqs == 2);
    // All 9 cells should be locked.
    int locked_count = 0;
    for (int c = 1; c <= 9; ++c)
        if (s.cell_chip(cell_of(5, c)) == SEQUENCE0) ++locked_count;
    CHECK(locked_count == 9);
}

// ---------------------------------------------------------------------------
// 2. Move-gen sanity
// ---------------------------------------------------------------------------

static void test_initial_move_count() {
    SECTION("initial position move count is reasonable");
    GameState s;
    s.reset(42);
    MoveList moves;
    s.legal_moves(moves);
    // At game start the board is empty, so every regular card has 2 legal
    // placements, every two-eyed jack has 96 (100 - 4 joker corners), and
    // every one-eyed jack is dead (no opponent chips on the board).
    // Lower bound: 7 cards * 2 = 14 (if all regular). Upper bound: dominated
    // by jacks.
    std::printf("  initial legal moves: %d\n", moves.size());
    CHECK(moves.size() >= 14);
    CHECK(moves.size() <= MAX_MOVES);
}

// ---------------------------------------------------------------------------
// 3. Random self-play soak
// ---------------------------------------------------------------------------

static void test_random_selfplay() {
    SECTION("random self-play soak");
    seq::Xoshiro256pp rng(12345);
    const int N_GAMES = 1000;
    const int MAX_STEPS = 2000;

    int wins[2] = {0, 0};
    int draws = 0;
    long long total_steps = 0;
    long long total_moves_seen = 0;
    int illegal_attempts = 0;
    int stalled = 0;

    for (int g = 0; g < N_GAMES; ++g) {
        GameState s;
        s.reset(rng());

        int steps;
        for (steps = 0; steps < MAX_STEPS && !s.done; ++steps) {
            MoveList moves;
            s.legal_moves(moves);
            total_moves_seen += moves.size();
            if (moves.size() == 0) break;
            int pick = std::uniform_int_distribution<int>(0, moves.size() - 1)(rng);
            if (!s.make_move(moves[pick])) {
                ++illegal_attempts;
                break;
            }
        }
        total_steps += steps;
        if (s.done) {
            wins[s.winner]++;
        } else {
            ++stalled;
            if (s.sequences[0] == s.sequences[1]) ++draws;
        }
    }

    std::printf("  games=%d  P0 wins=%d  P1 wins=%d  stalled=%d  draws-of-stalled=%d\n",
                N_GAMES, wins[0], wins[1], stalled, draws);
    std::printf("  avg steps/game=%.1f  avg moves/state=%.1f  illegal=%d\n",
                double(total_steps) / N_GAMES,
                double(total_moves_seen) / total_steps,
                illegal_attempts);

    // Hard requirements: no illegal moves, no stalls (random play in a
    // 104-card deck should always finish well before 2000 plies), and
    // roughly balanced wins (random vs random with no first-move advantage
    // worth speaking of).
    CHECK(illegal_attempts == 0);
    CHECK(stalled == 0);
    int total_wins = wins[0] + wins[1];
    CHECK(total_wins == N_GAMES);
    // Sanity-only — first player should be within ~5σ of 50%. With 1000
    // games, σ ≈ 16, so |wins[0] - 500| should be well under 100.
    int delta = wins[0] > 500 ? wins[0] - 500 : 500 - wins[0];
    CHECK(delta < 100);
}

// ---------------------------------------------------------------------------
// 4. MCTS sanity
// ---------------------------------------------------------------------------

// Construct a position where the player to move can win immediately, then
// verify that MCTS finds the winning move. The test fixture takes some
// setup but is the most honest test of "the search actually searches".
//
// Setup:
//   P0 has chips at (5,1), (5,2), (5,3), (5,4) — one chip short of a sequence.
//   P0 also already has 1 locked sequence elsewhere.
//   P0 holds a 6-of-hearts in hand, which maps to cell (5,5) or its other
//   copy. Placing it completes the row AND wins the game (2nd sequence).
//   P1's chips are scattered and irrelevant.
//
// We pin down P0's hand and the deck so the choice is forced — the only
// winning move is to place the 6-of-hearts at (5,5).
static void test_mcts_finds_obvious_win() {
    SECTION("MCTS spots an immediate winning move");

    // 6-of-hearts: cells are (1,5) and (6,4) per board_layout.py. Neither
    // is in the row we're building. We'll use 6-of-clubs instead, whose
    // board positions include (5,5) — let's verify by checking BOARD or
    // CARD_TO_CELLS at runtime rather than hardcoding the wrong card.
    //
    // To pick a card whose cells include (5,5), scan CARD_TO_CELLS.
    int target_cell = cell_of(5, 5);
    int chosen_card = -1;
    for (int c = 0; c < N_REGULAR_CARDS; ++c) {
        if (CARD_TO_CELLS[c][0] == target_cell || CARD_TO_CELLS[c][1] == target_cell) {
            chosen_card = c;
            break;
        }
    }
    CHECK(chosen_card >= 0);  // sanity — every cell maps to some card
    if (chosen_card < 0) return;

    GameState s;
    s.reset(0xCAFE);  // deal something, then we overwrite

    // Wipe board and rebuild the tactical layout from scratch.
    s.chips[0] = s.chips[1] = BB100{};
    s.locked[0] = s.locked[1] = BB100{};
    s.sequences[0] = 1;          // already has one locked sequence
    s.sequences[1] = 0;
    s.current_player = 0;
    s.done = false;
    s.winner = -1;
    s.dead_card_used[0] = s.dead_card_used[1] = false;

    // The "already locked" sequence — put it somewhere irrelevant, e.g.
    // row 0 cells (1..5). Marking them as locked is enough; we don't
    // need a window through them to also contain (5,5).
    for (int c = 1; c <= 5; ++c) s.locked[0].set(cell_of(0, c));

    // The almost-complete row at row 5, cells 1..4.
    for (int c = 1; c <= 4; ++c) s.chips[0].set(cell_of(5, c));

    // Pin hand[0] to contain chosen_card; everything else is filler we
    // already had from reset. (Filler doesn't matter: MCTS just needs to
    // see a winning move among legal options.)
    s.hands[0][0] = int8_t(chosen_card);

    // Sanity: confirm the move is legal.
    MoveList moves;
    s.legal_moves(moves);
    bool found = false;
    for (int i = 0; i < moves.count; ++i) {
        if (moves.moves[i].card_idx == 0 && moves.moves[i].cell == target_cell) {
            found = true;
            break;
        }
    }
    CHECK(found);
    if (!found) return;

    // Search and check.
    MCTSConfig cfg;
    cfg.iterations = 500;
    cfg.determinize = true;
    cfg.seed = 0xABCDEF;
    MCTSEngine eng(cfg);

    Move best = eng.suggest_move(s);
    std::printf("  MCTS chose card_idx=%d cell=%d, winrate=%.3f visits=%d/%d\n",
                int(best.card_idx), int(best.cell),
                eng.last_stats().best_move_winrate,
                eng.last_stats().best_move_visits,
                eng.last_stats().iterations);

    // The position is a guaranteed win for P0 — there's no defensive
    // response — so MCTS should be ≥95% confident. The exact move can
    // be either the direct winning placement OR a "free" move that
    // doesn't surrender the turn (e.g. declaring a dead one-eyed jack):
    // both lead to certain victory.
    CHECK(eng.last_stats().best_move_winrate >= 0.95);

    // Functional check: play out from this position with MCTS as P0 and
    // a random policy as P1. P0 must win within a few plies.
    {
        Xoshiro256pp opp_rng(0xFACE12);
        GameState g = s;
        MCTSEngine playthrough(cfg);
        for (int plies = 0; !g.done && plies < 40; ++plies) {
            Move m;
            if (g.current_player == 0) {
                m = playthrough.suggest_move(g);
            } else {
                MoveList ms;
                g.legal_moves(ms);
                if (ms.count == 0) break;
                m = ms.moves[opp_rng.rand_int(0, ms.count - 1)];
            }
            CHECK(g.make_move(m));
        }
        CHECK(g.done);
        CHECK(g.winner == 0);
    }
}

// Quick smoke: MCTS plays a full game against itself without crashing.
static void test_mcts_self_play_smoke() {
    SECTION("MCTS self-play smoke test");
    MCTSConfig cfg;
    cfg.iterations = 80;
    cfg.seed = 0xFEEDFACE;
    MCTSEngine a(cfg);
    cfg.seed = 0xBADFEED;
    MCTSEngine b(cfg);

    GameState s;
    s.reset(0x12345);
    int plies = 0;
    while (!s.done && plies < 600) {
        Move m = (s.current_player == 0) ? a.suggest_move(s) : b.suggest_move(s);
        CHECK(s.make_move(m));
        ++plies;
    }
    CHECK(s.done);                   // game terminated, not stalled out
    std::printf("  game finished in %d plies, winner=%d\n", plies, s.winner);
}

int main() {
    test_horizontal_sequence();
    test_joker_corner_in_sequence();
    test_diagonal_sequence();
    test_blocked_by_opponent();
    test_six_in_a_row_two_sequences();
    test_nine_in_a_row_two_sequences();

    test_initial_move_count();

    test_random_selfplay();

    test_mcts_finds_obvious_win();
    test_mcts_self_play_smoke();

    std::printf("\n----\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}

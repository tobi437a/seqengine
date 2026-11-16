// Window states, shared by the search (search.cpp), the training-data
// generator (tools/gen_data.cpp) and the generated src/learned_tables.hpp.
//
// A 5-window that can still matter is described by three small numbers:
// whether it contains a joker corner (j), and how many non-joker chips
// each of two parties has in it (a, b), with a + b + j <= 5. There are 36
// such states — 21 without a joker, 15 with. window_state_id numbers
// them densely; every learned table is indexed by that number.

#pragma once

namespace seq {

constexpr int N_WINDOW_STATES = 36;

// Dense id of the state (j, a, b), or -1 if a + b + j > 5.
constexpr int window_state_id(int j, int a, int b) {
    if (a + b + j > 5) return -1;
    int id = 0;
    for (int jj = 0; jj < 2; ++jj)
        for (int aa = 0; aa < 6; ++aa)
            for (int bb = 0; bb < 6; ++bb) {
                if (aa + bb + jj > 5) continue;
                if (jj == j && aa == a && bb == b) return id;
                ++id;
            }
    return -1;
}

// Hand support of a placement, per window: how many of the window's
// OTHER empty cells the mover could still fill from hand afterwards with
// a regular card — 0, 1, or 2-and-more. A line you hold the follow-up
// cards for is worth more than the same line you would need to draw for.
constexpr int N_HAND_LEVELS = 3;

// Game stage as seen by one player: 2·(I have a sequence) + (they do).
// Both the move scores and the evaluation are kept per stage — an open
// four means something different once it would be the winning sequence.
constexpr int N_STAGES = 4;
constexpr int stage_of(int my_sequences, int their_sequences) {
    return 2 * (my_sequences > 0) + (their_sequences > 0);
}

} // namespace seq

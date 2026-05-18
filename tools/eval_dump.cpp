// Cross-validation dump tool.
//
// Plays random games and emits for each position visited:
//   P  <100 chip ints> <phi_p0> <phi_p1>
//   M  <player> <card_type> <cell> <shaping_score>     [for one legal move]
//
// The `M` line(s) follow the `P` line they describe. validate_eval.py
// reconstructs each position in the Python engine and recomputes both Φ
// values and every shaping score, asserting bit-equality with the C++
// output. Used to confirm src/eval.hpp is a faithful port of the Python
// reference before MCTS starts depending on it.

#include "../src/state.hpp"
#include "../src/eval.hpp"
#include <cstdio>
#include <cstdlib>
#include <random>
#include "../src/rng.hpp"

using namespace seq;

int main(int argc, char** argv) {
    const int n_positions = argc >= 2 ? std::atoi(argv[1]) : 300;
    seq::Xoshiro256pp rng(0xBADC0FFEE12345ULL);

    int emitted = 0;
    while (emitted < n_positions) {
        GameState s;
        s.reset(rng());

        while (!s.done && emitted < n_positions) {
            // --- P line: the position itself ---
            std::printf("P");
            for (int cell = 0; cell < N_CELLS; ++cell) {
                std::printf(" %d", int(s.cell_chip(cell)));
            }
            std::printf(" %.10f %.10f\n", total_phi(s, 0), total_phi(s, 1));

            // --- M line(s): a sample of legal moves with their scores ---
            // Up to 5 random legal real moves (skip dead-card declarations —
            // shaping_score is zero by definition there).
            MoveList moves;
            s.legal_moves(moves);
            if (moves.size() == 0) break;

            int n_emit = std::min(moves.size(), 5);
            int player = s.current_player;
            for (int i = 0; i < n_emit; ++i) {
                int idx = std::uniform_int_distribution<int>(0, moves.size() - 1)(rng);
                Move m = moves[idx];
                if (m.is_dead()) continue;
                int card_type = s.hands[player][m.card_idx];
                double sc = shaping_score(s, player, card_type, m.cell);
                std::printf("M %d %d %d %.10f\n",
                            player, card_type, int(m.cell), sc);
            }

            ++emitted;

            // Step to next position with a uniformly random legal move.
            int pick = std::uniform_int_distribution<int>(0, moves.size() - 1)(rng);
            s.make_move(moves[pick]);
        }
    }
    return 0;
}

// A/B sweep: how does splitting the iteration budget across N root-
// parallel ISMCTS trees compare to a single tree with the full budget?
// Each iteration of ISMCTS already re-determinizes the opponent's hand,
// so extra trees don't add determinization diversity — they only trade
// per-tree depth for wall-clock parallelism. Each config plays N games
// vs the heuristic.

#include "../src/state.hpp"
#include "../src/search.hpp"
#include "../src/eval.hpp"
#include "../src/rng.hpp"
#include <cstdio>
#include <cstdlib>

using namespace seq;

static Move heuristic_action(GameState& s) {
    MoveList moves;
    s.legal_moves(moves);
    if (moves.count == 0) return Move{};
    int player = s.current_player;
    double best = -1e30;
    int best_idx = 0;
    for (int i = 0; i < moves.count; ++i) {
        Move m = moves.moves[i];
        double sc = m.is_dead() ? 0.0
            : shaping_score(s, player, s.hands[player][m.card_idx], m.cell);
        if (sc > best) { best = sc; best_idx = i; }
    }
    return moves.moves[best_idx];
}

static double match(MCTSConfig cfg, int n_games, Xoshiro256pp& seed_rng) {
    MCTSEngine mcts(cfg);
    int wins = 0;
    for (int g = 0; g < n_games; ++g) {
        GameState s;
        s.reset(seed_rng());
        int mcts_seat = g % 2;
        for (int plies = 0; !s.done && plies < 1000; ++plies) {
            Move m = (s.current_player == mcts_seat)
                ? mcts.suggest_move(s)
                : heuristic_action(s);
            if (!s.make_move(m)) break;
        }
        if (s.winner == mcts_seat) ++wins;
    }
    return double(wins) / n_games;
}

int main(int argc, char** argv) {
    int n_games = argc >= 2 ? std::atoi(argv[1]) : 16;
    int iters   = argc >= 3 ? std::atoi(argv[2]) : 800;

    struct Run {
        const char* label;
        int    n_trees;
        double c;
        double eps;
    };
    Run configs[] = {
        {"trees=1 (full budget)",    1, 2.0, 0.30},
        {"trees=2",                  2, 2.0, 0.30},
        {"trees=4",                  4, 2.0, 0.30},
        {"trees=8 (current dflt)",   8, 2.0, 0.30},
        {"trees=16",                16, 2.0, 0.30},
    };

    std::printf("Root-parallel-tree sweep at iters=%d total, n_games=%d, vs heuristic\n",
                iters, n_games);
    std::printf("%-30s | winrate\n", "config");
    for (Run r : configs) {
        Xoshiro256pp seed_rng(0xCAFE99);
        MCTSConfig cfg;
        cfg.iterations              = iters;
        cfg.ucb_c                   = r.c;
        cfg.rollout_epsilon         = r.eps;
        cfg.n_parallel_trees        = r.n_trees;
        cfg.seed                    = 0xDEADBEEF;
        double wr = match(cfg, n_games, seed_rng);
        std::printf("%-30s | %5.1f%%\n", r.label, wr * 100);
        std::fflush(stdout);
    }
    return 0;
}

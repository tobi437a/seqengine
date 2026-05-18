// Focused mini-sweep: MCTS vs heuristic at a handful of configs.

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
    int n_games = argc >= 2 ? std::atoi(argv[1]) : 20;

    struct Run { int iters; double c; double eps; };
    Run configs[] = {
        {500,  2.0, 0.30},   // baseline
        {500,  1.0, 0.30},
        {500,  1.0, 0.10},
        {500,  0.7, 0.10},
        {2000, 1.0, 0.10},
        {2000, 0.7, 0.10},
    };

    std::printf("MCTS vs heuristic, %d games each\n", n_games);
    std::printf("%-6s %-5s %-5s | winrate\n", "iters", "c", "eps");
    for (Run r : configs) {
        Xoshiro256pp seed_rng(0xCAFE99);  // same seed → same game sequence
        MCTSConfig cfg;
        cfg.iterations = r.iters;
        cfg.ucb_c = r.c;
        cfg.rollout_epsilon = r.eps;
        cfg.seed = 0xDEADBEEF;
        double wr = match(cfg, n_games, seed_rng);
        std::printf("%-6d %-5.2f %-5.2f | %5.1f%%\n", r.iters, r.c, r.eps, wr * 100);
        std::fflush(stdout);
    }
    return 0;
}

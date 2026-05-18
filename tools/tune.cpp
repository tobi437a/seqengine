// Tuning sweep over MCTS hyperparameters. Plays N games each against
// the random and heuristic baselines at a grid of (iterations, ucb_c,
// rollout_epsilon) settings.

#include "../src/state.hpp"
#include "../src/search.hpp"
#include "../src/eval.hpp"
#include "../src/rng.hpp"
#include <cstdio>
#include <cstdlib>
#include <chrono>

using namespace seq;

static Move random_action(GameState& s, Xoshiro256pp& rng) {
    MoveList moves;
    s.legal_moves(moves);
    if (moves.count == 0) return Move{};
    return moves.moves[rng.rand_int(0, moves.count - 1)];
}

static Move heuristic_action(GameState& s, Xoshiro256pp&) {
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

template <typename OppFn>
static double play_match(MCTSConfig cfg, OppFn opp_fn, int n_games,
                         Xoshiro256pp& seed_rng,
                         double& seconds_per_game)
{
    MCTSEngine mcts(cfg);
    Xoshiro256pp move_rng(seed_rng());
    int wins = 0;
    double total = 0.0;
    for (int g = 0; g < n_games; ++g) {
        GameState s;
        s.reset(seed_rng());
        int mcts_seat = g % 2;
        for (int plies = 0; !s.done && plies < 1000; ++plies) {
            Move m = (s.current_player == mcts_seat)
                ? mcts.suggest_move(s)
                : opp_fn(s, move_rng);
            if (s.current_player == mcts_seat) total += mcts.last_stats().seconds;
            if (!s.make_move(m)) break;
        }
        if (s.winner == mcts_seat) ++wins;
    }
    seconds_per_game = total / std::max(1, n_games);
    return double(wins) / n_games;
}

int main(int argc, char** argv) {
    int n_games = argc >= 2 ? std::atoi(argv[1]) : 30;
    std::printf("Tuning sweep, %d games per cell\n", n_games);
    std::printf("%-6s %-5s %-5s | %-12s %-12s | s/move\n",
                "iters", "c", "eps", "vs random", "vs heuristic");

    int iters_list[]   = {500, 2000, 5000};
    double c_list[]    = {0.7, 1.4, 2.0};
    double eps_list[]  = {0.10, 0.30};

    Xoshiro256pp seed_rng(0xBEEF1234);

    for (int it : iters_list)
    for (double c : c_list)
    for (double e : eps_list) {
        MCTSConfig cfg;
        cfg.iterations      = it;
        cfg.ucb_c           = c;
        cfg.rollout_epsilon = e;
        cfg.seed            = 0xC0FFEE99;

        double s_per_move = 0.0;
        double r_random = play_match(cfg, random_action,    n_games, seed_rng, s_per_move);
        double r_heur   = play_match(cfg, heuristic_action, n_games, seed_rng, s_per_move);

        std::printf("%-6d %-5.2f %-5.2f | %-12.1f %-12.1f | %.3f\n",
                    it, c, e, r_random * 100, r_heur * 100, s_per_move);
        std::fflush(stdout);
    }
    return 0;
}

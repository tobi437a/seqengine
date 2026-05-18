// Head-to-head competition tool.
//
// Plays MCTS against the existing baselines (random and heuristic) and
// reports win/loss/draw rates, alternating seats so first-move asymmetry
// averages out. Use this to confirm MCTS works and to tune the config.
//
// Usage:
//   ./build/compete [N_GAMES] [ITERATIONS]
//
// Defaults: 200 games, 2000 iterations per MCTS move. Should take around 100s to complete

#include "../src/state.hpp"
#include "../src/search.hpp"
#include "../src/eval.hpp"
#include "../src/rng.hpp"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

using namespace seq;

// Baseline policies — match the signatures in evaluate.py.

static Move random_action(GameState& s, Xoshiro256pp& rng) {
    MoveList moves;
    s.legal_moves(moves);
    if (moves.count == 0) return Move{};
    return moves.moves[rng.rand_int(0, moves.count - 1)];
}

static Move heuristic_action(GameState& s, Xoshiro256pp& rng) {
    (void)rng;
    MoveList moves;
    s.legal_moves(moves);
    if (moves.count == 0) return Move{};

    const int player = s.current_player;
    double best_score = -1e30;
    int best_idx = 0;
    for (int i = 0; i < moves.count; ++i) {
        Move m = moves.moves[i];
        double sc;
        if (m.is_dead()) {
            sc = 0.0;
        } else {
            int card_type = s.hands[player][m.card_idx];
            sc = shaping_score(s, player, card_type, m.cell);
        }
        if (sc > best_score) {
            best_score = sc;
            best_idx   = i;
        }
    }
    return moves.moves[best_idx];
}

struct MatchResult {
    int wins   = 0;
    int losses = 0;
    int draws  = 0;
    int games  = 0;
    double mcts_total_seconds = 0.0;
    long long mcts_total_iterations = 0;
};

// Parallelism is at the game level, not inside MCTS. Each game runs a
// single-threaded MCTSEngine (n_threads=1, the default) seeded
// deterministically up-front, so the aggregate W/L/D is reproducible
// regardless of which worker thread picks up which game. Game-level
// parallelism gives the same wall-clock win as 16-thread MCTS without
// the run-to-run variance that comes from threads racing inside one
// search tree.
constexpr int COMPETE_WORKERS = 16;

template <typename OpponentFn>
static MatchResult play_match(const MCTSConfig& base_cfg, OpponentFn opp_fn,
                              int n_games, Xoshiro256pp& seed_rng)
{
    // Pre-derive every per-game seed sequentially from seed_rng. This
    // happens before any worker starts, so the seeds — and therefore
    // the results — don't depend on thread scheduling.
    struct GameSeeds {
        uint64_t state_seed;
        uint64_t move_rng_seed;
        uint64_t mcts_seed;
    };
    std::vector<GameSeeds> seeds(n_games);
    for (int g = 0; g < n_games; ++g) {
        seeds[g].state_seed    = seed_rng();
        seeds[g].move_rng_seed = seed_rng();
        seeds[g].mcts_seed     = seed_rng();
    }

    std::vector<MatchResult> per_game(n_games);
    std::atomic<int> next{0};

    auto worker = [&]() {
        for (;;) {
            int g = next.fetch_add(1, std::memory_order_relaxed);
            if (g >= n_games) break;

            MCTSConfig cfg = base_cfg;
            cfg.seed = seeds[g].mcts_seed;
            MCTSEngine mcts(cfg);

            Xoshiro256pp move_rng(seeds[g].move_rng_seed);
            GameState s;
            s.reset(seeds[g].state_seed);
            int mcts_seat = g % 2;          // alternate seats

            MatchResult r;
            r.games = 1;
            const int MAX_PLIES = 1200;
            int plies = 0;
            while (!s.done && plies < MAX_PLIES) {
                Move m;
                if (s.current_player == mcts_seat) {
                    m = mcts.suggest_move(s);
                    r.mcts_total_seconds    += mcts.last_stats().seconds;
                    r.mcts_total_iterations += mcts.last_stats().iterations;
                } else {
                    m = opp_fn(s, move_rng);
                }
                if (!s.make_move(m)) break;
                ++plies;
            }
            if (s.winner == mcts_seat)     r.wins++;
            else if (s.winner != -1)       r.losses++;
            else                            r.draws++;

            per_game[g] = r;
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(COMPETE_WORKERS);
    for (int t = 0; t < COMPETE_WORKERS; ++t) threads.emplace_back(worker);
    for (auto& th : threads) th.join();

    MatchResult total;
    total.games = n_games;
    for (const auto& r : per_game) {
        total.wins                  += r.wins;
        total.losses                += r.losses;
        total.draws                 += r.draws;
        total.mcts_total_seconds    += r.mcts_total_seconds;
        total.mcts_total_iterations += r.mcts_total_iterations;
    }
    return total;
}

static void report(const char* label, const MatchResult& r) {
    double winrate = double(r.wins) / r.games;
    double avg_time = r.mcts_total_seconds / std::max(1, r.wins + r.losses + r.draws);
    double avg_iters_per_sec =
        r.mcts_total_seconds > 0
        ? double(r.mcts_total_iterations) / r.mcts_total_seconds
        : 0.0;
    std::printf("vs %-12s  W/L/D = %3d/%3d/%3d  winrate = %.1f%%   "
                "%.3fs/game  %.0f iters/s\n",
                label, r.wins, r.losses, r.draws, winrate * 100.0,
                avg_time, avg_iters_per_sec);
}

int main(int argc, char** argv) {
    int n_games    = argc >= 2 ? std::atoi(argv[1]) : 200;
    int iterations = argc >= 3 ? std::atoi(argv[2]) : 2000;

    MCTSConfig cfg;
    cfg.iterations = iterations;
    cfg.seed       = 0xDEADBEEFCAFE;
    // n_threads stays at the default (1): each game runs single-threaded
    // MCTS for deterministic results. Parallelism is across games — see
    // play_match.

    Xoshiro256pp seed_rng(0xC0FFEEBEEF);

    std::printf("MCTS config: iterations=%d, ucb_c=%.2f, eps=%.2f, "
                "trees=%d, determinize=%s\n",
                cfg.iterations, cfg.ucb_c, cfg.rollout_epsilon,
                cfg.n_parallel_trees,
                cfg.determinize ? "true" : "false");
    std::printf("Games per match: %d (alternating seats), %d parallel game workers\n\n",
                n_games, COMPETE_WORKERS);

    auto t0 = std::chrono::high_resolution_clock::now();

    auto r_rand = play_match(cfg, random_action,    n_games, seed_rng);
    report("random",    r_rand);

    auto r_heur = play_match(cfg, heuristic_action, n_games, seed_rng);
    report("heuristic", r_heur);

    double total_s = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t0).count();
    std::printf("\nTotal wall time: %.1fs\n", total_s);
    return 0;
}

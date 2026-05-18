// Microbenchmark for the engine and eval.
//
// Reports:
//   - total_phi throughput: relevant for a value-network-style leaf eval
//   - random rollout speed: relevant as a floor on MCTS rollout cost
//                           (using the heuristic as a rollout policy is
//                            slower per step but should be ≤ ~2× this)
//
// All four benchmarks are parallelized across BENCH_THREADS workers and
// sized so each phase takes a handful of seconds — short benches gave
// very noisy numbers run-to-run.

#include "../src/state.hpp"
#include "../src/eval.hpp"
#include "../src/search.hpp"
#include <chrono>
#include <random>
#include "../src/rng.hpp"
#include <cstdio>
#include <vector>
#include <thread>

using namespace seq;
using clk = std::chrono::high_resolution_clock;
template <typename Dur>
double secs(Dur d) { return std::chrono::duration<double>(d).count(); }

// Reported throughput is aggregate across these workers. Change here to
// match your machine.
constexpr int BENCH_THREADS = 16;

template <typename F>
static void run_parallel(int n, F&& fn) {
    std::vector<std::thread> threads;
    threads.reserve(n);
    for (int t = 0; t < n; ++t) threads.emplace_back(fn, t);
    for (auto& th : threads) th.join();
}

int main() {
    seq::Xoshiro256pp rng(0xBEEFCAFE);

    // --- Build a sample of ~1024 distinct positions from random play ---
    std::vector<GameState> positions;
    positions.reserve(1024);
    while (positions.size() < 1024) {
        GameState s;
        s.reset(rng());
        while (!s.done && positions.size() < 1024) {
            positions.push_back(s);
            MoveList moves;
            s.legal_moves(moves);
            if (moves.size() == 0) break;
            int pick = std::uniform_int_distribution<int>(0, moves.size() - 1)(rng);
            s.make_move(moves[pick]);
        }
    }

    // --- Benchmark 1: total_phi ---
    // total_phi is read-only on GameState, so workers all share `positions`.
    {
        const long long N_PER_THREAD = 5'000'000;
        const long long N_TOTAL = N_PER_THREAD * BENCH_THREADS;
        std::vector<double> sinks(BENCH_THREADS, 0.0);
        auto t0 = clk::now();
        run_parallel(BENCH_THREADS, [&](int tid) {
            double local = 0.0;
            const long long base = (long long)tid * 7919;
            for (long long i = 0; i < N_PER_THREAD; ++i) {
                local += total_phi(positions[(base + i) & 1023], int((base + i) & 1));
            }
            sinks[tid] = local;
        });
        double sec = secs(clk::now() - t0);
        volatile double sink = 0;
        for (auto v : sinks) sink += v;
        std::printf("total_phi:        %lld calls in %.3fs  =>  %6.1fM calls/s   (%.1f ns/call)   [%d threads]\n",
                    N_TOTAL, sec, double(N_TOTAL) / sec / 1e6,
                    sec / double(N_TOTAL) * 1e9 * BENCH_THREADS, BENCH_THREADS);
        (void)sink;
    }

    // --- Benchmark 2: full random rollouts (legal_moves + make_move loop) ---
    // Each worker has its own RNG and GameState.
    {
        const long long GAMES_PER_THREAD = 300'000;
        const long long N_GAMES_TOTAL = GAMES_PER_THREAD * BENCH_THREADS;
        std::vector<long long> moves_per_thread(BENCH_THREADS, 0);
        auto t0 = clk::now();
        run_parallel(BENCH_THREADS, [&](int tid) {
            seq::Xoshiro256pp lrng(0xBEEFCAFEull ^ (uint64_t(tid) * 0x9E3779B97F4A7C15ull));
            GameState s;
            MoveList moves;
            long long total = 0;
            for (long long g = 0; g < GAMES_PER_THREAD; ++g) {
                s.reset(lrng());
                while (!s.done) {
                    s.legal_moves(moves);
                    if (moves.size() == 0) break;
                    int pick = std::uniform_int_distribution<int>(0, moves.size() - 1)(lrng);
                    s.make_move(moves[pick]);
                    ++total;
                }
            }
            moves_per_thread[tid] = total;
        });
        double sec = secs(clk::now() - t0);
        long long total_moves = 0;
        for (auto v : moves_per_thread) total_moves += v;
        std::printf("random rollouts:  %lld games / %lld moves in %.3fs  =>  %5.0fk games/s, %5.1fM moves/s   [%d threads]\n",
                    N_GAMES_TOTAL, total_moves, sec,
                    double(N_GAMES_TOTAL) / sec / 1000.0,
                    double(total_moves) / sec / 1e6,
                    BENCH_THREADS);
    }

    // --- Benchmark 3: shaping_score over all legal moves at a position ---
    // Closer to the heuristic-rollout cost: at every state, score every
    // legal move and pick the argmax.
    //
    // shaping_score is pure (operates on a ShapingMasks snapshot built up
    // front), so each worker can share a single GameState — but we keep
    // a per-thread copy anyway for cache locality with the position array.
    {
        const long long DECS_PER_THREAD = 1'000'000;
        const long long N_DECISIONS_TOTAL = DECS_PER_THREAD * BENCH_THREADS;
        std::vector<long long> moves_scored_per_thread(BENCH_THREADS, 0);
        std::vector<double> sinks(BENCH_THREADS, 0.0);
        auto t0 = clk::now();
        run_parallel(BENCH_THREADS, [&](int tid) {
            double local = 0.0;
            long long scored = 0;
            const long long base = (long long)tid * 7919;
            for (long long i = 0; i < DECS_PER_THREAD; ++i) {
                GameState s = positions[(base + i) & 1023];  // local mutable copy
                int player = s.current_player;
                MoveList moves;
                s.legal_moves_for(player, moves);
                for (int k = 0; k < moves.size(); ++k) {
                    Move m = moves[k];
                    if (m.is_dead()) continue;
                    int card_type = s.hands[player][m.card_idx];
                    local += shaping_score(s, player, card_type, m.cell);
                    ++scored;
                }
            }
            moves_scored_per_thread[tid] = scored;
            sinks[tid] = local;
        });
        double sec = secs(clk::now() - t0);
        long long moves_scored = 0;
        for (auto v : moves_scored_per_thread) moves_scored += v;
        volatile double sink = 0;
        for (auto v : sinks) sink += v;
        std::printf("heuristic eval:   %lld decisions / %lld moves in %.3fs  =>  %5.0fk decisions/s   [%d threads]\n",
                    N_DECISIONS_TOTAL, moves_scored, sec,
                    double(N_DECISIONS_TOTAL) / sec / 1000.0,
                    BENCH_THREADS);
        (void)sink;
    }

    // --- Benchmark 4: MCTS iterations/s ---
    // Root-parallel: n_parallel_trees=BENCH_THREADS, n_threads=BENCH_THREADS,
    // so one independent tree per worker. iterations and N_MOVES are bumped
    // so the phase wall time matches the others.
    {
        const int N_MOVES = 40;
        MCTSConfig cfg;
        cfg.iterations       = 20000;
        cfg.n_parallel_trees = BENCH_THREADS;
        cfg.n_threads        = BENCH_THREADS;
        cfg.seed             = 0xDEADBEEFCAFE;
        MCTSEngine eng(cfg);

        // Warmup so the first call's thread/allocation cost doesn't skew.
        eng.suggest_move(positions[0]);

        double total_sec = 0.0;
        long long total_iters = 0;
        for (int i = 0; i < N_MOVES; ++i) {
            // Spread the sampled positions across the 1024-position buffer.
            const GameState& s = positions[(i * 53) & 1023];
            eng.suggest_move(s);
            total_sec   += eng.last_stats().seconds;
            total_iters += eng.last_stats().iterations;
        }

        std::printf("MCTS iters:       %lld iters / %d moves in %.3fs  =>  %5.1fk iters/s   (%.1f ms/move)   [%d threads, %d trees]\n",
                    total_iters, N_MOVES, total_sec,
                    double(total_iters) / total_sec / 1000.0,
                    total_sec / N_MOVES * 1000.0,
                    cfg.n_threads, cfg.n_parallel_trees);
    }

    return 0;
}

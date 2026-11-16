// Thread-count ablation: what does searching the one shared tree with N
// threads cost in playing strength at a fixed iteration budget, and
// what does it buy in speed?
//
// Threads descending the same tree at once work from slightly stale
// statistics and steer around each other with virtual loss, so N-thread
// search is a little less efficient per iteration than sequential
// search. This plays MCTS(n_threads = N) against MCTS(n_threads = 1) at
// the same budget — paired games, same deal with seats swapped — and
// reports the score and each side's speed.
//
// Usage:
//   ./build/ablation [N_PAIRS] [ITERATIONS] [THREADS]
//
// Defaults: 50 pairs, 20000 iterations, 16 threads. Games run one at a
// time (the N-thread side wants the whole machine), so this is slow.

#include "../src/state.hpp"
#include "../src/search.hpp"
#include "../src/rng.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>

using namespace seq;

struct Side {
    MCTSEngine eng;
    double     seconds = 0.0;
    long long  iters   = 0;
    explicit Side(const MCTSConfig& cfg) : eng(cfg) {}
    Move think(const GameState& s) {
        Move m = eng.suggest_move(s);
        seconds += eng.last_stats().seconds;
        iters   += eng.last_stats().iterations;
        return m;
    }
};

// Returns +1 if `multi` won, -1 if `single` won, 0 otherwise.
static int play_game(Side& multi, Side& single, int multi_seat, uint64_t deal) {
    GameState s;
    s.reset(deal);
    for (int plies = 0; !s.done && plies < 1200; ++plies) {
        Move m = (s.current_player == multi_seat) ? multi.think(s)
                                                  : single.think(s);
        multi.eng.advance(s, m);
        single.eng.advance(s, m);
        if (!s.make_move(m)) break;
    }
    if (s.winner < 0) return 0;
    return s.winner == multi_seat ? +1 : -1;
}

int main(int argc, char** argv) {
    const int n_pairs = argc >= 2 ? std::atoi(argv[1]) : 50;
    const int iters   = argc >= 3 ? std::atoi(argv[2]) : 20000;
    const int threads = argc >= 4 ? std::atoi(argv[3]) : 16;

    std::printf("%d threads vs 1 thread, %d iterations each, %d game pairs\n\n",
                threads, iters, n_pairs);

    Xoshiro256pp seed_rng(0xAB1A7105);
    int wins = 0, losses = 0, draws = 0;
    double sec_multi = 0, sec_single = 0;
    long long it_multi = 0, it_single = 0;

    for (int p = 0; p < n_pairs; ++p) {
        const uint64_t deal = seed_rng();
        for (int seat = 0; seat < 2; ++seat) {
            MCTSConfig cm, cs;
            cm.iterations = cs.iterations = iters;
            cm.n_threads  = threads;
            cs.n_threads  = 1;
            cm.seed = seed_rng();
            cs.seed = seed_rng();
            Side multi(cm), single(cs);
            const int r = play_game(multi, single, seat, deal);
            if (r > 0) ++wins; else if (r < 0) ++losses; else ++draws;
            sec_multi  += multi.seconds;  it_multi  += multi.iters;
            sec_single += single.seconds; it_single += single.iters;
        }
        const int n = wins + losses + draws;
        const double score = (wins + 0.5 * draws) / n;
        const double se = std::sqrt(score * (1.0 - score) / n);
        std::printf("pair %3d   %d-thread W/L/D = %d/%d/%d   score %.3f +/- %.3f   "
                    "%.0fk it/s vs %.0fk it/s  (x%.1f)\n",
                    p + 1, threads, wins, losses, draws, score, 1.96 * se,
                    it_multi / sec_multi / 1e3, it_single / sec_single / 1e3,
                    (it_multi / sec_multi) / (it_single / sec_single));
        std::fflush(stdout);
    }
    return 0;
}

// Profile driver — plays N complete MCTS-vs-MCTS games and reports
// where the engine actually spent its time across the run.
//
// Why not just use the bench tool? bench measures isolated functions at
// peak throughput. That tells you how fast each block CAN go, but not
// how often it runs during real play, which is what determines which
// optimization actually moves the needle. This tool answers
//
//     "during a normal MCTS game, how much wall time goes where,
//      and how many times per game does each block fire?"
//
// Build via `make profile` — that target compiles every translation
// unit with -DSEQ_PROFILE so the SEQ_PROFILE_SCOPE macros in search.cpp
// and state.cpp expand into real RAII timers. The default build leaves
// the macros as `(void)0` so there's zero overhead in production.
//
// Usage:
//   ./build/profile [N_GAMES] [ITERATIONS]
// Defaults: 5 games, 1000 iterations / move.

#include "../src/state.hpp"
#include "../src/search.hpp"
#include "../src/eval.hpp"
#include "../src/rng.hpp"
#include "../src/profile.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <vector>

using namespace seq;

// 4-significant-digit human-readable duration. uint64_t in ns.
static std::string fmt_ns(uint64_t ns) {
    char buf[32];
    if (ns < 1000ull)
        std::snprintf(buf, sizeof(buf), "%4llu ns", (unsigned long long)ns);
    else if (ns < 1'000'000ull)
        std::snprintf(buf, sizeof(buf), "%6.2f us", double(ns) / 1e3);
    else if (ns < 1'000'000'000ull)
        std::snprintf(buf, sizeof(buf), "%6.2f ms", double(ns) / 1e6);
    else
        std::snprintf(buf, sizeof(buf), "%7.3f s", double(ns) / 1e9);
    return buf;
}

// Comma-separated integer for the Calls column. Easier to eyeball
// scale than a bare digit string at 7+ figures.
static std::string fmt_count(uint64_t n) {
    std::string s = std::to_string(n);
    for (int i = int(s.size()) - 3; i > 0; i -= 3) s.insert(i, ",");
    return s;
}

int main(int argc, char** argv) {
    int n_games    = (argc > 1) ? std::atoi(argv[1]) : 5;
    int iterations = (argc > 2) ? std::atoi(argv[2]) : 1000;
    if (n_games    <= 0) n_games    = 5;
    if (iterations <= 0) iterations = 1000;

    std::printf("=== Sequence engine profile ===\n");
    std::printf("Games: %d   Iterations/move: %d   Threads: 1   Trees: 8 (default)\n\n",
                n_games, iterations);

#ifndef SEQ_PROFILE
    std::printf("WARNING: this binary was built WITHOUT -DSEQ_PROFILE; the\n"
                "         SEQ_PROFILE_SCOPE macros compiled to no-ops, so\n"
                "         no per-section timings will be reported. Rebuild\n"
                "         with `make profile`.\n\n");
#endif

    // Both seats run MCTS so the workload mirrors the case people
    // actually care about ("the engine playing the engine"). Distinct
    // seeds keep the two sides from making mirror-image decisions.
    MCTSConfig cfg_a;
    cfg_a.iterations = iterations;
    cfg_a.seed       = 0xA5A5A5A5A5A5A5A5ull;
    MCTSConfig cfg_b = cfg_a;
    cfg_b.seed       = 0x5A5A5A5A5A5A5A5Aull;

    MCTSEngine eng_a(cfg_a);
    MCTSEngine eng_b(cfg_b);

    Xoshiro256pp seed_rng(0xC0FFEEBABEull);

    // Reset counters AFTER engine construction so allocator / RNG seed
    // work doesn't get charged to suggest_move's first call.
    profile::Registry::instance().reset();

    auto t0 = std::chrono::high_resolution_clock::now();

    long long total_plies = 0;
    int wins[2] = {0, 0};
    int draws   = 0;

    for (int g = 0; g < n_games; ++g) {
        GameState s;
        s.reset(seed_rng());
        long long plies = 0;
        while (!s.done) {
            Move m = (s.current_player == 0)
                        ? eng_a.suggest_move(s)
                        : eng_b.suggest_move(s);
            if (!s.make_move(m)) break;
            ++plies;
        }
        total_plies += plies;
        if (s.winner == 0)      wins[0]++;
        else if (s.winner == 1) wins[1]++;
        else                    draws++;
        std::printf("  game %2d: %4lld plies, winner=%d\n",
                    g + 1, plies, s.winner);
        std::fflush(stdout);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double   wall_sec = std::chrono::duration<double>(t1 - t0).count();
    uint64_t wall_ns  = uint64_t(wall_sec * 1e9);

    std::printf("\nWall-clock:        %.3fs\n", wall_sec);
    std::printf("Games:             %d\n", n_games);
    std::printf("Total plies:       %lld\n", total_plies);
    std::printf("Avg game length:   %.1f plies\n",
                double(total_plies) / double(n_games));
    std::printf("Win split:         p0=%d  p1=%d  draws=%d\n\n",
                wins[0], wins[1], draws);

    auto entries = profile::Registry::instance().snapshot();
    std::sort(entries.begin(), entries.end(),
              [](const auto& a, const auto& b) {
                  return a.total_ns > b.total_ns;
              });

    // Column widths chosen so the values fit at our usual scales
    // (millions of calls, single-digit seconds total). Adjust if a
    // future workload blows past these.
    std::printf("%-28s %14s %11s %8s %12s %14s %11s\n",
                "Section", "Calls", "Total", "%wall",
                "ns/call", "Calls/game", "ms/game");
    std::printf("%-28s %14s %11s %8s %12s %14s %11s\n",
                "----------------------------",
                "--------------",
                "-----------",
                "--------",
                "------------",
                "--------------",
                "-----------");

    if (entries.empty()) {
        std::printf("(no profile data recorded)\n");
    }

    const double inv_games = 1.0 / double(n_games);
    for (const auto& e : entries) {
        double pct          = wall_ns > 0
                              ? 100.0 * double(e.total_ns) / double(wall_ns)
                              : 0.0;
        double ns_per_call  = e.calls > 0
                              ? double(e.total_ns) / double(e.calls)
                              : 0.0;
        double calls_per_g  = double(e.calls)    * inv_games;
        double ms_per_g     = double(e.total_ns) * 1e-6 * inv_games;

        std::string total_str = fmt_ns(e.total_ns);
        std::string calls_str = fmt_count(e.calls);

        std::printf("%-28s %14s %11s %7.2f%% %9.0f ns %14.1f %11.2f\n",
                    e.label.c_str(),
                    calls_str.c_str(),
                    total_str.c_str(),
                    pct,
                    ns_per_call,
                    calls_per_g,
                    ms_per_g);
    }

    std::printf(
        "\nReading the table:\n"
        "  * Timings are INCLUSIVE — an outer scope (e.g. MCTS::suggest_move)\n"
        "    contains the time of every inner scope it called, so the %%wall\n"
        "    column will sum to well over 100%%. Subtract inner from outer\n"
        "    when you want exclusive (\"self\") time.\n"
        "  * Calls/game and ms/game are the per-game averages. A scope with\n"
        "    a tiny ns/call but enormous Calls/game can dominate the\n"
        "    bottleneck just from frequency; that's the call pattern this\n"
        "    tool is meant to surface (vs. `make bench`, which only measures\n"
        "    peak throughput in isolation).\n");

    return 0;
}

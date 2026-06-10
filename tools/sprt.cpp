// Sequential Probability Ratio Test for the Sequence engine.
//
// Pairs of games (same initial deal, swapped seats) between two MCTS
// configurations. After each pair we update pentanomial outcome counts,
// compute the Generalized SPRT log-likelihood ratio under a normal model
// of pair scores, and stop when:
//
//   LLR ≥ log((1-β)/α)  → accept H1 (dev gains ≥ elo1)
//   LLR ≤ log(β/(1-α))  → accept H0 (dev gains ≤ elo0)
//
// Each periodic row also reports trinomial LOS over the underlying
// single games and an Elo point estimate + 95% normal-approx CI.
//
// In-process A/B: both configs run in the same binary. Useful for
// tuning hyperparameters (ucb_c, rollout_epsilon, iterations, trees).
// For testing code changes between revisions, build a separate binary
// at each revision and drive them externally — that's out of scope here.
//
// Usage:
//   ./build/sprt --tc stc --dev-eps 0.25         # default elo0=0, elo1=5
//   ./build/sprt --tc ltc --base-ucb-c 2.0 --dev-ucb-c 2.5
//   ./build/sprt --base-iters 800 --dev-iters 800 --elo0 0 --elo1 5
//
// Pairing: each pair shares one state_seed, dev sits 0 in game 1 and
// seat 1 in game 2. Halves the games needed for a given Elo resolution because most of
// the deal-luck variance cancels across the pair.

#include "../src/state.hpp"
#include "../src/search.hpp"
#include "../src/eval.hpp"
#include "../src/rng.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using namespace seq;

// --- args ------------------------------------------------------------------

struct Args {
    int      base_iters    = 800;
    int      dev_iters     = 800;
    double   base_ucb_c    = 2.0;
    double   dev_ucb_c     = 2.0;
    double   base_eps      = 0.30;
    double   dev_eps       = 0.30;
    int      base_trees    = 8;
    int      dev_trees     = 8;
    // Defaults A/B the new search features: base = legacy behavior,
    // dev = lazy expansion + tree reuse.
    int      base_lazy     = 0;
    int      dev_lazy      = 1;
    int      base_reuse    = 0;
    int      dev_reuse     = 1;
    double   elo0          = 0.0;
    double   elo1          = 5.0;
    double   alpha         = 0.05;
    double   beta          = 0.05;
    int      max_pairs     = 20000;
    int      workers       = 16;
    uint64_t seed          = 0xC0FFEEBEEFULL;
    int      report_every  = 16;
};

[[noreturn]] static void usage_and_exit(int code) {
    std::printf(
        "Usage: sprt [options]\n"
        "  --tc stc|ltc                   shortcut: STC=800 iters, LTC=8000 iters (both sides)\n"
        "  --base-iters N --dev-iters N   per-side iteration budgets\n"
        "  --base-ucb-c X --dev-ucb-c X   PUCT exploration constant\n"
        "  --base-eps X   --dev-eps X     epsilon for epsilon-greedy rollouts\n"
        "  --base-trees N --dev-trees N   number of root-parallel trees\n"
        "  --base-lazy 0|1 --dev-lazy 0|1   lazy expansion + FPU (default base=0 dev=1)\n"
        "  --base-reuse 0|1 --dev-reuse 0|1 tree reuse across moves (default base=0 dev=1)\n"
        "  --elo0 X --elo1 Y              SPRT bounds in Elo (default 0 / 5)\n"
        "  --alpha X --beta X             type-I / type-II error rates (default 0.05 / 0.05)\n"
        "  --max-pairs N                  hard cap on game pairs (default 20000)\n"
        "  --workers N                    parallel game-pair workers (default 16)\n"
        "  --seed S                       master RNG seed (default 0xC0FFEEBEEF)\n"
        "  --report-every N               print a status row every N pairs (default 16)\n"
        "  -h, --help                     this message\n"
    );
    std::exit(code);
}

static Args parse_args(int argc, char** argv) {
    Args a;
    auto need = [&](int& i, const char* flag) -> const char* {
        if (i + 1 >= argc) {
            std::fprintf(stderr, "%s: missing value\n", flag);
            usage_and_exit(2);
        }
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        std::string f = argv[i];
        if      (f == "-h" || f == "--help")    usage_and_exit(0);
        else if (f == "--tc") {
            std::string v = need(i, "--tc");
            if      (v == "stc") { a.base_iters = a.dev_iters = 800;  }
            else if (v == "ltc") { a.base_iters = a.dev_iters = 8000; }
            else { std::fprintf(stderr, "--tc expects stc or ltc\n"); usage_and_exit(2); }
        }
        else if (f == "--base-iters")   a.base_iters   = std::atoi(need(i, f.c_str()));
        else if (f == "--dev-iters")    a.dev_iters    = std::atoi(need(i, f.c_str()));
        else if (f == "--base-ucb-c")   a.base_ucb_c   = std::atof(need(i, f.c_str()));
        else if (f == "--dev-ucb-c")    a.dev_ucb_c    = std::atof(need(i, f.c_str()));
        else if (f == "--base-eps")     a.base_eps     = std::atof(need(i, f.c_str()));
        else if (f == "--dev-eps")      a.dev_eps      = std::atof(need(i, f.c_str()));
        else if (f == "--base-trees")   a.base_trees   = std::atoi(need(i, f.c_str()));
        else if (f == "--dev-trees")    a.dev_trees    = std::atoi(need(i, f.c_str()));
        else if (f == "--base-lazy")    a.base_lazy    = std::atoi(need(i, f.c_str()));
        else if (f == "--dev-lazy")     a.dev_lazy     = std::atoi(need(i, f.c_str()));
        else if (f == "--base-reuse")   a.base_reuse   = std::atoi(need(i, f.c_str()));
        else if (f == "--dev-reuse")    a.dev_reuse    = std::atoi(need(i, f.c_str()));
        else if (f == "--elo0")         a.elo0         = std::atof(need(i, f.c_str()));
        else if (f == "--elo1")         a.elo1         = std::atof(need(i, f.c_str()));
        else if (f == "--alpha")        a.alpha        = std::atof(need(i, f.c_str()));
        else if (f == "--beta")         a.beta         = std::atof(need(i, f.c_str()));
        else if (f == "--max-pairs")    a.max_pairs    = std::atoi(need(i, f.c_str()));
        else if (f == "--workers")      a.workers      = std::atoi(need(i, f.c_str()));
        else if (f == "--seed")         a.seed         = std::strtoull(need(i, f.c_str()), nullptr, 0);
        else if (f == "--report-every") a.report_every = std::atoi(need(i, f.c_str()));
        else { std::fprintf(stderr, "unknown flag: %s\n", argv[i]); usage_and_exit(2); }
    }
    return a;
}

// --- one game --------------------------------------------------------------

// Returns +1 if dev won, -1 if base won, 0 for draw / unresolved.
static int play_one_game(const MCTSConfig& base_cfg, const MCTSConfig& dev_cfg,
                         int dev_seat, uint64_t state_seed,
                         uint64_t base_mcts_seed, uint64_t dev_mcts_seed)
{
    MCTSConfig bc = base_cfg; bc.seed = base_mcts_seed;
    MCTSConfig dc = dev_cfg;  dc.seed = dev_mcts_seed;
    MCTSEngine base_eng(bc), dev_eng(dc);

    GameState s; s.reset(state_seed);
    const int MAX_PLIES = 1200;
    int plies = 0;
    while (!s.done && plies < MAX_PLIES) {
        Move m = (s.current_player == dev_seat)
                    ? dev_eng.suggest_move(s)
                    : base_eng.suggest_move(s);
        if (!s.make_move(m)) break;
        // Report every played move to both engines so tree reuse can
        // re-root at the next search (no-op when tree_reuse is off).
        base_eng.advance(m);
        dev_eng.advance(m);
        ++plies;
    }
    if (s.winner == -1)        return 0;
    if (s.winner == dev_seat)  return +1;
    return -1;
}

// --- SPRT math -------------------------------------------------------------

static double elo_to_score(double elo) {
    return 1.0 / (1.0 + std::pow(10.0, -elo / 400.0));
}

static double score_to_elo(double s) {
    s = std::clamp(s, 1e-9, 1.0 - 1e-9);
    return -400.0 * std::log10(1.0 / s - 1.0);
}

static double normal_cdf(double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

struct SprtState {
    // Pentanomial pair counts indexed by pair score * 2 → {0..4}, i.e.
    // LL, LD/DL, DD/WL/LW, WD/DW, WW.
    long long penta[5] = {0,0,0,0,0};
    long long wins = 0, losses = 0, draws = 0;   // underlying single-game tally
};

struct SprtReport {
    long long pairs;
    long long wins, losses, draws;
    double mu_pair;        // observed mean pair score (range 0..2)
    double sigma_pair_sq;  // observed variance of pair score
    double llr;
    double upper, lower;
    double elo;
    double elo_lo, elo_hi;
    double los;
};

static SprtReport compute(const SprtState& st, double elo0, double elo1,
                          double alpha, double beta)
{
    SprtReport r{};
    long long N = st.penta[0] + st.penta[1] + st.penta[2] + st.penta[3] + st.penta[4];
    r.pairs  = N;
    r.wins   = st.wins;
    r.losses = st.losses;
    r.draws  = st.draws;
    r.upper  = std::log((1.0 - beta)  / alpha);
    r.lower  = std::log(beta          / (1.0 - alpha));

    if (N == 0) {
        r.mu_pair = 1.0;
        r.sigma_pair_sq = 0.25;
        r.elo = r.elo_lo = r.elo_hi = 0.0;
        r.los = 0.5;
        return r;
    }

    const double sv[5] = {0.0, 0.5, 1.0, 1.5, 2.0};
    double sum = 0.0;
    for (int k = 0; k < 5; ++k) sum += double(st.penta[k]) * sv[k];
    r.mu_pair = sum / double(N);

    double vsum = 0.0;
    for (int k = 0; k < 5; ++k) {
        double d = sv[k] - r.mu_pair;
        vsum += double(st.penta[k]) * d * d;
    }
    r.sigma_pair_sq = vsum / double(N);
    // Guard the early degenerate case where every pair scored identically.
    double s2 = std::max(r.sigma_pair_sq, 1e-9);

    // GSPRT under a normal model of per-pair scores: H0:E[s]=2·score(elo0),
    // H1:E[s]=2·score(elo1). With known μ0,μ1 and plug-in variance s2,
    // LLR = N · (μ1-μ0) · (μ̂ - (μ0+μ1)/2) / σ². Asymptotically equivalent
    // to the BayesElo / logistic LLR Fishtest uses; cleaner to derive.
    double s0_pair = 2.0 * elo_to_score(elo0);
    double s1_pair = 2.0 * elo_to_score(elo1);
    r.llr = double(N) * (s1_pair - s0_pair) *
            (r.mu_pair - 0.5 * (s0_pair + s1_pair)) / s2;

    double per_game = 0.5 * r.mu_pair;
    r.elo = score_to_elo(per_game);
    // Per-pair score variance / N → per-game score variance / (4N).
    double sd_per_game = std::sqrt(s2 / (4.0 * double(N)));
    r.elo_lo = score_to_elo(per_game - 1.96 * sd_per_game);
    r.elo_hi = score_to_elo(per_game + 1.96 * sd_per_game);

    long long W = st.wins, L = st.losses;
    r.los = (W + L == 0) ? 0.5
                          : normal_cdf(double(W - L) / std::sqrt(double(W + L)));
    return r;
}

// --- output ----------------------------------------------------------------

static void print_header(const Args& a) {
    std::printf("SPRT: base [iters=%d ucb_c=%.2f eps=%.2f trees=%d lazy=%d reuse=%d]"
                "  vs  dev [iters=%d ucb_c=%.2f eps=%.2f trees=%d lazy=%d reuse=%d]\n",
                a.base_iters, a.base_ucb_c, a.base_eps, a.base_trees,
                a.base_lazy, a.base_reuse,
                a.dev_iters,  a.dev_ucb_c,  a.dev_eps,  a.dev_trees,
                a.dev_lazy, a.dev_reuse);
    std::printf("H0: elo <= %.1f   H1: elo >= %.1f   alpha=%.3f beta=%.3f\n",
                a.elo0, a.elo1, a.alpha, a.beta);
    std::printf("Bounds: LLR in [%.3f, %.3f]   max_pairs=%d   workers=%d\n\n",
                std::log(a.beta / (1.0 - a.alpha)),
                std::log((1.0 - a.beta) / a.alpha),
                a.max_pairs, a.workers);
    std::printf("%6s %5s %5s %5s   %4s %4s %4s %4s %4s   %7s   %7s   %18s   %5s\n",
                "pairs", "W", "L", "D",
                "LL", "LD", "DD", "WD", "WW",
                "LLR", "Elo", "Elo 95% CI", "LOS");
}

static void print_row(const SprtState& st, const SprtReport& r) {
    std::printf("%6lld %5lld %5lld %5lld   %4lld %4lld %4lld %4lld %4lld   "
                "%+7.3f   %+7.2f   %+7.2f .. %+7.2f   %5.3f\n",
                r.pairs, r.wins, r.losses, r.draws,
                st.penta[0], st.penta[1], st.penta[2], st.penta[3], st.penta[4],
                r.llr, r.elo, r.elo_lo, r.elo_hi, r.los);
    std::fflush(stdout);
}

// --- main ------------------------------------------------------------------

int main(int argc, char** argv) {
    Args a = parse_args(argc, argv);

    MCTSConfig base_cfg, dev_cfg;
    base_cfg.iterations       = a.base_iters;
    base_cfg.ucb_c            = a.base_ucb_c;
    base_cfg.rollout_epsilon  = a.base_eps;
    base_cfg.n_parallel_trees = a.base_trees;
    base_cfg.lazy_expansion   = a.base_lazy != 0;
    base_cfg.tree_reuse       = a.base_reuse != 0;
    base_cfg.n_threads        = 1;  // game-level parallelism only

    dev_cfg.iterations        = a.dev_iters;
    dev_cfg.ucb_c             = a.dev_ucb_c;
    dev_cfg.rollout_epsilon   = a.dev_eps;
    dev_cfg.n_parallel_trees  = a.dev_trees;
    dev_cfg.lazy_expansion    = a.dev_lazy != 0;
    dev_cfg.tree_reuse        = a.dev_reuse != 0;
    dev_cfg.n_threads         = 1;

    print_header(a);

    // Pre-derive every pair's seeds up-front so the result is reproducible
    // regardless of worker scheduling — same trick compete.cpp uses.
    Xoshiro256pp seed_rng(a.seed);
    struct PairSeeds {
        uint64_t state_seed;
        uint64_t g1_base, g1_dev;
        uint64_t g2_base, g2_dev;
    };
    std::vector<PairSeeds> seeds(a.max_pairs);
    for (int p = 0; p < a.max_pairs; ++p) {
        seeds[p].state_seed = seed_rng();
        seeds[p].g1_base    = seed_rng();
        seeds[p].g1_dev     = seed_rng();
        seeds[p].g2_base    = seed_rng();
        seeds[p].g2_dev     = seed_rng();
    }

    SprtState           state;
    std::mutex          st_mu;
    std::atomic<int>    next_pair{0};
    std::atomic<bool>   stop_flag{false};
    long long           last_reported_at = 0;  // protected by st_mu

    auto t0 = std::chrono::high_resolution_clock::now();

    auto worker = [&]() {
        for (;;) {
            if (stop_flag.load(std::memory_order_relaxed)) return;
            int p = next_pair.fetch_add(1, std::memory_order_relaxed);
            if (p >= a.max_pairs) return;
            const auto& s = seeds[p];

            int r1 = play_one_game(base_cfg, dev_cfg, /*dev_seat=*/0,
                                   s.state_seed, s.g1_base, s.g1_dev);
            int r2 = play_one_game(base_cfg, dev_cfg, /*dev_seat=*/1,
                                   s.state_seed, s.g2_base, s.g2_dev);

            // r ∈ {-1,0,+1} → per-game dev score (r+1)/2; pair sum in {0..2}.
            int bucket = r1 + r2 + 2;   // 0..4 by construction

            std::lock_guard<std::mutex> g(st_mu);
            state.penta[bucket]++;
            for (int r : {r1, r2}) {
                if      (r > 0) state.wins++;
                else if (r < 0) state.losses++;
                else            state.draws++;
            }

            long long N = state.penta[0] + state.penta[1] + state.penta[2]
                        + state.penta[3] + state.penta[4];

            if (N >= last_reported_at + a.report_every) {
                last_reported_at = N;
                auto rep = compute(state, a.elo0, a.elo1, a.alpha, a.beta);
                print_row(state, rep);
                if (rep.llr >= rep.upper) {
                    if (!stop_flag.exchange(true))
                        std::printf("==> H1 accepted (dev gains >= %.1f Elo)\n", a.elo1);
                } else if (rep.llr <= rep.lower) {
                    if (!stop_flag.exchange(true))
                        std::printf("==> H0 accepted (dev gains <= %.1f Elo)\n", a.elo0);
                }
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(a.workers);
    for (int t = 0; t < a.workers; ++t) threads.emplace_back(worker);
    for (auto& th : threads) th.join();

    auto rep = compute(state, a.elo0, a.elo1, a.alpha, a.beta);
    std::printf("\nFinal:\n");
    print_row(state, rep);

    if (rep.llr < rep.upper && rep.llr > rep.lower) {
        std::printf("==> Inconclusive at %lld pairs; "
                    "LLR=%+.3f in (%.3f, %.3f). Re-run with --max-pairs higher.\n",
                    rep.pairs, rep.llr, rep.lower, rep.upper);
    }
    double total_s = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t0).count();
    std::printf("Total wall time: %.1fs (%.3fs/pair)\n",
                total_s, total_s / double(std::max(1LL, rep.pairs)));
    return 0;
}

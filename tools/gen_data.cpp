// Self-play data for fitting the search's score and value tables
// (tools/fit_tables.py turns the output into src/learned_tables.hpp).
//
// Plays engine-vs-engine games and, at every searched decision, writes
//
//   policy.bin  one row per legal move: which window states the move's
//               cell sits in (split by how well the mover's hand supports
//               each window), and how many visits the search gave it;
//   value.bin   one row per position: the board's window-state counts,
//               sequences, jacks in hand, the search's root value and
//               the final result of the game.
//
// Window states are the same (joker, own chips, opponent chips) triples
// the search scores moves and positions with, so what is learned drops
// straight into its lookup tables.
//
// Usage:
//   ./build/gen_data OUT_DIR [N_GAMES] [ITERATIONS] [SEED] [WORKERS]
//
// Defaults: 2000 games, 3000 iterations per move, 16 workers.

#include "../src/state.hpp"
#include "../src/search.hpp"
#include "../src/rng.hpp"
#include "../src/window_states.hpp"
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using namespace seq;

constexpr int N_STATES = N_WINDOW_STATES;

#pragma pack(push, 1)
struct PolicyRow {
    int32_t decision;           // rows of one decision share this id
    int32_t visits;
    uint8_t kind;               // 0 regular, 1 two-eyed, 2 one-eyed, 3 dead
    uint8_t stage;              // 2 * (mover has a sequence) + (opp has one)
    // windows through the cell, by hand support and (j, mine, theirs)
    uint8_t hist[N_HAND_LEVELS][N_STATES];
    uint8_t dead;               // dead windows through the cell
    uint8_t pad;
};
struct ValueRow {
    int32_t game;
    uint8_t ply, stm, seq0, seq1;
    uint8_t j2_0, j1_0, j2_1, j1_1;
    float   root_q_p0;          // search's root value, player 0's view, [-1, 1]
    uint8_t winner;             // 0, 1, or 2 = no result
    uint8_t hist[N_STATES];     // all 192 windows, by (j, n0, n1)
    uint8_t dead;
    uint8_t pad[2];
};
#pragma pack(pop)

struct WindowView {
    uint8_t j[N_WINDOWS], n0[N_WINDOWS], n1[N_WINDOWS], dead[N_WINDOWS];

    explicit WindowView(const GameState& s) {
        const BB100 b0 = s.chips[0] | s.locked[0];
        const BB100 b1 = s.chips[1] | s.locked[1];
        for (int w = 0; w < N_WINDOWS; ++w) {
            const BB100 m = WINDOW_MASKS[w];
            j[w]    = (m & JOKER_MASK).any();
            n0[w]   = uint8_t((m & b0).popcount());
            n1[w]   = uint8_t((m & b1).popcount());
            dead[w] = (m & s.locked[0]).popcount() >= 2
                   || (m & s.locked[1]).popcount() >= 2;
        }
    }
};

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: gen_data OUT_DIR [N_GAMES] [ITERATIONS] [SEED] [WORKERS]\n");
        return 2;
    }
    const std::string out_dir = argv[1];
    const int n_games  = argc >= 3 ? std::atoi(argv[2]) : 2000;
    const int iters    = argc >= 4 ? std::atoi(argv[3]) : 3000;
    const uint64_t seed = argc >= 5 ? std::strtoull(argv[4], nullptr, 0) : 0xDA7A5EEDULL;
    const int workers  = argc >= 6 ? std::atoi(argv[5]) : 16;

    FILE* fp = std::fopen((out_dir + "/policy.bin").c_str(), "wb");
    FILE* fv = std::fopen((out_dir + "/value.bin").c_str(), "wb");
    if (!fp || !fv) { std::fprintf(stderr, "cannot write to %s\n", out_dir.c_str()); return 1; }

    Xoshiro256pp seed_rng(seed);
    struct Seeds { uint64_t deal, a, b; };
    std::vector<Seeds> seeds(static_cast<size_t>(n_games));
    for (auto& s : seeds) { s.deal = seed_rng(); s.a = seed_rng(); s.b = seed_rng(); }

    std::mutex mu;
    std::atomic<int> next_game{0};
    std::atomic<int32_t> next_decision{0};
    long long n_policy = 0, n_value = 0;
    int done_games = 0;

    auto worker = [&]() {
        std::vector<PolicyRow> prow;
        std::vector<ValueRow>  vrow;
        for (;;) {
            const int g = next_game.fetch_add(1);
            if (g >= n_games) return;
            prow.clear();
            vrow.clear();

            MCTSConfig cfg;
            cfg.iterations = iters;
            cfg.n_threads  = 1;
            cfg.seed = seeds[size_t(g)].a;
            MCTSEngine e0(cfg);
            cfg.seed = seeds[size_t(g)].b;
            MCTSEngine e1(cfg);

            GameState s;
            s.reset(seeds[size_t(g)].deal);
            for (int ply = 0; !s.done && ply < 1200; ++ply) {
                const int p = s.current_player;
                MCTSEngine& eng = p == 0 ? e0 : e1;
                const Move m = eng.suggest_move(s);
                const auto& rm = eng.last_root_moves();

                if (!rm.empty()) {
                    const WindowView wv(s);
                    const int stage = stage_of(s.sequences[p], s.sequences[1 - p]);
                    const int32_t decision = next_decision.fetch_add(1);
                    const BB100 occupied = s.chips[0] | s.chips[1]
                                         | s.locked[0] | s.locked[1];
                    for (const RootMoveStat& r : rm) {
                        PolicyRow row{};
                        row.decision = decision;
                        row.visits   = r.visits;
                        row.stage    = uint8_t(stage);
                        const int card = s.hands[p][r.move.card_idx];
                        if (r.move.is_dead()) {
                            row.kind = 3;
                        } else {
                            row.kind = card == TWO_EYED_JACK ? 1
                                     : card == ONE_EYED_JACK ? 2 : 0;
                            // Empty cells the mover could still fill from
                            // hand after this move (regular cards only, the
                            // played card and the target cell excluded).
                            BB100 support;
                            if (row.kind != 2) {
                                for (int i = 0; i < HAND_SIZE; ++i) {
                                    const int c = s.hands[p][i];
                                    if (i == r.move.card_idx || c < 0 || c >= N_REGULAR_CARDS) continue;
                                    for (int k = 0; k < 2; ++k) {
                                        const int cc = CARD_TO_CELLS[c][k];
                                        if (cc >= 0 && !occupied.test(cc)) support.set(cc);
                                    }
                                }
                                support.reset(r.move.cell);
                            }
                            const int16_t* ws = CELL_WINDOWS[r.move.cell];
                            for (int i = 0; ws[i] >= 0; ++i) {
                                const int w = ws[i];
                                if (wv.dead[w]) { ++row.dead; continue; }
                                const int me  = p == 0 ? wv.n0[w] : wv.n1[w];
                                const int opp = p == 0 ? wv.n1[w] : wv.n0[w];
                                const int h = std::min(
                                    (WINDOW_MASKS[w] & support).popcount(),
                                    N_HAND_LEVELS - 1);
                                ++row.hist[h][window_state_id(wv.j[w], me, opp)];
                            }
                        }
                        prow.push_back(row);
                    }

                    ValueRow v{};
                    v.game = g;
                    v.ply  = uint8_t(std::min(ply, 255));
                    v.stm  = uint8_t(p);
                    v.seq0 = uint8_t(s.sequences[0]);
                    v.seq1 = uint8_t(s.sequences[1]);
                    for (int q = 0; q < 2; ++q) {
                        for (int i = 0; i < HAND_SIZE; ++i) {
                            const int c = s.hands[q][i];
                            if (c == TWO_EYED_JACK) ++(q == 0 ? v.j2_0 : v.j2_1);
                            if (c == ONE_EYED_JACK) ++(q == 0 ? v.j1_0 : v.j1_1);
                        }
                    }
                    const double q_mover = 2.0 * eng.last_stats().root_winrate - 1.0;
                    v.root_q_p0 = float(p == 0 ? q_mover : -q_mover);
                    v.winner = 2;
                    for (int w = 0; w < N_WINDOWS; ++w) {
                        if (wv.dead[w]) { ++v.dead; continue; }
                        ++v.hist[window_state_id(wv.j[w], wv.n0[w], wv.n1[w])];
                    }
                    vrow.push_back(v);
                }

                e0.advance(s, m);
                e1.advance(s, m);
                if (!s.make_move(m)) break;
            }
            for (auto& v : vrow) v.winner = uint8_t(s.winner < 0 ? 2 : s.winner);

            std::lock_guard<std::mutex> lock(mu);
            std::fwrite(prow.data(), sizeof(PolicyRow), prow.size(), fp);
            std::fwrite(vrow.data(), sizeof(ValueRow), vrow.size(), fv);
            n_policy += (long long)prow.size();
            n_value  += (long long)vrow.size();
            if (++done_games % 100 == 0) {
                std::printf("%d/%d games, %lld positions, %lld move rows\n",
                            done_games, n_games, n_value, n_policy);
                std::fflush(stdout);
            }
        }
    };

    std::vector<std::thread> threads;
    for (int t = 0; t < workers; ++t) threads.emplace_back(worker);
    for (auto& th : threads) th.join();
    std::fclose(fp);
    std::fclose(fv);
    std::printf("done: %d games, %lld positions, %lld move rows "
                "(policy row %zu bytes, value row %zu bytes)\n",
                n_games, n_value, n_policy, sizeof(PolicyRow), sizeof(ValueRow));
    return 0;
}

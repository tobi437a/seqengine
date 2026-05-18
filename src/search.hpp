// Monte-Carlo Tree Search for Sequence — single-observer Information-
// Set MCTS (Cowling–Powley–Whitehouse 2012) with AlphaZero-style PUCT
// selection.
//
// One tree shared across all iterations; each iteration re-determinizes
// the opponent's hand and the deck order, then walks the tree using only
// the children whose move is legal in the current determinization. Tree
// nodes store only the move from parent — state is rebuilt by replaying
// moves from the determinized root each iteration. Lower memory per
// node and biases less toward one wrong guess of the opponent's hand
// than the classic PIMC fix for strategy fusion.
//
// Design points:
//
//   * PUCT selection: argmax Q(a) + c · P(a) · √Σ_b N(b) / (1 + N(a)).
//     The prior P(a) is a softmax over the shaping score of each
//     legal action at the current node — the same handcrafted signal
//     the rollout policy uses to break out of uniform random. With 50+
//     legal moves at the root, a uniform UCB1 spreads early visits
//     thinly; the prior funnels them onto plausible moves first.
//     c configurable (default 2.0).
//
//   * ε-greedy heuristic rollouts. Pure-greedy would make every rollout
//     from a state identical, defeating MCTS. ε=0.30 by default; gives
//     enough diversity that two rollouts diverge within a few plies.
//
//   * Robust-child final selection: argmax visits over root children
//     (not argmax mean). Visit count is the more stable signal because
//     it incorporates the bandit's own confidence in its estimates.

#pragma once
#include "state.hpp"
#include "eval.hpp"
#include "rng.hpp"
#include <memory>
#include <vector>

namespace seq {

struct MCTSConfig {
    int    iterations        = 2000;
    double ucb_c             = 2.0;
    double rollout_epsilon   = 0.30;    // ε for ε-greedy rollout policy
    int    max_rollout_steps = 400;
    bool   determinize       = true;

    // --- Length-discounted terminal value ---
    // Rollout value = terminal_value × length_decay^plies. With decay
    // = 1.0 (default), this is just terminal ±1. With decay < 1.0, MCTS
    // prefers fast wins / slow losses. Ablation showed decay < 1 didn't
    // help overall winrate vs the heuristic at moderate iter budgets,
    // but the knob is useful for tactical positions where multiple
    // winning continuations exist and we want the shortest one.
    double length_decay      = 1.0;

    // --- Rollout policy ---
    // When true, the rollout policy will take a sequence-completing
    // move whenever one is available, overriding the ε-greedy choice.
    // Default OFF: ablation showed it actively hurt, because forcing
    // both rollout players to grab obvious sequences makes rollouts
    // collapse into deterministic tempo races that amplify tiny initial
    // differences. Left as a feature flag for experimentation.
    bool   prefer_completing_moves = false;

    uint64_t seed            = 0;       // 0 → random_device

    // --- Number of independent root-parallel trees ---
    // Each tree is an independent ISMCTS search (own root, own RNG);
    // every iteration within a tree re-determinizes the opponent's
    // hand. We aggregate root visits across trees at the end. Since
    // each tree already varies its determinization per iteration,
    // running N trees doesn't add determinization diversity — it's
    // purely a parallelism knob.
    // n=1 → single-threaded one-tree search.
    // n>1 → root-parallel; total budget is split as iterations/n per
    //       tree. Trades per-tree depth for wall-clock speedup.
    int      n_parallel_trees   = 8;

    // --- Parallelism ---
    // Number of worker threads to distribute the n_parallel_trees trees
    // across. Trees are independent (no shared state), so this gives
    // near-linear scaling up to min(n_threads, n_parallel_trees) cores.
    // Effective threads will be clamped to n_parallel_trees.
    //
    //   n_threads = 1  : sequential (default; deterministic for tests).
    //   n_threads = N  : spawn N threads via std::async.
    //   n_threads = 0  : auto — std::thread::hardware_concurrency(),
    //                    capped at min(n_parallel_trees, 16).
    //
    // The Python wrapper (MCTSOpponent) defaults to n_threads=0 so
    // end-users get parallelism for free; the C++ default of 1 keeps
    // tests, benchmarks, and the compete harness deterministic.
    int      n_threads          = 1;
};

struct MCTSStats {
    int    iterations         = 0;
    int    max_depth          = 0;
    double seconds            = 0.0;
    Move   best_move          = {};
    int    best_move_visits   = 0;
    double best_move_winrate  = 0.0;   // mean value for the player to move
                                       // at root, mapped to [0, 1]
    int    root_legal_moves   = 0;
};

class MCTSEngine {
public:
    explicit MCTSEngine(MCTSConfig cfg = {});

    // Choose a move for `s.current_player` to play. `s` is observed in
    // full; if cfg.determinize, the opponent's hand and deck are
    // resampled before search so we don't cheat on hidden info.
    Move suggest_move(const GameState& s);

    const MCTSStats& last_stats() const { return stats_; }
    MCTSConfig&       config()           { return cfg_; }
    const MCTSConfig& config() const     { return cfg_; }

private:
    MCTSConfig    cfg_;
    Xoshiro256pp  rng_;
    MCTSStats     stats_;
};

} // namespace seq

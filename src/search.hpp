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

    // --- Truncated rollouts ---
    // rollout_cutoff = 0 plays every rollout to terminal. With N > 0 (default
    // 12) a
    // rollout that is still running after N plies stops and scores the
    // position statically (P0 perspective):
    //     tanh(cutoff_scale · (Φ0 − Φ1 + cutoff_seq_bonus · (seq0 − seq1)))
    // where Φp = total_phi(s, p). Shorter rollouts are cheaper, and the
    // static eval is lower-variance than a long ε-greedy playout.
    int    rollout_cutoff    = 12;
    double cutoff_scale      = 0.2;
    double cutoff_seq_bonus  = 4.0;

    // --- Jack hand value ---
    // Φ is board-only, so without these a jack in hand is worth nothing
    // at a cutoff leaf, and the greedy rollout/prior (whose best two-eyed
    // jack placement always scores ≥ any regular card's) spends jacks on
    // the first positional gain — e.g. a two-eyed jack on move 1. Each
    // held jack adds its value (Φ units) to the cutoff eval, and playing
    // one subtracts the same value from its shaping score in the rollout
    // policy and the PUCT prior. 0 = off. SPRT at 8000 iters, 3.0/1.5
    // vs 0/0: +26 Elo (95% CI +9..+44, 465 pairs).
    double jack2_value       = 3.0;     // two-eyed (wild)
    double jack1_value       = 1.5;     // one-eyed (remove)

    // --- Where the search's scoring departs from eval.hpp ---
    // Each of these fixes something the shaping score / Φ get wrong for
    // search purposes; turn one off to get the plain eval.hpp behavior
    // back for an A/B.
    //
    // dead_windows: a new sequence may share at most one chip with an
    //   existing one, so a window holding two or more of a player's
    //   locked chips can never complete. Score it as zero for both sides
    //   instead of as a live four/five.
    // shaping_seq_bonus: added (Φ units) to a placement's score when it
    //   completes a five. The shaping score alone gives that ≈ −0.03
    //   (the 4 → 5 weights are equal), so rollout players would walk
    //   past their own sequences. Hand-tuned scores only.
    // random_ply_jacks: false = a random rollout ply picks among the
    //   non-jack moves. A two-eyed jack contributes ~90 of ~105 legal
    //   moves, so a uniformly random ply throws it away almost every
    //   time one is in hand.
    bool   dead_windows      = true;
    double shaping_seq_bonus = 4.0;
    bool   random_ply_jacks  = false;

    // --- Learned tables ---
    // Use the move-score / value tables fitted to the engine's own
    // searches (src/learned_tables.hpp, tools/fit_tables.py) instead of
    // the hand-set weights from eval.hpp. The learned move score also
    // knows which cards the mover holds: a line they can keep extending
    // from hand scores higher. With learned_policy the jack values and
    // shaping_seq_bonus no longer affect move scores; with learned_value
    // the cutoff_* and jack values no longer affect the leaf evaluation.
    //
    // Per iteration these are stronger than the hand-set tables (+23 Elo
    // at 3000 iterations, +9 at 20000) but iterations take ~1.8x as long,
    // because the sharper prior drives the tree deeper.
    bool   learned_policy    = true;
    bool   learned_value     = true;

    // --- First-play urgency ---
    // Selection runs PUCT over ALL legal moves at a node: visited
    // children use their empirical Q, unvisited moves use parent_q −
    // fpu_reduction, and a node is only materialized when selected.
    double fpu_reduction     = 0.10;

    // --- Tree reuse across moves ---
    // When true, the engine keeps its tree between suggest_move calls.
    // The harness reports every actually-played move via advance(); at
    // the next suggest_move the tree is re-rooted at the node reached by
    // those moves, carrying over the subtree's statistics. If a played
    // move has no node (or advance() was never called), the search
    // starts from an empty root.
    bool   tree_reuse        = true;

    uint64_t seed            = 0;       // 0 → random_device

    // --- Parallelism ---
    // Worker threads, all searching the one shared tree (lock-free, with
    // virtual loss).
    //
    //   n_threads = 1  : sequential (default; deterministic given a seed).
    //   n_threads = N  : N threads.
    //   n_threads = 0  : auto — std::thread::hardware_concurrency().
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
    double root_winrate       = 0.0;   // same, averaged over the whole search
    int    root_legal_moves   = 0;     // distinct (card, target) moves
    int    reused_visits      = 0;     // root visits carried over by tree reuse
    int    threads            = 0;     // worker threads this search used
};

// One legal root move and what the last search thought of it.
struct RootMoveStat {
    Move   move;
    int    visits  = 0;
    double winrate = 0.0;      // mean value for the mover, mapped to [0, 1]
};

class MCTSEngine {
public:
    explicit MCTSEngine(MCTSConfig cfg = {});
    ~MCTSEngine();
    MCTSEngine(MCTSEngine&&) noexcept;
    MCTSEngine& operator=(MCTSEngine&&) noexcept;

    // Choose a move for `s.current_player` to play. `s` is observed in
    // full; if cfg.determinize, the opponent's hand and the deck are
    // resampled every iteration so we don't cheat on hidden info.
    Move suggest_move(const GameState& s);

    // Tree reuse: report a move that was actually played in the game
    // (by either side, in order). `before` is the state the move was
    // played from — only the mover's card in slot m.card_idx is read.
    // At the next suggest_move the kept tree is re-rooted along the
    // reported moves. Optional — if it's never called, every search
    // starts fresh.
    void advance(const GameState& before, Move m);

    // Same, for callers that know the card but not the hand slot (e.g.
    // an opponent's move at a physical table). `card_type` is the 0..49
    // card index; `cell` is the board cell, or Move::DEAD for a dead-
    // card declaration.
    void advance_card(int card_type, int cell);

    const MCTSStats& last_stats() const { return stats_; }

    // Every legal move at the root of the last search with its visit
    // count and mean value (unvisited moves included, with 0 visits).
    // Empty if the last call didn't search (0 or 1 legal moves).
    const std::vector<RootMoveStat>& last_root_moves() const { return root_moves_; }
    MCTSConfig&       config()           { return cfg_; }
    const MCTSConfig& config() const     { return cfg_; }

private:
    struct Tree;                        // persistent search tree
    int prepare_tree(int budget);

    MCTSConfig    cfg_;
    Xoshiro256pp  rng_;
    MCTSStats     stats_;
    std::unique_ptr<Tree> tree_;
    std::vector<uint16_t> pending_;     // keys of moves played since last search
    std::vector<RootMoveStat> root_moves_;
};

// Test hook: plays `n_games` random games checking the search's internal
// fast move generation / make-move / scoring against GameState and
// eval.hpp. Returns the number of mismatches (0 = all consistent).
int search_selfcheck(int n_games, uint64_t seed);

} // namespace seq

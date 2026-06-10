#include "search.hpp"
#include "profile.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <future>
#include <random>
#include <thread>

namespace seq {

// Outcome of a terminal state from player 0's perspective.
static double terminal_value_p0(const GameState& s) {
    if (s.winner == 0) return  1.0;
    if (s.winner == 1) return -1.0;
    return 0.0;
}

// ---------------------------------------------------------------------------
// Determinization
// ---------------------------------------------------------------------------
//
// From the engine's perspective the opponent's hand is "known" because
// we hold a full GameState. To play fair on the hidden-info side, we
// treat the opponent's hand as unknown: dump it back into the deck,
// reshuffle the deck, deal HAND_SIZE fresh cards. The shuffle also
// randomizes the deck order, which we'd otherwise be implicitly assuming.
//
// Called once per ISMCTS iteration on the iteration's local state copy,
// so the search averages over many distinct guesses of the hidden info.

static void determinize_opponent_hand(GameState& s, int our_player,
                                      Xoshiro256pp& rng)
{
    SEQ_PROFILE_SCOPE("MCTS::determinize");
    const int opp = 1 - our_player;
    for (int i = 0; i < HAND_SIZE; ++i) {
        if (s.hands[opp][i] >= 0) {
            s.deck.push_back(s.hands[opp][i]);
            s.hands[opp][i] = -1;
        }
    }
    std::shuffle(s.deck.begin(), s.deck.end(), rng);
    for (int i = 0; i < HAND_SIZE; ++i) {
        if (!s.deck.empty()) {
            s.hands[opp][i] = s.deck.back();
            s.deck.pop_back();
        }
    }
}

// ---------------------------------------------------------------------------
// Rollout policy
// ---------------------------------------------------------------------------
//
// Three-tier policy, in priority order:
//
//   1. If cfg.prefer_completing_moves, scan for a move that completes a
//      sequence (or wins). Take the first one found. This is the
//      single biggest fix to the rollout's blindness, because the
//      shaping score gives Δ = 0 for a 4-window → 5-window transition
//      (WINDOW_WEIGHTS[4] == WINDOW_WEIGHTS[5] == 1.0).
//   2. With probability ε, pick a uniformly random move (diversity —
//      pure greedy collapses every rollout to the same trajectory).
//   3. Otherwise, greedy on shaping score.
//
// Tier 1's check is cheap: a placement at `cell` completes a sequence
// iff some window through `cell` already had popcount-of-mine == 4 and
// no opponent chips. We can verify without applying the move.

// Does placing `player`'s chip at `cell` complete a sequence?
// (Pure read — does not mutate state.)
static bool placement_completes_sequence(const GameState& s, int player, int cell) {
    const int opp = 1 - player;
    const BB100 mine_total = s.chips[player] | s.locked[player] | JOKER_MASK;
    const BB100 opp_total  = s.chips[opp]    | s.locked[opp];
    const int16_t* ws = CELL_WINDOWS[cell];
    for (int i = 0; ws[i] >= 0; ++i) {
        const BB100 w = WINDOW_MASKS[ws[i]];
        if (!w.test(cell)) continue;             // (safety; CELL_WINDOWS only lists windows through cell)
        if ((w & opp_total).any()) continue;     // blocked, can't form a sequence here
        // cell is currently empty (move is a legal placement), so it isn't
        // already in mine_total. If 4 are already there, adding cell makes 5.
        if ((w & mine_total).popcount() == 4) return true;
    }
    return false;
}

// Returns -1 if no completing move is found.
static int find_completing_move(GameState& s, const MoveList& moves, int player) {
    for (int i = 0; i < moves.count; ++i) {
        Move m = moves.moves[i];
        if (m.is_dead()) continue;
        int card_type = s.hands[player][m.card_idx];
        if (card_type == ONE_EYED_JACK) continue;  // removes, doesn't place
        if (placement_completes_sequence(s, player, m.cell)) return i;
    }
    return -1;
}

static Move pick_rollout_move(GameState& s, const MoveList& moves,
                              Xoshiro256pp& rng, const MCTSConfig& cfg)
{
    SEQ_PROFILE_SCOPE("MCTS::pick_rollout_move");
    if (moves.count == 1) return moves.moves[0];

    const int player = s.current_player;

    // Tier 1: complete a sequence if we can.
    if (cfg.prefer_completing_moves) {
        int idx = find_completing_move(s, moves, player);
        if (idx >= 0) return moves.moves[idx];
    }

    // Tier 2: ε random.
    if (rng.rand_double() < cfg.rollout_epsilon) {
        return moves.moves[rng.rand_int(0, moves.count - 1)];
    }

    // Tier 3: greedy on shaping. Dead-card declarations score 0 by
    // definition, so they only get picked if no real move scores
    // positive — useful as a "punt" when nothing helps.
    //
    // Cache by (flavor, cell): the loop is dominated by two-eyed-jack
    // slots that all emit moves at the same set of empty cells, and
    // duplicate hand slots otherwise revisit the same (flavor, cell)
    // pairs. See ShapingCache for the full justification.
    const ShapingMasks masks = build_shaping_masks(s, player);
    ShapingCache cache(masks);
    double best_score = -std::numeric_limits<double>::infinity();
    int best_idx = 0;
    for (int i = 0; i < moves.count; ++i) {
        Move m = moves.moves[i];
        double sc;
        if (m.is_dead()) {
            sc = 0.0;
        } else {
            int card_type = s.hands[player][m.card_idx];
            sc = cache.score(card_type, m.cell);
        }
        if (sc > best_score) {
            best_score = sc;
            best_idx   = i;
        }
    }
    return moves.moves[best_idx];
}

// ---------------------------------------------------------------------------
// Rollout
// ---------------------------------------------------------------------------
//
// Roll the position out to terminal (or truncation), then return
//
//     terminal_value × length_decay^plies
//
// where terminal_value is ±1 (or 0 if truncated). With decay=0.99 a
// 1-ply win is +0.99, an 80-ply win is +0.45. This gives MCTS a clear
// preference for fast wins / slow losses without the high variance of
// summing per-step rewards (which an earlier ablation showed actively
// hurt play — variance dominated signal in mid-budget searches).
//
// The decay also breaks the ties that pure ±1 produces between
// "winning this move" and "winning eventually", which is what was
// confusing MCTS at near-win positions.

static double rollout(GameState& state, Xoshiro256pp& rng,
                      const MCTSConfig& cfg)
{
    SEQ_PROFILE_SCOPE("MCTS::rollout");
    int plies = 0;
    MoveList moves;
    for (; plies < cfg.max_rollout_steps && !state.done; ++plies) {
        state.legal_moves(moves);
        if (moves.count == 0) break;
        Move m = pick_rollout_move(state, moves, rng, cfg);
        if (!state.make_move(m)) break;
    }
    double terminal = terminal_value_p0(state);   // ±1 or 0
    if (terminal == 0.0) return 0.0;              // truncated / draw
    double decay = std::pow(cfg.length_decay, double(plies));
    return terminal * decay;
}

// ---------------------------------------------------------------------------
// ISMCTS — single-observer Information-Set MCTS
// ---------------------------------------------------------------------------
//
// One tree shared across all iterations of a worker. Each iteration:
//
//   1. Determinize the opponent's hand and the deck order at the root.
//   2. Walk the tree from the root, applying each chosen move to a
//      live GameState copy. At each step:
//        - Compute legal moves in the current determinization.
//        - If any legal move has no matching child, expand one of those
//          untried legal moves and proceed to rollout.
//        - Else: compute a softmax prior over the legal children's
//          shaping scores and pick the legal child maximizing PUCT
//             Q(a) + c · P(a) · √Σ_b N(b) / (1 + N(a))
//          where the sum runs over the currently-legal children only,
//          so children that aren't legal in this determinization
//          don't dilute the exploration bonus.
//   3. Rollout from the leaf to terminal (or truncation).
//   4. Backup visits + value along the descent path.
//
// Node memory is small (just the move + counts) because we don't store
// state. The cost is that descent has to make moves on a live state.
//
// On ISMCTS + PUCT: with plain UCB1 the standard ISMCTS fix is a per-
// child "availability count" so moves that are rarely legal aren't
// unfairly punished for having few visits. PUCT folds that concern in
// differently: the prior P(a) directly biases selection toward
// plausible moves regardless of how often they've been legal, and the
// exploration sum is over the currently-legal children only, so the
// confidence bound a child accumulates is paced by how often it's
// actually been in contention.

// Pack a Move into a 16-bit key. Used to sort node children for binary-
// search lookup during descent.
//   card_idx ∈ [-1, HAND_SIZE)   (we only emit ≥ 0 in practice)
//   cell     ∈ [-1, N_CELLS)
// (+1 shifts keep keys non-negative; 128 > N_CELLS + 1 so the packing
//  is one-to-one.)
static inline uint16_t move_key(Move m) {
    return uint16_t((int(m.card_idx) + 1) * 128 + (int(m.cell) + 1));
}

// One entry in a node's flat children list. 8 bytes per entry: tightly
// packed so a binary search through hundreds of root children stays in
// just a handful of cache lines.
struct ChildEntry {
    uint16_t key;       // move_key(move) — sorted, used as binary-search key
    Move     move;      // 2 bytes (card_idx, cell)
    int32_t  node_idx;  // index into ISTree::nodes
};

// ISMCTS tree node. No parent pointer — descent collects the path into
// a local array and backup walks it. No std::unordered_map for children
// — instead a flat std::vector<ChildEntry> kept sorted by move_key, so
// lookups are a cache-friendly binary search and insertions are a
// memmove on contiguous 8-byte entries.
struct ISNode {
    Move    move_from_parent{};
    int     visits       = 0;
    double  value_p0     = 0.0;
    std::vector<ChildEntry> children;
};

// All nodes for one ISMCTS tree live in this arena (one std::vector per
// tree). We reserve up-front so node indices remain stable for the
// lifetime of the tree — every iteration expands at most one node, so
// the upper bound is (1 root) + (per_tree iterations).
struct ISTree {
    std::vector<ISNode> nodes;

    explicit ISTree(int budget_hint) {
        nodes.reserve(size_t(budget_hint) + 2);
        nodes.emplace_back();   // root at index 0
    }
    ISNode&       root()       { return nodes[0]; }
    ISNode&       at(int32_t i)       { return nodes[i]; }
    const ISNode& at(int32_t i) const { return nodes[i]; }
};

// Persistent root-parallel forest, kept across suggest_move calls so
// trees can be re-rooted along the actually-played moves (tree reuse).
struct MCTSEngine::Forest {
    std::vector<ISTree> trees;
};

// Binary search by move_key. Returns the index into node.children whose
// key equals `key`, or -1 if not found.
static inline int find_child_slot(const ISNode& node, uint16_t key) {
    int lo = 0, hi = int(node.children.size());
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        uint16_t k = node.children[mid].key;
        if (k < key)      lo = mid + 1;
        else if (k > key) hi = mid;
        else              return mid;
    }
    return -1;
}

// Returns the lowest index i such that node.children[i].key >= key.
// Used to find the sorted-insertion point when adding a new child.
static inline int lower_bound_slot(const ISNode& node, uint16_t key) {
    int lo = 0, hi = int(node.children.size());
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (node.children[mid].key < key) lo = mid + 1;
        else                              hi = mid;
    }
    return lo;
}

// Reset a tree to a single fresh root.
static void reset_tree(ISTree& tree) {
    tree.nodes.clear();
    tree.nodes.emplace_back();
}

// Re-root `tree` at the node reached by following `path` (the moves
// actually played since the tree's root was current) and drop everything
// outside that subtree. Returns false — leaving the tree to be reset by
// the caller — if any move on the path has no child node, or if the
// destination has no children worth carrying.
static bool reroot_tree(ISTree& tree, const std::vector<Move>& path) {
    int32_t idx = 0;
    for (Move m : path) {
        const ISNode& node = tree.at(idx);
        int slot = find_child_slot(node, move_key(m));
        if (slot < 0) return false;
        idx = node.children[slot].node_idx;
    }
    if (tree.at(idx).children.empty()) return false;

    // Compact the kept subtree into a fresh arena in BFS order,
    // remapping child node indices as we go. Each node has exactly one
    // parent, so every kept node is moved exactly once. (Indexed access
    // throughout — out.push_back may not invalidate out[i] here because
    // of the reserve, but don't hold references across it anyway.)
    std::vector<ISNode> out;
    out.reserve(tree.nodes.size());
    out.push_back(std::move(tree.nodes[idx]));
    for (size_t i = 0; i < out.size(); ++i) {
        for (size_t k = 0; k < out[i].children.size(); ++k) {
            int32_t old = out[i].children[k].node_idx;
            out[i].children[k].node_idx = int32_t(out.size());
            out.push_back(std::move(tree.nodes[old]));
        }
    }
    out[0].move_from_parent = Move{};
    tree.nodes = std::move(out);
    return true;
}

static int ismcts_iteration(ISTree& tree, GameState state,
                            Xoshiro256pp& rng, const MCTSConfig& cfg)
{
    SEQ_PROFILE_SCOPE("MCTS::ismcts_iteration");

    // 1. Re-determinize at the root.
    if (cfg.determinize) {
        determinize_opponent_hand(state, state.current_player, rng);
    }

    // Path of node indices traversed this descent. Used for backup.
    // Bounded by the max rollout depth in practice; MAX_MOVES is a
    // comfortable upper bound that costs only a few KB on the stack.
    int32_t path[MAX_MOVES];
    int     path_len = 0;
    path[path_len++] = 0;   // root

    int     depth    = 0;
    int32_t node_idx = 0;

    // 2. Selection + expansion.
    while (!state.done) {
        MoveList legal;
        state.legal_moves(legal);
        if (legal.count == 0) break;

        // First pass: for each legal move, find its matching child (if
        // any). We cache the result so the UCB pass below doesn't have
        // to repeat the lookup.
        int    child_slot[MAX_MOVES];   // index into node.children, or -1
        int    untried[MAX_MOVES];      // legal-move indices with no child
        int    n_untried = 0;
        {
            const ISNode& node = tree.at(node_idx);
            for (int i = 0; i < legal.count; ++i) {
                int slot = find_child_slot(node, move_key(legal.moves[i]));
                child_slot[i] = slot;
                if (slot < 0) untried[n_untried++] = i;
            }
        }

        if (!cfg.lazy_expansion && n_untried > 0) {
            SEQ_PROFILE_SCOPE("MCTS::expand");
            // Legacy width-first expansion: expand a uniformly random
            // untried legal move, then roll out.
            int  pick = untried[rng.rand_int(0, n_untried - 1)];
            Move m    = legal.moves[pick];
            uint16_t k = move_key(m);

            // Append a fresh node to the arena. The reserve in ISTree's
            // ctor guarantees no reallocation, so existing indices stay
            // valid. We bind `node` *after* the push_back to avoid a
            // stale reference if the (already-reserved) vector grew.
            int32_t new_idx = int32_t(tree.nodes.size());
            tree.nodes.emplace_back();
            tree.nodes.back().move_from_parent = m;

            // Insert child entry in sorted position.
            ISNode& node = tree.at(node_idx);
            int    pos  = lower_bound_slot(node, k);
            node.children.insert(node.children.begin() + pos,
                                 ChildEntry{k, m, new_idx});

            if (!state.make_move(m)) break;
            node_idx = new_idx;
            path[path_len++] = node_idx;
            ++depth;
            break;   // proceed to rollout from this fresh leaf
        }

        SEQ_PROFILE_SCOPE("MCTS::select_puct");

        // Compute a softmax prior over each legal move's shaping score,
        // then pick the PUCT-best. (Legacy mode only reaches here once
        // all legal moves have children; lazy mode scores unvisited
        // moves with an FPU value and expands one only when selected.) Priors are recomputed per descent step because
        // the player's hand (and thus the per-move shaping score)
        // diverges across determinizations after a few plies.
        // Ties broken uniformly at random — without this, dead-card
        // declarations get expanded first (because legal_moves emits them
        // last and untried popping was historically back-to-front) and
        // accumulate visits whenever multiple moves tie at winrate ≈ 1.0.
        const int player = state.current_player;

        const ShapingMasks prior_masks = build_shaping_masks(state, player);
        ShapingCache prior_cache(prior_masks);
        double scores[MAX_MOVES];
        for (int i = 0; i < legal.count; ++i) {
            Move m = legal.moves[i];
            if (m.is_dead()) {
                scores[i] = 0.0;
            } else {
                int card_type = state.hands[player][m.card_idx];
                scores[i] = prior_cache.score(card_type, m.cell);
            }
        }
        // Softmax (numerically stable: subtract max).
        double max_s = -std::numeric_limits<double>::infinity();
        for (int i = 0; i < legal.count; ++i) {
            if (scores[i] > max_s) max_s = scores[i];
        }
        double sum_exp = 0.0;
        double priors[MAX_MOVES];
        for (int i = 0; i < legal.count; ++i) {
            priors[i] = std::exp(scores[i] - max_s);
            sum_exp += priors[i];
        }
        const double inv_sum_exp = 1.0 / sum_exp;

        int32_t best_idx = -1;   // node index of the chosen child, -1 if unvisited
        int     best_i   = -1;   // index into legal.moves
        Move    best_move{};
        double  best_score = -std::numeric_limits<double>::infinity();
        int     n_tied = 0;

        {
            ISNode& node = tree.at(node_idx);

            // Σ_b N(s,b) over the currently-legal children. In legacy
            // mode every legal move has been expanded by the time we
            // reach this branch, so sum_visits ≥ legal.count ≥ 1. In
            // lazy mode unvisited moves contribute 0.
            double sum_visits = 0.0;
            for (int i = 0; i < legal.count; ++i) {
                if (child_slot[i] >= 0) {
                    sum_visits += double(
                        tree.at(node.children[child_slot[i]].node_idx).visits);
                }
            }
            // Lazy mode adds +1 under the √ so the very first selection
            // at a node follows the prior instead of tying at explore=0.
            const double sqrt_sum = cfg.lazy_expansion
                ? std::sqrt(sum_visits + 1.0)
                : std::sqrt(sum_visits);

            // First-play urgency (lazy mode): unvisited moves score the
            // node's own mean from the mover's perspective minus a small
            // reduction, so they compete on prior + optimism instead of
            // being force-expanded width-first.
            double fpu_q = 0.0;
            if (cfg.lazy_expansion) {
                double parent_mean_p0 =
                    node.visits > 0 ? node.value_p0 / node.visits : 0.0;
                double parent_q = (player == 0) ? parent_mean_p0
                                                : -parent_mean_p0;
                fpu_q = parent_q - cfg.fpu_reduction;
            }

            for (int i = 0; i < legal.count; ++i) {
                double  q, nvis;
                int32_t ci = -1;
                if (child_slot[i] >= 0) {
                    ChildEntry& ce = node.children[child_slot[i]];
                    ISNode& c = tree.at(ce.node_idx);
                    double mean_p0 = c.value_p0 / std::max(1, c.visits);
                    q    = (player == 0) ? mean_p0 : -mean_p0;
                    nvis = double(c.visits);
                    ci   = ce.node_idx;
                } else {
                    q    = fpu_q;
                    nvis = 0.0;
                }
                double prior   = priors[i] * inv_sum_exp;
                double explore = cfg.ucb_c * prior * sqrt_sum
                                 / (1.0 + nvis);
                double score   = q + explore;

                if (score > best_score + 1e-12) {
                    best_score = score;
                    best_idx   = ci;
                    best_i     = i;
                    best_move  = legal.moves[i];
                    n_tied     = 1;
                } else if (score > best_score - 1e-12) {
                    ++n_tied;
                    if (rng.rand_int(0, n_tied - 1) == 0) {
                        best_idx  = ci;
                        best_i    = i;
                        best_move = legal.moves[i];
                    }
                }
            }
        }

        if (best_i < 0) break;                         // pathological

        if (best_idx < 0) {
            // Lazy expansion: the selected move has no node yet —
            // materialize it now, then roll out from the fresh leaf.
            SEQ_PROFILE_SCOPE("MCTS::expand");
            uint16_t k = move_key(best_move);
            int32_t new_idx = int32_t(tree.nodes.size());
            tree.nodes.emplace_back();
            tree.nodes.back().move_from_parent = best_move;

            ISNode& node = tree.at(node_idx);
            int pos = lower_bound_slot(node, k);
            node.children.insert(node.children.begin() + pos,
                                 ChildEntry{k, best_move, new_idx});

            if (!state.make_move(best_move)) break;
            node_idx = new_idx;
            path[path_len++] = node_idx;
            ++depth;
            break;   // proceed to rollout from this fresh leaf
        }

        if (!state.make_move(best_move)) break;
        node_idx = best_idx;
        path[path_len++] = node_idx;
        ++depth;
    }

    // 3. Rollout from the leaf (or use terminal value if state.done).
    double value_p0 = state.done
        ? terminal_value_p0(state)
        : rollout(state, rng, cfg);

    // 4. Backup along the descent path.
    {
        SEQ_PROFILE_SCOPE("MCTS::backup");
        for (int i = 0; i < path_len; ++i) {
            ISNode& n = tree.at(path[i]);
            n.visits   += 1;
            n.value_p0 += value_p0;
        }
    }

    return depth;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

MCTSEngine::MCTSEngine(MCTSConfig cfg) : cfg_(cfg) {
    if (cfg_.seed == 0) {
        std::random_device rd;
        uint64_t s = (uint64_t(rd()) << 32) ^ rd();
        rng_.seed(s);
    } else {
        rng_.seed(cfg_.seed);
    }
}

// Out-of-line where Forest is complete.
MCTSEngine::~MCTSEngine() = default;
MCTSEngine::MCTSEngine(MCTSEngine&&) noexcept = default;
MCTSEngine& MCTSEngine::operator=(MCTSEngine&&) noexcept = default;

void MCTSEngine::advance(Move m) {
    if (!cfg_.tree_reuse) return;
    pending_.push_back(m);
}

// Run `budget` iterations on one ISMCTS tree and append its root-level
// visit counts to the aggregator vectors. Each iteration re-determinizes
// the opponent's hand and deck; multiple trees per call are purely a
// parallelism knob (each tree builds independently and we aggregate at
// the end). The tree is owned by the caller: fresh after reset_tree, or
// carrying a re-rooted subtree from the previous move (tree reuse).
struct PerTreeStats { int iterations; int max_depth; };

static PerTreeStats run_one_tree(const GameState& s_in,
                                 Xoshiro256pp& rng,
                                 const MCTSConfig& cfg,
                                 int budget,
                                 ISTree& tree,
                                 std::vector<int>& root_visits,
                                 std::vector<double>& root_value_p0,
                                 std::vector<Move>& root_moves)
{
    int max_depth  = 0;
    int iters_done = 0;

    // Every iteration expands at most one node; reserving up-front keeps
    // node indices valid across the emplace_backs inside the iteration.
    tree.nodes.reserve(tree.nodes.size() + size_t(budget) + 2);
    for (int i = 0; i < budget; ++i) {
        int d = ismcts_iteration(tree, s_in, rng, cfg);
        if (d > max_depth) max_depth = d;
        ++iters_done;
    }

    // Aggregate root-level visit counts into the caller's accumulators.
    // Moves are matched on (card_idx, cell) — workers see the same legal
    // moves at root since the player's own hand is fixed; only the
    // opponent's hand and deck order differ across determinizations.
    const ISNode& root = tree.root();
    for (const ChildEntry& ce : root.children) {
        const ISNode& child = tree.at(ce.node_idx);
        Move m = ce.move;
        int idx = -1;
        for (size_t k = 0; k < root_moves.size(); ++k) {
            if (root_moves[k].card_idx == m.card_idx &&
                root_moves[k].cell     == m.cell) {
                idx = int(k); break;
            }
        }
        if (idx < 0) {
            root_moves.push_back(m);
            root_visits.push_back(child.visits);
            root_value_p0.push_back(child.value_p0);
        } else {
            root_visits[idx]   += child.visits;
            root_value_p0[idx] += child.value_p0;
        }
    }

    return {iters_done, max_depth};
}

// One worker's slice of the root-parallel aggregation: visit counts,
// summed rollout values, and the moves they refer to. Each worker
// thread fills its own instance; we merge them sequentially on the
// calling thread after all workers complete. Merging is O(W·M·K) where
// W=workers, M=root moves, K=children seen per tree — tiny compared to
// rollouts.
struct WorkerResult {
    std::vector<int>    visits;
    std::vector<double> value_p0;
    std::vector<Move>   moves;
    int                 iterations = 0;
    int                 max_depth  = 0;
};

// Given an initial state and a seed, run `per_tree` iterations on each
// of the `n_trees` trees in the caller-owned slice and return their
// aggregated root-move statistics. Workers operate on disjoint tree
// slices of the persistent forest — no shared state, no locks; safe to
// call concurrently on independent slices.
//
// The function builds its own RNG from the given seed, so worker threads
// are reproducible from (seed, config) alone.
static WorkerResult run_worker(const GameState& s_in,
                               const MCTSConfig& cfg,
                               ISTree* trees,
                               int n_trees,
                               int per_tree,
                               uint64_t seed)
{
    Xoshiro256pp rng(seed);
    WorkerResult r;
    for (int t = 0; t < n_trees; ++t) {
        auto pt = run_one_tree(s_in, rng, cfg, per_tree, trees[t],
                               r.visits, r.value_p0, r.moves);
        r.iterations += pt.iterations;
        if (pt.max_depth > r.max_depth) r.max_depth = pt.max_depth;
    }
    return r;
}

// Determine the effective worker count and the per-worker tree split.
// Returns a vector of (n_trees_for_this_worker) sized to the number of
// workers we'll actually spawn. If n_threads is auto (0) we cap at 16;
// otherwise we honor the request, clamped to [1, n_total_trees].
static std::vector<int> partition_trees(int n_total_trees, int n_threads_cfg) {
    int n_threads;
    if (n_threads_cfg <= 0) {
        unsigned hc = std::thread::hardware_concurrency();
        if (hc == 0) hc = 1;
        n_threads = std::min<int>(int(hc), 16);
    } else {
        n_threads = n_threads_cfg;
    }
    n_threads = std::max(1, std::min(n_threads, n_total_trees));

    // Distribute n_total_trees trees across n_threads workers as evenly
    // as possible. Extra trees go to the first few workers.
    std::vector<int> sizes(n_threads, n_total_trees / n_threads);
    int leftover = n_total_trees % n_threads;
    for (int i = 0; i < leftover; ++i) sizes[i]++;
    return sizes;
}

Move MCTSEngine::suggest_move(const GameState& s_in) {
    SEQ_PROFILE_SCOPE("MCTS::suggest_move");
    using clk = std::chrono::high_resolution_clock;
    auto t0 = clk::now();

    stats_ = {};

    // Quick edge cases — read off the un-determinized state since our
    // own legal moves don't depend on the opponent's hand.
    {
        MoveList ms;
        GameState peek = s_in;
        peek.legal_moves(ms);
        stats_.root_legal_moves = ms.count;
        if (ms.count == 0) { stats_.best_move = Move{}; return Move{}; }
        if (ms.count == 1) {
            Move only = ms.moves[0];
            stats_.best_move        = only;
            stats_.best_move_visits = 1;
            stats_.iterations       = 0;
            stats_.seconds = std::chrono::duration<double>(clk::now() - t0).count();
            return only;
        }
    }

    // Split the iteration budget across n root-parallel trees. With
    // ISMCTS each tree already re-determinizes per iteration, so extra
    // trees are purely for parallelism.
    int n_trees  = std::max(1, cfg_.n_parallel_trees);
    int per_tree = std::max(1, cfg_.iterations / n_trees);

    // Prepare the persistent forest. If tree reuse is on and the harness
    // reported the moves played since the last search (via advance()),
    // re-root each tree along that path, carrying its subtree statistics
    // over. Any tree where the path dies — and every tree when there's
    // no pending path (advance() never called, or first search) — resets
    // to a fresh root, which reproduces the old search-from-scratch
    // behavior exactly.
    if (!forest_) forest_ = std::make_unique<Forest>();
    Forest& forest = *forest_;
    const bool try_reuse = cfg_.tree_reuse && !pending_.empty()
                        && int(forest.trees.size()) == n_trees;
    if (int(forest.trees.size()) != n_trees) {
        forest.trees.clear();
        forest.trees.reserve(size_t(n_trees));
        for (int t = 0; t < n_trees; ++t) forest.trees.emplace_back(per_tree);
    }
    for (auto& tree : forest.trees) {
        if (!(try_reuse && reroot_tree(tree, pending_))) reset_tree(tree);
    }
    pending_.clear();

    // Decide how many threads to use and how many trees each handles.
    // Pre-generate a seed for each worker from the engine RNG, then
    // hand off — workers don't share rng_. Each worker gets a disjoint
    // contiguous slice of the forest's trees.
    auto split = partition_trees(n_trees, cfg_.n_threads);
    std::vector<uint64_t> seeds(split.size());
    for (auto& s : seeds) s = rng_();

    std::vector<WorkerResult> results(split.size());

    if (split.size() == 1) {
        // Sequential fast path — no async overhead.
        results[0] = run_worker(s_in, cfg_, forest.trees.data(),
                                split[0], per_tree, seeds[0]);
    } else {
        // Parallel path — std::async with explicit launch::async so each
        // task actually gets a fresh thread (deferred would serialize).
        std::vector<std::future<WorkerResult>> futures;
        futures.reserve(split.size());
        int tree_off = 0;
        for (size_t i = 0; i < split.size(); ++i) {
            futures.push_back(std::async(std::launch::async,
                                         run_worker, std::cref(s_in),
                                         std::cref(cfg_),
                                         forest.trees.data() + tree_off,
                                         split[i], per_tree, seeds[i]));
            tree_off += split[i];
        }
        for (size_t i = 0; i < futures.size(); ++i) {
            results[i] = futures[i].get();
        }
    }

    // Merge all worker results into a single (moves, visits, value_p0).
    std::vector<int>    root_visits;
    std::vector<double> root_value_p0;
    std::vector<Move>   root_moves;
    int total_iters = 0;
    int max_depth   = 0;

    for (const auto& r : results) {
        total_iters += r.iterations;
        if (r.max_depth > max_depth) max_depth = r.max_depth;
        for (size_t i = 0; i < r.moves.size(); ++i) {
            Move m = r.moves[i];
            int idx = -1;
            for (size_t k = 0; k < root_moves.size(); ++k) {
                if (root_moves[k].card_idx == m.card_idx &&
                    root_moves[k].cell     == m.cell) {
                    idx = int(k);
                    break;
                }
            }
            if (idx < 0) {
                root_moves.push_back(m);
                root_visits.push_back(r.visits[i]);
                root_value_p0.push_back(r.value_p0[i]);
            } else {
                root_visits[idx]   += r.visits[i];
                root_value_p0[idx] += r.value_p0[i];
            }
        }
    }

    // Pick the move with the most aggregate visits; ties broken at
    // random so we don't have a structural bias toward whatever move
    // legal_moves emits first.
    int best_idx = 0;
    int best_v   = root_visits[0];
    int n_tied   = 1;
    for (size_t i = 1; i < root_visits.size(); ++i) {
        if (root_visits[i] > best_v) {
            best_v   = root_visits[i];
            best_idx = int(i);
            n_tied   = 1;
        } else if (root_visits[i] == best_v) {
            ++n_tied;
            if (rng_.rand_int(0, n_tied - 1) == 0) best_idx = int(i);
        }
    }

    // Aggregate mean value of the chosen move, P0 perspective.
    double mean_p0 = root_value_p0[best_idx] / std::max(1, root_visits[best_idx]);
    int player     = s_in.current_player;
    double q_for_player = (player == 0) ? mean_p0 : -mean_p0;
    double q_clipped    = std::max(-1.0, std::min(1.0, q_for_player));

    stats_.iterations         = total_iters;
    stats_.max_depth          = max_depth;
    stats_.best_move          = root_moves[best_idx];
    stats_.best_move_visits   = best_v;
    stats_.best_move_winrate  = 0.5 * (q_clipped + 1.0);
    stats_.seconds = std::chrono::duration<double>(clk::now() - t0).count();
    return stats_.best_move;
}

} // namespace seq

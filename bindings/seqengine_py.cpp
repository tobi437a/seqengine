// pybind11 binding for the C++ MCTS engine.
//
// Strategy: keep the C++ side oblivious to Python's SequenceGame. The
// binding accepts primitive arrays (flat board, hands as int lists,
// deck as int list, etc.) and translates them into a fresh GameState
// on each call. State translation happens once per move — far cheaper
// than the rollouts. The Python side is responsible for the int card
// encoding, which must match tools/gen_board_data.py:
//
//   suit_idx * 12 + rank_idx       -> 0..47   regular cards
//   48 = ONE_EYED_JACK
//   49 = TWO_EYED_JACK
//   suit_idx:  spades=0, hearts=1, diamonds=2, clubs=3
//   rank_idx:  2,3,4,5,6,7,8,9,10,Q,K,A  (no J on board)

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "state.hpp"
#include "search.hpp"
#include "board_data.hpp"
#include "types.hpp"

namespace py = pybind11;
using namespace seq;

// Python board_chips encoding:
//   EMPTY=0, PLAYER0=1, PLAYER1=2, SEQUENCE0=3, SEQUENCE1=4
// Locked = SEQUENCE0/SEQUENCE1; unlocked = PLAYER0/PLAYER1. Joker corners
// stay EMPTY in Python's board_chips (the JOKER status is held in BOARD,
// not board_chips).
static constexpr int PY_EMPTY     = 0;
static constexpr int PY_PLAYER0   = 1;
static constexpr int PY_PLAYER1   = 2;
static constexpr int PY_SEQUENCE0 = 3;
static constexpr int PY_SEQUENCE1 = 4;


static GameState build_state_from_py(
    const std::vector<int>& board_chips,
    const std::vector<std::vector<int>>& hands,
    const std::vector<int>& deck,
    int current_player,
    const std::vector<int>& sequences,
    const std::vector<bool>& dead_card_used)
{
    if (board_chips.size() != size_t(N_CELLS)) {
        throw std::invalid_argument("board_chips must have 100 entries");
    }
    if (hands.size() != 2) {
        throw std::invalid_argument("hands must have 2 sub-lists");
    }
    if (sequences.size() != 2 || dead_card_used.size() != 2) {
        throw std::invalid_argument("sequences and dead_card_used must have 2 entries");
    }

    GameState s;
    // Default-construct gives us clean bitboards, empty deck/discard,
    // sequences=0, etc. We overwrite the bits we care about.

    for (int cell = 0; cell < N_CELLS; ++cell) {
        // Joker corners stay outside chips/locked bitboards — JOKER_MASK
        // handles them as "anyone's chip" inside window_phi.
        if (JOKER_MASK.test(cell)) continue;

        int v = board_chips[cell];
        switch (v) {
            case PY_PLAYER0:   s.chips[0].set(cell);  break;
            case PY_PLAYER1:   s.chips[1].set(cell);  break;
            case PY_SEQUENCE0: s.locked[0].set(cell); break;
            case PY_SEQUENCE1: s.locked[1].set(cell); break;
            case PY_EMPTY:     /* no-op */            break;
            default:
                throw std::invalid_argument("unknown board_chips value");
        }
    }

    for (int p = 0; p < 2; ++p) {
        if (hands[p].size() > size_t(HAND_SIZE)) {
            throw std::invalid_argument("hand too large");
        }
        for (int i = 0; i < HAND_SIZE; ++i) {
            int v = (i < int(hands[p].size())) ? hands[p][i] : -1;
            s.hands[p][i] = int8_t(v);
        }
    }

    if (deck.size() > size_t(DECK_SIZE)) {
        throw std::invalid_argument("deck has more than 104 cards");
    }
    s.deck.clear();
    for (int c : deck) s.deck.push_back(int8_t(c));

    s.current_player    = current_player;
    s.sequences[0]      = sequences[0];
    s.sequences[1]      = sequences[1];
    s.dead_card_used[0] = dead_card_used[0];
    s.dead_card_used[1] = dead_card_used[1];

    // We don't reconstruct discard. legal_moves doesn't read it; the only
    // place the deck-reshuffle reads it is during make_move when the deck
    // empties — which can happen during search rollouts. To be safe,
    // start with whatever cards the determinization can dump back in.
    s.discard.clear();

    s.done   = false;
    s.winner = -1;
    // rng is default-seeded from the GameState ctor; the search has its
    // own RNG, so this doesn't bite us.

    return s;
}


PYBIND11_MODULE(_seqengine, m) {
    m.doc() = "C++ MCTS engine for Sequence (1v1).";

    py::class_<MCTSConfig>(m, "MCTSConfig")
        .def(py::init<>())
        .def_readwrite("iterations",              &MCTSConfig::iterations)
        .def_readwrite("ucb_c",                   &MCTSConfig::ucb_c)
        .def_readwrite("rollout_epsilon",         &MCTSConfig::rollout_epsilon)
        .def_readwrite("max_rollout_steps",       &MCTSConfig::max_rollout_steps)
        .def_readwrite("determinize",             &MCTSConfig::determinize)
        .def_readwrite("length_decay",            &MCTSConfig::length_decay)
        .def_readwrite("prefer_completing_moves", &MCTSConfig::prefer_completing_moves)
        .def_readwrite("lazy_expansion",          &MCTSConfig::lazy_expansion)
        .def_readwrite("fpu_reduction",           &MCTSConfig::fpu_reduction)
        .def_readwrite("tree_reuse",              &MCTSConfig::tree_reuse)
        .def_readwrite("n_parallel_trees",        &MCTSConfig::n_parallel_trees)
        .def_readwrite("n_threads",               &MCTSConfig::n_threads)
        .def_readwrite("seed",                    &MCTSConfig::seed)
        .def("__repr__", [](const MCTSConfig& c) {
            return "<MCTSConfig iters=" + std::to_string(c.iterations) +
                   " c=" + std::to_string(c.ucb_c) +
                   " ε=" + std::to_string(c.rollout_epsilon) +
                   " trees=" + std::to_string(c.n_parallel_trees) + ">";
        });

    py::class_<MCTSStats>(m, "MCTSStats")
        .def_readonly("iterations",        &MCTSStats::iterations)
        .def_readonly("max_depth",         &MCTSStats::max_depth)
        .def_readonly("seconds",           &MCTSStats::seconds)
        .def_readonly("best_move_visits",  &MCTSStats::best_move_visits)
        .def_readonly("best_move_winrate", &MCTSStats::best_move_winrate)
        .def_readonly("root_legal_moves",  &MCTSStats::root_legal_moves);

    py::class_<MCTSEngine>(m, "MCTSEngine")
        .def(py::init<MCTSConfig>(), py::arg("config") = MCTSConfig{})
        .def("suggest_move",
             [](MCTSEngine& eng,
                const std::vector<int>& board_chips,
                const std::vector<std::vector<int>>& hands,
                const std::vector<int>& deck,
                int current_player,
                const std::vector<int>& sequences,
                const std::vector<bool>& dead_card_used) {
                 GameState s = build_state_from_py(
                     board_chips, hands, deck, current_player,
                     sequences, dead_card_used);
                 Move m = eng.suggest_move(s);
                 int row, col;
                 if (m.is_dead()) { row = -1; col = -1; }
                 else { row = row_of(m.cell); col = col_of(m.cell); }
                 return py::make_tuple(int(m.card_idx), row, col);
             },
             py::arg("board_chips"),
             py::arg("hands"),
             py::arg("deck"),
             py::arg("current_player"),
             py::arg("sequences"),
             py::arg("dead_card_used"),
             "Pick a move for current_player. Returns (card_idx, row, col); "
             "row=col=-1 means a dead-card declaration.")
        .def("advance",
             [](MCTSEngine& eng, int card_idx, int row, int col) {
                 Move m;
                 m.card_idx = int8_t(card_idx);
                 if (row < 0 || col < 0) {
                     m.cell = Move::DEAD;
                 } else {
                     if (row >= N_ROWS || col >= N_COLS) {
                         throw std::invalid_argument("row/col out of range");
                     }
                     m.cell = int8_t(cell_of(row, col));
                 }
                 eng.advance(m);
             },
             py::arg("card_idx"),
             py::arg("row"),
             py::arg("col"),
             "Report a move actually played in the game (by either side, "
             "in order) so tree_reuse can re-root the kept forest at the "
             "next suggest_move. row=col=-1 means a dead-card declaration. "
             "Optional: if never called, every search starts fresh.")
        .def("last_stats", [](MCTSEngine& eng) { return eng.last_stats(); });
}

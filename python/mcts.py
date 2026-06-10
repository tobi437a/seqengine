"""
Python wrapper around the C++ MCTS engine.

Two layers:
  * `card_to_int` / `extract_state` translate a Python SequenceGame
    instance into the primitive arrays the C++ binding accepts.
  * `MCTSOpponent` adapts the engine to the
    `(game, player, legal_mask) -> action_int` signature used by the
    other opponents in this project.

The card encoding must match tools/gen_board_data.py exactly:
    suit_idx (spades=0, hearts=1, diamonds=2, clubs=3) * 12 + rank_idx
    (2..10 → 0..8, Q=9, K=10, A=11). 48 = ONE_EYED_JACK, 49 = TWO_EYED_JACK.
"""
import os
import sys

# Allow the compiled extension to live next to this file.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

try:
    import _seqengine as _ext
except ImportError as e:
    # The actual failure could be any of:
    #   - File missing: build wasn't run yet
    #   - Wrong filename: .pyd is for a different Python ABI than the
    #     interpreter we're running under
    #   - DLL load failure on Windows: the .pyd built fine but a dependent
    #     DLL (libstdc++-6.dll, libgcc_s_seh-1.dll, libwinpthread-1.dll
    #     for MinGW builds without static-linking) isn't on PATH
    # Pass the underlying exception through so callers can see which.
    _py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    _diag = (
        f"Failed to import _seqengine extension: {e}\n"
        f"  Diagnostics:\n"
        f"    - Python ABI tag expected: {_py_tag}\n"
        f"    - Looking in: {HERE}\n"
        f"    - Files found here matching _seqengine*: "
        f"{[f for f in os.listdir(HERE) if f.startswith('_seqengine')] if os.path.isdir(HERE) else '(dir missing)'}\n"
        f"  Most common causes:\n"
        f"    1. Build wasn't run: `make python` from the project root.\n"
        f"    2. .pyd built for a different Python version than the one "
        f"running this script. Rebuild with the same `python` you'll use.\n"
        f"    3. (Windows) DLL load failed — the .pyd needs a runtime DLL "
        f"that isn't on PATH. `make python` should statically link these "
        f"and, as a safety net, copy any remaining MinGW runtime DLLs "
        f"into this directory. Run `make pydeps` to see what DLLs the .pyd "
        f"depends on; anything beyond KERNEL32, api-ms-win-*, and "
        f"python{sys.version_info.major}{sys.version_info.minor}.dll "
        f"should be next to the .pyd in this directory."
    )
    raise ImportError(_diag) from e

from board_layout import ONE_EYED_JACK, TWO_EYED_JACK, SUITS, RANKS
from game_engine import HAND_SIZE
from seq_actions import action_to_int, int_to_action


SUIT_IDX = {s: i for i, s in enumerate(SUITS)}
RANK_IDX = {r: i for i, r in enumerate(RANKS)}

ONE_EYED_JACK_INT = 48
TWO_EYED_JACK_INT = 49


def card_to_int(card):
    """Translate a card (suit, rank) tuple or jack constant to a 0..49 int.
    `None` (empty hand slot, exhausted deck) maps to -1."""
    if card is None:
        return -1
    if card == ONE_EYED_JACK:
        return ONE_EYED_JACK_INT
    if card == TWO_EYED_JACK:
        return TWO_EYED_JACK_INT
    suit, rank = card
    return SUIT_IDX[suit] * 12 + RANK_IDX[rank]


def extract_state(game):
    """Pull the C++-binding inputs out of a Python SequenceGame instance.

    Returns a tuple of (board_chips, hands, deck, current_player,
    sequences, dead_card_used) matching MCTSEngine.suggest_move's args.
    """
    # Flat 100-int board.
    board = [int(game.board_chips[r][c]) for r in range(10) for c in range(10)]

    # 2 × HAND_SIZE list, padded with -1 for empty slots.
    hands = []
    for p in range(2):
        slot = list(game.hands[p])
        ints = [card_to_int(c) for c in slot]
        while len(ints) < HAND_SIZE:
            ints.append(-1)
        hands.append(ints[:HAND_SIZE])

    deck = [card_to_int(c) for c in game.deck if c is not None]

    sequences = [int(game.sequences[0]), int(game.sequences[1])]

    # `dead_card_used` may be missing on older SequenceGame versions;
    # default to false so we don't break.
    dcu = getattr(game, 'dead_card_used', [False, False])
    dead_card_used = [bool(dcu[0]), bool(dcu[1])]

    return (board, hands, deck, int(game.current_player),
            sequences, dead_card_used)


class MCTSOpponent:
    """
    MCTS opponent matching the `(game, player, legal_mask) -> action_int`
    signature used by random_action and heuristic_action.

    Configurable via kwargs that pass straight through to MCTSConfig:
        iterations               default 2000   total search budget
        n_parallel_trees         default 8      root-parallel ISMCTS trees
        n_threads                default 0      0 = auto (hardware_concurrency,
                                                capped at min(16, n_parallel_trees))
        ucb_c                    default 2.0
        rollout_epsilon          default 0.30
        seed                     default 0      (0 → random_device)
    """

    def __init__(self, **kwargs):
        cfg = _ext.MCTSConfig()
        # Default to auto-threading so end-users get parallelism without
        # having to know about it. The C++ default is single-threaded to
        # keep tests/benchmarks deterministic.
        cfg.n_threads = 0
        for key, value in kwargs.items():
            if not hasattr(cfg, key):
                raise ValueError(f"unknown MCTSConfig option: {key}")
            setattr(cfg, key, value)
        self._cfg    = cfg
        self.engine  = _ext.MCTSEngine(cfg)

    def __call__(self, game, player, legal_mask):
        # MCTS only plays from the perspective of the side-to-move; if
        # the dispatcher asked us to play for the wrong player, the
        # opponent-hand determinization would invert which hand we
        # treat as hidden — silently wrong. Fail loudly instead.
        if player != game.current_player:
            raise RuntimeError(
                f"MCTSOpponent called for player {player} but "
                f"game.current_player={game.current_player}")

        args = extract_state(game)
        card_idx, row, col = self.engine.suggest_move(*args)
        return action_to_int(card_idx, row, col)

    def advance(self, action):
        """Report a move actually played in the game — by either side, in
        the order played — so tree_reuse can re-root the kept search
        forest at the next call instead of starting from scratch.

        Accepts a flat action int or a (card_idx, row, col) tuple;
        row=col=-1 means a dead-card declaration. Optional: if never
        called, every search starts from a fresh tree. Only report moves
        whose card_idx is the true hand slot (a wrong slot can match the
        wrong subtree); harnesses that don't track opponent slots should
        simply not call this."""
        if not isinstance(action, (tuple, list)):
            action = int_to_action(int(action))
        card_idx, row, col = action
        self.engine.advance(int(card_idx), int(row), int(col))

    @property
    def last_stats(self):
        return self.engine.last_stats()


# Convenience function: build a one-shot opponent.
def mcts_action_factory(**kwargs):
    """Return an opponent_fn closure (game, player, mask) -> action_int."""
    return MCTSOpponent(**kwargs)

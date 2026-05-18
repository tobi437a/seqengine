"""
play_irl.py — engine plays Sequence on a physical board.

You are the engine's hands and eyes:
  * Draw cards from the deck for the engine and type each one in.
  * Read the engine's chosen move and play it on the physical board.
  * After your opponent plays, type their move.

Run: python play_irl.py

This is a sibling of play.py (which is for digital head-to-head). The
big design difference is that we only know *one* hand (the engine's),
plus everything that's been visibly played. The opponent's hand and the
remaining physical deck are merged into a single "unseen pool" — that's
what game.deck represents for us. The C++ MCTS already treats the
opponent's hand as hidden info (it dumps it into the deck and reshuffles
per rollout), so passing -1s for the opponent's hand is exactly what the
search expects.
"""

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from game_engine import (SequenceGame, EMPTY, PLAYER0, PLAYER1,
                         SEQUENCE0, SEQUENCE1, HAND_SIZE,
                         SEQUENCES_TO_WIN)
from board_layout import (BOARD, JOKER, card_str, TWO_EYED_JACK,
                          ONE_EYED_JACK, SUIT_SYMBOLS, CARD_POSITIONS,
                          SUITS, RANKS)
from seq_actions import int_to_action, get_legal_action_mask
from seq_opponents import heuristic_action


# --- terminal styling -----------------------------------------------------

BLUE   = '\033[94m'
GREEN  = '\033[92m'
RED    = '\033[91m'
YELLOW = '\033[93m'
CYAN   = '\033[96m'
BOLD   = '\033[1m'
DIM    = '\033[2m'
RESET  = '\033[0m'
CLEAR  = '\033[2J\033[H'

CHIP_DISPLAY = {
    EMPTY:     DIM   + '·' + RESET,
    PLAYER0:   BLUE  + '●' + RESET,
    PLAYER1:   GREEN + '●' + RESET,
    SEQUENCE0: BLUE  + '★' + RESET,
    SEQUENCE1: GREEN + '★' + RESET,
}

SUIT_COLOR = {
    'spades':   DIM,
    'clubs':    DIM,
    'hearts':   RED,
    'diamonds': RED,
}


def clear():
    print(CLEAR, end='')


# --- rendering ------------------------------------------------------------

def render_board(game, highlight_cells=None):
    if highlight_cells is None:
        highlight_cells = set()

    print(f"\n  {BOLD}Board{RESET}")
    print("     " + "  ".join(f"{c:2}" for c in range(10)))
    print("    " + "─" * 41)
    for r in range(10):
        row_str = f" {r:2}│"
        for c in range(10):
            cell  = BOARD[r][c]
            chip  = game.board_chips[r][c]
            is_hl = (r, c) in highlight_cells

            prefix = YELLOW + '[' if is_hl else ' '
            suffix = ']' + RESET if is_hl else ' '

            if cell == JOKER:
                row_str += prefix + BOLD + 'J*' + RESET + suffix
            elif chip != EMPTY:
                row_str += prefix + ' ' + CHIP_DISPLAY[chip] + suffix
            else:
                suit, rank = cell
                col_code  = SUIT_COLOR[suit]
                sym       = SUIT_SYMBOLS[suit]
                row_str  += prefix + col_code + f"{rank}{sym}".rjust(2) + RESET + suffix
        print(row_str)
    print("    " + "─" * 41)


def render_hand(game, engine_player):
    print(f"\n  {BOLD}Engine's hand:{RESET}")
    for i, card in enumerate(game.hands[engine_player]):
        if card is None:
            print(f"    [{i}] {DIM}(empty){RESET}")
        else:
            print(f"    [{i}] {card_str(card)}")


def render_status(game, engine_player):
    eng_color = BLUE if engine_player == 0 else GREEN
    opp_color = GREEN if engine_player == 0 else BLUE
    eng_label = 'P0/Blue' if engine_player == 0 else 'P1/Green'
    opp_label = 'P1/Green' if engine_player == 0 else 'P0/Blue'
    print(f"\n  {eng_color}●{RESET} Engine ({eng_label}): {game.sequences[engine_player]} seq   "
          f"{opp_color}●{RESET} Opponent ({opp_label}): {game.sequences[1 - engine_player]} seq   "
          f"{DIM}unseen pool: {len(game.deck)}{RESET}")


# --- card parsing ---------------------------------------------------------

SUIT_LETTERS = {
    'h': 'hearts', 'd': 'diamonds', 's': 'spades', 'c': 'clubs',
}
RANK_ALIASES = {
    '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7',
    '8': '8', '9': '9', '10': '10', 't': '10', 'q': 'Q', 'k': 'K', 'a': 'A',
}


def parse_card(text):
    """Parse one card token. Returns (suit, rank) tuple, ONE_EYED_JACK,
    TWO_EYED_JACK, or None if it can't be parsed."""
    if text is None:
        return None
    s = text.strip().lower()
    if not s:
        return None
    if s in ('j1', '1j', 'oj', 'one-eyed', 'oneeyed', 'one_eyed'):
        return ONE_EYED_JACK
    if s in ('j2', '2j', 'tj', 'two-eyed', 'twoeyed', 'two_eyed'):
        return TWO_EYED_JACK

    # Strip punctuation/spaces inside the token.
    s = ''.join(ch for ch in s if ch.isalnum())

    suit_letter = None
    rank_part   = None
    if s and s[-1] in SUIT_LETTERS:
        suit_letter = s[-1]
        rank_part   = s[:-1]
    elif s and s[0] in SUIT_LETTERS:
        suit_letter = s[0]
        rank_part   = s[1:]
    if suit_letter is None:
        return None

    rank = RANK_ALIASES.get(rank_part)
    if rank is None:
        return None
    return (SUIT_LETTERS[suit_letter], rank)


def prompt_card(label):
    while True:
        raw = input(label).strip()
        if raw.lower() == 'q':
            sys.exit(0)
        if not raw:
            print(f"  {RED}Empty input. Try again.{RESET}")
            continue
        card = parse_card(raw)
        if card is None:
            print(f"  {RED}Couldn't parse '{raw}'. Try e.g. 5h, 10s, qd, j1, j2.{RESET}")
            continue
        return card


def prompt_card_or_none(label):
    while True:
        raw = input(label).strip()
        if raw.lower() == 'q':
            sys.exit(0)
        if raw.lower() in ('none', 'n', '-', ''):
            return None
        card = parse_card(raw)
        if card is None:
            print(f"  {RED}Couldn't parse '{raw}'. Try again, or 'none' if deck is empty.{RESET}")
            continue
        return card


# --- deck bookkeeping -----------------------------------------------------

def build_full_deck():
    """Same composition as game_engine.build_deck() but unshuffled.

    Two decks of 48 regular cards + 4 one-eyed jacks + 4 two-eyed jacks = 104."""
    deck = []
    for _ in range(2):
        for suit in SUITS:
            for rank in RANKS:
                deck.append((suit, rank))
    for _ in range(4):
        deck.append(ONE_EYED_JACK)
    for _ in range(4):
        deck.append(TWO_EYED_JACK)
    return deck


def make_game(engine_player, engine_hand):
    """Construct a SequenceGame whose 'engine' seat holds engine_hand and
    'opponent' seat is empty (-1s). game.deck is the unseen pool — every
    card we haven't seen yet (opponent's hand + remaining physical deck)."""
    game = SequenceGame()  # reset() runs; we overwrite everything below.
    game.board_chips    = [[EMPTY] * 10 for _ in range(10)]
    game.hands          = [[None] * HAND_SIZE, [None] * HAND_SIZE]
    for i, c in enumerate(engine_hand):
        game.hands[engine_player][i] = c
    game.sequences      = [0, 0]
    game.done           = False
    game.winner         = None
    game.dead_card_used = [False, False]
    game.current_player = 0   # P0 always plays first in SequenceGame.
    game.discard        = []

    # Build the unseen pool: full 104-card deck minus the engine's hand.
    unseen = build_full_deck()
    for c in engine_hand:
        try:
            unseen.remove(c)
        except ValueError:
            # User entered more copies of a card than the deck contains.
            # Don't blow up — likely a typo and not worth derailing setup.
            pass
    game.deck = unseen
    return game


# --- move application -----------------------------------------------------

def apply_move(game, player, card, card_idx, row, col):
    """
    Apply a move. Updates board, sequences, current_player, dead-card flag,
    and removes the played card from the unseen pool (game.deck).

    For the engine's moves, pass card_idx (the slot it came from); we'll
    clear that slot, and the caller is expected to refill it with whatever
    the user actually drew.

    For the opponent, pass card_idx=None — we don't track their hand.

    row==col==-1 declares a dead card (no board change, no turn switch).
    """
    info = {}

    # Dead-card declaration: discard, mark swap-used, no turn switch.
    if row == -1 and col == -1:
        info['dead_card'] = True
        game.dead_card_used[player] = True
        try:
            game.deck.remove(card)
        except ValueError:
            pass
        if card_idx is not None:
            game.hands[player][card_idx] = None
        return info

    my_chip  = PLAYER0 if player == 0 else PLAYER1
    opp_chip = PLAYER1 if player == 0 else PLAYER0

    if card == ONE_EYED_JACK:
        if game.board_chips[row][col] != opp_chip:
            raise ValueError(
                f"one-eyed jack: ({row},{col}) isn't an unlocked enemy chip."
            )
        game.board_chips[row][col] = EMPTY
    elif card == TWO_EYED_JACK:
        if BOARD[row][col] == JOKER:
            raise ValueError(f"({row},{col}) is a JOKER corner.")
        if game.board_chips[row][col] != EMPTY:
            raise ValueError(f"({row},{col}) is not empty.")
        game.board_chips[row][col] = my_chip
    else:
        positions = CARD_POSITIONS.get(card, [])
        if (row, col) not in positions:
            raise ValueError(f"{card_str(card)} doesn't map to cell ({row},{col}).")
        if game.board_chips[row][col] != EMPTY:
            raise ValueError(f"({row},{col}) is not empty.")
        game.board_chips[row][col] = my_chip

    # Sequence detection (and locking of cells) — reuse the engine's logic.
    new_seqs = game._check_sequences(player, row, col)
    if new_seqs > 0:
        game.sequences[player] += new_seqs
        info['sequences_formed'] = new_seqs

    if game.sequences[player] >= SEQUENCES_TO_WIN:
        game.done   = True
        game.winner = player
        info['winner'] = player

    try:
        game.deck.remove(card)
    except ValueError:
        info['card_not_in_pool'] = True

    if card_idx is not None:
        game.hands[player][card_idx] = None

    game.dead_card_used[player] = False
    game.current_player         = 1 - player
    return info


# --- opponent selection ---------------------------------------------------

def _try_load_mcts(iterations, n_parallel_trees):
    try:
        from mcts import MCTSOpponent
    except Exception as e:
        print(f"  {YELLOW}MCTS unavailable:{RESET}")
        cur = e
        while cur is not None:
            for line in str(cur).splitlines():
                print(f"    {line}")
            nxt = getattr(cur, '__cause__', None) or getattr(cur, '__context__', None)
            if nxt is None or nxt is cur:
                break
            print(f"  {YELLOW}caused by:{RESET}")
            cur = nxt
        return None
    return MCTSOpponent(iterations=iterations,
                        n_parallel_trees=n_parallel_trees)


# --- setup phase ----------------------------------------------------------

INTRO = f"""
{BOLD}═══════════════════════════════════════════════════════════════{RESET}
{BOLD}  SEQUENCE — engine plays alongside you on a physical board{RESET}
{BOLD}═══════════════════════════════════════════════════════════════{RESET}

  You are the engine's hands and eyes.

  {BOLD}Card format:{RESET}
    Regular:   {CYAN}5h, 10s, qd, ah, kc{RESET}   (rank + h/d/s/c)
    Jacks:     {CYAN}j1{RESET} = one-eyed   |   {CYAN}j2{RESET} = two-eyed

  {BOLD}When it's the opponent's turn, type their move as:{RESET}
    {CYAN}<card> <row> <col>{RESET}     e.g.  {CYAN}5h 1 6{RESET}
    {CYAN}dead <card>{RESET}            opponent declared a dead card
    {CYAN}pass{RESET}                   no legal move (deck exhausted)

  Type {CYAN}q{RESET} at any prompt to quit.
"""


def setup():
    clear()
    print(INTRO)

    print(f"  {BOLD}Who goes first?{RESET}")
    print(f"    [1] Engine")
    print(f"    [2] Opponent")
    while True:
        choice = input("  > ").strip().lower()
        if choice == 'q':
            sys.exit(0)
        if choice in ('1', 'e', 'engine'):
            engine_player = 0
            break
        if choice in ('2', 'o', 'opponent'):
            engine_player = 1
            break
        print(f"  {RED}Pick 1 or 2.{RESET}")

    print(f"\n  {BOLD}Choose the engine's strategy:{RESET}")
    print(f"    [1] MCTS         (C++ search, strongest; needs the .pyd built)")
    print(f"    [2] Heuristic    (one-ply greedy; pure Python, no build needed)")
    while True:
        choice = input("  > ").strip().lower()
        if choice == 'q':
            sys.exit(0)
        if choice in ('1', 'm', 'mcts'):
            iters = 100000
            opp_fn = _try_load_mcts(iterations=iters, n_parallel_trees=8)
            if opp_fn is None:
                print(f"\n  {YELLOW}Falling back to Heuristic.{RESET}")
                input("  Press Enter to continue...")
                opp_fn      = heuristic_action
                opp_label   = "Heuristic (MCTS unavailable)"
                show_stats  = False
            else:
                opp_label   = f"MCTS ({iters} iterations)"
                show_stats  = True
            break
        if choice in ('2', 'h', 'heuristic'):
            opp_fn      = heuristic_action
            opp_label   = "Heuristic"
            show_stats  = False
            break
        print(f"  {RED}Pick 1 or 2.{RESET}")

    print(f"\n  {BOLD}Enter the engine's 7 starting cards.{RESET}")
    print(f"  {DIM}(All on one line or one at a time — example: 5h 10s qd ah j1 j2 7c){RESET}")
    hand = []
    while len(hand) < HAND_SIZE:
        raw = input(f"  ({len(hand)}/{HAND_SIZE}): ").strip()
        if raw.lower() == 'q':
            sys.exit(0)
        if not raw:
            continue
        for tok in raw.split():
            if len(hand) >= HAND_SIZE:
                break
            card = parse_card(tok)
            if card is None:
                print(f"  {RED}Skipping '{tok}' (couldn't parse).{RESET}")
                continue
            hand.append(card)

    return engine_player, opp_fn, opp_label, show_stats, hand


# --- engine turn ----------------------------------------------------------

def describe_engine_move(card, card_idx, row, col, engine_player):
    """Big readable instruction for the human to execute on the board."""
    eng_color = BLUE  if engine_player == 0 else GREEN
    color_word = 'BLUE' if engine_player == 0 else 'GREEN'

    lines = []
    if row == -1:
        lines.append(f"  {BOLD}{YELLOW}>>> DEAD CARD:{RESET} discard {BOLD}{card_str(card)}{RESET} "
                     f"(slot [{card_idx}]) and draw a replacement.")
        lines.append(f"  Engine will play again this turn.")
    elif card == ONE_EYED_JACK:
        lines.append(f"  {BOLD}{YELLOW}>>> ONE-EYED JACK:{RESET} "
                     f"{BOLD}remove{RESET} the opponent chip at "
                     f"{BOLD}row {row}, col {col}{RESET}.")
        lines.append(f"  (Slot [{card_idx}] in hand.)")
    elif card == TWO_EYED_JACK:
        lines.append(f"  {BOLD}{YELLOW}>>> TWO-EYED JACK:{RESET} "
                     f"place {eng_color}{BOLD}{color_word}{RESET} chip at "
                     f"{BOLD}row {row}, col {col}{RESET}.")
        lines.append(f"  (Slot [{card_idx}] in hand.)")
    else:
        lines.append(f"  {BOLD}{YELLOW}>>> PLAY{RESET} {BOLD}{card_str(card)}{RESET}: "
                     f"place {eng_color}{BOLD}{color_word}{RESET} chip at "
                     f"{BOLD}row {row}, col {col}{RESET}.")
        lines.append(f"  (Slot [{card_idx}] in hand.)")
    return "\n".join(lines)


def engine_turn(game, engine_player, opp_fn, opp_label, show_stats):
    clear()
    render_board(game)
    render_hand(game, engine_player)
    render_status(game, engine_player)

    print(f"\n  {BOLD}{CYAN}Engine ({opp_label}) is thinking...{RESET}")

    # Sanity: MCTS needs current_player == engine_player.
    if game.current_player != engine_player:
        # Defensive — should never happen in normal flow.
        game.current_player = engine_player

    legal_mask = get_legal_action_mask(game, engine_player)
    if not legal_mask.any():
        print(f"  {YELLOW}Engine has no legal moves. Passing.{RESET}")
        game.current_player = 1 - engine_player
        input("  Press Enter to continue...")
        return

    t0         = time.time()
    action_int = opp_fn(game, engine_player, legal_mask)
    dt         = time.time() - t0
    card_idx, row, col = int_to_action(action_int)
    card       = game.hands[engine_player][card_idx]

    # Show the move highlighted on the board.
    clear()
    highlight = {(row, col)} if row >= 0 else set()
    render_board(game, highlight_cells=highlight)
    render_hand(game, engine_player)
    render_status(game, engine_player)

    print()
    print("  " + "━" * 60)
    print(describe_engine_move(card, card_idx, row, col, engine_player))
    print("  " + "━" * 60)

    if show_stats and hasattr(opp_fn, 'last_stats'):
        try:
            st = opp_fn.last_stats
            print(f"  {DIM}{st.iterations} rollouts in {dt:.2f}s "
                  f"(best move: {st.best_move_visits} visits, "
                  f"{st.best_move_winrate*100:.1f}% win){RESET}")
        except Exception:
            print(f"  {DIM}decided in {dt:.2f}s{RESET}")
    else:
        print(f"  {DIM}decided in {dt:.2f}s{RESET}")

    input(f"\n  {BOLD}Press Enter once you've made the play on the physical board...{RESET}")

    info = apply_move(game, engine_player, card, card_idx, row, col)

    if info.get('sequences_formed'):
        clear()
        render_board(game)
        render_status(game, engine_player)
        print(f"\n  {BOLD}{(BLUE if engine_player == 0 else GREEN)}"
              f"★ Engine formed {info['sequences_formed']} sequence(s)!{RESET}")
        time.sleep(1.5)

    if game.done:
        return

    # The played slot is now empty — refill from the user-supplied draw.
    print(f"\n  {BOLD}What card did the engine draw to replace its play?{RESET}")
    print(f"  {DIM}(or 'none' if the physical deck is empty){RESET}")
    drawn = prompt_card_or_none(f"  drew: ")
    if drawn is not None:
        try:
            game.deck.remove(drawn)
        except ValueError:
            print(f"  {YELLOW}Note: both copies of {card_str(drawn)} are already "
                  f"accounted for (typo, or our tracking has drifted). "
                  f"Using it anyway.{RESET}")
            time.sleep(1.2)
    game.hands[engine_player][card_idx] = drawn


# --- opponent turn --------------------------------------------------------

def opponent_turn(game, engine_player):
    opponent_player = 1 - engine_player
    clear()
    render_board(game)
    render_hand(game, engine_player)
    render_status(game, engine_player)

    print(f"\n  {BOLD}Opponent's turn — type what they played.{RESET}")
    print(f"  {DIM}Format: <card> <row> <col>   |   dead <card>   |   pass{RESET}")

    while True:
        raw = input("  > ").strip()
        if raw.lower() == 'q':
            sys.exit(0)
        if not raw:
            continue

        low = raw.lower()
        if low == 'pass':
            # Opponent passes (no legal move / deck exhausted). Just hand off.
            game.current_player = engine_player
            return {}

        if low.startswith('dead'):
            parts = raw.split(maxsplit=1)
            if len(parts) != 2:
                print(f"  {RED}Format: dead <card>{RESET}")
                continue
            card = parse_card(parts[1])
            if card is None:
                print(f"  {RED}Couldn't parse card '{parts[1]}'.{RESET}")
                continue
            # Dead card: opponent stays on turn (free swap). But from our
            # bookkeeping it's identical to an engine dead-card: the played
            # card is now seen, and the opponent draws something we don't
            # learn about (still in their hidden hand).
            info = apply_move(game, opponent_player, card, None, -1, -1)
            if info.get('card_not_in_pool'):
                print(f"  {YELLOW}Note: both copies of {card_str(card)} are already "
                      f"accounted for (typo, or our tracking has drifted).{RESET}")
                time.sleep(1.0)
            # Re-prompt: opponent still has the turn after a dead card.
            print(f"  {DIM}Dead card recorded. Waiting for opponent's real move...{RESET}")
            continue

        parts = raw.split()
        if len(parts) < 3:
            print(f"  {RED}Format: <card> <row> <col>  (or 'dead <card>', 'pass'){RESET}")
            continue

        card = parse_card(parts[0])
        if card is None:
            print(f"  {RED}Couldn't parse card '{parts[0]}'.{RESET}")
            continue
        try:
            row = int(parts[1])
            col = int(parts[2])
        except ValueError:
            print(f"  {RED}Row and col must be integers.{RESET}")
            continue
        if not (0 <= row < 10 and 0 <= col < 10):
            print(f"  {RED}Row/col out of range (0..9).{RESET}")
            continue

        try:
            info = apply_move(game, opponent_player, card, None, row, col)
        except ValueError as e:
            print(f"  {RED}{e}{RESET}")
            continue

        if info.get('card_not_in_pool'):
            print(f"  {YELLOW}Note: {card_str(card)} wasn't in the unseen pool "
                  f"(typo, or our tracking has drifted). Move applied anyway.{RESET}")
            time.sleep(1.2)
        if info.get('sequences_formed'):
            clear()
            render_board(game)
            render_status(game, engine_player)
            print(f"\n  {BOLD}{(GREEN if engine_player == 0 else BLUE)}"
                  f"★ Opponent formed {info['sequences_formed']} sequence(s).{RESET}")
            time.sleep(2)
        return info


# --- main loop ------------------------------------------------------------

def play():
    engine_player, opp_fn, opp_label, show_stats, hand = setup()
    game = make_game(engine_player, hand)

    while not game.done:
        if game.current_player == engine_player:
            engine_turn(game, engine_player, opp_fn, opp_label, show_stats)
        else:
            opponent_turn(game, engine_player)

    clear()
    render_board(game)
    render_status(game, engine_player)
    print(f"\n{'═' * 60}")
    if game.winner == engine_player:
        eng_color = BLUE if engine_player == 0 else GREEN
        print(f"  {BOLD}{eng_color}The engine wins.{RESET}")
    else:
        opp_color = GREEN if engine_player == 0 else BLUE
        print(f"  {BOLD}{opp_color}The opponent wins. Good game!{RESET}")
    print(f"{'═' * 60}\n")


if __name__ == '__main__':
    try:
        play()
    except (KeyboardInterrupt, EOFError):
        print("\nQuitting.")
        sys.exit(0)

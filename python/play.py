"""
Play Sequence against a fixed opponent.
Run: python play.py

Three opponents:
  * MCTS        — the C++ engine via pybind11. Strongest of the three; needs
                  the binding compiled (`make python` from project root).
  * Heuristic   — one-ply greedy on the shaping potential. Pure Python,
                  no compilation needed.
  * Random      — uniform over legal moves.
"""

import os
import sys
import time

# Make the python/ directory importable regardless of where this is run from.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from game_engine import (SequenceGame, EMPTY, PLAYER0, PLAYER1,
                          SEQUENCE0, SEQUENCE1, HAND_SIZE)
from board_layout import (BOARD, JOKER, card_str, TWO_EYED_JACK,
                          ONE_EYED_JACK, SUIT_SYMBOLS)
from seq_actions import int_to_action, get_legal_action_mask
from seq_opponents import heuristic_action, random_action


# --- terminal styling ------------------------------------------------------

BLUE   = '\033[94m'
GREEN  = '\033[92m'
RED    = '\033[91m'
YELLOW = '\033[93m'
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


def render_board(game, highlight_cells=None, show_coords=True):
    if highlight_cells is None:
        highlight_cells = set()

    print(f"\n  {BOLD}Sequence Board{RESET}")
    print("     " + "  ".join(f"{c:2}" for c in range(10)))
    print("    " + "─" * 41)

    for r in range(10):
        row_str = f" {r:2}│"
        for c in range(10):
            cell = BOARD[r][c]
            chip = game.board_chips[r][c]
            is_hl = (r, c) in highlight_cells

            prefix = YELLOW + '[' if is_hl else ' '
            suffix = ']' + RESET if is_hl else ' '

            if cell == JOKER:
                row_str += prefix + BOLD + 'J*' + RESET + suffix
            elif chip != EMPTY:
                row_str += prefix + ' ' + CHIP_DISPLAY[chip] + suffix
            else:
                suit, rank = cell
                col = SUIT_COLOR[suit]
                sym = SUIT_SYMBOLS[suit]
                display = f"{rank}{sym}"
                row_str += prefix + col + f"{display:>2}" + RESET + suffix
        print(row_str)

    print("    " + "─" * 41)
    print(f"  {BLUE}●{RESET} You (P0): {game.sequences[0]} seq  |  "
          f"{GREEN}●{RESET} AI (P1): {game.sequences[1]} seq  |  "
          f"Deck: {len(game.deck)}")


def render_hand(hand, player=0, selected=None):
    print(f"\n  {BOLD}Your Hand:{RESET}")
    for i, card in enumerate(hand):
        marker = YELLOW + '→' + RESET if i == selected else ' '
        if card is None:
            print(f"  {marker} [{i}] (empty)")
        else:
            print(f"  {marker} [{i}] {card_str(card)}")


def get_input(prompt, valid_choices=None):
    while True:
        try:
            val = input(prompt).strip()
            if valid_choices is None:
                return val
            if val in valid_choices:
                return val
            print(f"  Please enter one of: {', '.join(valid_choices)}")
        except (KeyboardInterrupt, EOFError):
            print("\nQuitting...")
            sys.exit(0)


def human_turn(game):
    """Handle human player's turn. Returns action tuple."""
    player = 0
    hand   = game.hands[player]
    legal_actions = game.get_legal_actions(player)

    # Group legal actions by card index.
    card_actions = {}
    for (cidx, r, c) in legal_actions:
        card_actions.setdefault(cidx, []).append((r, c))

    while True:
        clear()
        render_board(game)
        render_hand(hand, player=0)

        print(f"\n  {BOLD}Your turn!{RESET}")
        print(f"  Choose a card to play (0-{HAND_SIZE-1}), or 'q' to quit:")

        choice = get_input("  > ")
        if choice == 'q':
            sys.exit(0)

        try:
            card_idx = int(choice)
        except ValueError:
            continue

        if card_idx < 0 or card_idx >= HAND_SIZE:
            print("  Invalid card index.")
            time.sleep(1)
            continue

        card = hand[card_idx]
        if card is None:
            print("  That slot is empty.")
            time.sleep(1)
            continue

        if card_idx not in card_actions:
            print(f"  {RED}No legal moves for {card_str(card)}.{RESET}")
            time.sleep(1.5)
            continue

        positions = card_actions[card_idx]

        # Dead card with no real placements.
        if positions == [(-1, -1)]:
            print(f"  {YELLOW}Dead card! Playing {card_str(card)} as dead card.{RESET}")
            time.sleep(1.5)
            return (card_idx, -1, -1)

        # Dead card might be offered alongside real moves (one-eyed jack
        # with no opponent targets, etc.).
        has_dead       = (-1, -1) in positions
        real_positions = [(r, c) for r, c in positions if r >= 0]

        highlight = set(real_positions)
        clear()
        render_board(game, highlight_cells=highlight)
        render_hand(hand, player=0, selected=card_idx)
        print(f"\n  Playing: {BOLD}{card_str(card)}{RESET}")

        if card == ONE_EYED_JACK:
            print("  Choose opponent chip to remove (row col):")
        elif card == TWO_EYED_JACK:
            print("  Choose where to place your chip (row col):")
        else:
            print("  Available positions are highlighted. Choose (row col):")

        if has_dead:
            print("  Or type 'd' to play as dead card.")

        while True:
            pos_input = get_input("  > ").strip()

            if pos_input == 'q':
                sys.exit(0)

            if pos_input == 'd' and has_dead:
                return (card_idx, -1, -1)

            parts = pos_input.split()
            if len(parts) == 2:
                try:
                    r, c = int(parts[0]), int(parts[1])
                    if (r, c) in real_positions:
                        return (card_idx, r, c)
                    print(f"  {RED}Invalid position. Choose a highlighted cell.{RESET}")
                except ValueError:
                    print("  Enter row and column as two numbers (e.g. '3 5')")
            else:
                print("  Enter row and column (e.g. '3 5')")


def report_move(opponent_fn, action):
    """Tell a tree-reusing opponent (MCTS) what was just played, so its
    next search re-roots the kept tree instead of starting fresh. No-op
    for plain-function opponents (heuristic, random)."""
    advance = getattr(opponent_fn, 'advance', None)
    if advance is not None:
        advance(action)


def ai_turn(game, opponent_fn):
    """Run opponent turn, return action tuple and info dict."""
    player = 1
    legal_mask = get_legal_action_mask(game, player)
    if not legal_mask.any():
        return None, {}

    action_int = opponent_fn(game, player, legal_mask)
    action     = int_to_action(action_int)
    return action, {}


# --- opponent selection ----------------------------------------------------

def _try_load_mcts(iterations, n_parallel_trees):
    """Build an MCTS opponent if the binding is available; else return None.

    On failure we print the full exception chain so users can see why
    the load failed (missing .pyd vs Windows DLL load error vs ABI
    mismatch — they look identical from a thumbs-up/down perspective
    but the fixes are different)."""
    try:
        from mcts import MCTSOpponent
    except Exception as e:
        print(f"  {YELLOW}MCTS unavailable:{RESET}")
        # Walk the cause chain so the actual error (typically wrapped by
        # mcts.py's ImportError) is visible to the user.
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


def play():
    clear()
    print(f"\n{BOLD}{'='*50}{RESET}")
    print(f"{BOLD}       SEQUENCE - You vs AI{RESET}")
    print(f"{BOLD}{'='*50}{RESET}")
    print(f"\n  {BLUE}●{RESET} You are Player 0 (Blue)")
    print(f"  {GREEN}●{RESET} AI  is Player 1 (Green)")
    print(f"\n  First to {2} sequences wins!")
    print(f"\n  Board legend: {BLUE}★{RESET}=your sequence, "
          f"{GREEN}★{RESET}=AI sequence, J* = JOKER corner (free for all)")

    print(f"\n  {BOLD}Choose your opponent:{RESET}")
    print(f"    [1] MCTS         (C++ search)")
    print(f"    [2] Heuristic    (one-ply greedy on shaping potential)")
    print(f"    [3] Random       (uniform over legal moves)")
    choice = get_input("\n  > ", ['1', '2', '3'])

    if choice == '1':
        opponent_fn = _try_load_mcts(iterations=100000, n_parallel_trees=8)
        if opponent_fn is None:
            print(f"\n  {YELLOW}Falling back to Heuristic.{RESET}")
            input("  Press Enter to continue...")
            opponent_fn = heuristic_action
            opponent_label = "Heuristic (fallback)"
        else:
            opponent_label = "MCTS"
    elif choice == '2':
        opponent_fn   = heuristic_action
        opponent_label = "Heuristic"
    else:
        opponent_fn   = random_action
        opponent_label = "Random"

    print(f"\n  Opponent: {GREEN}{BOLD}{opponent_label}{RESET}")
    input("  Press Enter to start the game...")

    game = SequenceGame()
    game.reset()

    while not game.done:
        player = game.current_player

        if player == 0:
            action = human_turn(game)
            _, _, _, info = game.step(action)

            if 'error' in info:
                print(f"  {RED}Error: {info['error']}{RESET}")
                time.sleep(1.5)
                continue

            report_move(opponent_fn, action)

            if 'sequences_formed' in info:
                clear()
                render_board(game)
                print(f"\n  {BLUE}{BOLD}You formed {info['sequences_formed']} sequence(s)!{RESET}")
                time.sleep(2)

            if 'dead_card' in info:
                clear()
                render_board(game)
                print(f"\n  {YELLOW}Dead card played - drew a new card.{RESET}")
                time.sleep(1.5)
        else:
            clear()
            render_board(game)
            print(f"\n  {GREEN}{BOLD}{opponent_label} is thinking...{RESET}")
            t0 = time.time()
            action, _ = ai_turn(game, opponent_fn)
            dt = time.time() - t0

            if action is None:
                print("  Opponent has no legal moves!")
                time.sleep(1)
                game.current_player = 0
                continue

            card_idx, row, col = action
            ai_card = game.hands[1][card_idx]
            _, _, _, info = game.step(action)
            report_move(opponent_fn, action)

            clear()
            highlight = {(row, col)} if row >= 0 else set()
            render_board(game, highlight_cells=highlight)

            if row == -1:
                print(f"\n  {GREEN}{opponent_label} played dead card: "
                      f"{card_str(ai_card)}{RESET} ({dt:.2f}s)")
            else:
                print(f"\n  {GREEN}{opponent_label} played {card_str(ai_card)} "
                      f"at ({row},{col}){RESET} ({dt:.2f}s)")

            if 'sequences_formed' in info:
                print(f"  {GREEN}{BOLD}{opponent_label} formed "
                      f"{info['sequences_formed']} sequence(s)!{RESET}")

            time.sleep(2)

    # Game over.
    clear()
    render_board(game)
    print(f"\n{'='*50}")
    if game.winner == 0:
        print(f"  {BLUE}{BOLD}🎉 YOU WIN! Congratulations!{RESET}")
    else:
        print(f"  {GREEN}{BOLD}Opponent wins! Better luck next time.{RESET}")
    print(f"{'='*50}")

    again = get_input("\n  Play again? (y/n): ", ['y', 'n', 'Y', 'N'])
    if again.lower() == 'y':
        play()


if __name__ == '__main__':
    play()

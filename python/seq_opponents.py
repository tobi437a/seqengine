"""
Baseline opponents for Sequence — random and one-ply heuristic.

Both match the (game, player, legal_mask) -> action_int signature so they
plug into the same dispatch as the C++ MCTS opponent.

The heuristic is one-ply greedy on the same shaping potential
SequenceGame.step uses (γ·Φ_me_after − Φ_me_before + Φ_opp_before −
γ·Φ_opp_after). Known weakness: WINDOW_WEIGHTS gives a 4-threat and a
completed sequence the same weight (1.0), so it's indifferent between
"extend best threat" and "finish a sequence". That's exactly the gap
MCTS is supposed to fill.
"""
import numpy as np

from game_engine import PLAYER0, PLAYER1, EMPTY
from board_layout import ONE_EYED_JACK
from seq_actions import int_to_action


# Matches SHAPING_GAMMA in game_engine.py.
SHAPING_GAMMA = 0.97


def random_action(game, player, legal_mask):
    """Uniform over legal action indices."""
    legal_indices = np.where(legal_mask)[0]
    return int(np.random.choice(legal_indices))


def _shaping_score(game, player, card_idx, row, col):
    """
    Net potential-based score for playing (card_idx, row, col) as `player`.
    Mutates game.board_chips[row][col] transiently and restores it.
    """
    if row == -1:
        # Dead-card swap: no board change, no shaping reward. Score 0 so
        # the heuristic never prefers a swap over a positive-Φ board move.
        return 0.0

    card = game.hands[player][card_idx]
    my_chip = PLAYER0 if player == 0 else PLAYER1

    phi_me_before  = game._affected_windows_value(row, col, player)
    phi_opp_before = game._affected_windows_value(row, col, 1 - player)

    saved = game.board_chips[row][col]
    try:
        if card == ONE_EYED_JACK:
            game.board_chips[row][col] = EMPTY
        else:
            # Regular and two-eyed jacks both place the player's chip.
            game.board_chips[row][col] = my_chip
        phi_me_after  = game._affected_windows_value(row, col, player)
        phi_opp_after = game._affected_windows_value(row, col, 1 - player)
    finally:
        game.board_chips[row][col] = saved

    return ((SHAPING_GAMMA * phi_me_after  - phi_me_before)
          + (phi_opp_before - SHAPING_GAMMA * phi_opp_after))


def heuristic_action(game, player, legal_mask):
    """
    One-ply greedy on shaping potential. Deterministic; ties broken by
    lowest action_int. The deck shuffle supplies all per-game variance.
    """
    legal_indices = np.where(legal_mask)[0]
    best_score  = -float('inf')
    best_action = int(legal_indices[0])
    for ai in legal_indices:
        card_idx, row, col = int_to_action(int(ai))
        score = _shaping_score(game, player, card_idx, row, col)
        if score > best_score:
            best_score  = score
            best_action = int(ai)
    return best_action

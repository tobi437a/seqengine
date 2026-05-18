"""
Action encoding for Sequence.

The action space is flat-indexed by `card_slot * 101 + position`, where
position is `row * 10 + col` for a board placement and 100 for a dead-card
declaration. Originally lived in rl_agent.py but lifted out here so play.py
can use opponents without pulling in torch.
"""
import numpy as np
from game_engine import HAND_SIZE


N_CARD_SLOTS = HAND_SIZE
N_POSITIONS  = 101                # 0..99 = board cells, 100 = dead card
ACTION_SIZE  = N_CARD_SLOTS * N_POSITIONS


def action_to_int(card_idx, row, col):
    if row == -1:
        pos = 100
    else:
        pos = row * 10 + col
    return card_idx * N_POSITIONS + pos


def int_to_action(action_int):
    card_idx = action_int // N_POSITIONS
    pos      = action_int %  N_POSITIONS
    if pos == 100:
        return (card_idx, -1, -1)
    row, col = divmod(pos, 10)
    return (card_idx, row, col)


def get_legal_action_mask(game, player):
    """Boolean array of size ACTION_SIZE, True for each legal (slot, pos)."""
    mask = np.zeros(ACTION_SIZE, dtype=bool)
    for action in game.get_legal_actions(player):
        idx = action_to_int(*action)
        if idx < ACTION_SIZE:
            mask[idx] = True
    return mask

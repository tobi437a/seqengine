"""
Sequence board layout - 10x10 grid.
Each cell is either:
  - (suit, rank) e.g. ('hearts', '5')
  - 'JOKER' for the four corner free spaces
  - None (unused corners already handled as JOKER)

The board is read from the image left-to-right, top-to-bottom.
Suits: 'spades', 'hearts', 'diamonds', 'clubs'
Ranks: '2','3','4','5','6','7','8','9','10','J','Q','K','A'

Note: Jacks do NOT appear on the board (they are special action cards).
"""

JOKER = 'JOKER'

BOARD = [
    # Row 0
    [JOKER,          ('spades','10'),  ('spades','Q'),   ('spades','K'),   ('spades','A'),   ('diamonds','2'), ('diamonds','3'), ('diamonds','4'), ('diamonds','5'),  JOKER],
    # Row 1
    [('spades','9'),  ('hearts','10'),  ('hearts','9'),   ('hearts','8'),   ('hearts','7'),   ('hearts','6'),   ('hearts','5'),   ('hearts','4'),   ('hearts','3'),   ('diamonds','6')],
    # Row 2
    [('spades','8'),  ('hearts','Q'),   ('diamonds','7'), ('diamonds','8'), ('diamonds','9'), ('diamonds','10'),('diamonds','Q'), ('diamonds','K'), ('hearts','2'),   ('diamonds','7')],
    # Row 3
    [('spades','7'),  ('hearts','K'),   ('diamonds','6'), ('clubs','2'),    ('hearts','A'),   ('hearts','K'),   ('hearts','Q'),   ('diamonds','A'),   ('spades','2'),     ('diamonds','8')],
    # Row 4
    [('spades','6'),  ('hearts','A'),   ('diamonds','5'), ('clubs','3'),    ('hearts','4'),    ('hearts','3'),   ('hearts','10'),  ('clubs','A'),    ('spades','3'),    ('diamonds','9')],
    # Row 5
    [('spades','5'),  ('clubs','2'),    ('diamonds','4'), ('clubs','4'),    ('hearts','5'),   ('hearts','2'),   ('hearts','9'),   ('clubs','K'),    ('spades','4'),    ('diamonds','10')],
    # Row 6
    [('spades','4'),  ('clubs','3'),    ('diamonds','3'), ('clubs','5'),    ('hearts','6'),   ('hearts','7'),   ('hearts','8'),   ('clubs','Q'),    ('spades','5'),     ('diamonds','Q')],
    # Row 7
    [('spades','3'),  ('clubs','4'),    ('diamonds','2'), ('clubs','6'),    ('clubs','7'),    ('clubs','8'),    ('clubs','9'),   ('clubs','10'),   ('spades','6'),    ('diamonds','K')],
    # Row 8
    [('spades','2'),  ('clubs','5'),    ('spades','A'),    ('spades','K'),    ('spades','Q'),    ('spades','10'),   ('spades','9'),   ('spades','8'),   ('spades','7'),     ('diamonds','A')],
    # Row 9 (bottom)
    [JOKER,          ('clubs','6'),    ('clubs','7'),    ('clubs','8'),    ('clubs','9'),    ('clubs','10'),   ('clubs','Q'),    ('clubs','K'),    ('clubs','A'),    JOKER],
]

# Build lookup: card -> list of (row, col) positions on board
def build_card_positions():
    positions = {}
    for r, row in enumerate(BOARD):
        for c, cell in enumerate(row):
            if cell == JOKER:
                continue
            if cell not in positions:
                positions[cell] = []
            positions[cell].append((r, c))
    return positions

CARD_POSITIONS = build_card_positions()

# All unique cards that appear on board
BOARD_CARDS = set(CARD_POSITIONS.keys())

# Suits and ranks
SUITS = ['spades', 'hearts', 'diamonds', 'clubs']
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'Q', 'K', 'A']  # no J on board

# Special jack types
TWO_EYED_JACK = 'two_eyed_jack'   # place anywhere
ONE_EYED_JACK = 'one_eyed_jack'   # remove opponent chip

SUIT_SYMBOLS = {
    'spades': '♠',
    'hearts': '♥',
    'diamonds': '♦',
    'clubs': '♣',
}

RANK_DISPLAY = {r: r for r in RANKS}

def card_str(card):
    if card == TWO_EYED_JACK:
        return 'J2👁 (place anywhere)'
    if card == ONE_EYED_JACK:
        return 'J1👁 (remove chip)'
    suit, rank = card
    return f"{rank}{SUIT_SYMBOLS[suit]}"
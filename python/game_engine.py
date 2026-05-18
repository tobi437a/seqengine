"""
Sequence Game Engine - 1v1
Player 0 = Blue chips, Player 1 = Green chips
"""

import random
from board_layout import (
    BOARD, CARD_POSITIONS, SUITS, RANKS,
    TWO_EYED_JACK, ONE_EYED_JACK, JOKER
)

# Chip states
EMPTY = 0
PLAYER0 = 1
PLAYER1 = 2
SEQUENCE0 = 3   # locked sequence chips for player 0
SEQUENCE1 = 4   # locked sequence chips for player 1

HAND_SIZE = 7   # 1v1 hand size
SEQUENCES_TO_WIN = 2
SEQUENCE_LENGTH = 5

# Number of channels in the CNN's board input (see get_state_vector).
N_BOARD_CHANNELS = 8

# -----------------------------------------------------------------------------
# Reward shaping: threat-potential weights
# -----------------------------------------------------------------------------
# Each of the 192 possible 5-in-a-row windows on the board contributes to a
# player's "threat potential" Φ. The contribution depends on how many of that
# player's chips (counting jokers) sit in the window; any opponent chip blocks
# the window outright (weight 0).
#
# Placement reward becomes:
#     (Φ_me_after − Φ_me_before) + (Φ_opp_before − Φ_opp_after)
#
# i.e. credit for building my own threats AND for reducing the opponent's.
# Sequence completion (count=5) is given the same weight as a 4-threat so
# that finishing a sequence never produces a negative shaping spike — the
# +10 sequence bonus handles the upside.
WINDOW_WEIGHTS = {
    0: 0.00,
    1: 0.01,   # one chip in an open window — barely anything
    2: 0.05,   # pair
    3: 0.20,   # open three
    4: 1.00,   # one chip from a sequence — strong threat
    5: 1.00,   # already a sequence (don't drop when completing)
}

# Build the full deck: 2 full decks, 4 one-eyed jacks, 4 two-eyed jacks
def build_deck():
    deck = []
    for _ in range(2):
        for suit in SUITS:
            for rank in RANKS:
                deck.append((suit, rank))
            # 2 jacks per suit: one 1-eyed, one 2-eyed
            # Spades & Hearts jacks = one-eyed; Diamonds & Clubs = two-eyed
            # (standard Sequence rules)
        for suit in ['spades', 'hearts']:
            deck.append(ONE_EYED_JACK)
        for suit in ['diamonds', 'clubs']:
            deck.append(TWO_EYED_JACK)
    random.shuffle(deck)
    return deck


class SequenceGame:
    def __init__(self):
        self.reset()

    def reset(self):
        self.deck = build_deck()
        self.discard = []
        # 10x10 board of chip states
        self.board_chips = [[EMPTY] * 10 for _ in range(10)]
        # Mark joker corners as permanent (they count as wildcards)
        # Hands
        self.hands = [[], []]
        for p in range(2):
            for _ in range(HAND_SIZE):
                self.hands[p].append(self._draw())
        self.current_player = 0
        self.sequences = [0, 0]   # completed sequences per player
        self.done = False
        self.winner = None
        self.dead_card_used = [False, False]
        return self._get_state()

    def _draw(self):
        if not self.deck:
            # reshuffle discard
            self.deck = self.discard[:]
            self.discard = []
            random.shuffle(self.deck)
        if not self.deck:
            return None
        return self.deck.pop()

    def _get_state(self):
        """Return a dict with full game state."""
        return {
            'board': [row[:] for row in self.board_chips],
            'hands': [h[:] for h in self.hands],
            'current_player': self.current_player,
            'sequences': self.sequences[:],
            'done': self.done,
            'winner': self.winner,
            'deck_size': len(self.deck),
        }

    def get_legal_actions(self, player=None):
        """
        Returns list of (card_index, row, col) tuples.
        For dead cards: (card_index, -1, -1) — always legal for any actually-
        dead card, up to one declaration per turn (tracked via
        self.dead_card_used). After the per-turn swap has been used, dead-card
        declarations remain available only as a fallback when the player has
        no real moves left, to avoid a deadlock.
        For one-eyed jack: (card_index, row, col) where (row,col) has opponent chip (not locked)
        For two-eyed jack: (card_index, row, col) any empty non-JOKER cell
        """
        if player is None:
            player = self.current_player
        hand = self.hands[player]
        opponent_chip = PLAYER1 if player == 0 else PLAYER0

        real_actions = []
        dead_card_actions = []

        for idx, card in enumerate(hand):
            if card is None:
                continue

            if card == TWO_EYED_JACK:
                card_actions = []
                for r in range(10):
                    for c in range(10):
                        if BOARD[r][c] == JOKER:
                            continue
                        if self.board_chips[r][c] == EMPTY:
                            card_actions.append((idx, r, c))
                if card_actions:
                    real_actions.extend(card_actions)
                else:
                    dead_card_actions.append((idx, -1, -1))

            elif card == ONE_EYED_JACK:
                card_actions = []
                for r in range(10):
                    for c in range(10):
                        # Can remove opponent chips that aren't part of a locked sequence
                        if self.board_chips[r][c] == opponent_chip:
                            card_actions.append((idx, r, c))
                if card_actions:
                    real_actions.extend(card_actions)
                else:
                    # No opponent chips on the board → jack has no target → dead
                    dead_card_actions.append((idx, -1, -1))

            else:
                positions = CARD_POSITIONS.get(card, [])
                card_is_dead = True
                for (r, c) in positions:
                    if self.board_chips[r][c] == EMPTY:
                        card_is_dead = False
                        real_actions.append((idx, r, c))
                if card_is_dead:
                    dead_card_actions.append((idx, -1, -1))

        # One dead-card swap per turn. Before it's used, dead-card declarations
        # are freely available alongside any real moves. After it's used, they
        # only come back as a fallback when the player has no real moves at all
        # (preventing a deadlock where the entire hand is dead).
        if not self.dead_card_used[player]:
            return real_actions + dead_card_actions
        if not real_actions:
            return dead_card_actions
        return real_actions

    def step(self, action):
        """
        action: (card_index, row, col)
        Returns (state, reward, done, info)
        """
        if self.done:
            return self._get_state(), 0, True, {'error': 'game over'}

        card_idx, row, col = action
        player = self.current_player
        hand = self.hands[player]

        if card_idx >= len(hand) or hand[card_idx] is None:
            return self._get_state(), -1, False, {'error': 'invalid card index'}

        card = hand[card_idx]
        reward = 0
        info = {}

        my_chip = PLAYER0 if player == 0 else PLAYER1
        opponent_chip = PLAYER1 if player == 0 else PLAYER0

        # Dead card declaration: discard, draw a replacement, and CONTINUE
        # the same player's turn (per official Sequence rules, declaring a
        # dead card is a free swap, not a turn-ending action). Mark the swap
        # as used so the player can't declare another dead card this turn
        # (unless their remaining hand is entirely dead — see get_legal_actions).
        if row == -1 and col == -1:
            info['dead_card'] = True
            self.discard.append(card)
            hand[card_idx] = self._draw()
            self.dead_card_used[player] = True
            # NOTE: do not switch current_player — same player acts again.
            return self._get_state(), 0, False, info

        # Validate the action first — we want to compute the potential
        # snapshot only on legal moves, after all validation has passed,
        # so that we can do an atomic state change with bookkeeping.
        if card == TWO_EYED_JACK:
            if BOARD[row][col] == JOKER or self.board_chips[row][col] != EMPTY:
                return self._get_state(), -1, False, {'error': 'invalid two-eyed jack placement'}
        elif card == ONE_EYED_JACK:
            if self.board_chips[row][col] != opponent_chip:
                return self._get_state(), -1, False, {'error': 'invalid one-eyed jack target'}
        else:
            # Regular card
            positions = CARD_POSITIONS.get(card, [])
            if (row, col) not in positions:
                return self._get_state(), -1, False, {'error': 'card does not match position'}
            if self.board_chips[row][col] != EMPTY:
                return self._get_state(), -1, False, {'error': 'cell not empty'}

        # --- Potential-based reward shaping (snapshot BEFORE the change) ---
        # Only windows passing through (row, col) can have their value altered
        # by this move, so we only re-score those ~20 windows per perspective.
        phi_me_before  = self._affected_windows_value(row, col, player)
        phi_opp_before = self._affected_windows_value(row, col, 1 - player)

        # Apply the change.
        if card == ONE_EYED_JACK:
            self.board_chips[row][col] = EMPTY
        else:
            # Both regular cards and two-eyed jacks place my chip.
            self.board_chips[row][col] = my_chip

        phi_me_after  = self._affected_windows_value(row, col, player)
        phi_opp_after = self._affected_windows_value(row, col, 1 - player)

        # Ng, Harada, Russell 1999
        # $$F(s, s') = \gamma \Phi(s') - \Phi(s)$$
        GAMMA = 0.97
        reward += (GAMMA * phi_me_after - phi_me_before) + (phi_opp_before - GAMMA * phi_opp_after)

        # Check for new sequences
        new_seqs = self._check_sequences(player, row, col)
        if new_seqs > 0:
            self.sequences[player] += new_seqs
            reward += new_seqs * 10
            info['sequences_formed'] = new_seqs

        # Discard and draw
        self.discard.append(card)
        hand[card_idx] = self._draw()

        # Check win
        if self.sequences[player] >= SEQUENCES_TO_WIN:
            self.done = True
            self.winner = player
            reward += 100
            info['winner'] = player

        # Real move completed: the player's turn is over. Clear their
        # dead-card-used flag so they get a fresh swap allotment next turn.
        self.dead_card_used[player] = False
        self.current_player = 1 - player
        return self._get_state(), reward, self.done, info

    def _window_weight(self, start_r, start_c, dr, dc, player):
        """
        Weight of a single 5-window starting at (start_r, start_c) in
        direction (dr, dc), from `player`'s perspective.

        Returns WINDOW_WEIGHTS[k] where k is the number of player's chips
        (counting jokers) in the window, or 0 if the window is blocked
        (contains an opponent chip) or extends off the board.
        """
        end_r = start_r + 4 * dr
        end_c = start_c + 4 * dc
        if not (0 <= start_r < 10 and 0 <= start_c < 10
                and 0 <= end_r < 10 and 0 <= end_c < 10):
            return 0.0

        my_chip  = PLAYER0   if player == 0 else PLAYER1
        my_seq   = SEQUENCE0 if player == 0 else SEQUENCE1
        opp_chip = PLAYER1   if player == 0 else PLAYER0
        opp_seq  = SEQUENCE1 if player == 0 else SEQUENCE0

        my_count = 0
        for i in range(SEQUENCE_LENGTH):
            nr = start_r + dr * i
            nc = start_c + dc * i
            cell_chip = self.board_chips[nr][nc]
            if BOARD[nr][nc] == JOKER:
                my_count += 1
            elif cell_chip == my_chip or cell_chip == my_seq:
                my_count += 1
            elif cell_chip == opp_chip or cell_chip == opp_seq:
                return 0.0  # opponent blocks the window
            # else EMPTY — counts as 0 for both sides
        return WINDOW_WEIGHTS[my_count]

    def _affected_windows_value(self, row, col, player):
        """
        Sum of weights of every 5-window passing through (row, col), from
        `player`'s perspective. Used for potential-based reward shaping:
        since only windows containing the changed cell can have their
        weight altered by a move, we only need to score these ~20 windows
        (4 directions × up to 5 offsets) instead of all 192 board-wide.
        """
        total = 0.0
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            for offset in range(SEQUENCE_LENGTH):
                # window in which (row, col) sits at position `offset`
                start_r = row - dr * offset
                start_c = col - dc * offset
                total += self._window_weight(start_r, start_c, dr, dc, player)
        return total

    def _check_sequences(self, player, last_row, last_col):
        """Count newly formed sequences for player around the last placed chip."""
        # Dead cards use (-1, -1) coordinates
        if last_row < 0 or last_col < 0:
            return 0
            
        my_chip = PLAYER0 if player == 0 else PLAYER1
        seq_chip = SEQUENCE0 if player == 0 else SEQUENCE1

        # If the cell doesn't have my chip or a locked sequence chip, 
        # it can't be part of a new sequence (e.g., One-Eyed Jack removed a chip)
        if self.board_chips[last_row][last_col] not in (my_chip, seq_chip):
            return 0

        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        new_sequences = 0

        # Check only sequences that include the newly placed chip
        for dr, dc in directions:
            # Shift the starting point up to 4 cells backward in the current direction
            for offset in range(-4, 1):
                start_r = last_row + dr * offset
                start_c = last_col + dc * offset
                
                seq_cells = []
                for i in range(SEQUENCE_LENGTH):
                    nr = start_r + dr * i
                    nc = start_c + dc * i
                    if 0 <= nr < 10 and 0 <= nc < 10:
                        cell_chip = self.board_chips[nr][nc]
                        cell_board = BOARD[nr][nc]
                        # Counts if: my chip, my locked chip, or JOKER corner
                        if cell_chip in (my_chip, seq_chip) or cell_board == JOKER:
                            seq_cells.append((nr, nc))
                        else:
                            break
                    else:
                        break

                if len(seq_cells) == SEQUENCE_LENGTH:
                    # Get all non-joker cells in the sequence
                    non_joker = [(nr, nc) for nr, nc in seq_cells if BOARD[nr][nc] != JOKER]

                    # Count how many chips are already locked to a previous sequence
                    locked_count = sum(1 for nr, nc in non_joker if self.board_chips[nr][nc] == seq_chip)
                    
                    # A sequence can share at most 1 chip with a previous sequence
                    if locked_count <= 1:
                        new_sequences += 1
                        # Lock non-joker cells
                        for nr, nc in non_joker:
                            self.board_chips[nr][nc] = seq_chip

        return new_sequences

    def get_state_vector(self, player):
        """
        State representation for the CNN agent.

        Returns a single flat float32 vector laid out as:
          [ board_channels (N_BOARD_CHANNELS * 10 * 10),
            hand encoding (HAND_SIZE * 50),
            sequence counts (2) ]

        The network unflattens the leading N_BOARD_CHANNELS*100 entries back
        into a (C, 10, 10) tensor before applying convolutions. Keeping it
        flat means the replay buffer and the rest of the pipeline don't have
        to change.

        Board channels (all from `player`'s perspective — channel 0 is always
        "mine" regardless of whether player is 0 or 1):
          0: my unlocked chips
          1: opponent's unlocked chips
          2: my locked (sequence) chips
          3: opponent's locked (sequence) chips
          4: joker corners (constant per game)
          5: cells reachable by some regular card currently in my hand
             (i.e., the cell is empty AND a card in my hand maps to it)
          6: legal one-eyed-jack targets (opponent's unlocked chips, but
             only if I hold a one-eyed jack — zero otherwise)
          7: legal two-eyed-jack targets (empty non-joker cells, but only
             if I hold a two-eyed jack — zero otherwise)
        """
        import numpy as np

        my_chip = PLAYER0 if player == 0 else PLAYER1
        opp_chip = PLAYER1 if player == 0 else PLAYER0
        my_seq = SEQUENCE0 if player == 0 else SEQUENCE1
        opp_seq = SEQUENCE1 if player == 0 else SEQUENCE0

        board = np.zeros((N_BOARD_CHANNELS, 10, 10), dtype=np.float32)

        hand = self.hands[player]
        have_one_eyed = any(c == ONE_EYED_JACK for c in hand)
        have_two_eyed = any(c == TWO_EYED_JACK for c in hand)

        # Pre-compute which empty cells are reachable by a regular card in hand.
        playable_by_hand = set()
        for card in hand:
            if card is None or card == ONE_EYED_JACK or card == TWO_EYED_JACK:
                continue
            for (r, c) in CARD_POSITIONS.get(card, []):
                if self.board_chips[r][c] == EMPTY:
                    playable_by_hand.add((r, c))

        for r in range(10):
            for c in range(10):
                chip = self.board_chips[r][c]
                cell_board = BOARD[r][c]

                if chip == my_chip:
                    board[0, r, c] = 1.0
                elif chip == opp_chip:
                    board[1, r, c] = 1.0
                elif chip == my_seq:
                    board[2, r, c] = 1.0
                elif chip == opp_seq:
                    board[3, r, c] = 1.0

                if cell_board == JOKER:
                    board[4, r, c] = 1.0

                if (r, c) in playable_by_hand:
                    board[5, r, c] = 1.0

                if have_one_eyed and chip == opp_chip:
                    board[6, r, c] = 1.0

                if have_two_eyed and chip == EMPTY and cell_board != JOKER:
                    board[7, r, c] = 1.0

        # Hand encoding (one-hot card-type per slot) — unchanged from before.
        all_cards = [(s, rk) for s in ['spades', 'hearts', 'diamonds', 'clubs']
                     for rk in ['2','3','4','5','6','7','8','9','10','Q','K','A']]
        all_cards += [ONE_EYED_JACK, TWO_EYED_JACK]
        card_to_idx = {c: i for i, c in enumerate(all_cards)}
        n_types = len(all_cards)

        hand_vec = np.zeros(HAND_SIZE * n_types, dtype=np.float32)
        for i, card in enumerate(self.hands[player]):
            if card is not None:
                cidx = card_to_idx.get(card, 0)
                hand_vec[i * n_types + cidx] = 1.0

        seq_vec = np.array([self.sequences[player], self.sequences[1-player]])

        return np.concatenate([board.flatten(), hand_vec, seq_vec])

    @property
    def state_size(self):
        n_types = 50  # 48 regular + 2 jacks
        return N_BOARD_CHANNELS * 100 + HAND_SIZE * n_types + 2

    def render(self, highlight_actions=None):
        """Print board to terminal."""
        chip_chars = {
            EMPTY: '·',
            PLAYER0: '\033[94m●\033[0m',    # blue
            PLAYER1: '\033[92m●\033[0m',    # green
            SEQUENCE0: '\033[94m★\033[0m',  # blue star (locked sequence)
            SEQUENCE1: '\033[92m★\033[0m',  # green star
        }
        highlight_cells = set()
        if highlight_actions:
            for _, r, c in highlight_actions:
                if r >= 0:
                    highlight_cells.add((r, c))

        suit_colors = {
            'spades': '\033[90m',
            'clubs': '\033[90m',
            'hearts': '\033[91m',
            'diamonds': '\033[91m',
        }
        reset = '\033[0m'

        print()
        print('    ' + '  '.join(f'{c:2}' for c in range(10)))
        print('   ' + '─' * 42)
        for r in range(10):
            row_str = f'{r:2} │'
            for c in range(10):
                cell = BOARD[r][c]
                chip = self.board_chips[r][c]
                if (r, c) in highlight_cells:
                    row_str += '\033[43m'  # yellow highlight
                if cell == JOKER:
                    row_str += ' J '
                elif chip != EMPTY:
                    row_str += f' {chip_chars[chip]} '
                else:
                    suit, rank = cell
                    col = suit_colors[suit]
                    row_str += f'{col}{rank:>2}{reset} '
                if (r, c) in highlight_cells:
                    row_str += reset
                row_str += '│'
            print(row_str)
        print('   ' + '─' * 42)
        print(f'  🔵 Player 0 sequences: {self.sequences[0]}  |  🟢 Player 1 sequences: {self.sequences[1]}')
        print(f'  Deck: {len(self.deck)} cards remaining')
        print()
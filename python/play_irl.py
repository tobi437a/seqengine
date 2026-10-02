"""
play_irl.py — the engine plays Sequence on a physical board, with a GUI.

You are the engine's hands and eyes:
  * At the start, tap in the engine's 7 cards.
  * When the engine moves, it highlights the square on the on-screen board
    and tells you what to do. Make the play, draw a card for the engine and
    tap the card you drew.
  * When your opponent moves, click the square they played on (or the
    engine chip they removed with a one-eyed jack).

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

import copy
import os
import sys
import time
import tkinter as tk
from tkinter import messagebox

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from game_engine import (SequenceGame, EMPTY, PLAYER0, PLAYER1,
                         SEQUENCE0, SEQUENCE1, HAND_SIZE,
                         SEQUENCES_TO_WIN)
from board_layout import (BOARD, JOKER, TWO_EYED_JACK, ONE_EYED_JACK,
                          SUIT_SYMBOLS, CARD_POSITIONS, SUITS, RANKS)
from seq_actions import int_to_action, get_legal_action_mask
from seq_opponents import heuristic_action


# --- card helpers ---------------------------------------------------------

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


def card_name(card):
    if card == ONE_EYED_JACK:
        return 'One-eyed Jack'
    if card == TWO_EYED_JACK:
        return 'Two-eyed Jack'
    suit, rank = card
    return f"{rank}{SUIT_SYMBOLS[suit]}"


def card_short(card):
    """Fits on a small card tile."""
    if card == ONE_EYED_JACK:
        return 'Jack\n1-eye'
    if card == TWO_EYED_JACK:
        return 'Jack\n2-eye'
    return card_name(card)


def is_red(card):
    return isinstance(card, tuple) and card[0] in ('hearts', 'diamonds')


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
            # More copies of a card than the deck contains — likely a
            # misclick and not worth derailing setup over.
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
            info['card_not_in_pool'] = True
        if card_idx is not None:
            game.hands[player][card_idx] = None
        return info

    my_chip  = PLAYER0 if player == 0 else PLAYER1
    opp_chip = PLAYER1 if player == 0 else PLAYER0

    if card == ONE_EYED_JACK:
        if game.board_chips[row][col] != opp_chip:
            raise ValueError("That square doesn't hold an unlocked enemy chip.")
        game.board_chips[row][col] = EMPTY
    elif card == TWO_EYED_JACK:
        if BOARD[row][col] == JOKER:
            raise ValueError("That's a free corner.")
        if game.board_chips[row][col] != EMPTY:
            raise ValueError("That square is not empty.")
        game.board_chips[row][col] = my_chip
    else:
        positions = CARD_POSITIONS.get(card, [])
        if (row, col) not in positions:
            raise ValueError(f"{card_name(card)} doesn't go on that square.")
        if game.board_chips[row][col] != EMPTY:
            raise ValueError("That square is not empty.")
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


# --- engine selection -----------------------------------------------------

def try_load_mcts(iterations, n_parallel_trees=8):
    """Returns (opponent, None) or (None, error message)."""
    try:
        from mcts import MCTSOpponent
    except Exception as e:
        return None, str(e)
    return MCTSOpponent(iterations=iterations,
                        n_parallel_trees=n_parallel_trees), None


STRENGTHS = {
    'quick':  ('Quick',  'Simple heuristic, instant', None),
    'normal': ('Normal', 'MCTS, 100k simulations',    100_000),
    'strong': ('Strong', 'MCTS, 400k simulations',    400_000),
}


# --- look & feel ----------------------------------------------------------

FONT       = 'Segoe UI'
BG         = '#1d2230'
PANEL      = '#272e3f'
PANEL_2    = '#323a4f'
TEXT       = '#eef0f5'
MUTED      = '#9aa3b8'
ACCENT     = '#f5b301'
DANGER     = '#ff6b6b'
CARD_BG    = '#fbf8f0'
CARD_DIM   = '#8b8f99'
CORNER_BG  = '#e9dcbc'
RED_INK    = '#c62828'
BLACK_INK  = '#1b1b1b'

CHIP_COLORS = {
    'Blue':  ('#2563eb', '#173f99'),
    'Green': ('#16a34a', '#0e5e2b'),
    'Red':   ('#dc2626', '#7f1414'),
}


def f(size, weight='normal'):
    return (FONT, size, weight)


def make_button(parent, text, command, primary=False, **kw):
    bg = ACCENT if primary else PANEL_2
    fg = '#1b1b1b' if primary else TEXT
    opts = dict(text=text, command=command, bg=bg, fg=fg,
                activebackground='#ffd24d' if primary else '#46506a',
                activeforeground=fg, relief='flat', bd=0, cursor='hand2',
                font=f(11, 'bold'), padx=14, pady=8)
    opts.update(kw)
    return tk.Button(parent, **opts)


class Segmented(tk.Frame):
    """Row of toggle buttons bound to a StringVar."""

    def __init__(self, parent, var, options, command=None):
        super().__init__(parent, bg=PANEL)
        for value, label in options:
            tk.Radiobutton(self, text=label, value=value, variable=var,
                           indicatoron=0, command=command,
                           bg=PANEL_2, fg=TEXT, selectcolor=ACCENT,
                           activebackground='#46506a', activeforeground=TEXT,
                           relief='flat', bd=0, font=f(11, 'bold'),
                           padx=16, pady=7, cursor='hand2'
                           ).pack(side='left', padx=(0, 6))
        # Selected button gets dark text so it reads on the yellow.
        var.trace_add('write', lambda *_: self._restyle(var))
        self._restyle(var)

    def _restyle(self, var):
        for b in self.winfo_children():
            sel = b.cget('value') == var.get()
            b.configure(fg='#1b1b1b' if sel else TEXT)


# --- card picker ----------------------------------------------------------

class CardPicker(tk.Frame):
    """Grid of every card in the deck. Click one (or type e.g. '5h' and
    press Enter) to pick it. count_fn(card) -> how many copies are still
    unseen; cards with none left are greyed but still clickable, in case
    our tracking has drifted from the real table."""

    def __init__(self, parent, on_pick, count_fn):
        super().__init__(parent, bg=PANEL)
        self.on_pick  = on_pick
        self.count_fn = count_fn
        self.buttons  = {}

        grid = tk.Frame(self, bg=PANEL)
        grid.pack(anchor='w')
        for r, suit in enumerate(SUITS):
            for c, rank in enumerate(RANKS):
                card = (suit, rank)
                b = tk.Button(grid, text=card_name(card), width=4,
                              font=f(12, 'bold'), relief='flat', bd=0,
                              cursor='hand2', pady=4,
                              command=lambda k=card: self.on_pick(k))
                b.grid(row=r, column=c, padx=2, pady=2)
                self.buttons[card] = b

        jacks = tk.Frame(self, bg=PANEL)
        jacks.pack(anchor='w', pady=(4, 0))
        for card, label in ((TWO_EYED_JACK, 'Jack — two-eyed (wild)'),
                            (ONE_EYED_JACK, 'Jack — one-eyed (remove)')):
            b = tk.Button(jacks, text=label, font=f(11, 'bold'),
                          relief='flat', bd=0, cursor='hand2', pady=5, padx=10,
                          command=lambda k=card: self.on_pick(k))
            b.pack(side='left', padx=2)
            self.buttons[card] = b

        typed = tk.Frame(self, bg=PANEL)
        typed.pack(anchor='w', pady=(8, 0))
        tk.Label(typed, text='or type it (5h, 10s, qd, j1, j2) + Enter:',
                 bg=PANEL, fg=MUTED, font=f(10)).pack(side='left')
        self.entry = tk.Entry(typed, width=8, font=f(12), bg=PANEL_2, fg=TEXT,
                              insertbackground=TEXT, relief='flat')
        self.entry.pack(side='left', padx=6, ipady=3)
        self.entry.bind('<Return>', self._typed)
        self.refresh()

    def _typed(self, _event):
        card = parse_card(self.entry.get())
        if card is None:
            self.entry.configure(bg='#6b2a2a')
            self.after(400, lambda: self.entry.configure(bg=PANEL_2))
            return
        self.entry.delete(0, 'end')
        self.on_pick(card)

    def refresh(self):
        for card, b in self.buttons.items():
            left = self.count_fn(card)
            if left > 0:
                ink = RED_INK if is_red(card) else BLACK_INK
                b.configure(bg=CARD_BG, fg=ink, activebackground='#fff3c4',
                            activeforeground=ink)
            else:
                b.configure(bg='#4a5168', fg=CARD_DIM,
                            activebackground='#5a6280',
                            activeforeground=CARD_DIM)

    def focus_entry(self):
        self.entry.focus_set()


# --- board ----------------------------------------------------------------

class BoardView(tk.Canvas):
    """Scales to fit its space. Clicks are reported as (row, col) to
    on_click when clickable(row, col) is true."""

    def __init__(self, parent, colors):
        super().__init__(parent, bg=BG, highlightthickness=0)
        self.colors     = colors      # player -> colour name
        self.game       = None
        self.marks      = {}          # (r, c) -> 'engine' | 'remove' | 'opp' | 'pending'
        self.clickable  = None
        self.on_click   = None
        self.hover      = None
        self._pulse_on  = False
        self.bind('<Configure>', lambda e: self.redraw())
        self.bind('<Motion>', self._motion)
        self.bind('<Leave>', lambda e: self._set_hover(None))
        self.bind('<Button-1>', self._click)
        self._pulse()

    def geometry_(self):
        w, h = self.winfo_width(), self.winfo_height()
        s = max(10, min(w, h) // 10)
        return s, (w - 10 * s) // 2, (h - 10 * s) // 2

    def cell_at(self, x, y):
        s, ox, oy = self.geometry_()
        c, r = (x - ox) // s, (y - oy) // s
        if 0 <= r < 10 and 0 <= c < 10:
            return int(r), int(c)
        return None

    def redraw(self):
        self.delete('all')
        if self.game is None:
            return
        s, ox, oy = self.geometry_()
        pad = max(2, s // 22)
        for r in range(10):
            for c in range(10):
                x0, y0 = ox + c * s + pad, oy + r * s + pad
                x1, y1 = ox + (c + 1) * s - pad, oy + (r + 1) * s - pad
                cell = BOARD[r][c]
                chip = self.game.board_chips[r][c]
                if cell == JOKER:
                    self.create_rectangle(x0, y0, x1, y1, fill=CORNER_BG,
                                          outline='')
                    self.create_text((x0 + x1) / 2, (y0 + y1) / 2, text='★',
                                     fill='#9c7a2c', font=(FONT, -int(s * .42)))
                    continue
                lit = self.marks.get((r, c)) in ('engine', 'remove', 'pending')
                self.create_rectangle(x0, y0, x1, y1, outline='',
                                      fill='#ffe58a' if lit else CARD_BG)
                ink = RED_INK if is_red(cell) else BLACK_INK
                if chip == EMPTY:
                    self.create_text((x0 + x1) / 2, (y0 + y1) / 2,
                                     text=card_name(cell), fill=ink,
                                     font=(FONT, -int(s * .30), 'bold'))
                else:
                    self.create_text(x0 + 3, y0 + 1, text=card_name(cell),
                                     anchor='nw', fill=ink,
                                     font=(FONT, -int(s * .17), 'bold'))
                    owner = 0 if chip in (PLAYER0, SEQUENCE0) else 1
                    fill, edge = CHIP_COLORS[self.colors[owner]]
                    cx, cy, rad = (x0 + x1) / 2, (y0 + y1) / 2 + s * .04, s * .31
                    self.create_oval(cx - rad, cy - rad, cx + rad, cy + rad,
                                     fill=fill, outline=edge, width=2)
                    if chip in (SEQUENCE0, SEQUENCE1):
                        ir = rad * .72
                        self.create_oval(cx - ir, cy - ir, cx + ir, cy + ir,
                                         outline='white', width=2)
                        self.create_text(cx, cy, text='★', fill='white',
                                         font=(FONT, -int(s * .28)))

        for (r, c), kind in self.marks.items():
            x0, y0 = ox + c * s + 1, oy + r * s + 1
            x1, y1 = ox + (c + 1) * s - 1, oy + (r + 1) * s - 1
            if kind == 'opp':
                self.create_rectangle(x0, y0, x1, y1, outline='white',
                                      width=3, dash=(6, 4))
            elif kind == 'remove':
                cx, cy, rad = (x0 + x1) / 2, (y0 + y1) / 2, s * .32
                self.create_rectangle(x0, y0, x1, y1, outline=ACCENT, width=5,
                                      tags='pulse')
                self.create_line(cx - rad, cy - rad, cx + rad, cy + rad,
                                 fill=DANGER, width=5)
                self.create_line(cx - rad, cy + rad, cx + rad, cy - rad,
                                 fill=DANGER, width=5)
            else:   # 'engine' / 'pending'
                self.create_rectangle(x0, y0, x1, y1, outline=ACCENT, width=5,
                                      tags='pulse')
        self._draw_hover()

    def _draw_hover(self):
        self.delete('hover')
        if self.hover is None:
            return
        s, ox, oy = self.geometry_()
        r, c = self.hover
        self.create_rectangle(ox + c * s + 2, oy + r * s + 2,
                              ox + (c + 1) * s - 2, oy + (r + 1) * s - 2,
                              outline='#7fb3ff', width=3, tags='hover')

    def _set_hover(self, cell):
        if cell != self.hover:
            self.hover = cell
            self.configure(cursor='hand2' if cell else '')
            self._draw_hover()

    def _motion(self, e):
        cell = self.cell_at(e.x, e.y)
        if cell and self.clickable and self.clickable(*cell):
            self._set_hover(cell)
        else:
            self._set_hover(None)

    def _click(self, e):
        cell = self.cell_at(e.x, e.y)
        if cell and self.on_click and self.clickable and self.clickable(*cell):
            self.on_click(*cell)

    def _pulse(self):
        self._pulse_on = not self._pulse_on
        self.itemconfigure('pulse', outline=ACCENT if self._pulse_on else '#ff5a1f')
        self.after(450, self._pulse)


# --- setup screen ---------------------------------------------------------

class SetupScreen(tk.Frame):
    def __init__(self, app, prev=None):
        super().__init__(app, bg=BG)
        self.app  = app
        prev      = prev or {}
        self.hand = []

        self.first     = tk.StringVar(value=prev.get('first', 'engine'))
        self.eng_color = tk.StringVar(value=prev.get('eng_color', 'Blue'))
        self.opp_color = tk.StringVar(value=prev.get('opp_color', 'Green'))
        self.strength  = tk.StringVar(value=prev.get('strength', 'normal'))

        box = tk.Frame(self, bg=PANEL, padx=32, pady=24)
        box.place(relx=.5, rely=.5, anchor='center')

        tk.Label(box, text='Sequence', bg=PANEL, fg=TEXT,
                 font=f(28, 'bold')).pack(anchor='w')
        tk.Label(box, text='The engine plays on your real board. '
                           'You move its chips and tell it what happens.',
                 bg=PANEL, fg=MUTED, font=f(11)).pack(anchor='w', pady=(0, 16))

        def row(label, widget_fn):
            fr = tk.Frame(box, bg=PANEL)
            fr.pack(anchor='w', fill='x', pady=5)
            tk.Label(fr, text=label, bg=PANEL, fg=TEXT, font=f(11, 'bold'),
                     width=16, anchor='w').pack(side='left')
            widget_fn(fr).pack(side='left')

        row('Who goes first?', lambda p: Segmented(
            p, self.first, [('engine', 'Engine'), ('opponent', 'Opponent')]))
        row("Engine's chips", lambda p: Segmented(
            p, self.eng_color, [(k, k) for k in CHIP_COLORS],
            command=lambda: self._fix_colors(self.eng_color)))
        row("Opponent's chips", lambda p: Segmented(
            p, self.opp_color, [(k, k) for k in CHIP_COLORS],
            command=lambda: self._fix_colors(self.opp_color)))
        row('Strength', lambda p: Segmented(
            p, self.strength, [(k, f"{v[0]} · {v[1]}")
                               for k, v in STRENGTHS.items()]))

        tk.Label(box, text="Deal 7 cards to the engine and tap them in:",
                 bg=PANEL, fg=TEXT, font=f(12, 'bold')).pack(anchor='w',
                                                             pady=(18, 6))
        self.hand_row = tk.Frame(box, bg=PANEL)
        self.hand_row.pack(anchor='w', pady=(0, 10))

        self.picker = CardPicker(box, self._add, self._count)
        self.picker.pack(anchor='w')

        bottom = tk.Frame(box, bg=PANEL)
        bottom.pack(fill='x', pady=(18, 0))
        self.start_btn = make_button(bottom, 'Start game  ▶', self._start,
                                     primary=True, font=f(13, 'bold'))
        self.start_btn.pack(side='right')
        self.hint = tk.Label(bottom, text='', bg=PANEL, fg=MUTED, font=f(10))
        self.hint.pack(side='left')

        self._render_hand()
        self.picker.focus_entry()

    def _fix_colors(self, changed):
        other = self.opp_color if changed is self.eng_color else self.eng_color
        if other.get() == changed.get():
            other.set(next(k for k in CHIP_COLORS if k != changed.get()))

    def _count(self, card):
        return build_full_deck().count(card) - self.hand.count(card)

    def _add(self, card):
        if len(self.hand) < HAND_SIZE and self._count(card) > 0:
            self.hand.append(card)
            self._render_hand()

    def _remove(self, i):
        del self.hand[i]
        self._render_hand()

    def _render_hand(self):
        for w in self.hand_row.winfo_children():
            w.destroy()
        for i in range(HAND_SIZE):
            if i < len(self.hand):
                card = self.hand[i]
                tk.Button(self.hand_row, text=card_short(card), width=7,
                          height=2, bg=CARD_BG, relief='flat', bd=0,
                          fg=RED_INK if is_red(card) else BLACK_INK,
                          font=f(12, 'bold'), cursor='hand2',
                          command=lambda k=i: self._remove(k)
                          ).pack(side='left', padx=3)
            else:
                tk.Label(self.hand_row, text='?', width=7, height=2,
                         bg=PANEL_2, fg=MUTED, font=f(12, 'bold')
                         ).pack(side='left', padx=3)
        self.picker.refresh()
        n = len(self.hand)
        self.start_btn.configure(state='normal' if n == HAND_SIZE else 'disabled')
        self.hint.configure(text=f'{n}/{HAND_SIZE} cards — click a card '
                                 'above to take it back' if n else
                                 f'0/{HAND_SIZE} cards')

    def _start(self):
        if len(self.hand) != HAND_SIZE:
            return
        settings = dict(first=self.first.get(), eng_color=self.eng_color.get(),
                        opp_color=self.opp_color.get(),
                        strength=self.strength.get())
        self.app.start_game(settings, list(self.hand))


# --- game screen ----------------------------------------------------------

class GameScreen(tk.Frame):
    """Phases (what we're waiting on):
         thinking     engine is searching
         engine_show  engine's move is on the board; waiting for the draw
         opp          waiting for the opponent's move to be clicked
         opp_choose   empty square clicked; which card was it?
         opp_dead     picking the card the opponent discarded as dead
         over         game finished
    Every waiting phase except the opp_* sub-steps pushes a snapshot, so
    Undo can rewind misclicks."""

    def __init__(self, app, settings, hand, opp_fn, label, has_stats):
        super().__init__(app, bg=BG)
        self.app       = app
        self.settings  = settings
        self.opp_fn    = opp_fn
        self.label     = label
        self.has_stats = has_stats

        self.me   = 0 if settings['first'] == 'engine' else 1
        self.opp  = 1 - self.me
        self.colors = {self.me: settings['eng_color'],
                       self.opp: settings['opp_color']}
        self.game    = make_game(self.me, hand)
        self.marks   = {}
        self.log     = []
        self.pending = None
        self.phase   = None
        self.history = []
        self.hide_hand = tk.BooleanVar(value=False)

        # Layout: board on the left, everything else on the right.
        self.board = BoardView(self, self.colors)
        self.board.pack(side='left', fill='both', expand=True, padx=16, pady=16)
        self.board.clickable = self._clickable
        self.board.on_click  = self._board_click

        side = tk.Frame(self, bg=BG)
        side.pack(side='right', fill='y', padx=(0, 16), pady=16)
        self.wrap = app.px(520)

        # Score line.
        self.score = tk.Frame(side, bg=PANEL, padx=14, pady=10)
        self.score.pack(fill='x')

        # Engine hand.
        hand_box = tk.Frame(side, bg=PANEL, padx=14, pady=10)
        hand_box.pack(fill='x', pady=(10, 0))
        top = tk.Frame(hand_box, bg=PANEL)
        top.pack(fill='x')
        tk.Label(top, text="Engine's hand", bg=PANEL, fg=TEXT,
                 font=f(11, 'bold')).pack(side='left')
        tk.Checkbutton(top, text='hide', variable=self.hide_hand,
                       command=self._render_hand, bg=PANEL, fg=MUTED,
                       selectcolor=PANEL_2, activebackground=PANEL,
                       activeforeground=TEXT, font=f(10)).pack(side='right')
        self.hand_row = tk.Frame(hand_box, bg=PANEL)
        self.hand_row.pack(anchor='w', pady=(6, 0))

        # Action panel (rebuilt per phase).
        self.action = tk.Frame(side, bg=PANEL, padx=16, pady=14)
        self.action.pack(fill='x', pady=(10, 0))

        # Bottom: log + controls.
        controls = tk.Frame(side, bg=BG)
        controls.pack(side='bottom', fill='x', pady=(10, 0))
        self.undo_btn = make_button(controls, '↶ Undo  (Ctrl+Z)', self.undo)
        self.undo_btn.pack(side='left')
        make_button(controls, 'New game', self._new_game).pack(side='right')

        log_box = tk.Frame(side, bg=PANEL, padx=14, pady=10)
        log_box.pack(fill='both', expand=True, pady=(10, 0))
        tk.Label(log_box, text='Moves', bg=PANEL, fg=TEXT,
                 font=f(11, 'bold')).pack(anchor='w')
        self.log_list = tk.Listbox(log_box, bg=PANEL, fg=MUTED, bd=0,
                                   highlightthickness=0, font=f(10),
                                   activestyle='none', selectbackground=PANEL)
        self.log_list.pack(fill='both', expand=True)

        if self.game.current_player == self.me:
            self.start_engine_turn()
        else:
            self.enter_opp()

    # --- naming -------------------------------------------------------

    def cname(self, player):
        return self.colors[player].upper()

    def who(self, player):
        return 'Engine' if player == self.me else 'Opponent'

    # --- snapshots / undo ----------------------------------------------

    def push(self):
        self.history.append(copy.deepcopy(
            (self.game, self.phase, self.pending, self.marks, self.log)))

    def restore(self, snap):
        (self.game, self.phase, self.pending,
         self.marks, self.log) = copy.deepcopy(snap)
        self.render()

    def undo(self):
        if self.phase in ('thinking', None):
            return
        if self.phase in ('opp_choose', 'opp_dead'):
            self.restore(self.history[-1])     # just cancel the sub-step
            return
        if len(self.history) < 2:
            return
        self.history.pop()
        self.restore(self.history[-1])

    def _new_game(self):
        if self.phase != 'over' and not messagebox.askyesno(
                'New game', 'Abandon this game and start a new one?'):
            return
        self.app.show_setup(self.settings)

    # --- rendering ------------------------------------------------------

    def render(self):
        self.board.game  = self.game
        self.board.marks = self.marks
        self.board.redraw()
        self._render_score()
        self._render_hand()
        self._render_log()
        for w in self.action.winfo_children():
            w.destroy()
        getattr(self, f'_panel_{self.phase}')()
        self.undo_btn.configure(
            state='normal' if len(self.history) > 1 or
            self.phase in ('opp_choose', 'opp_dead') else 'disabled')

    def _render_score(self):
        for w in self.score.winfo_children():
            w.destroy()
        for p in (self.me, self.opp):
            fr = tk.Frame(self.score, bg=PANEL)
            fr.pack(side='left', padx=(0, 22))
            c = tk.Canvas(fr, width=18, height=18, bg=PANEL,
                          highlightthickness=0)
            fill, edge = CHIP_COLORS[self.colors[p]]
            c.create_oval(2, 2, 16, 16, fill=fill, outline=edge)
            c.pack(side='left')
            n = self.game.sequences[p]
            tk.Label(fr, text=f' {self.who(p)} ', bg=PANEL, fg=TEXT,
                     font=f(12, 'bold')).pack(side='left')
            for i in range(SEQUENCES_TO_WIN):
                tk.Label(fr, text='★', bg=PANEL, font=f(14),
                         fg=ACCENT if i < n else '#4a5168').pack(side='left')
        tk.Label(self.score, text=self.label, bg=PANEL, fg=MUTED,
                 font=f(10)).pack(side='right')

    def _render_hand(self):
        for w in self.hand_row.winfo_children():
            w.destroy()
        hand = self.game.hands[self.me]
        playing = self.pending['card_idx'] if (
            self.pending and self.phase == 'engine_show') else None
        for i, card in enumerate(hand):
            if card is None:
                txt, bg, fg = ('?' if i == playing else '–'), PANEL_2, MUTED
            elif self.hide_hand.get():
                txt, bg, fg = '', '#3b5aa8', MUTED
            else:
                txt, bg = card_short(card), CARD_BG
                fg = RED_INK if is_red(card) else BLACK_INK
            border = ACCENT if i == playing else PANEL
            tk.Label(self.hand_row, text=txt, width=6, height=2, bg=bg, fg=fg,
                     font=f(11, 'bold'), highlightthickness=3,
                     highlightbackground=border).pack(side='left', padx=2)

    def _render_log(self):
        self.log_list.delete(0, 'end')
        for line in self.log:
            self.log_list.insert('end', line)
        self.log_list.see('end')

    def _title(self, text, color=TEXT):
        tk.Label(self.action, text=text, bg=PANEL, fg=color,
                 wraplength=self.wrap, justify='left',
                 font=f(16, 'bold')).pack(anchor='w')

    def _text(self, text, color=MUTED, size=11, pady=(4, 0)):
        tk.Label(self.action, text=text, bg=PANEL, fg=color,
                 wraplength=self.wrap, justify='left',
                 font=f(size)).pack(anchor='w', pady=pady)

    def _picker(self, on_pick):
        p = CardPicker(self.action, on_pick, self.game.deck.count)
        p.pack(anchor='w', pady=(10, 0))
        p.focus_entry()
        return p

    def _warn(self, text):
        self._text(text, color=DANGER)

    # --- engine turn ----------------------------------------------------

    def start_engine_turn(self):
        self.phase = 'thinking'
        self.render()
        # The search holds the GIL, so let Tk paint "thinking" first.
        self.update_idletasks()
        self.after(60, self._think)

    def _panel_thinking(self):
        self._title('Engine is thinking…', ACCENT)
        self._text(self.label)

    def _think(self):
        g, me = self.game, self.me
        g.current_player = me      # MCTS needs to be the side to move.
        mask = get_legal_action_mask(g, me)
        if not mask.any():
            self.log.append('Engine: no legal move — passes')
            g.current_player = self.opp
            self.enter_opp()
            return

        t0 = time.time()
        action = self.opp_fn(g, me, mask)
        dt = time.time() - t0
        card_idx, row, col = int_to_action(action)
        card = g.hands[me][card_idx]

        stats = f'Decided in {dt:.1f}s'
        if self.has_stats:
            try:
                st = self.opp_fn.last_stats
                stats = (f"Engine's win estimate: {st.best_move_winrate * 100:.0f}%"
                         f"   ·   {st.iterations:,} simulations in {dt:.1f}s")
            except Exception:
                pass

        info = apply_move(g, me, card, card_idx, row, col)
        self.pending = dict(card=card, card_idx=card_idx, row=row, col=col,
                            info=info, stats=stats)
        self.marks = {}
        if row >= 0:
            self.marks[(row, col)] = 'remove' if card == ONE_EYED_JACK else 'engine'

        if row < 0:
            self.log.append(f'Engine: discards dead {card_name(card)}')
        elif card == ONE_EYED_JACK:
            self.log.append(f'Engine: one-eyed Jack removes '
                            f'{card_name(BOARD[row][col])}')
        elif card == TWO_EYED_JACK:
            self.log.append(f'Engine: two-eyed Jack on '
                            f'{card_name(BOARD[row][col])}')
        else:
            self.log.append(f'Engine: {card_name(card)}')
        if info.get('sequences_formed'):
            self.log.append('   ★ Engine completes a sequence!')

        self.phase = 'over' if g.done else 'engine_show'
        self.push()
        self.render()

    def _engine_instruction(self):
        p = self.pending
        card, row, col = p['card'], p['row'], p['col']
        where = f'(row {row + 1}, column {col + 1})' if row >= 0 else ''
        if row < 0:
            return (f'Discard the dead {card_name(card)}',
                    "Both of its squares are taken. Put it on the discard "
                    "pile — the engine moves again after it draws.")
        if card == ONE_EYED_JACK:
            return (f'One-eyed Jack: REMOVE the {self.cname(self.opp)} chip '
                    f'from the {card_name(BOARD[row][col])}',
                    f'The square crossed out on the board {where}.')
        if card == TWO_EYED_JACK:
            return (f'Two-eyed Jack: put a {self.cname(self.me)} chip '
                    f'on the {card_name(BOARD[row][col])}',
                    f'The highlighted square {where}.')
        return (f'Play the {card_name(card)}: put a {self.cname(self.me)} '
                f'chip on it',
                f'The highlighted square {where}.')

    def _panel_engine_show(self):
        title, sub = self._engine_instruction()
        tk.Label(self.action, text="ENGINE'S MOVE", bg=PANEL, fg=ACCENT,
                 font=f(10, 'bold')).pack(anchor='w')
        self._title(title)
        self._text(sub)
        self._text(self.pending['stats'], size=10)
        if self.pending['info'].get('sequences_formed'):
            self._text('★ That completes a sequence!', color=ACCENT, size=13)
        if self.pending['info'].get('card_not_in_pool'):
            self._warn('(Card tracking looks off — that card was already '
                       'counted as seen.)')

        tk.Frame(self.action, bg=PANEL_2, height=2).pack(fill='x', pady=12)
        tk.Label(self.action, text='Then draw a card for the engine and tap it:',
                 bg=PANEL, fg=TEXT, font=f(12, 'bold')).pack(anchor='w')
        self._picker(self._engine_drew)
        make_button(self.action, 'Deck is empty — no card',
                    lambda: self._engine_drew(None),
                    font=f(10)).pack(anchor='w', pady=(10, 0))

    def _engine_drew(self, card):
        g, slot = self.game, self.pending['card_idx']
        if card is not None:
            try:
                g.deck.remove(card)
            except ValueError:
                pass   # both copies already seen; trust the table anyway
        g.hands[self.me][slot] = card
        if g.current_player == self.me:     # dead card: engine goes again
            self.start_engine_turn()
        else:
            self.enter_opp()

    # --- opponent turn --------------------------------------------------

    def enter_opp(self):
        self.phase = 'opp'
        self.pending = None
        self.push()
        self.render()

    def _panel_opp(self):
        tk.Label(self.action, text="OPPONENT'S TURN", bg=PANEL, fg=MUTED,
                 font=f(10, 'bold')).pack(anchor='w')
        self._title('Click the square they played on')
        self._text(f'Placed a chip? Click that square.\n'
                   f'Removed a {self.cname(self.me)} chip with a one-eyed '
                   f'Jack? Click that chip.')
        btns = tk.Frame(self.action, bg=PANEL)
        btns.pack(anchor='w', pady=(14, 0))
        make_button(btns, 'They discarded a dead card',
                    self._enter_opp_dead).pack(side='left', padx=(0, 8))
        make_button(btns, "They couldn't move", self._opp_pass
                    ).pack(side='left')

    def _clickable(self, r, c):
        if self.phase != 'opp' or BOARD[r][c] == JOKER:
            return False
        my_chip = PLAYER0 if self.me == 0 else PLAYER1
        return self.game.board_chips[r][c] in (EMPTY, my_chip)

    def _board_click(self, r, c):
        chip = self.game.board_chips[r][c]
        if chip == EMPTY:
            self.phase = 'opp_choose'
            self.pending = dict(row=r, col=c)
            self.marks = dict(self.marks)
            self.marks[(r, c)] = 'pending'
            self.render()
        else:
            self._opp_play(ONE_EYED_JACK, r, c)

    def _panel_opp_choose(self):
        r, c = self.pending['row'], self.pending['col']
        cell = BOARD[r][c]
        self._title(f'Which card did they play on the {card_name(cell)}?')
        if self.game.deck.count(cell) == 0:
            self._text(f'Both {card_name(cell)} cards have already been '
                       f'played, so it must be a Jack.', color=ACCENT)
        btns = tk.Frame(self.action, bg=PANEL)
        btns.pack(anchor='w', pady=(14, 0))
        make_button(btns, f'  {card_name(cell)}  ', primary=True,
                    font=f(16, 'bold'),
                    command=lambda: self._opp_play(cell, r, c)
                    ).pack(side='left', padx=(0, 10))
        make_button(btns, 'Two-eyed Jack', font=f(14, 'bold'),
                    command=lambda: self._opp_play(TWO_EYED_JACK, r, c)
                    ).pack(side='left', padx=(0, 10))
        make_button(self.action, 'Cancel', self.undo, font=f(10)
                    ).pack(anchor='w', pady=(12, 0))

    def _opp_play(self, card, r, c):
        g = self.game
        try:
            info = apply_move(g, self.opp, card, None, r, c)
        except ValueError as e:
            messagebox.showerror('Not a legal move', str(e))
            self.restore(self.history[-1])
            return
        cell = card_name(BOARD[r][c])
        if card == ONE_EYED_JACK:
            self.log.append(f'Opponent: one-eyed Jack removes {cell}')
        elif card == TWO_EYED_JACK:
            self.log.append(f'Opponent: two-eyed Jack on {cell}')
        else:
            self.log.append(f'Opponent: {cell}')
        if info.get('sequences_formed'):
            self.log.append('   ★ Opponent completes a sequence')
        self.marks = {(r, c): 'opp'}
        if g.done:
            self.phase = 'over'
            self.pending = None
            self.push()
            self.render()
        else:
            self.start_engine_turn()

    def _opp_pass(self):
        self.log.append("Opponent: can't move — passes")
        self.game.current_player = self.me
        self.start_engine_turn()

    def _enter_opp_dead(self):
        self.phase = 'opp_dead'
        self.render()

    def _panel_opp_dead(self):
        self._title('Which card did they discard?')
        self._text('They keep the turn and draw a replacement — click their '
                   'real move afterwards.')
        self._picker(self._opp_dead)
        make_button(self.action, 'Cancel', self.undo, font=f(10)
                    ).pack(anchor='w', pady=(12, 0))

    def _opp_dead(self, card):
        apply_move(self.game, self.opp, card, None, -1, -1)
        self.log.append(f'Opponent: discards dead {card_name(card)}')
        self.enter_opp()

    # --- game over ------------------------------------------------------

    def _panel_over(self):
        won = self.game.winner == self.me
        if self.pending and self.pending.get('card') is not None:
            # Engine's winning move still has to be made on the table.
            title, sub = self._engine_instruction()
            tk.Label(self.action, text="ENGINE'S MOVE", bg=PANEL, fg=ACCENT,
                     font=f(10, 'bold')).pack(anchor='w')
            self._title(title)
            self._text(sub)
            tk.Frame(self.action, bg=PANEL_2, height=2).pack(fill='x', pady=12)
        fill, _ = CHIP_COLORS[self.colors[self.game.winner]]
        self._title('The engine wins! ★★' if won else
                    'The opponent wins. Good game!', fill)
        make_button(self.action, 'Play again', primary=True,
                    command=lambda: self.app.show_setup(self.settings)
                    ).pack(anchor='w', pady=(14, 0))


# --- app ------------------------------------------------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Sequence — engine vs. the table')
        self.configure(bg=BG)
        # Tk scales fonts for DPI but not pixel sizes; px() does that.
        self.k = float(self.tk.call('tk', 'scaling')) / (96 / 72)
        self.geometry(f'{self.px(1400)}x{self.px(900)}')
        self.minsize(self.px(1100), self.px(720))
        try:
            self.state('zoomed')
        except tk.TclError:
            pass
        self.screen   = None
        self.engines  = {}     # strength -> loaded MCTSOpponent
        self.show_setup()

    def px(self, n):
        return int(n * self.k)

    def _swap(self, screen):
        if self.screen is not None:
            self.screen.destroy()
        self.screen = screen
        screen.pack(fill='both', expand=True)
        if isinstance(screen, GameScreen):
            self.bind('<Control-z>', lambda e: screen.undo())
        else:
            self.unbind('<Control-z>')

    def show_setup(self, prev=None):
        self._swap(SetupScreen(self, prev))

    def start_game(self, settings, hand):
        name, desc, iters = STRENGTHS[settings['strength']]
        opp_fn, label, has_stats = heuristic_action, 'Quick (heuristic)', False
        if iters is not None:
            if settings['strength'] not in self.engines:
                eng, err = try_load_mcts(iters)
                if eng is None:
                    messagebox.showwarning(
                        'MCTS unavailable',
                        'Couldn\'t load the C++ engine, so the simple '
                        'heuristic will play instead.\n\n'
                        'Build it with `make python`.\n\n' + err[:1200])
                else:
                    self.engines[settings['strength']] = eng
            if settings['strength'] in self.engines:
                opp_fn = self.engines[settings['strength']]
                label, has_stats = f'{name} · {desc}', True
        self._swap(GameScreen(self, settings, hand, opp_fn, label, has_stats))


def main():
    if sys.platform == 'win32':
        try:   # crisp text on high-DPI screens
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    App().mainloop()


if __name__ == '__main__':
    main()

"""
play.py — play Sequence against the engine on screen, with a GUI.

Run: python play.py

Pick who goes first, chip colours and the opponent:
  * Random   uniform over legal moves
  * Quick    one-ply greedy on the shaping potential (pure Python)
  * Normal   C++ MCTS, 100k simulations   (needs `make python`)
  * Strong   C++ MCTS, 400k simulations
If the C++ binding isn't built, the MCTS levels fall back to Quick.

Click a card in your hand, then one of the highlighted squares (keys 1-7
pick a card, Esc puts it back). A card whose squares are both taken can
be discarded as dead once per turn — you draw a replacement and keep
the turn.

Every game is recorded to games/<date>_<time>_<opponent>.json at the
project root, rewritten after each move so a game cut short is kept too.
See GameRecorder for the format.

The board, card picker and look & feel are shared with play_irl.py (the
version for playing on a physical board).
"""

import json
import os
import subprocess
import sys
import time
import tkinter as tk
from datetime import datetime
from tkinter import messagebox

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from game_engine import (SequenceGame, EMPTY, PLAYER0, PLAYER1,
                         SEQUENCE0, SEQUENCE1, HAND_SIZE, SEQUENCES_TO_WIN)
from board_layout import BOARD, JOKER, TWO_EYED_JACK, ONE_EYED_JACK
from seq_actions import int_to_action, get_legal_action_mask
from seq_opponents import heuristic_action, random_action, _shaping_score
from play_irl import (BG, PANEL, PANEL_2, TEXT, MUTED, ACCENT, CARD_BG,
                      CARD_DIM, RED_INK, BLACK_INK, CHIP_COLORS, f,
                      make_button, Segmented, BoardView, card_name,
                      card_short, is_red, try_load_mcts)


GAMES_DIR = os.path.normpath(os.path.join(HERE, '..', 'games'))

OPPONENTS = {
    'random': ('Random', 'uniform over legal moves', None),
    'quick':  ('Quick',  'one-ply heuristic',        None),
    'normal': ('Normal', 'MCTS, 100k simulations',   100_000),
    'strong': ('Strong', 'MCTS, 400k simulations',   400_000),
}


# --- game record ------------------------------------------------------------

def card_code(card):
    """Compact, parse_card()-compatible name: '10h', 'Qs', 'J1', 'J2'."""
    if card is None:
        return None
    if card == ONE_EYED_JACK:
        return 'J1'
    if card == TWO_EYED_JACK:
        return 'J2'
    suit, rank = card
    return f'{rank}{suit[0]}'


def move_kind(card, row):
    if row < 0:
        return 'dead'
    if card == ONE_EYED_JACK:
        return 'one_eyed'
    if card == TWO_EYED_JACK:
        return 'two_eyed'
    return 'place'


def git_version():
    """`git describe` of the engine source, so a game can be matched to the
    engine that played it. None outside a git checkout."""
    try:
        out = subprocess.run(
            ['git', 'describe', '--always', '--dirty'], cwd=HERE,
            capture_output=True, text=True, timeout=5,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        return out.stdout.strip() or None
    except Exception:
        return None


def heuristic_review(game, player, action, legal):
    """Score every legal move with the one-ply shaping potential (the
    'Quick' opponent's evaluation) and say where the chosen one ranks.
    Cheap, and a useful first filter for blunders on either side."""
    scores = {a: _shaping_score(game, player, *a) for a in legal}
    best   = max(scores, key=scores.get)
    mine   = scores[action]
    return {
        'score':      round(mine, 4),
        'best_score': round(scores[best], 4),
        'best_move':  {'slot': best[0], 'row': best[1], 'col': best[2]},
        'rank':       1 + sum(s > mine + 1e-9 for s in scores.values()),
        'of':         len(scores),
    }


class GameRecorder:
    """
    Writes one JSON file per game:

      format        'seqengine-game/1'
      started, finished          ISO timestamps
      engine_version             git describe of this checkout
      human_seat                 0 or 1 (seat 0 moves first)
      opponent                   strength, label, kind, MCTS config
      legend                     board characters (see below)
      initial                    deck order (drawn from the END of the
                                 list), hands, board before the first move
      moves[]                    one entry per action, dead cards included:
        ply, actor ('human' | 'engine'), seat
        before                   board, both hands, sequences, deck size,
                                 dead_card_used for the mover
        legal_moves              number of legal actions
        move                     slot, card, kind (place | two_eyed |
                                 one_eyed | dead), row, col, square
        drew                     card drawn into the emptied slot
        think_seconds            wall time to decide (humans too)
        heuristic                one-ply shaping score of the move, the
                                 best available score/move and its rank
        engine                   MCTS stats (engine moves, MCTS only):
                                 winrate is the engine's own win estimate
        sequences_formed         if any
        after                    board, both hands, sequences, deck size
      result                     winner ('human' | 'engine' | null),
                                 reason ('sequences' | 'abandoned'), plies

    Board rows are 10-char strings: '*' corner, '.' empty, 'h' / 'e'
    human / engine chip, 'H' / 'E' the same locked into a sequence.
    Cards use card_code(): '10h', 'Qs', 'J1' (one-eyed), 'J2' (two-eyed).
    """

    FORMAT = 'seqengine-game/1'

    def __init__(self, game, human, opponent):
        self.human = human
        h, e = ('h', 'e') if human == 0 else ('e', 'h')
        self.chars = {EMPTY: '.', PLAYER0: h, PLAYER1: e,
                      SEQUENCE0: h.upper(), SEQUENCE1: e.upper()}
        started = datetime.now()
        os.makedirs(GAMES_DIR, exist_ok=True)
        self.path = os.path.join(
            GAMES_DIR, f"{started:%Y-%m-%d_%H-%M-%S}_{opponent['strength']}.json")
        self.data = {
            'format':         self.FORMAT,
            'started':        started.isoformat(timespec='seconds'),
            'finished':       None,
            'engine_version': git_version(),
            'human_seat':     human,
            'opponent':       opponent,
            'legend': {'*': 'corner', '.': 'empty', 'h': 'human chip',
                       'e': 'engine chip', 'H': 'human chip in a sequence',
                       'E': 'engine chip in a sequence'},
            'initial': dict(deck=[card_code(c) for c in game.deck],
                            **self.snapshot(game)),
            'moves':   [],
            'result':  None,
        }
        self.flush()

    def side(self, player):
        return 'human' if player == self.human else 'engine'

    def board(self, game):
        return [''.join('*' if BOARD[r][c] == JOKER
                        else self.chars[game.board_chips[r][c]]
                        for c in range(10)) for r in range(10)]

    def snapshot(self, game):
        return {
            'board': self.board(game),
            'hands': {self.side(p): [card_code(c) for c in game.hands[p]]
                      for p in (0, 1)},
            'sequences': {self.side(p): game.sequences[p] for p in (0, 1)},
            'deck_size': len(game.deck),
        }

    def add_move(self, entry):
        entry['ply'] = len(self.data['moves'])
        self.data['moves'].append(entry)
        self.flush()

    def finish(self, winner, reason):
        if self.data['result'] is not None:
            return
        self.data['finished'] = datetime.now().isoformat(timespec='seconds')
        self.data['result'] = {
            'winner': None if winner is None else self.side(winner),
            'reason': reason,
            'plies':  len(self.data['moves']),
        }
        self.flush()

    def flush(self):
        tmp = self.path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as fh:
            json.dump(self.data, fh, indent=1, ensure_ascii=False)
        os.replace(tmp, self.path)


# --- setup screen -----------------------------------------------------------

class SetupScreen(tk.Frame):
    def __init__(self, app, prev=None):
        super().__init__(app, bg=BG)
        self.app = app
        prev     = prev or {}

        self.first     = tk.StringVar(value=prev.get('first', 'human'))
        self.my_color  = tk.StringVar(value=prev.get('my_color', 'Blue'))
        self.eng_color = tk.StringVar(value=prev.get('eng_color', 'Green'))
        self.strength  = tk.StringVar(value=prev.get('strength', 'normal'))
        self.record    = tk.BooleanVar(value=prev.get('record', True))

        box = tk.Frame(self, bg=PANEL, padx=32, pady=24)
        box.place(relx=.5, rely=.5, anchor='center')

        tk.Label(box, text='Sequence', bg=PANEL, fg=TEXT,
                 font=f(28, 'bold')).pack(anchor='w')
        tk.Label(box, text='You against the engine. First to '
                           f'{SEQUENCES_TO_WIN} sequences wins.',
                 bg=PANEL, fg=MUTED, font=f(11)).pack(anchor='w', pady=(0, 16))

        def row(label, widget_fn):
            fr = tk.Frame(box, bg=PANEL)
            fr.pack(anchor='w', fill='x', pady=5)
            tk.Label(fr, text=label, bg=PANEL, fg=TEXT, font=f(11, 'bold'),
                     width=16, anchor='w').pack(side='left')
            widget_fn(fr).pack(side='left')

        row('Who goes first?', lambda p: Segmented(
            p, self.first, [('human', 'You'), ('engine', 'Engine')]))
        row('Your chips', lambda p: Segmented(
            p, self.my_color, [(k, k) for k in CHIP_COLORS],
            command=lambda: self._fix_colors(self.my_color)))
        row("Engine's chips", lambda p: Segmented(
            p, self.eng_color, [(k, k) for k in CHIP_COLORS],
            command=lambda: self._fix_colors(self.eng_color)))
        row('Opponent', lambda p: Segmented(
            p, self.strength, [(k, v[0]) for k, v in OPPONENTS.items()]))
        self.desc = tk.Label(box, text='', bg=PANEL, fg=MUTED, font=f(10))
        self.desc.pack(anchor='w', padx=(self.app.px(150), 0))
        self.strength.trace_add('write', lambda *_: self._describe())
        self._describe()

        tk.Checkbutton(box, text='Record this game for later analysis',
                       variable=self.record, bg=PANEL, fg=TEXT,
                       selectcolor=PANEL_2, activebackground=PANEL,
                       activeforeground=TEXT, font=f(11)
                       ).pack(anchor='w', pady=(16, 0))
        tk.Label(box, text=f'Saved to {GAMES_DIR}', bg=PANEL, fg=MUTED,
                 font=f(9)).pack(anchor='w', padx=(24, 0))

        make_button(box, 'Start game  ▶', self._start, primary=True,
                    font=f(13, 'bold')).pack(anchor='e', pady=(18, 0))

    def _describe(self):
        self.desc.configure(text=OPPONENTS[self.strength.get()][1])

    def _fix_colors(self, changed):
        other = self.eng_color if changed is self.my_color else self.my_color
        if other.get() == changed.get():
            other.set(next(k for k in CHIP_COLORS if k != changed.get()))

    def _start(self):
        self.app.start_game(dict(
            first=self.first.get(), my_color=self.my_color.get(),
            eng_color=self.eng_color.get(), strength=self.strength.get(),
            record=self.record.get()))


# --- game screen ------------------------------------------------------------

class GameScreen(tk.Frame):
    """Phases: 'human' (your move; self.sel is the picked hand slot),
    'thinking' (engine searching) and 'over'."""

    def __init__(self, app, settings, opp_fn, opponent):
        super().__init__(app, bg=BG)
        self.app      = app
        self.settings = settings
        self.opp_fn   = opp_fn
        self.label    = opponent['label']
        self.has_stats = opponent['kind'] == 'mcts'

        self.human  = 0 if settings['first'] == 'human' else 1
        self.eng    = 1 - self.human
        self.colors = {self.human: settings['my_color'],
                       self.eng:   settings['eng_color']}
        self.game   = SequenceGame()
        self.rec    = (GameRecorder(self.game, self.human, opponent)
                       if settings['record'] else None)
        self.marks   = {}      # last moves: (r, c) -> BoardView mark kind
        self.log     = []
        self.news    = []      # what the engine did since your last move
        self.phase   = None
        self.sel     = None
        self.by_slot = {}      # your legal moves: slot -> [(r, c), ...]
        self.turn_t0 = time.time()

        self.board = BoardView(self, self.colors)
        self.board.pack(side='left', fill='both', expand=True, padx=16, pady=16)
        self.board.clickable = self._clickable
        self.board.on_click  = lambda r, c: self._human_move((self.sel, r, c))

        side = tk.Frame(self, bg=BG)
        side.pack(side='right', fill='y', padx=(0, 16), pady=16)
        self.wrap = app.px(520)

        self.score = tk.Frame(side, bg=PANEL, padx=14, pady=10)
        self.score.pack(fill='x')

        eng_box = tk.Frame(side, bg=PANEL, padx=14, pady=10)
        eng_box.pack(fill='x', pady=(10, 0))
        tk.Label(eng_box, text="Engine's hand", bg=PANEL, fg=TEXT,
                 font=f(11, 'bold')).pack(anchor='w')
        self.eng_row = tk.Frame(eng_box, bg=PANEL)
        self.eng_row.pack(anchor='w', pady=(6, 0))

        self.action = tk.Frame(side, bg=PANEL, padx=16, pady=14)
        self.action.pack(fill='x', pady=(10, 0))

        my_box = tk.Frame(side, bg=PANEL, padx=14, pady=10)
        my_box.pack(fill='x', pady=(10, 0))
        tk.Label(my_box, text='Your hand', bg=PANEL, fg=TEXT,
                 font=f(11, 'bold')).pack(anchor='w')
        self.my_row = tk.Frame(my_box, bg=PANEL)
        self.my_row.pack(anchor='w', pady=(6, 0))

        controls = tk.Frame(side, bg=BG)
        controls.pack(side='bottom', fill='x', pady=(10, 0))
        if self.rec:
            tk.Label(controls, text='● recording', bg=BG, fg=MUTED,
                     font=f(10)).pack(side='left')
        make_button(controls, 'New game', self._new_game).pack(side='right')

        log_box = tk.Frame(side, bg=PANEL, padx=14, pady=10)
        log_box.pack(fill='both', expand=True, pady=(10, 0))
        tk.Label(log_box, text='Moves', bg=PANEL, fg=TEXT,
                 font=f(11, 'bold')).pack(anchor='w')
        self.log_list = tk.Listbox(log_box, bg=PANEL, fg=MUTED, bd=0,
                                   highlightthickness=0, font=f(10),
                                   activestyle='none', selectbackground=PANEL)
        self.log_list.pack(fill='both', expand=True)

        if self.game.current_player == self.human:
            self.start_human_turn()
        else:
            self.start_engine_turn()

    def who(self, player):
        return 'You' if player == self.human else 'Engine'

    # --- making moves ---------------------------------------------------

    def _commit(self, action, think, stats=None):
        """Play `action` for the side to move, tell the engine, record it.
        Returns (card, info)."""
        g    = self.game
        p    = g.current_player
        slot, row, col = action
        card = g.hands[p][slot]
        rec  = self.rec
        if rec:
            legal  = g.get_legal_actions(p)
            before = rec.snapshot(g)
            before['dead_card_used'] = g.dead_card_used[p]
            review = heuristic_review(g, p, action, legal)

        _, _, _, info = g.step(action)
        if 'error' in info:
            raise RuntimeError(f"illegal move {action}: {info['error']}")
        advance = getattr(self.opp_fn, 'advance', None)
        if advance is not None:
            advance(action)

        if rec:
            entry = {
                'actor': rec.side(p), 'seat': p,
                'before': before,
                'legal_moves': len(legal),
                'move': {'slot': slot, 'card': card_code(card),
                         'kind': move_kind(card, row), 'row': row, 'col': col,
                         'square': card_code(BOARD[row][col]) if row >= 0
                                   else None},
                'drew': card_code(g.hands[p][slot]),
                'think_seconds': round(think, 2),
                'heuristic': review,
            }
            if stats is not None:
                entry['engine'] = stats
            if info.get('sequences_formed'):
                entry['sequences_formed'] = info['sequences_formed']
            entry['after'] = rec.snapshot(g)
            rec.add_move(entry)
        return card, info

    def describe(self, player, card, row, col):
        who = self.who(player)
        if row < 0:
            return f'{who}: discards dead {card_name(card)}'
        cell = card_name(BOARD[row][col])
        if card == ONE_EYED_JACK:
            return f'{who}: one-eyed Jack removes {cell}'
        if card == TWO_EYED_JACK:
            return f'{who}: two-eyed Jack on {cell}'
        return f'{who}: {cell}'

    def _after_move(self, player, card, info, row, col):
        self.log.append(self.describe(player, card, row, col))
        if info.get('sequences_formed'):
            self.log.append(f'   ★ {self.who(player)} completes a sequence!')
        if self.game.done:
            if self.rec:
                self.rec.finish(self.game.winner, 'sequences')
            self.phase = 'over'
            self.sel = None
            self.render()
        elif self.game.current_player == self.human:
            self.start_human_turn()
        else:
            self.start_engine_turn()

    # --- your turn ------------------------------------------------------

    def start_human_turn(self):
        g = self.game
        self.by_slot = {}
        for slot, r, c in g.get_legal_actions(self.human):
            self.by_slot.setdefault(slot, []).append((r, c))
        if not self.by_slot:
            self.log.append('You: no legal move — pass')
            g.current_player = self.eng
            self.start_engine_turn()
            return
        self.phase   = 'human'
        self.sel     = None
        self.turn_t0 = time.time()
        self.render()

    def select(self, slot):
        if self.phase != 'human' or slot not in self.by_slot:
            return
        self.sel = None if slot == self.sel else slot
        self.render()

    def key(self, event):
        if event.keysym == 'Escape':
            if self.sel is not None:
                self.select(self.sel)
        elif event.char and event.char in '1234567':
            self.select(int(event.char) - 1)

    def _clickable(self, r, c):
        return (self.phase == 'human' and self.sel is not None
                and (r, c) in self.by_slot[self.sel])

    def _human_move(self, action):
        think = time.time() - self.turn_t0
        card, info = self._commit(action, think)
        _, row, col = action
        if row >= 0:
            self.marks = {(row, col): 'opp'}
            self.news  = []
        self._after_move(self.human, card, info, row, col)

    # --- engine turn ----------------------------------------------------

    def start_engine_turn(self):
        self.phase = 'thinking'
        self.sel   = None
        self.render()
        # The search holds the GIL, so let Tk paint "thinking" first.
        self.update_idletasks()
        self.after(60, self._think)

    def _think(self):
        g = self.game
        mask = get_legal_action_mask(g, self.eng)
        if not mask.any():
            self.log.append('Engine: no legal move — pass')
            self.news.append('The engine had no legal move and passed.')
            g.current_player = self.human
            self.start_human_turn()
            return

        t0 = time.time()
        action = int_to_action(self.opp_fn(g, self.eng, mask))
        dt = time.time() - t0

        stats, summary = None, f'Decided in {dt:.1f}s'
        if self.has_stats:
            st = self.opp_fn.last_stats
            stats = {'winrate':          round(st.best_move_winrate, 4),
                     'iterations':       st.iterations,
                     'best_move_visits': st.best_move_visits,
                     'root_legal_moves': st.root_legal_moves,
                     'max_depth':        st.max_depth,
                     'seconds':          round(st.seconds, 3)}
            summary = (f"Engine's win estimate: {st.best_move_winrate * 100:.0f}%"
                       f"   ·   {st.iterations:,} simulations in {dt:.1f}s")

        card, info = self._commit(action, dt, stats)
        _, row, col = action
        if row >= 0:
            self.marks[(row, col)] = ('remove' if card == ONE_EYED_JACK
                                      else 'engine')
        if row < 0:
            what = f'Discarded a dead {card_name(card)} and drew again'
        elif card == ONE_EYED_JACK:
            what = (f'One-eyed Jack: removed your chip from the '
                    f'{card_name(BOARD[row][col])}')
        elif card == TWO_EYED_JACK:
            what = f'Two-eyed Jack on the {card_name(BOARD[row][col])}'
        else:
            what = f'Played the {card_name(card)}'
        self.news.append((what, summary))
        self._after_move(self.eng, card, info, row, col)

    # --- controls -------------------------------------------------------

    def abandon(self):
        if self.rec and self.phase != 'over':
            self.rec.finish(None, 'abandoned')

    def _new_game(self):
        if self.phase != 'over' and not messagebox.askyesno(
                'New game', 'Abandon this game and start a new one?'):
            return
        self.abandon()
        self.app.show_setup(self.settings)

    # --- rendering ------------------------------------------------------

    def render(self):
        marks = dict(self.marks)
        if self.phase == 'human' and self.sel is not None:
            for rc in self.by_slot[self.sel]:
                if rc[0] >= 0:
                    marks[rc] = 'legal'
        self.board.game  = self.game
        self.board.marks = marks
        self.board.redraw()
        self._render_score()
        self._render_hands()
        self.log_list.delete(0, 'end')
        for line in self.log:
            self.log_list.insert('end', line)
        self.log_list.see('end')
        for w in self.action.winfo_children():
            w.destroy()
        getattr(self, f'_panel_{self.phase}')()

    def _render_score(self):
        for w in self.score.winfo_children():
            w.destroy()
        for p in (self.human, self.eng):
            fr = tk.Frame(self.score, bg=PANEL)
            fr.pack(side='left', padx=(0, 22))
            c = tk.Canvas(fr, width=18, height=18, bg=PANEL,
                          highlightthickness=0)
            fill, edge = CHIP_COLORS[self.colors[p]]
            c.create_oval(2, 2, 16, 16, fill=fill, outline=edge)
            c.pack(side='left')
            tk.Label(fr, text=f' {self.who(p)} ', bg=PANEL, fg=TEXT,
                     font=f(12, 'bold')).pack(side='left')
            for i in range(SEQUENCES_TO_WIN):
                tk.Label(fr, text='★', bg=PANEL, font=f(14),
                         fg=ACCENT if i < self.game.sequences[p] else '#4a5168'
                         ).pack(side='left')
        tk.Label(self.score, text=f'{self.label}  ·  deck {len(self.game.deck)}',
                 bg=PANEL, fg=MUTED, font=f(10)).pack(side='right')

    def _tile(self, parent, text, bg, fg, border=PANEL, caption='',
              on_click=None):
        fr = tk.Frame(parent, bg=PANEL)
        fr.pack(side='left', padx=2)
        lab = tk.Label(fr, text=text, width=6, height=2, bg=bg, fg=fg,
                       font=f(11, 'bold'), highlightthickness=3,
                       highlightbackground=border)
        lab.pack()
        tk.Label(fr, text=caption, bg=PANEL, fg=MUTED, font=f(9)).pack()
        if on_click is not None:
            for w in (fr, lab):
                w.configure(cursor='hand2')
                w.bind('<Button-1>', lambda e: on_click())

    def _render_hands(self):
        for row in (self.eng_row, self.my_row):
            for w in row.winfo_children():
                w.destroy()

        reveal = self.phase == 'over'
        for card in self.game.hands[self.eng]:
            if card is None:
                self._tile(self.eng_row, '–', PANEL_2, MUTED)
            elif reveal:
                self._tile(self.eng_row, card_short(card), CARD_BG,
                           RED_INK if is_red(card) else BLACK_INK)
            else:
                self._tile(self.eng_row, '', '#3b5aa8', MUTED)

        for i, card in enumerate(self.game.hands[self.human]):
            if card is None:
                self._tile(self.my_row, '–', PANEL_2, MUTED)
                continue
            moves  = self.by_slot.get(i, []) if self.phase == 'human' else None
            usable = moves is None or bool(moves)
            dead   = bool(moves) and all(r < 0 for r, _ in moves)
            ink    = RED_INK if is_red(card) else BLACK_INK
            self._tile(self.my_row, card_short(card),
                       CARD_BG if usable else '#4a5168',
                       ink if usable else CARD_DIM,
                       border=ACCENT if i == self.sel else PANEL,
                       caption=f'{i + 1} · dead' if dead else str(i + 1),
                       on_click=(lambda k=i: self.select(k)) if usable else None)

    def _title(self, text, color=TEXT):
        tk.Label(self.action, text=text, bg=PANEL, fg=color,
                 wraplength=self.wrap, justify='left',
                 font=f(16, 'bold')).pack(anchor='w')

    def _text(self, text, color=MUTED, size=11, pady=(4, 0)):
        tk.Label(self.action, text=text, bg=PANEL, fg=color,
                 wraplength=self.wrap, justify='left',
                 font=f(size)).pack(anchor='w', pady=pady)

    def _small_header(self, text, color=MUTED):
        tk.Label(self.action, text=text, bg=PANEL, fg=color,
                 font=f(10, 'bold')).pack(anchor='w')

    def _show_news(self):
        if not self.news:
            return
        self._small_header("ENGINE'S MOVE", ACCENT)
        for item in self.news:
            if isinstance(item, tuple):
                self._text(item[0], color=TEXT, size=13)
                self._text(item[1], size=10, pady=(0, 2))
            else:
                self._text(item, color=TEXT)
        tk.Frame(self.action, bg=PANEL_2, height=2).pack(fill='x', pady=12)

    def _panel_human(self):
        self._show_news()
        self._small_header('YOUR TURN')
        if self.sel is None:
            self._title('Pick a card from your hand')
            self._text('Click a card (or press 1–7). The squares you can play '
                       'it on light up green.')
            return
        card  = self.game.hands[self.human][self.sel]
        moves = self.by_slot[self.sel]
        real  = [rc for rc in moves if rc[0] >= 0]
        if card == ONE_EYED_JACK:
            self._title('One-eyed Jack: click an engine chip to remove')
        elif card == TWO_EYED_JACK:
            self._title('Two-eyed Jack: click any open square')
        elif real:
            self._title(f'Play the {card_name(card)}: click a green square')
        else:
            self._title(f'The {card_name(card)} is dead')
            self._text('Both of its squares are taken.')
        if (-1, -1) in moves:
            make_button(self.action, 'Discard as dead card and draw',
                        lambda: self._human_move((self.sel, -1, -1)),
                        primary=not real).pack(anchor='w', pady=(12, 0))
            self._text('You keep the turn — once per turn.', size=10)
        self._text('Esc or click the card again to put it back.', size=10,
                   pady=(10, 0))

    def _panel_thinking(self):
        self._title('Engine is thinking…', ACCENT)
        self._text(self.label)

    def _panel_over(self):
        self._show_news()
        won = self.game.winner == self.human
        fill, _ = CHIP_COLORS[self.colors[self.game.winner]]
        self._title('You win! ★★' if won else 'The engine wins.', fill)
        if self.rec:
            self._text(f'Game saved to {self.rec.path}', size=10, pady=(8, 0))
        make_button(self.action, 'Play again', primary=True,
                    command=lambda: self.app.show_setup(self.settings)
                    ).pack(anchor='w', pady=(14, 0))


# --- app --------------------------------------------------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Sequence — you vs. the engine')
        self.configure(bg=BG)
        # Tk scales fonts for DPI but not pixel sizes; px() does that.
        self.k = float(self.tk.call('tk', 'scaling')) / (96 / 72)
        self.geometry(f'{self.px(1400)}x{self.px(900)}')
        self.minsize(self.px(1100), self.px(720))
        try:
            self.state('zoomed')
        except tk.TclError:
            pass
        self.screen = None
        self.protocol('WM_DELETE_WINDOW', self._close)
        self.show_setup()

    def px(self, n):
        return int(n * self.k)

    def _swap(self, screen):
        if self.screen is not None:
            self.screen.destroy()
        self.screen = screen
        screen.pack(fill='both', expand=True)
        if isinstance(screen, GameScreen):
            self.bind('<Key>', screen.key)
        else:
            self.unbind('<Key>')

    def _close(self):
        if isinstance(self.screen, GameScreen):
            self.screen.abandon()
        self.destroy()

    def show_setup(self, prev=None):
        self._swap(SetupScreen(self, prev))

    def start_game(self, settings):
        key = settings['strength']
        name, desc, iters = OPPONENTS[key]
        opponent = dict(strength=key, label=f'{name} · {desc}', kind=key)
        opp_fn = random_action if key == 'random' else heuristic_action
        if key == 'quick':
            opponent['kind'] = 'heuristic'
        if iters is not None:
            # A fresh engine per game: its kept search tree belongs to the
            # previous game.
            eng, err = try_load_mcts(iters)
            if eng is None:
                messagebox.showwarning(
                    'MCTS unavailable',
                    'Couldn\'t load the C++ engine, so the Quick heuristic '
                    'will play instead.\n\nBuild it with `make python`.\n\n'
                    + err[:1200])
                opponent.update(kind='heuristic',
                                label='Quick · heuristic (MCTS fallback)')
            else:
                opp_fn = eng
                cfg = eng._cfg
                opponent.update(kind='mcts', config={
                    k: getattr(cfg, k) for k in dir(cfg)
                    if not k.startswith('_')
                    and isinstance(getattr(cfg, k), (bool, int, float))})
        self._swap(GameScreen(self, settings, opp_fn, opponent))


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

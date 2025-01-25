import traceback
import chess
from flask import Flask, request
from torch.functional import Tensor
import torch
import collections
import copy
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


# -------------------- UTILS ---------------------------

import base64
import chess.svg


def to_svg(state):
    return base64.b64encode(chess.svg.board(board=state.board).encode('utf-8')).decode('utf-8')


def stockfish_treshold(x):
    if x > 1.5:
        return 0
    elif x < -1.5:
        return 1
    else:
        return 2
    

# ------------------- STATES ----------------------------

class State(object):

    def __init__(self, board=None):
        self.board = chess.Board() if board is None else board

    @property
    def legal_moves(self):
        return list(self.board.legal_moves)

    def serialize(self):
        import numpy as np
        piece_dict = {"P": 1, "N": 1, "B": 1, "R": 1, "Q": 1, "K": 1,
                      "p": -1, "n": -1, "b": -1, "r": -1, "q": -1, "k": -1}

        state = np.zeros(768)
        idx = 0
        for k in piece_dict.keys():
            for i in range(64):
                piece = self.board.piece_at(i)
                if piece is not None:
                    turn = 1 if self.board.turn else -1
                    state[idx] = piece_dict[k] * turn if k == piece.symbol() else 0
                else:
                    state[idx] = 0

                idx += 1

        return state



# ----------------- MONTE CARLO TREE SEARCH ----------------------
current_child = 0


class Node:
    def __init__(self, state, move, parent=None):
        self.state = state
        self.move = move
        self.parent = parent
        self.expanded = False
        self.children = {}
        self.children_priors = np.zeros([self.state.children_len], dtype=np.float32)
        self.children_priors.fill(2)
        self.children_total_values = np.zeros([self.state.children_len], dtype=np.float32)
        self.children_number_visits = np.zeros([self.state.children_len], dtype=np.float32)
        self.win = False

    @property
    def number_visits(self):
        return self.parent.children_number_visits[self.move]

    @number_visits.setter
    def number_visits(self, value):
        self.parent.children_number_visits[self.move] = value

    @property
    def total_value(self):
        return self.parent.children_total_values[self.move]

    @total_value.setter
    def total_value(self, value):
        self.parent.children_total_values[self.move] = value

    def children_q(self): 
        return self.children_total_values / (1 + self.children_number_visits)

    def children_u(self):  
        return self.children_priors * np.sqrt(self.number_visits / (1.0 + self.children_number_visits))

    def best_child(self):
        return np.argmax(self.children_q() + self.children_u())

    def select_leaf(self):
        current = self
        if current.parent.parent is None:
            current.expanded = True
        while current.expanded:
            try:
                global current_child
                best_move = current.best_child()
                current_child = best_move
                current = current.maybe_add_child(best_move)
            except:
                continue
        return current

    def expand(self, children_priors):
        self.expanded = True
        state = copy.deepcopy(self.state.state)
        while state.board.is_game_over() is not True:
            move = state.board.san(random_move(state))
            state.board.push_san(move)
        winner = False
        result = state.board.result()
        if (result == '1-0' and state.board.turn) or (result == '0-1' and not state.board.turn) or result == '1/2-1/2':
            winner = True
        if winner is True:
            self.win = True
            self.total_value += 1
        self.number_visits += 1

    def maybe_add_child(self, move):
        if move >= len(self.children) or self.children[move] is None:
            self.children[move] = Node(self.state.play(move), move, parent=self)
        return self.children[move]

    def backup(self, value_estimate: float):
        current = self
        while current.parent.parent is not None:
            current.parent.total_value = np.sum(current.parent.children_total_values, dtype=np.float32)
            if current.parent.win is True:
                current.parent.total_value += 1
            current.parent.number_visits = np.sum(current.parent.children_number_visits, dtype=np.float32) + 1
            current = current.parent
           

def random_move(state):
    return random.choice([move for move in state.legal_moves])


def net_evaluation_move(state):
    successors = []
    for move in state.legal_moves:
        state.board.push_san(str(move))
        successors.append(torch.argmax(model(Tensor(state.serialize()))))
        state.board.pop()

    if 0 in successors:
        return state.legal_moves[successors.index(0)]
    elif 2 in successors:
        return state.legal_moves[successors.index(2)]
    elif successors:
        return successors[0]
    else:
        return []


class TestNode(object):
    def __init__(self):
        self.parent = None
        self.children_total_values = collections.defaultdict(float)
        self.children_number_visits = collections.defaultdict(float)


class NeuralNet:
    @classmethod
    def evaluate(self, state):
        return 2.0, 1.0


def visit_children(root, to_range):
    for i in range(to_range):
        root.children[i] = Node(root.state.play(i), i, parent=root)
        root.children[i].expanded = True
        state1 = copy.deepcopy(root.children[i].state.state)
        while state1.board.is_game_over() is not True:
            move = state1.board.san(random_move(state1))
            state1.board.push_san(move)
        winner = False 
        result = state1.board.result()
        if (result == '1-0' and state1.board.turn) or (result == '0-1' and not state1.board.turn) or result == '1/2-1/2':
            winner = True
        if winner is True:
            root.children[i].win = True
            root.children[i].total_value += 1
        root.children[i].number_visits += 1
        root.children[i].backup(1)


def uct_search(state, n_simulations):
    root = Node(state, move=None, parent=TestNode())
    root.expanded = True
    visit_children(root, root.state.children_len)

    for _ in range(n_simulations):  
        leaf = root.select_leaf()
        children_priors, value_estimate = NeuralNet.evaluate(leaf.state)
        leaf.expand(children_priors)
        leaf.backup(value_estimate)
    return state.state.legal_moves[np.argmax(root.children_total_values)]


class GameState:
    def __init__(self, turn=1, state=None):
        self.turn = turn
        self.state = state
        self.children_len = len(state.legal_moves) if state.legal_moves else 0

    def play(self, move):
        state = copy.deepcopy(self.state)
        move_str = self.state.board.san(self.state.legal_moves[move])
        state.board.push_san(move_str)
        return GameState(-self.turn, state)
    

# -------------------- MODEL ARCHITECTURE ----------------------
class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 64)

        self.last = nn.Linear(64, 3)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.dropout(x)

        return self.last(x)



# ---------------- LOAD MODEL ------------------------

model = Model()
model.load_state_dict(torch.load('./model.pth', map_location=torch.device('cpu'))['state_dict'])
model.eval()





# ---------- MAIN FLASK APP --------------
    
app = Flask(__name__)


STATE = State()


def random_move(state):
    return random.choice([move for move in state.legal_moves])


@app.route("/")
def home():
    return open("static/index.html").read().replace('start', STATE.board.fen())


@app.route("/new-game")
def new_game():
    STATE.board.reset()
    return app.response_class(response=STATE.board.fen(), status=200)


@app.route("/self-play")
def self_play():
    state = State()
    ret = '<html><head>'

    while not state.board.is_game_over():
        move = state.board.san(random_move(state))
        state.board.push_san(move)
        ret += '<img width=600 height=600 src="data:image/svg+xml;base64,%s"></img><br/>' % to_svg(state)

    print(state.board.result())
    return ret
        

@app.route("/move")
def move():
    if STATE.board.is_game_over():
        return app.response_class(response="Game over!", status=200)

    source = int(request.args.get('from', default=''))
    target = int(request.args.get('to', default=''))
    promotion = True if request.args.get('promotion', default='') == 'true' else False

    move = STATE.board.san(chess.Move(source, target, promotion=chess.QUEEN if promotion else None))

    if move is None or move == '':
        return app.response_class(response=STATE.board.fen(), status=200)

    try:
        STATE.board.push_san(move)
        if STATE.board.is_game_over():
            return app.response_class(response="Game over!", status=200)

        computer_move = uct_search(GameState(state=copy.deepcopy(STATE)), n_simulations=50)
        if chess.Move.from_uci(str(computer_move) + 'q') in STATE.board.legal_moves:
            computer_move.promotion = chess.QUEEN

        STATE.board.push_san(STATE.board.san(computer_move))
        if STATE.board.is_game_over():
            return app.response_class(response="Game over!", status=200)

    except Exception:
        traceback.print_exc()

    return app.response_class(response=STATE.board.fen(), status=200)


if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0",port=5000)
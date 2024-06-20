import chess
import math
import numpy as np
from os import listdir

EXPORT_DIR = "Export/"
BEST_VERSION_FILE = "best_version.txt"
PORT_FILE = "port.txt"
ADDRESS_FILE = "address.txt"

PIECE_SYMBOLS = ['p', 'n', 'b', 'r', 'q', 'k', 'P', 'N', 'B', 'R', 'Q', 'K']
QUEEN_MOVES = {'N': 0, 'NE': 1, 'E': 2, 'SE': 3, 'S': 4, 'SW': 5, 'W': 6, 'NW': 7}
KNIGHT_MOVES = {'NWF': 0, 'NWS': 1, 'NEF': 2, 'NES': 3, 'SEF': 4, 'SES': 5, 'SWF': 6, 'SWS': 7}

def decode_result(result, turn):
    if result == '1-0':
        return float(turn)
    if result == '0-1':
        return float(not turn)
    if result == '1/2-1/2':
        return 0.5
    return None

def decode_symbol(symbol):
    return PIECE_SYMBOLS.index(symbol)

def num_squares_attacking(square, board):
    return len(board.attacks(square))

def build_heatmap(board, is_defensive=True):
    heatmap = [0] * 64
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            color = piece.color
            if is_defensive:
                heatmap[square] = len(board.attackers(not color, square)) * (1 if color else -1)
            else:
                for attack_square in board.attacks(square):
                    heatmap[attack_square] += (1 if color else -1)
    return heatmap

def print_heatmap(heatmap):
    reshaped_heatmap = np.reshape(np.array(heatmap), (8, 8))
    for row in reversed(reshaped_heatmap):
        print(row)

def expand_position(board):
    if not isinstance(board, chess.Board):
        print("Tried to expand non-Board object")
        return []

    expanded = [0] * (64 * 13)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            offset = decode_symbol(piece.symbol())
            expanded[offset * 64 + square] = 1
    if board.turn:
        expanded[12 * 64:] = [1] * 64
    return expanded

def direction_and_distance(square1, square2):
    rank1, file1 = chess.square_rank(square1), chess.square_file(square1)
    rank2, file2 = chess.square_rank(square2), chess.square_file(square2)

    horiz = 'W' if file1 > file2 else 'E'
    vert = 'S' if rank1 > rank2 else 'N'
    file_dist, rank_dist = abs(file1 - file2), abs(rank1 - rank2)
    is_diag, is_straight = file_dist == rank_dist, file_dist == 0 or rank_dist == 0
    is_knight = (file_dist == 1 and rank_dist == 2) or (file_dist == 2 and rank_dist == 1)
    far = 'F' if rank_dist == 2 else 'S'

    if is_diag:
        return {'dir': f"{vert}{horiz}", 'dist': file_dist}
    elif is_straight:
        return {'dir': horiz if file_dist else vert, 'dist': max(file_dist, rank_dist)}
    elif is_knight:
        return {'dir': f"{vert}{horiz}{far}", 'dist': -1}

def get_prediction_index(move):
    from_index = move.from_square * 73
    dir_dist = direction_and_distance(move.from_square, move.to_square)
    direction, distance = dir_dist['dir'], dir_dist['dist']

    if distance == -1:
        offset = 56 + KNIGHT_MOVES[direction]
    elif move.promotion:
        left, right, straight = direction in {'NW', 'SE'}, direction in {'NE', 'SW'}, direction in {'N', 'S'}
        promo_offset = (move.promotion - 2) * 3
        if left:
            offset = 64 + promo_offset
        elif right:
            offset = 65 + promo_offset
        elif straight:
            offset = 66 + promo_offset
    else:
        offset = QUEEN_MOVES[direction] * 7 + distance

    return from_index + offset

def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read().strip()

def best_version():
    return int(read_file(BEST_VERSION_FILE))

def update_best_version(version):
    if isinstance(version, int):
        with open(BEST_VERSION_FILE, 'w') as file:
            file.write(str(version))
    else:
        print("Invalid version number for best player")

def latest_version():
    return int(sorted(listdir(EXPORT_DIR), reverse=True)[0])

def get_address():
    return read_file(ADDRESS_FILE)

def get_port():
    return int(read_file(PORT_FILE))

def logit_to_prob(logit):
    return math.exp(logit) / (1 + math.exp(logit))

def prob_to_logit(prob):
    return math.log(prob / (1 - prob))

import sys
sys.path.append(sys.path[0] + "/..")

import chess
import os
import time
import datetime
import numpy as np
import pychess_utils as chess_utils
from chess import pgn
from deepmind_mcts import MCTS

PGN_DIRECTORY = "ACZData/pgn/"
DATA_FILE = "ACZData/self_play.csv"
BATCH_SIZE = 1
ALLOW_DRAW = True
ENGINE_ID = "ACZ"

def array_to_csv_string(array):
    return ','.join(map(str, array))

def save_board_data(boards, policies, result):
    if len(boards) != len(policies):
        print("Mismatch between boards and policy lengths.")
    with open(DATA_FILE, "a") as file:
        for board, policy in zip(boards, policies):
            board_state = chess_utils.expand_position(board)
            game_result = chess_utils.decode_result(result, board.turn)
            file.write(array_to_csv_string(board_state + [policy] + [game_result]) + "\n")

def save_game_data(game_pgn):
    version_path = os.path.join(PGN_DIRECTORY, str(chess_utils.best_version()))
    if not os.path.exists(version_path):
        os.makedirs(version_path)
    game_count = len(os.listdir(version_path))
    game_path = os.path.join(version_path, f"{game_count}.pgn")
    with open(game_path, "w") as file:
        file.write(game_pgn)

def execute_game():
    game = chess.pgn.Game()
    board = chess.Board()
    mcts = MCTS(version=chess_utils.best_version(), startpos=board)

    boards = []
    policies = []
    move_count = 0
    temperature_active = True

    current_node = game

    while not board.is_game_over(claim_draw=ALLOW_DRAW) and move_count < 200:
        start_time = time.time()
        mcts.build()
        boards.append(board.copy())
        policies.append(mcts.get_policy_string())
        move = mcts.best_move()
        board.push(move)
        move_count += 1
        if move_count == 15:
            temperature_active = False
        print(f"Move {move_count}: {move.uci()}")
        print(board)
        current_node = current_node.add_variation(move)
        mcts = MCTS(startpos=board, prev_mcts=mcts, temperature=temperature_active)
        elapsed_time = time.time() - start_time
        print(f"Time for move {move_count}: {elapsed_time:.2f} seconds")

    result = board.result(claim_draw=ALLOW_DRAW) if move_count < 200 else "1/2-1/2"
    save_board_data(boards, policies, result)

    game.headers["White"] = ENGINE_ID
    game.headers["Black"] = ENGINE_ID
    game.headers["Date"] = datetime.date.today().isoformat()
    game.headers["Event"] = "N/A"
    game.headers["Result"] = result
    save_game_data(game.accept(chess.pgn.StringExporter(headers=True, variations=True, comments=True)))
    return board

def main():
    for _ in range(BATCH_SIZE):
        execute_game()

if __name__ == "__main__":
    main()

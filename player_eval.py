import sys
sys.path.append(sys.path[0] + "/..")

import chess
import os
import datetime
import pychess_utils as util
from chess import pgn
from deepmind_mcts import MCTS

EVAL_GAMES = 10

latest_player = MCTS(startpos=chess.Board())
best_player = MCTS(startpos=chess.Board(), version=util.best_version())

def play_game(best_player_starts=True):
    player1, player2 = (best_player, latest_player) if best_player_starts else (latest_player, best_player)
    board = player1.startpos
    move_count = 0
    next_temp = True
    turn = True

    while not board.is_game_over(claim_draw=True) and move_count < 200:
        current_player = player1 if turn else player2
        current_player.build()
        move = current_player.best_move()
        board.push(move)
        move_count += 1
        
        if move_count == 20:
            next_temp = False

        print(f"Move {move_count}: {move.uci()}")
        print(board)

        if turn:
            player2 = MCTS(startpos=board, prev_mcts=current_player, temperature=next_temp, startcolor=board.turn)
        else:
            player1 = MCTS(startpos=board, prev_mcts=current_player, temperature=next_temp, startcolor=board.turn)

        turn = not turn

    result = util.decode_result(board.result(claim_draw=True), not best_player_starts) if move_count < 200 else 0.5
    return result

def main():
    results = 0
    for i in range(EVAL_GAMES):
        results += play_game(best_player_starts=(i % 2 == 0))

    if results >= (EVAL_GAMES * 0.55):
        print("New player won!")
        util.update_best_player(latest_player.version)
    else:
        print("New player did not reach 55% wins, best player unchanged.")

if __name__ == "__main__":
    main()

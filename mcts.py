import chess
import sys
import math
import numpy as np
import argparse
import datetime
import operator
import random
from random import choice
import copy
from chess import pgn, uci
from collections import defaultdict

DRAW = 'draw'

def get_sibling_nodes(node):
    return [sibling for sibling in node.parent.children if sibling != node]

def get_total_simulations_at_depth(node):
    return sum(sibling.simulations for sibling in get_sibling_nodes(node)) + node.simulations

def ucb1_formula(node):
    if node.simulations == 0:
        return float('inf')
    win_ratio = node.wins / node.simulations
    total_sims = get_total_simulations_at_depth(node)
    exploration_term = math.sqrt(2) * math.sqrt(math.log(total_sims) / node.simulations)
    return win_ratio + exploration_term

class TreeNode:
    def __init__(self, color, parent=None, board_position=None):
        self.color = color
        self.parent = parent
        self.children = []
        self.board_position = board_position if isinstance(board_position, chess.Board) else None
        self.move = None
        self.wins = 0
        self.simulations = 0

    def add_child(self, child_node):
        self.children.append(child_node)

class MonteCarloTreeSearch:
    def __init__(self, start_position=chess.Board(), iteration_time=100):
        self.start_position = start_position
        self.root = TreeNode(True, board_position=start_position)
        self.iteration_time = max(iteration_time, 1)
        self.states = []

    def add_state(self, state):
        if isinstance(state, chess.Board):
            self.states.append(state)

    def run_simulation(self):
        start_time = datetime.datetime.utcnow()
        while datetime.datetime.utcnow() - start_time < datetime.timedelta(seconds=self.iteration_time):
            leaf_node = self.select_leaf(self.root)
            new_leaves = self.expand_leaf(leaf_node)
            results = [(leaf, self.simulate_game(leaf)) for leaf in new_leaves]
            self.backpropagate(results)

    def select_leaf(self, node):
        while node.children:
            node = max(node.children, key=ucb1_formula)
        return node

    def expand_leaf(self, node):
        if node.board_position.is_game_over(claim_draw=True):
            return []
        
        new_leaves = []
        for move in node.board_position.legal_moves:
            new_board = node.board_position.copy()
            new_board.push(move)
            child_node = TreeNode(not node.color, parent=node, board_position=new_board)
            child_node.move = move
            node.add_child(child_node)
            new_leaves.append(child_node)
        return new_leaves

    def interpret_result(self, result):
        if result == '1-0':
            return True
        if result == '0-1':
            return False
        return DRAW

    def simulate_game(self, node):
        board = node.board_position.copy()
        while not board.is_game_over(claim_draw=True):
            board.push(choice(list(board.legal_moves)))
        return self.interpret_result(board.result(claim_draw=True))

    def backpropagate(self, results):
        for node, result in results:
            while node:
                if result == DRAW:
                    node.wins += 0.5
                elif node.color == result:
                    node.wins += 1
                node.simulations += 1
                node = node.parent

    def get_best_move(self):
        if not self.root.children:
            return None
        best_child = max(self.root.children, key=lambda child: child.wins / child.simulations)
        return best_child.move

    def visualize_tree(self):
        with open("/Users/evanmdoyle/Programming/ChessAI/MCTS.dot", 'w') as f:
            f.write("digraph G { \n")
            self._write_subtree(self.root, f)
            f.write("}\n")

    def _write_subtree(self, node, file_handle):
        file_handle.write(f'{id(node)} [label="sims: {node.simulations}, wins: {node.wins}, color: {node.color}"];\n')
        for child in node.children:
            file_handle.write(f'{id(node)} -> {id(child)};\n')
            self._write_subtree(child, file_handle)


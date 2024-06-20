import sys
sys.path.append(sys.path[0] + "/..")

import chess
import math
import numpy as np
import argparse
import datetime
import operator
import random
import copy
import pychess_utils as util

from random import choice
from chess import pgn, uci
from collections import defaultdict
from rpc_client import PredictClient

DRAW = 'draw'
CPUCT = 1.5

prediction_cache = {}
value_cache = {}

ADDRESS = util.get_address()
PORT = util.get_port()

class Edge:
    def __init__(self, node, move, prob, simulations=0, total_action_value=0, action_value=0):
        self.node = node
        self.move = move if isinstance(move, chess.Move) else None
        self.prob = max(prob, 0)
        self.total_action_value = total_action_value
        self.action_value = action_value
        self.simulations = max(simulations, 0)

    def get_siblings(self):
        return [y for x, y in self.node.children if y != self]

    def total_sims_at_depth(self):
        return sum(sibling.simulations for sibling in self.get_siblings()) + self.simulations

    def get_confidence(self):
        term1 = CPUCT * self.prob
        term2 = math.sqrt(self.total_sims_at_depth()) / (1 + self.simulations)
        return term1 * term2

class Node:
    def __init__(self, color, parent=None, position=None):
        self.position = position if isinstance(position, chess.Board) else None
        self.color = color
        self.parent = parent
        self.children = []

class MCTS:
    ITERATIONS_PER_BUILD = 100
    ITER_TIME = 5

    def __init__(self, startpos=chess.Board(), iterations=None, iter_time=None, prev_mcts=None, temperature=True, version=0, startcolor=True):
        self.version = version if version else util.latest_version()
        self.client = PredictClient(ADDRESS, PORT, 'ACZ', int(self.version))
        self.startpos = startpos if isinstance(startpos, chess.Board) else chess.Board()
        self.root = prev_mcts.child_matching(self.startpos) if prev_mcts else Node(startcolor, position=self.startpos)
        self.iterations = max(iterations, self.ITERATIONS_PER_BUILD) if iterations else self.ITERATIONS_PER_BUILD
        self.iter_time = max(iter_time, self.ITER_TIME) if iter_time else self.ITER_TIME
        self.temperature = temperature

    def child_matching(self, position):
        for child, edge in self.root.children:
            if child.position == position:
                return child
        return None

    def max_action_val_child(self, root):
        max_child, max_edge, max_val = None, None, -float("inf")
        for child, edge in root.children:
            val = edge.action_value + edge.get_confidence()
            if val >= max_val:
                max_child, max_edge, max_val = child, edge, val
        return max_child, max_edge

    def most_visited_child(self, root):
        max_visits, choices = 0, []
        for child, edge in root.children:
            if edge.simulations >= max_visits:
                choices.append(edge)
        return random.choice(choices)

    def total_child_visits(self, root):
        return sum(edge.simulations for _, edge in root.children)

    def search(self):
        leaf = self.select_leaf(self.root)
        self.expand_tree(leaf)
        position_fen = leaf.position.fen()
        if position_fen not in value_cache:
            try:
                value_cache[position_fen] = self.client.predict(util.expand_position(leaf.position))[0]
            except:
                value_cache[position_fen] = self.client.predict(util.expand_position(leaf.position))[0]
        self.backpropagate(leaf, value_cache[position_fen])

    def build(self, timed=False):
        if timed:
            start_time = datetime.datetime.utcnow()
            while datetime.datetime.utcnow() - start_time < datetime.timedelta(seconds=self.iter_time):
                self.search()
        else:
            for _ in range(self.iterations):
                self.search()

    def select_leaf(self, root):
        while root.children:
            root = self.max_action_val_child(root)[0]
        return root

    def expand_tree(self, leaf):
        if not leaf.position:
            print("MCTS tried to expand with empty position.")
            return

        board = leaf.position
        if board.fen() not in prediction_cache:
            try:
                prediction_cache[board.fen()] = self.client.predict(util.expand_position(board), 'policy')
            except:
                prediction_cache[board.fen()] = self.client.predict(util.expand_position(board), 'policy')

        new_leaves = []
        for move in board.legal_moves:
            new_board = board.copy()
            pred_index = util.get_prediction_index(move)
            new_edge = Edge(leaf, move, util.logit_to_prob(prediction_cache[board.fen()][pred_index]))
            new_board.push(move)
            new_node = Node(not leaf.color, parent=leaf, position=new_board)
            leaf.children.append((new_node, new_edge))
            new_leaves.append(new_node)
        return new_leaves

    def backpropagate(self, leaf, value):
        while leaf:
            path_edge = self.max_action_val_child(leaf)[1]
            path_edge.simulations += 1
            path_edge.total_action_value += value
            path_edge.action_value = path_edge.total_action_value / path_edge.simulations
            leaf = leaf.parent

    def get_policy_string(self):
        total_visits = self.total_child_visits(self.root)
        policy = [
            f"({util.get_prediction_index(edge.move)}:{edge.simulations / total_visits})"
            for _, edge in self.root.children
        ]
        return '#'.join(policy)

    def best_move(self):
        if self.temperature:
            choices = []
            for _, edge in self.root.children:
                choices.extend([edge.move] * edge.simulations)
            return random.choice(choices)
        return self.most_visited_child(self.root).move

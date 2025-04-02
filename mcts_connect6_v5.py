#!/usr/bin/env python3
import sys
import numpy as np
import random
import copy
import math
import itertools
from tqdm import tqdm

class Eval:
    def __init__(self, board):
        self.board = board
        self.size = board.shape[0]
        self.directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

    def evaluation(self):
        total_reward = 0

        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] != 0:
                    total_reward += self.evaluate_position(r, c, self.board[r, c])
        
        return total_reward
        
    def evaluate_position(self, r, c, player):
        reward = 0

        for dr, dc in self.directions:
            prev_r, prev_c = r - dr, c - dc
            if 0 <= prev_r < self.size and 0 <= prev_c < self.size and self.board[prev_r, prev_c] == player:
                continue
            count = 0
            rr, cc = r, c
            while 0 <= rr < self.size and 0 <= cc < self.size and self.board[rr, cc] == player:
                count += 1
                rr += dr
                cc += dc
            
            if 0 <= rr < self.size and 0 <= cc < self.size:
                next_pos = self.board[rr, cc]
            else:
                next_pos = None

            if 0 <= prev_r < self.size and 0 <= prev_c < self.size:
                prev_pos = self.board[prev_r, prev_c]
            else:
                prev_pos = None

            if count == 2:
                reward += self.connect_two(player, prev_pos, next_pos)
            elif count == 3:
                reward += self.connect_three(player, prev_pos, next_pos)
            elif count == 4:
                reward += self.connect_four(player, prev_pos, next_pos)
            elif count == 5:
                reward += self.connect_five(player, prev_pos, next_pos)
            elif count >= 6:
                if player == 1:
                    reward += 1
                else:
                    reward -= 1

        return reward

    def connect_two(self, player, prev_pos, next_pos):
        if prev_pos == 0 and next_pos == 0:
            reward = 0.005
        elif prev_pos != 0 and next_pos == 0:
            reward = 0.002
        else:
            reward = 0
        reward = reward if player == 1 else -reward
        return reward
    
    def connect_three(self, player, prev_pos, next_pos):
        if prev_pos == 0 and next_pos == 0:
            reward = 0.05
        elif prev_pos != 0 and next_pos == 0:
            reward = 0.01
        else:
            reward = 0.001
        reward = reward if player == 1 else -reward
        return reward

    def connect_four(self, player, prev_pos, next_pos):
        if prev_pos == 0 and next_pos == 0:
            reward = 0.15
        elif prev_pos != 0 and next_pos == 0:
            reward = 0.1
        else:
            reward = 0.001
        reward = reward if player == 1 else -reward
        return reward

    def connect_five(self, player, prev_pos, next_pos):
        if prev_pos == 0 and next_pos == 0:
            reward = 0.5
        elif prev_pos != 0 and next_pos == 0:
            reward = 0.3
        else:
            reward = 0.001
        reward = reward if player == 1 else -reward
        return reward
    
# UCT Node for MCTS
class UCTNode:
    def __init__(self, board, parent=None):
        self.size = board.shape[0]
        self.state = self.create_state(board)
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.total_reward = 0
        # self.empty_positions = [(r, c) for r in range(size) for c in range(size) if self.state[r, c] == 0]
        self.candidate_positions = self.get_candidate_positions(self.size)
        
        self.untried_actions = list(itertools.combinations(self.candidate_positions, 2))
        self.candidate_actions = self.get_candidate_actions(self.untried_actions)
    
    def create_state(self, board):
        state = {"size": self.size, "B":[], "W":[]}
        for r in range(self.size):
            for c in range(self.size):
                if board[r, c] == 1:
                    state["B"].append((r, c))
                elif board[r, c] == 2:
                    state["W"].append((r, c))
        
        return state

    def fully_expanded(self):
		# A node is fully expanded if no legal actions remain untried.
        return len(self.candidate_actions) == 0
    
    def get_candidate_positions(self, size, radius=1):
        # stones = np.argwhere(self.state != 0)  # Get all placed stones
        stones = self.state["B"] + self.state["W"]
        candidates = set()
        for r, c in stones:
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < size and 0 <= nc < size and (nr, nc) not in stones:
                        candidates.add((nr, nc))
        return list(candidates)
    
    def get_candidate_actions(self, untried_actions):
        candidate_actions = []
        for idx, (pos0, pos1) in enumerate(untried_actions):
            if abs(pos1[0]-pos0[0])+abs(pos1[1]-pos0[1]) <= 4:
                candidate_actions.append(untried_actions[idx])
        return candidate_actions

class UCTMCTS:
    def __init__(self, env, iterations=500, exploration_constant=1.41, rollout_depth=10):
        self.env = env
        self.iterations = iterations
        self.c = exploration_constant  # Balances exploration and exploitation
        self.rollout_depth = rollout_depth

    def create_env_from_state(self, state, turn):
        new_env = copy.deepcopy(self.env)
        board = self.state_to_board(state)
        new_env.board = np.copy(board)
        new_env.turn = turn
        return new_env
    
    def state_to_board(self, state):
        size = state["size"]
        board = np.zeros((size, size), dtype=int)
        for b in state["B"]:
            board[b[0], b[1]] = 1
        for w in state["W"]:
            board[w[0], w[1]] = 2
        return board

    def select_child(self, node):
        # TODO: Use the UCT formula: Q + c * sqrt(log(parent_visits)/child_visits) to select the child
        best_uct = -np.inf
        best_child = None
        
        for action, child in node.children.items():
            uct = child.total_reward / child.visits + self.c * np.sqrt(np.log(node.visits)/child.visits+1e-6)
            if uct > best_uct:
                best_uct = uct
                best_child = child

        return best_child

    def rollout(self, sim_env):
        # TODO: Perform a random rollout from the current state up to the specified depth.
        end_game = False
        for _ in range(self.rollout_depth):
        # while sim_env.check_win() == 0 and (not end_game):
            sim_env, end_game = self.random_move(sim_env)
            if end_game:
                break
        
        eval = Eval(sim_env.board)
        reward = eval.evaluation()

        return reward

    def backpropagate(self, node, reward):
        # TODO: Propagate the reward up the tree, updating visit counts and total rewards.
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    def run_simulation(self, root, turn):
        player = {1: "B", 2: "W"}
        node = root

        # TODO: Selection: Traverse the tree until reaching a non-fully expanded node.
        while node.fully_expanded():
            node = self.select_child(node)

        # TODO: Expansion: if the node has untried actions, expand one.
        # check the node has untried actions or the node is leaf
        if node.candidate_actions:
            sim_env = self.create_env_from_state(node.state, turn)
            
            action = random.choice(node.candidate_actions)
            node.candidate_actions.remove(action)
            
            move_str = f"{sim_env.index_to_label(action[0][1])}{action[0][0]+1},{sim_env.index_to_label(action[1][1])}{action[1][0]+1}"
            sim_env.play_move(player[turn], move_str)

            sim_env, _ = self.random_move(sim_env)
            new_state = np.copy(sim_env.board)

            child_node = UCTNode(new_state, parent=node)
            node.children[tuple(action)] = child_node
            node = child_node

        # Rollout: Simulate a random game from the expanded node.
        reward = self.rollout(sim_env)
        # Backpropagation: Update the tree with the rollout reward.
        self.backpropagate(node, reward)

    def random_move(self, sim_env):
        player = {1: "B", 2: "W"}
        
        state = np.copy(sim_env.board)
        candidate_positions = self.get_candidate_positions(state)
        
        # empty_positions = [(r, c) for r in range(sim_env.size) for c in range(sim_env.size) if state[r, c] == 0]
        if len(candidate_positions) < 2:
            return sim_env, True
        
        pos = random.sample(candidate_positions, 2)
        move_str = ",".join(f"{sim_env.index_to_label(c)}{r+1}" for r, c in pos) 
        sim_env.play_move(player[sim_env.turn], move_str)

        return sim_env ,False

    def get_candidate_positions(self, state, radius=1):
        size = state.shape[0]
        stones = np.argwhere(state != 0)  # Get all placed stones
        candidates = set()
        for r, c in stones:
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < size and 0 <= nc < size and state[nr, nc] == 0:
                        candidates.add((nr, nc))
        return list(candidates)

    def best_action_distribution(self, root):
        '''
        Computes the visit count distribution for each action at the root node.
        '''
        # total_visits = sum(child.visits for child in root.children.values())
        # distribution = np.zeros(len((total_visits)))
        best_visits = -1
        best_action = None
        # print(root.children.items())
        for action, child in root.children.items():
            # distribution[action] = child.visits / total_visits if total_visits > 0 else 0
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action
        return best_action


class Connect6Game:
    def __init__(self, size=19):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)  # 0: Empty, 1: Black, 2: White
        self.turn = 1  # 1: Black, 2: White
        self.game_over = False

    def reset_board(self):
        """Clears the board and resets the game."""
        self.board.fill(0)
        self.turn = 1
        self.game_over = False
        print("= ", flush=True)

    def set_board_size(self, size):
        """Sets the board size and resets the game."""
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.turn = 1
        self.game_over = False
        print("= ", flush=True)

    def check_win(self):
        """Checks if a player has won.
        Returns:
        0 - No winner yet
        1 - Black wins
        2 - White wins
        """
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] != 0:
                    current_color = self.board[r, c]
                    for dr, dc in directions:
                        prev_r, prev_c = r - dr, c - dc
                        if 0 <= prev_r < self.size and 0 <= prev_c < self.size and self.board[prev_r, prev_c] == current_color:
                            continue
                        count = 0
                        rr, cc = r, c
                        while 0 <= rr < self.size and 0 <= cc < self.size and self.board[rr, cc] == current_color:
                            count += 1
                            rr += dr
                            cc += dc
                        if count >= 6:
                            return current_color
        return 0

    def index_to_label(self, col):
        """Converts column index to letter (skipping 'I')."""
        return chr(ord('A') + col + (1 if col >= 8 else 0))  # Skips 'I'

    def label_to_index(self, col_char):
        """Converts letter to column index (accounting for missing 'I')."""
        col_char = col_char.upper()
        if col_char >= 'J':  # 'I' is skipped
            return ord(col_char) - ord('A') - 1
        else:
            return ord(col_char) - ord('A')

    def play_move(self, color, move):
        """Places stones and checks the game status."""
        if self.game_over:
            print("? Game over")
            return

        stones = move.split(',')
        positions = []

        for stone in stones:
            stone = stone.strip()
            if len(stone) < 2:
                print("? Invalid format")
                return
            col_char = stone[0].upper()
            if not col_char.isalpha():
                print("? Invalid format")
                return
            col = self.label_to_index(col_char)
            try:
                row = int(stone[1:]) - 1
            except ValueError:
                print("? Invalid format")
                return
            if not (0 <= row < self.size and 0 <= col < self.size):
                print("? Move out of board range")
                return
            if self.board[row, col] != 0:
                print("? Position already occupied")
                return "error"
            positions.append((row, col))

        for row, col in positions:
            self.board[row, col] = 1 if color.upper() == 'B' else 2

        self.turn = 3 - self.turn
        print('= ', end='', flush=True)

    def generate_move(self, color, uct_mcts, first_step):
        """Generates a random move for the computer."""
        if self.game_over:
            print("? Game over")
            return

        if first_step:
            empty_positions = [(r, c) for r in range(self.size) for c in range(self.size) if self.board[r, c] == 0]
            selected = random.sample(empty_positions, 1)
            move_str = ",".join(f"{self.index_to_label(c)}{r+1}" for r, c in selected)
        
        else:
            state = np.copy(self.board)
            root = UCTNode(state)
            uct_mcts.iterations = len(root.candidate_actions) * 4
        
            for _ in tqdm(range(uct_mcts.iterations)):
                uct_mcts.run_simulation(root, self.turn)

                best_action = uct_mcts.best_action_distribution(root)
                move_str = ",".join(f"{self.index_to_label(c)}{r+1}" for r, c in best_action)

        self.play_move(color, move_str)

        print(f"{move_str}\n\n", end='', flush=True)
        print(move_str, file=sys.stderr)

    def show_board(self):
        """Displays the board as text."""
        # print("= ")
        for row in range(self.size - 1, -1, -1):
            line = f"{row+1:2} " + " ".join("X" if self.board[row, col] == 1 else "O" if self.board[row, col] == 2 else "." for col in range(self.size))
            print(line)
        col_labels = "   " + " ".join(self.index_to_label(i) for i in range(self.size))
        print(col_labels)
        print(flush=True)

    def list_commands(self):
        """Lists all available commands."""
        print("= ", flush=True)  

    def process_command(self, command,  uct_mcts, first_step):
        """Parses and executes GTP commands."""
        command = command.strip()
        if command == "get_conf_str env_board_size:":
            print(flush=True)
            return "env_board_size=19"

        if not command:
            return
        
        parts = command.split()
        cmd = parts[0].lower()

        if cmd == "boardsize":
            try:
                size = int(parts[1])
                self.set_board_size(size)
            except ValueError:
                print("? Invalid board size")
        elif cmd == "clear_board":
            self.reset_board()
        elif cmd == "play":
            if len(parts) < 3:
                print("? Invalid play command format")
            else:
                self.play_move(parts[1], parts[2])
                print('', flush=True)
        elif cmd == "genmove":
            if len(parts) < 2:
                print("? Invalid genmove command format")
            else:
                self.generate_move(parts[1], uct_mcts, first_step)
        elif cmd == "showboard":
            self.show_board()
        elif cmd == "list_commands":
            self.list_commands()
        elif cmd == "quit":
            print("= ", flush=True)
            sys.exit(0)
        else:
            print("? Unsupported command")

    def run(self, uct_mcts):
        """Main loop that reads GTP commands from standard input."""
        first_step = True
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                self.process_command(line, uct_mcts, first_step)
                first_step = False
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"? Error: {str(e)}")

if __name__ == "__main__":
    game = Connect6Game()
    uct_mcts = UCTMCTS(game, iterations=100, exploration_constant=1.41, rollout_depth=4)
    game.run(uct_mcts)

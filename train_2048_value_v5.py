import copy
import random
import math
import numpy as np
from collections import defaultdict
import pickle
from student_agent import Game2048Env
import time

import tempfile
import os
import pickle
import shutil

def simulate_row_move(row, size):
    """Simulate a left move for a single row"""
    # Compress: move non-zero numbers to the left
    new_row = row[row != 0]
    
    new_row_padded = np.zeros(size, dtype=new_row.dtype)
    new_row_padded[:len(new_row)] = new_row
    
    # Merge: merge adjacent equal numbers (do not update score)
    for i in range(len(new_row_padded) - 1):
        if new_row_padded[i] == new_row_padded[i + 1] and new_row_padded[i] != 0:
            new_row_padded[i] *= 2
            new_row_padded[i + 1] = 0
    
    # Compress again
    new_row_padded = new_row_padded[new_row_padded != 0]
    
    new_row_padded_final = np.zeros(size, dtype=new_row_padded.dtype)
    new_row_padded_final[:len(new_row_padded)] = new_row_padded
    
    return new_row_padded_final

# def simulate_row_move(row, size):
#     """Simulate a left move for a single row"""
#     # Compress: move non-zero numbers to the left
#     new_row = row[row != 0]
#     new_row = np.pad(new_row, (0, size - len(new_row)), mode='constant')
#     # Merge: merge adjacent equal numbers (do not update score)
#     for i in range(len(new_row) - 1):
#         if new_row[i] == new_row[i + 1] and new_row[i] != 0:
#             new_row[i] *= 2
#             new_row[i + 1] = 0
#     # Compress again
#     new_row = new_row[new_row != 0]
#     new_row = np.pad(new_row, (0, size - len(new_row)), mode='constant')
#     return new_row

def create_env_from_state(env, state):
    """
    Creates a deep copy of the environment with a given board state and score.
    """
    new_env = copy.deepcopy(env)
    new_env.board = state.copy()
    new_env.score = env.score
    return new_env

class Action:
    def __init__(self, board):
        self.board = board.copy()
        self.score = 0
        self.size = 4

    def compress(self, row):
        """Compress the row: move non-zero values to the left"""
        new_row = row[row != 0]  # Remove zeros
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')  # Pad with zeros on the right
        return new_row

    def merge(self, row):
        """Merge adjacent equal numbers in the row"""
        for i in range(len(row) - 1):
            if row[i] == row[i + 1] and row[i] != 0:
                row[i] *= 2
                row[i + 1] = 0
                self.score += row[i]
        return row

    def move_left(self):
        """Move the board left"""
        # moved = False
        for i in range(self.size):
            # original_row = self.board[i].copy()
            new_row = self.compress(self.board[i])
            new_row = self.merge(new_row)
            new_row = self.compress(new_row)
            self.board[i] = new_row
            # if not np.array_equal(original_row, self.board[i]):
            #     moved = True
        # return moved

    def move_right(self):
        """Move the board right"""
        # moved = False
        for i in range(self.size):
            # original_row = self.board[i].copy()
            # Reverse the row, compress, merge, compress, then reverse back
            reversed_row = self.board[i][::-1]
            reversed_row = self.compress(reversed_row)
            reversed_row = self.merge(reversed_row)
            reversed_row = self.compress(reversed_row)
            self.board[i] = reversed_row[::-1]
            # if not np.array_equal(original_row, self.board[i]):
            #     moved = True
        # return moved

    def move_up(self):
        """Move the board up"""
        # moved = False
        for j in range(self.size):
            # original_col = self.board[:, j].copy()
            col = self.compress(self.board[:, j])
            col = self.merge(col)
            col = self.compress(col)
            self.board[:, j] = col
        #     if not np.array_equal(original_col, self.board[:, j]):
        #         moved = True
        # return moved

    def move_down(self):
        """Move the board down"""
        # moved = False
        for j in range(self.size):
            # original_col = self.board[:, j].copy()
            # Reverse the column, compress, merge, compress, then reverse back
            reversed_col = self.board[:, j][::-1]
            reversed_col = self.compress(reversed_col)
            reversed_col = self.merge(reversed_col)
            reversed_col = self.compress(reversed_col)
            self.board[:, j] = reversed_col[::-1]
        #     if not np.array_equal(original_col, self.board[:, j]):
        #         moved = True
        # return moved
    
    def fake_step(self, action):
        """Execute one action"""

        if action == 0:
            self.move_up()
        elif action == 1:
            self.move_down()
        elif action == 2:
            self.move_left()
        elif action == 3:
            self.move_right()

        return self.board, self.score
    
    def is_move_legal(self, action):
        """Check if the specified move is legal (i.e., changes the board)"""
        # Create a copy of the current board state
        temp_board = self.board.copy()

        if action == 0:  # Move up
            for j in range(self.size):
                col = temp_board[:, j]
                new_col = simulate_row_move(col, self.size)
                temp_board[:, j] = new_col
        elif action == 1:  # Move down
            for j in range(self.size):
                # Reverse the column, simulate, then reverse back
                col = temp_board[:, j][::-1]
                new_col = simulate_row_move(col, self.size)
                temp_board[:, j] = new_col[::-1]
        elif action == 2:  # Move left
            for i in range(self.size):
                row = temp_board[i]
                temp_board[i] = simulate_row_move(row, self.size)
        elif action == 3:  # Move right
            for i in range(self.size):
                row = temp_board[i][::-1]
                new_row = simulate_row_move(row, self.size)
                temp_board[i] = new_row[::-1]
        else:
            raise ValueError("Invalid action")

        # If the simulated board is different from the current board, the move is legal
        return not np.array_equal(self.board, temp_board)

# -------------------------------
# TODO: Define transformation functions (rotation and reflection), i.e., rot90, rot180, ..., etc.
# -------------------------------
def rotation(pattern, angle):
    new_pattern = []
    if angle == 90:
        for element in pattern:
            new_pattern.append((element[1], 3 - element[0]))
    elif angle == 180:
        for element in pattern:
            new_pattern.append((3 - element[0], 3 - element[1]))
    elif angle == 270:
        for element in pattern:
            new_pattern.append((3 - element[1], element[0]))
    
    return new_pattern

def reflection(pattern):
    new_pattern = []
    for element in pattern:
        new_pattern.append((element[0], 3 - element[1]))
    return new_pattern



def tile_to_index(tile):
    """
    Converts tile values to an index for the lookup table.
    """
    if tile == 0:
        return 0
    else:
        # 2^n -> n, like 2->1, 4->2, 8->3, ...
        return int(np.log2(tile))


def tuple_id(values, POWER):
    # values = values[::-1]
    length = len(values)

    reversed_values = np.empty(length, dtype=np.int64) 
    for i in range(length):
        reversed_values[i] = values[length - 1 - i]
    k = 1
    n = 0
    for v in reversed_values:
        n += v * k
        k *= POWER
    return n

class NTupleApproximator:
    def __init__(self, board_size, patterns):
        """
        Initializes the N-Tuple approximator.
        Hint: you can adjust these if you want
        """
        self.board_size = board_size
        self.POWER = 15
        self.patterns = patterns
        # Create a weight dictionary for each pattern (shared within a pattern group)
        # self.weights = [defaultdict(float) for _ in patterns]
        
        # Generate symmetrical transformations for each pattern
        self.symmetry_patterns = []

        for pattern in self.patterns:
            symmetric_pattern = []
            syms = self.generate_symmetries(pattern)
            for syms_ in syms:
                if syms_ not in symmetric_pattern:
                    symmetric_pattern.append(syms_)
        
            self.symmetry_patterns.extend(symmetric_pattern)

        self.weights = []
        for pattern in self.symmetry_patterns:
            self.weights.append(np.zeros((self.POWER + 1) ** len(pattern)))

    def generate_symmetries(self, pattern):
        # TODO: Generate 8 symmetrical transformations of the given pattern.
        syms = [sorted(pattern)]
        syms.append(sorted(rotation(pattern, 90)))
        syms.append(sorted(rotation(pattern, 180)))
        syms.append(sorted(rotation(pattern, 270)))
        
        # horizontal flip for each rotations
        syms.append(sorted(reflection(pattern)))
        syms.append(sorted(reflection(rotation(pattern, 90))))
        syms.append(sorted(reflection(rotation(pattern, 180))))
        syms.append(sorted(reflection(rotation(pattern, 270))))

        return syms

    def value(self, board, delta=None):
        vals = []
        for i, (pattern, weight) in enumerate(zip(self.symmetry_patterns, self.weights)):
            tiles = [board[i][j] for i, j in pattern]
            index = [tile_to_index(int(t)) for t in tiles]
            tpid = tuple_id(index, self.POWER)
            if delta is not None:
                weight[tpid] += delta
            v = weight[tpid]
            vals.append(v)
        return np.mean(vals)
    
    def best_action(self, state, legal_moves):
        values = []
        for a in legal_moves:
            # create a new env for fake step to look the value of each next_state
            sim_env = Action(state)
            state_after, reward = sim_env.fake_step(a)
            v = reward + self.value(state_after)
            values.append(v)

        action = legal_moves[np.argmax(values)]

        return action

def td_learning(env, approximator, num_episodes=50000, alpha=0.1, gamma=1, epsilon=0.1):
    """
    Trains the 2048 agent using TD-Learning.

    Args:
        env: The 2048 game environment.
        approximator: NTupleApproximator instance.
        num_episodes: Number of training episodes.
        alpha: Learning rate.
        gamma: Discount factor.
        epsilon: Epsilon-greedy exploration rate.
    """
    final_scores = []
    success_flags = []
    global_max_tile = 0

    for episode in range(num_episodes):
        state = env.reset()
        trajectory = []  # Store trajectory data if needed
        previous_score = 0
        done = False
        max_tile = np.max(state) # remember the max value one the board

        start_time = time.time()

        while not done:
            legal_moves = [a for a in range(4) if env.is_move_legal(a)]
            if not legal_moves: # means the game is over
                break
            # TODO: action selection
            # Note: TD learning works fine on 2048 without explicit exploration, but you can still try some exploration methods.
            action = approximator.best_action(state, legal_moves)

            next_state, new_score, done, _ = env.step(action)
            # print(new_score-previous_score)

            sim_env = Action(state)
            after_state, after_score = sim_env.fake_step(action)
            # print(after_score, "\n")
            # incremental_reward = new_score - previous_score # the reward get in this step

            # previous_score = new_score
            max_tile = max(max_tile, np.max(next_state))

            # TODO: Store trajectory or just update depending on the implementation
            trajectory.append([state.copy(), action, after_score, after_state.copy(), next_state.copy()])

            state = copy.deepcopy(next_state)

        # TODO: If you are storing the trajectory, consider updating it now depending on your implementation.

        for exp in trajectory:
            s, a, r, s_after, s_next = exp
            sim_env = Action(s_next)

            legal_moves = [a for a in range(4) if sim_env.is_move_legal(a)]
            if not legal_moves: # means the game is over
                v_after_next = 0
                r_next = 0
            else:
                a_next = approximator.best_action(s_next, legal_moves)
                s_after_next, r_next = sim_env.fake_step(a_next)
                v_after_next = approximator.value(s_after_next)
            
            # print(r_next, v_after_next, approximator.value(s_after))
            # TD error: r + gamma * v(s') - v(s)
            delta = r_next + gamma * v_after_next - approximator.value(s_after)
            approximator.value(s_after, delta * alpha)

        end_time = time.time()
        print("one episode time: ", end_time-start_time)

        final_scores.append(env.score)
        success_flags.append(1 if max_tile >= 2048 else 0)
        global_max_tile = max(max_tile, global_max_tile)

        if (episode + 1) % 100 == 0:
            avg_score = np.mean(final_scores[-100:])
            success_rate = np.sum(success_flags[-100:]) / 100
            print(f"Episode {episode+1}/{num_episodes} | Avg Score: {avg_score:.2f} | Success Rate: {success_rate:.2f} | Max Tile: {global_max_tile}")
            global_max_tile = 0
            with open('value_approximator_4.pkl', 'wb') as file:
                pickle.dump(approximator, file)

    return final_scores


class TD_MCTS_Node:
    def __init__(self, state, score, state_type, update_reward, parent=None, action=None):
        """
        state: current board state (numpy array)
        score: cumulative score at this node
        parent: parent node (None for root)
        action: action taken from parent to reach this node
        """
        self.state = state
        self.score = score
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
        self.update_reward = update_reward

        self.state_type = state_type
        # List of untried actions based on the current state's legal moves
        if self.state_type == "next":
            self.untried_actions = [a for a in range(4) if env.is_move_legal(a)]
        elif self.state_type == "after":
            self.untried_positions = {2: list(zip(*np.where(self.state == 0))), 4:list(zip(*np.where(self.state == 0)))}

    def fully_expanded(self):
        # A node is fully expanded if no legal actions remain untried.
        if self.state_type == "next":
            return len(self.untried_actions) == 0
        elif self.state_type == "after":
            return (len(self.untried_positions[2])+len(self.untried_positions[4])) == 0


# TD-MCTS class utilizing a trained approximator for leaf evaluation
class TD_MCTS:
    def __init__(self, env, approximator, iterations=500, exploration_constant=1.41, rollout_depth=10, gamma=0.99):
        self.env = env
        self.approximator = approximator
        self.iterations = iterations
        self.c = exploration_constant
        self.rollout_depth = rollout_depth
        self.gamma = gamma

    def create_env_from_state(self, state, score):
        # Create a deep copy of the environment with the given state and score.
        new_env = copy.deepcopy(self.env)
        new_env.board = state.copy()
        new_env.score = score
        return new_env

    def select_child(self, node):
        # TODO: Use the UCT formula: Q + c * sqrt(log(parent.visits)/child.visits) to select the best child.
        best_uct = -np.inf
        best_child = None
        
        for action, child in node.children.items():
            # print("Q value: ",child.total_reward / child.visits, "explore term: ", self.c * np.sqrt(np.log(node.visits)/(child.visits+1e-6)))
            uct = child.total_reward / child.visits + self.c * np.sqrt(np.log(node.visits)/(child.visits))
            if uct > best_uct:
                best_uct = uct
                best_child = child

        return best_child

    def rollout(self, sim_env, after_state, state, state_type, depth):
        # TODO: Perform a random rollout until reaching the maximum depth or a terminal state.
        # TODO: Use the approximator to evaluate the final state.
        done = False
        rewards = 0.0

        if state_type == "after":
            empty_cells = list(zip(*np.where(state == 0)))
            if empty_cells:
                x, y = random.choice(empty_cells)
                num = 2 if random.random() < 0.9 else 4
                sim_env.add_tile(x, y, num)
            else:
                return rewards + self.approximator.value(after_state)
            
        legal_moves = [a for a in range(4) if sim_env.is_move_legal(a)]
        if not legal_moves:
            return rewards + self.approximator.value(after_state)

        for _ in range(depth):
            if not done:
                legal_moves = [a for a in range(4) if sim_env.is_move_legal(a)]
                if not legal_moves:
                    break
                action = random.choice(legal_moves)

                after_state, reward = sim_env.fake_step(action)
                rewards += reward

                empty_cells = list(zip(*np.where(after_state == 0)))
                if empty_cells:
                    x, y = random.choice(empty_cells)
                    num = 2 if random.random() < 0.9 else 4
                    sim_env.add_tile(x, y, num)
                else:
                    break
            else:
                break

        return rewards + self.approximator.value(after_state)

    def backpropagate(self, node, rollout_reward):
        # TODO: Propagate the obtained reward back up the tree.
        update_rewards = rollout_reward

        while node is not None:
            node.visits += 1
            node.total_reward += update_rewards
            
            update_rewards += node.update_reward
            node = node.parent

    def run_simulation(self, root):
        node = root
        score = root.score
        # sim_env = self.create_env_from_state(node.state, node.score)

        # TODO: Selection: Traverse the tree until reaching an unexpanded node.
        while node.fully_expanded():
            new_node = self.select_child(node)
            if new_node == None:
                return
            node = new_node

        # TODO: Expansion: If the node is not terminal, expand an untried action.
        if node.state_type == "next":
            if node.untried_actions:        
                action = random.choice(node.untried_actions)
                node.untried_actions.remove(action)

                sim_env = Action(node.state)
                after_state, reward = sim_env.fake_step(action)
                score = node.score + reward

                ### Update the search tree
                child_node = TD_MCTS_Node(copy.copy(after_state), score, "after", reward, parent=node, action=action)
                node.children[action] = child_node
                node = child_node

        elif node.state_type == "after":
            num = 2 if random.random() < 0.5 else 4
            if node.untried_positions[num]:
                x, y = random.choice(node.untried_positions[num])
            else:
                if num == 2:
                    num = 4
                else:
                    num = 2
                if node.untried_positions[num]:
                    x, y = random.choice(node.untried_positions[num])
                    
            node.untried_positions[num].remove((x, y))

            sim_env = Action(node.state)
            sim_env.add_tile(x, y, num)
            next_state = sim_env.board

            ### Update the search tree
            child_node = TD_MCTS_Node(copy.copy(next_state), node.score, "next", 0, parent=node, action=None)
            node.children[(x, y)] = child_node
            node = child_node

        # Rollout: Simulate a random game from the expanded node.
        if node.state_type == "next":
            after_state = copy.copy(node.parent.state)
        else:
            after_state = copy.copy(node.state)

        rollout_reward = self.rollout(sim_env, after_state, copy.copy(node.state), node.state_type, self.rollout_depth)
        # rollout_reward = reward + self.approximator.value(after_state)

        # Backpropagate the obtained reward.
        self.backpropagate(node, rollout_reward)

    def best_action_distribution(self, root):
        # Compute the normalized visit count distribution for each child of the root.
        total_visits = sum(child.visits for child in root.children.values())
        distribution = np.zeros(4)
        best_visits = -1
        best_action = None
        for action, child in root.children.items():
            distribution[action] = child.visits / total_visits if total_visits > 0 else 0
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action
        return best_action, distribution


# TODO: Define your own n-tuple patterns
patterns = [
    [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
    [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)],
    [(1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1)],
    [(0, 0), (0, 1), (1, 1), (1, 2), (1, 3), (2, 2)],
    [(0, 0), (0, 1), (0, 2), (1, 1), (2, 1), (2, 2)],
    [(0, 0), (0, 1), (1, 1), (2, 1), (3, 1), (3, 2)],
    [(0, 0), (0, 1), (1, 1), (2, 0), (2, 1), (3, 1)],
    [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 2)]
]

# approximator = NTupleApproximator(board_size=4, patterns=patterns)
with open('value_approximator_4.pkl', 'rb') as file:
    approximator = pickle.load(file)

env = Game2048Env()

# Run TD-Learning training
# Note: To achieve significantly better performance, you will likely need to train for over 100,000 episodes.
# However, to quickly verify that your implementation is working correctly, you can start by running it for 1,000 episodes before scaling up.
final_scores = td_learning(env, approximator, num_episodes=200000, alpha=0.1, gamma=1, epsilon=0.1)

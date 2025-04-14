# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import copy
import random
import math
import os

COLOR_MAP = {
    0: "#cdc1b4", 2: "#eee4da", 4: "#ede0c8", 8: "#f2b179",
    16: "#f59563", 32: "#f67c5f", 64: "#f65e3b", 128: "#edcf72",
    256: "#edcc61", 512: "#edc850", 1024: "#edc53f", 2048: "#edc22e",
    4096: "#3c3a32", 8192: "#3c3a32", 16384: "#3c3a32", 32768: "#3c3a32"
}
TEXT_COLOR = {
    2: "#776e65", 4: "#776e65", 8: "#f9f6f2", 16: "#f9f6f2",
    32: "#f9f6f2", 64: "#f9f6f2", 128: "#f9f6f2", 256: "#f9f6f2",
    512: "#f9f6f2", 1024: "#f9f6f2", 2048: "#f9f6f2", 4096: "#f9f6f2"
}

class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()

        self.size = 4  # 4x4 2048 board
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0

        # Action space: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)
        self.actions = ["up", "down", "left", "right"]

        self.last_move_valid = True  # Record if the last move was valid

        self.reset()

    def reset(self):
        """Reset the environment"""
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board

    def add_random_tile(self):
        """Add a random tile (2 or 4) to an empty cell"""
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[x, y] = 2 if random.random() < 0.9 else 4

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
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            new_row = self.compress(self.board[i])
            new_row = self.merge(new_row)
            new_row = self.compress(new_row)
            self.board[i] = new_row
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_right(self):
        """Move the board right"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            # Reverse the row, compress, merge, compress, then reverse back
            reversed_row = self.board[i][::-1]
            reversed_row = self.compress(reversed_row)
            reversed_row = self.merge(reversed_row)
            reversed_row = self.compress(reversed_row)
            self.board[i] = reversed_row[::-1]
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_up(self):
        """Move the board up"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            col = self.compress(self.board[:, j])
            col = self.merge(col)
            col = self.compress(col)
            self.board[:, j] = col
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def move_down(self):
        """Move the board down"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            # Reverse the column, compress, merge, compress, then reverse back
            reversed_col = self.board[:, j][::-1]
            reversed_col = self.compress(reversed_col)
            reversed_col = self.merge(reversed_col)
            reversed_col = self.compress(reversed_col)
            self.board[:, j] = reversed_col[::-1]
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def is_game_over(self):
        """Check if there are no legal moves left"""
        # If there is any empty cell, the game is not over
        if np.any(self.board == 0):
            return False

        # Check horizontally
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j+1]:
                    return False

        # Check vertically
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.board[i, j] == self.board[i+1, j]:
                    return False

        return True

    def step(self, action):
        """Execute one action"""
        assert self.action_space.contains(action), "Invalid action"

        if action == 0:
            moved = self.move_up()
        elif action == 1:
            moved = self.move_down()
        elif action == 2:
            moved = self.move_left()
        elif action == 3:
            moved = self.move_right()
        else:
            moved = False

        self.last_move_valid = moved  # Record if the move was valid

        if moved:
            self.add_random_tile()

        done = self.is_game_over()

        return self.board, self.score, done, {}

    def render(self, mode="human", action=None):
        """
        Render the current board using Matplotlib.
        This function does not check if the action is valid and only displays the current board state.
        """
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)

        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i, j]
                color = COLOR_MAP.get(value, "#3c3a32")  # Default dark color
                text_color = TEXT_COLOR.get(value, "white")
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor="black")
                ax.add_patch(rect)

                if value != 0:
                    ax.text(j, i, str(value), ha='center', va='center',
                            fontsize=16, fontweight='bold', color=text_color)
        title = f"score: {self.score}"
        if action is not None:
            title += f" | action: {self.actions[action]}"
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.show()

    def simulate_row_move(self, row):
        """Simulate a left move for a single row"""
        # Compress: move non-zero numbers to the left
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        # Merge: merge adjacent equal numbers (do not update score)
        for i in range(len(new_row) - 1):
            if new_row[i] == new_row[i + 1] and new_row[i] != 0:
                new_row[i] *= 2
                new_row[i + 1] = 0
        # Compress again
        new_row = new_row[new_row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row

    def is_move_legal(self, action):
        """Check if the specified move is legal (i.e., changes the board)"""
        # Create a copy of the current board state
        temp_board = self.board.copy()

        if action == 0:  # Move up
            for j in range(self.size):
                col = temp_board[:, j]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col
        elif action == 1:  # Move down
            for j in range(self.size):
                # Reverse the column, simulate, then reverse back
                col = temp_board[:, j][::-1]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col[::-1]
        elif action == 2:  # Move left
            for i in range(self.size):
                row = temp_board[i]
                temp_board[i] = self.simulate_row_move(row)
        elif action == 3:  # Move right
            for i in range(self.size):
                row = temp_board[i][::-1]
                new_row = self.simulate_row_move(row)
                temp_board[i] = new_row[::-1]
        else:
            raise ValueError("Invalid action")

        # If the simulated board is different from the current board, the move is legal
        return not np.array_equal(self.board, temp_board)

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
    
    def simulate_row_move(self, row):
        """Simulate a left move for a single row"""
        # Compress: move non-zero numbers to the left
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        # Merge: merge adjacent equal numbers (do not update score)
        for i in range(len(new_row) - 1):
            if new_row[i] == new_row[i + 1] and new_row[i] != 0:
                new_row[i] *= 2
                new_row[i + 1] = 0
        # Compress again
        new_row = new_row[new_row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row
    
    def is_move_legal(self, action):
        """Check if the specified move is legal (i.e., changes the board)"""
        # Create a copy of the current board state
        temp_board = self.board.copy()

        if action == 0:  # Move up
            for j in range(self.size):
                col = temp_board[:, j]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col
        elif action == 1:  # Move down
            for j in range(self.size):
                # Reverse the column, simulate, then reverse back
                col = temp_board[:, j][::-1]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col[::-1]
        elif action == 2:  # Move left
            for i in range(self.size):
                row = temp_board[i]
                temp_board[i] = self.simulate_row_move(row)
        elif action == 3:  # Move right
            for i in range(self.size):
                row = temp_board[i][::-1]
                new_row = self.simulate_row_move(row)
                temp_board[i] = new_row[::-1]
        else:
            raise ValueError("Invalid action")

        # If the simulated board is different from the current board, the move is legal
        return not np.array_equal(self.board, temp_board)
    
    def add_tile(self, x, y, num):
        self.board[x, y] = num

class NTupleApproximator:
    def __init__(self, board_size, patterns):
        """
        Initializes the N-Tuple approximator.
        Hint: you can adjust these if you want
        """
        self.board_size = board_size
        self.POWER = 15
        self.patterns = patterns
        
        # Generate symmetrical transformations for each pattern
        self.symmetry_patterns = []

        for pattern in self.patterns:
            symmetric_pattern = []
            syms = self.generate_symmetries(pattern)
            for syms_ in syms:
                if syms_ not in symmetric_pattern:
                    symmetric_pattern.append(syms_)
        
            self.symmetry_patterns.extend(symmetric_pattern)

        # Create a weight dictionary for each pattern (shared within a pattern group)
        # self.weights = [defaultdict(float) for _ in patterns]
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


    def tile_to_index(self, tile):
        """
        Converts tile values to an index for the lookup table.
        """
        if tile == 0:
            return 0
        else:
            # 2^n -> n, like 2->1, 4->2, 8->3, ...
            return int(math.log(tile, 2))

    def get_feature(self, board, coords):
        # TODO: Extract tile values from the board based on the given coordinates and convert them into a feature tuple.
        pass
    #     features = []
    #     # each pattern have N-tuples(coords)
    #     for coord in coords:
    #         tile = board[coord[0]][coord[1]]
    #         index = self.tile_to_index(tile) # convert to power expression
    #         features.append(index)
    #     return tuple(features)

    def tuple_id(self, values):
        values = values[::-1]
        k = 1
        n = 0
        for v in values:
            n += v * k
            k *= self.POWER
        return n

    def value(self, board):
        # TODO: Estimate the board value: sum the evaluations from all patterns.
        vals = []
        for i, (pattern, weight) in enumerate(zip(self.symmetry_patterns, self.weights)):
            tiles = [board[i][j] for i, j in pattern]
            index = [self.tile_to_index(t) for t in tiles]
            tpid = self.tuple_id(index)
            v = weight[tpid]
            vals.append(v)
        return np.mean(vals)

    def update(self, board, delta, alpha):
        # TODO: Update weights based on the TD error.
        for i, (pattern, weight) in enumerate(zip(self.symmetry_patterns, self.weights)):
            tiles = [board[i][j] for i, j in pattern]
            index = [self.tile_to_index(t) for t in tiles]
            tpid = self.tuple_id(index)

            weight[tpid] += alpha * delta
    
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


# Note: This MCTS implementation is almost identical to the previous one,
# except for the rollout phase, which now incorporates the approximator.

# Node for TD-MCTS using the TD-trained value approximator
class TD_MCTS_Node:
    def __init__(self, state, score, state_type, update_reward, parent=None):
        """
        state: current board state (numpy array)
        score: cumulative score at this node
        parent: parent node (None for root)
        action: action taken from parent to reach this node
        """
        self.state = state
        self.score = score
        self.parent = parent
        
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
        self.update_reward = update_reward

        self.state_type = state_type
        # List of untried actions based on the current state's legal moves
        if self.state_type == "next":
            self.untried_actions = [a for a in range(4) if self.is_move_legal(a)]

            self.random_tile = {'num': None, 'pos': None}

        
        elif self.state_type == "after":
            random_space_2 = self.select_random_tile(num=5)
            random_space_4 = self.select_random_tile(num=1)
            self.untried_positions = {2: list(random_space_2), 4: list(random_space_4)}

            self.action = None

        
    def fully_expanded(self):
        # A node is fully expanded if no legal actions remain untried.
        if self.state_type == "next":
            return len(self.untried_actions) == 0
        elif self.state_type == "after":
            return (len(self.untried_positions[2])+len(self.untried_positions[4])) == 0
    
    def select_random_tile(self, num=5):
        empty_space = [(i, j) for i, row in enumerate(self.state) for j, val in enumerate(row) if val == 0]
        if len(empty_space) >= num:
            return random.sample(empty_space, num)
        else:
            return empty_space
        
    def is_move_legal(self, action, size=4):
        """Check if the specified move is legal (i.e., changes the board)"""
        # Create a copy of the current board state
        temp_board = self.state.copy()

        if action == 0:  # Move up
            for j in range(size):
                col = temp_board[:, j]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col
        elif action == 1:  # Move down
            for j in range(size):
                # Reverse the column, simulate, then reverse back
                col = temp_board[:, j][::-1]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col[::-1]
        elif action == 2:  # Move left
            for i in range(size):
                row = temp_board[i]
                temp_board[i] = self.simulate_row_move(row)
        elif action == 3:  # Move right
            for i in range(size):
                row = temp_board[i][::-1]
                new_row = self.simulate_row_move(row)
                temp_board[i] = new_row[::-1]
        else:
            raise ValueError("Invalid action")

        # If the simulated board is different from the current board, the move is legal
        return not np.array_equal(self.state, temp_board)

    def simulate_row_move(self, row, size=4):
        """Simulate a left move for a single row"""
        # Compress: move non-zero numbers to the left
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, size - len(new_row)), mode='constant')
        # Merge: merge adjacent equal numbers (do not update score)
        for i in range(len(new_row) - 1):
            if new_row[i] == new_row[i + 1] and new_row[i] != 0:
                new_row[i] *= 2
                new_row[i + 1] = 0
        # Compress again
        new_row = new_row[new_row != 0]
        new_row = np.pad(new_row, (0, size - len(new_row)), mode='constant')
        return new_row


# TD-MCTS class utilizing a trained approximator for leaf evaluation
class TD_MCTS:
    def __init__(self, env, approximator, iterations=500, exploration_constant=1.41, rollout_depth=10, gamma=0.99):
        self.env = env
        self.approximator = approximator
        self.iterations = iterations
        self.c = exploration_constant
        self.rollout_depth = rollout_depth
        self.gamma = gamma

    def select_child(self, node):
        # TODO: Use the UCT formula: Q + c * sqrt(log(parent.visits)/child.visits) to select the best child.
        if node.state_type == "next":
            best_uct = -np.inf
            best_child = None
            
            for action, child in node.children.items():
                # print("Q value: ",child.total_reward / child.visits, "explore term: ", self.c * np.sqrt(np.log(node.visits)/(child.visits)))
                uct = child.total_reward / child.visits + self.c * np.sqrt(np.log(node.visits)/(child.visits))
                if uct > best_uct:
                    best_uct = uct
                    best_child = child
            # print("\n")
            return best_child
        else:
            select_child = None
            # num = random.choice([2, 4])
            num = 2 if random.random() < 0.9 else 4
            children = []
            for child in node.children.values():
                if child.random_tile['num'] == num:
                    children.append(child)
            if children:
                select_child = random.choice(children)

            return select_child


    def rollout(self, sim_env, after_state, state, state_type, score, depth):
        # TODO: Perform a random rollout until reaching the maximum depth or a terminal state.
        # TODO: Use the approximator to evaluate the final state.
        done = False
        rewards = 0.0

        V_norm = 30000
        # print(self.approximator.value(after_state), score)
        # print((self.approximator.value(after_state) + score)/V_norm)

        # print("Before " ,self.approximator.value(after_state)/V_norm)
        # print(after_state)

        if state_type == "after":
            empty_cells = list(zip(*np.where(state == 0)))
            if empty_cells:
                x, y = random.choice(empty_cells)
                num = 2 if random.random() < 0.9 else 4
                sim_env.add_tile(x, y, num)
            else:
                return (self.approximator.value(after_state) + score)/V_norm
            
        legal_moves = [a for a in range(4) if sim_env.is_move_legal(a)]
        if not legal_moves:
            return (self.approximator.value(after_state) + score)/V_norm 

        # real rollout
        for _ in range(depth):
            if not done:
                legal_moves = [a for a in range(4) if sim_env.is_move_legal(a)]
                if not legal_moves:
                    break
                # action = random.choice(legal_moves)

                values = []
                sim_sim_env = Action(state)
                for a in legal_moves:
                    a_s, r = sim_sim_env.fake_step(a)
                    values.append(self.approximator.value(a_s))
                action = legal_moves[np.argmax(values)]

                
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

        return (self.approximator.value(after_state) + rewards + score)/V_norm 

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
                child_node = TD_MCTS_Node(copy.copy(after_state), score, "after", reward, parent=node)
                child_node.action = action

                node.children[action] = child_node
                node = child_node

        elif node.state_type == "after":
            num = random.choice([2, 4])

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
            score = node.score

            ### Update the search tree
            child_node = TD_MCTS_Node(copy.copy(next_state), score, "next", 0, parent=node)
            child_node.random_tile['num'] = num
            child_node.random_tile['pos'] = (x, y)

            node.children[(x, y)] = child_node
            node = child_node

        
        # Rollout: Simulate a random game from the expanded node.
        if node.state_type == "next":
            after_state = copy.copy(node.parent.state)
        else:
            after_state = copy.copy(node.state)

        rollout_reward = self.rollout(sim_env, after_state, copy.copy(node.state), node.state_type, node.score, self.rollout_depth)

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
    
        # best_score = -1
        # best_act = None
        # for action, child in root.children.items():
        #     score = child.total_reward / child.visits if child.visits > 0 else 0
        #     if score > best_score:
        #         best_score = score
        #         best_act = action

        # if best_act == None:
        #     best_act = random.choice([0,1,2,3])
        
        # return best_act, None

# patterns = [
#     [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
#     [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)],
#     [(1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1)],
#     [(0, 0), (0, 1), (1, 1), (1, 2), (1, 3), (2, 2)],
#     [(0, 0), (0, 1), (0, 2), (1, 1), (2, 1), (2, 2)],
#     [(0, 0), (0, 1), (1, 1), (2, 1), (3, 1), (3, 2)],
#     [(0, 0), (0, 1), (1, 1), (2, 0), (2, 1), (3, 1)],
#     [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 2)]
# ]
# patterns = [
#     [(0, 0), (0, 1), (0, 2), (0, 3)],
#     [(1, 0), (1, 1), (1, 2), (1, 3)],
#     [(0, 0), (0, 1), (1, 0), (1, 1)],
#     [(1, 1), (1, 2), (2, 1), (2, 2)],
#     [(1, 0), (1, 1), (2, 0), (2, 1)],
#     [(0, 0), (0, 1), (1, 1), (1, 2)],
# ]

patterns = [
    [(0, 0), (0, 1), (0, 2), (0, 3)],
    [(1, 0), (1, 1), (1, 2), (1, 3)],
    [(0, 0), (0, 1), (1, 0), (1, 1)],
    [(1, 1), (1, 2), (2, 1), (2, 2)],
    [(1, 0), (1, 1), (2, 0), (2, 1)],
    [(0, 0), (0, 1), (1, 1), (1, 2)],
    [(0, 0), (0, 1), (0, 2), (1, 2)],
    [(0, 0), (0, 1), (0, 2), (1, 1)],
]

# import gdown
# # os.system("gdown --fuzzy https://drive.google.com/file/d/1G9xRMK6oniVhs_EZYo8S7Mo4eqVbv7jT/view?usp=sharing -O value_approximator_5_weights.pkl")
# url = "https://drive.google.com/file/d/1G9xRMK6oniVhs_EZYo8S7Mo4eqVbv7jT/view?usp=sharing"
# output = "value_approximator_5_weights.pkl"
# gdown.download(url, output, quiet=False, fuzzy=True, resume=True)

approximator = NTupleApproximator(board_size=4, patterns=patterns)
with open('value_approximator_9_weights.pkl', 'rb') as file:
    approximator.weights = pickle.load(file)


def get_action(state, score):
    
    root = TD_MCTS_Node(state, score, "next", 0)

    env = Action(state)
    # print([a for a in range(4) if root.is_move_legal(a)])

    td_mcts = TD_MCTS(env, approximator, iterations=5, exploration_constant=0.00001, rollout_depth=0, gamma=0.99)

    # Run multiple simulations to build the MCTS tree
    for _ in range(td_mcts.iterations):
        td_mcts.run_simulation(root)
    
    # Select the best action (based on highest visit count)
    best_act, _ = td_mcts.best_action_distribution(root)


    return best_act

    # return random.choice([0, 1, 2, 3]) # Choose a random action
    
    # You can submit this random agent to evaluate the performance of a purely random strategy.



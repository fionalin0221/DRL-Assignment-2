import copy
import random
import math
import numpy as np
from collections import defaultdict
import pickle
from student_agent import Game2048Env
import time

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


    def tile_to_index(self, tile):
        """
        Converts tile values to an index for the lookup table.
        """
        if tile == 0:
            return 0
        else:
            # 2^n -> n, like 2->1, 4->2, 8->3, ...
            return int(math.log(tile, 2))

    def tuple_id(self, values):
        values = values[::-1]
        k = 1
        n = 0
        for v in values:
            if v >= self.POWER:
                raise ValueError(
                    "digit %d should be smaller than the base %d" % (v, self.POWER)
                )
            n += v * k
            k *= self.POWER
        return n

    def value(self, board, delta=None):
        vals = []
        for i, (pattern, weight) in enumerate(zip(self.symmetry_patterns, self.weights)):
            tiles = [board[i][j] for i, j in pattern]
            index = [self.tile_to_index(t) for t in tiles]
            tpid = self.tuple_id(index)
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


        final_scores.append(env.score)
        success_flags.append(1 if max_tile >= 2048 else 0)
        global_max_tile = max(max_tile, global_max_tile)

        if (episode + 1) % 100 == 0:
            avg_score = np.mean(final_scores[-100:])
            success_rate = np.sum(success_flags[-100:]) / 100
            print(f"Episode {episode+1}/{num_episodes} | Avg Score: {avg_score:.2f} | Success Rate: {success_rate:.2f} | Max Tile: {global_max_tile}")
            global_max_tile = 0
            with open('value_approximator_9.pkl', 'wb') as file:
                pickle.dump(approximator, file)

    return final_scores


# TODO: Define your own n-tuple patterns
patterns = [
    [(0, 0), (0, 1), (0, 2), (0, 3)],
    [(1, 0), (1, 1), (1, 2), (1, 3)],
    [(0, 0), (0, 1), (1, 0), (1, 1)],
    [(1, 1), (1, 2), (2, 1), (2, 2)],
    [(1, 0), (1, 1), (2, 0), (2, 1)],
    [(0, 0), (0, 1), (1, 1), (1, 2)],
]

approximator = NTupleApproximator(board_size=4, patterns=patterns)

env = Game2048Env()

# Run TD-Learning training
# Note: To achieve significantly better performance, you will likely need to train for over 100,000 episodes.
# However, to quickly verify that your implementation is working correctly, you can start by running it for 1,000 episodes before scaling up.
final_scores = td_learning(env, approximator, num_episodes=200000, alpha=0.1, gamma=1, epsilon=0.1)

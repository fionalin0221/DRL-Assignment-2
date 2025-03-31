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




def compress(row):
    """Compress the row: move non-zero values to the left"""
    new_row = row[row != 0]  # Remove zeros
    new_row = np.pad(new_row, (0, 4 - len(new_row)), mode='constant')  # Pad with zeros on the right
    return new_row

def merge(row, score):
    """Merge adjacent equal numbers in the row"""
    for i in range(len(row) - 1):
        if row[i] == row[i + 1] and row[i] != 0:
            row[i] *= 2
            row[i + 1] = 0
            score += row[i]
    return row, score

def move_left(board, score):
    """Move the board left"""
    new_board = np.zeros_like(board)
    moved = False
    for i in range(4):
        original_row = board[i].copy()
        new_row = compress(board[i])
        new_row, new_score = merge(new_row, score)
        new_row = compress(new_row)
        new_board[i] = new_row
        if not np.array_equal(original_row, board[i]):
            moved = True
    return new_board, new_score

def move_right(board, score):
    """Move the board right"""
    new_board = np.zeros_like(board)
    moved = False
    for i in range(4):
        original_row = board[i].copy()
        # Reverse the row, compress, merge, compress, then reverse back
        reversed_row = board[i][::-1]
        reversed_row = compress(reversed_row)
        reversed_row, new_score = merge(reversed_row, score)
        reversed_row = compress(reversed_row)
        new_board[i] = reversed_row[::-1]
        if not np.array_equal(original_row, board[i]):
            moved = True
    return new_board, new_score

def move_up(board, score):
    """Move the board up"""
    new_board = np.zeros_like(board)
    moved = False
    for j in range(4):
        original_col = board[:, j].copy()
        col = compress(board[:, j])
        col, new_score = merge(col, score)
        col = compress(col)
        new_board[:, j] = col
        if not np.array_equal(original_col, board[:, j]):
            moved = True
    return new_board, new_score

def move_down(board, score):
    """Move the board down"""
    new_board = np.zeros_like(board)
    moved = False
    for j in range(4):
        original_col = board[:, j].copy()
        # Reverse the column, compress, merge, compress, then reverse back
        reversed_col = board[:, j][::-1]
        reversed_col = compress(reversed_col)
        reversed_col, new_score = merge(reversed_col, score)
        reversed_col = compress(reversed_col)
        new_board[:, j] = reversed_col[::-1]
        if not np.array_equal(original_col, board[:, j]):
            moved = True
    return new_board, new_score

def fake_step(board, score, action):
    """Execute one action"""

    if action == 0:
        return move_up(board, score)
    elif action == 1:
        return move_down(board, score) 
    elif action == 2:
        return move_left(board, score)
    elif action == 3:
        return move_right(board, score)


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

def td_learning(env, approximator, num_episodes=50000, alpha=0.01, gamma=0.99, epsilon=0.1):
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
            # if random.random() < epsilon:
            #     action = random.choice(legal_moves)
            # else:
            start_best_a = time.time()
            values = []
            for a in legal_moves:
                # create a new env for fake step to look the value of each next_state
                sim_env = create_env_from_state(env, state)
                test_next_state, new_score, done, _ = sim_env.step(a)

                incremental_reward = new_score - previous_score
                value = incremental_reward + approximator.value(test_next_state)
                values.append(value)

            action = legal_moves[np.argmax(values)]
            end_best_a = time.time()
            # print(f"select action time:{end_best_a-start_best_a}")

            start_update = time.time()

            next_state, new_score, done, _ = env.step(action)

            incremental_reward = new_score - previous_score # the reward get in this step
            previous_score = new_score
            max_tile = max(max_tile, np.max(next_state))

            # TODO: Store trajectory or just update depending on the implementation
            # TD error: r + gamma * v(s') - v(s)
            delta = incremental_reward + gamma * approximator.value(next_state) - approximator.value(state)
            
            # update the weights table
            # approximator.update(state, delta, alpha)
            approximator.value(state, delta*alpha)

            state = copy.deepcopy(next_state)

            end_update = time.time()
            # print(f"upate time: {end_update-start_update}")

        # TODO: If you are storing the trajectory, consider updating it now depending on your implementation.

        final_scores.append(env.score)
        success_flags.append(1 if max_tile >= 2048 else 0)
        global_max_tile = max(max_tile, global_max_tile)

        if (episode + 1) % 100 == 0:
            avg_score = np.mean(final_scores[-100:])
            success_rate = np.sum(success_flags[-100:]) / 100
            print(f"Episode {episode+1}/{num_episodes} | Avg Score: {avg_score:.2f} | Success Rate: {success_rate:.2f} | Max Tile: {global_max_tile}")
            global_max_tile = 0
            with open('value_approximator_6.pkl', 'wb') as file:
                pickle.dump(approximator, file)

    return final_scores


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

approximator = NTupleApproximator(board_size=4, patterns=patterns)
# with open('value_approximator_6.pkl', 'rb') as file:
#     approximator = pickle.load(file)
# for syms in approximator.symmetry_patterns:
#     for pattern in syms:
#         feature = []
#         for element in pattern:
#             a = element[0] * 4 + element[1]
#             feature.append(a)
#         print(feature)

env = Game2048Env()

# Run TD-Learning training
# Note: To achieve significantly better performance, you will likely need to train for over 100,000 episodes.
# However, to quickly verify that your implementation is working correctly, you can start by running it for 1,000 episodes before scaling up.
final_scores = td_learning(env, approximator, num_episodes=200000, alpha=0.1, gamma=1, epsilon=0.1)

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

class NTupleApproximator:
    def __init__(self, board_size, patterns):
        """
        Initializes the N-Tuple approximator.
        Hint: you can adjust these if you want
        """
        self.board_size = board_size
        # self.POWER = 15
        self.patterns = patterns
        # Create a weight dictionary for each pattern (shared within a pattern group)
        self.weights = [defaultdict(float) for _ in patterns]
        
        # Generate symmetrical transformations for each pattern
        self.symmetry_patterns = []

        for pattern in self.patterns:
            symmetric_pattern = []
            syms = self.generate_symmetries(pattern)
            for syms_ in syms:
                if syms_ not in symmetric_pattern:
                    symmetric_pattern.append(syms_)

            self.symmetry_patterns.append(symmetric_pattern)
            # self.symmetry_patterns.extend(symmetric_pattern)

        # self.weights = []
        # for pattern in self.symmetry_patterns:
        #     self.weights.append(np.zeros((self.POWER + 1) ** len(pattern)))

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

    # def tuple_id(self, values):
    #     values = values[::-1]
    #     k = 1
    #     n = 0
    #     for v in values:
    #         if v >= self.POWER:
    #             raise ValueError(
    #                 "digit %d should be smaller than the base %d" % (v, self.POWER)
    #             )
    #         n += v * k
    #         k *= self.POWER
    #     return n

    def get_feature(self, board, coords):
        # TODO: Extract tile values from the board based on the given coordinates and convert them into a feature tuple.
        features = []
        # each pattern have N-tuples(coords)
        for coord in coords:
            tile = board[coord[0]][coord[1]]
            index = self.tile_to_index(tile) # convert to power expression
            features.append(index)
        return tuple(features)

    # def value(self, board, delta=None):
    #     vals = []
    #     for i, (pattern, weight) in enumerate(zip(self.symmetry_patterns, self.weights)):
    #         tiles = [board[i][j] for i, j in pattern]
    #         index = [self.tile_to_index(t) for t in tiles]
    #         tpid = self.tuple_id(index)
    #         if delta is not None:
    #             weight[tpid] += delta
    #         v = weight[tpid]
    #         vals.append(v)
    #     return np.mean(vals)

    def value(self, board):
        # TODO: Estimate the board value: sum the evaluations from all patterns.
        values = []

        # sum up all patterns' weight
        for idx, symmetry_pattern in enumerate(self.symmetry_patterns):
            for pattern in symmetry_pattern: # each pattern in one class of symmteric pattern
                features = self.get_feature(board, pattern)
                # use features as index to search weights table
                value = self.weights[idx][features]
                values.append(value)

        return np.mean(values)

    def update(self, board, delta, alpha):
        # TODO: Update weights based on the TD error.
        # update each pattern use in this board
        for idx, symmetry_pattern in enumerate(self.symmetry_patterns):
            for pattern in symmetry_pattern:
                features = self.get_feature(board, pattern)
                self.weights[idx][features] += alpha * delta

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
            values = []
            for a in legal_moves:
                # create a new env for fake step to look the value of each next_state
                sim_env = create_env_from_state(env, state)
                test_next_state, new_score, done, _ = sim_env.step(a)

                incremental_reward = new_score - previous_score
                value = incremental_reward + approximator.value(test_next_state)
                values.append(value)

            action = legal_moves[np.argmax(values)]

            next_state, new_score, done, _ = env.step(action)

            incremental_reward = new_score - previous_score # the reward get in this step
            previous_score = new_score
            max_tile = max(max_tile, np.max(next_state))

            # TODO: Store trajectory or just update depending on the implementation
            # TD error: r + gamma * v(s') - v(s)
            delta = incremental_reward + gamma * approximator.value(next_state) - approximator.value(state)
            
            # update the weights table
            approximator.update(state, delta, alpha)
            # approximator.value(state, delta*alpha)

            state = copy.deepcopy(next_state)

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

env = Game2048Env()

# Run TD-Learning training
# Note: To achieve significantly better performance, you will likely need to train for over 100,000 episodes.
# However, to quickly verify that your implementation is working correctly, you can start by running it for 1,000 episodes before scaling up.
final_scores = td_learning(env, approximator, num_episodes=200000, alpha=0.1, gamma=1, epsilon=0.1)

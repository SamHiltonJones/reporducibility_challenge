import numpy as np
import gym
from gym import spaces
import time
import pickle
import copy

class GridWorldEnv(gym.Env):
    def __init__(self, grid_matrix, seed=np.random.randint(int(1e5)), max_steps=100):
        super(GridWorldEnv, self).__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple((spaces.Discrete(len(grid_matrix)), spaces.Discrete(len(grid_matrix[0]))))
        self.grid_matrix = grid_matrix
        self.state = None
        self.max_steps = max_steps
        self.num_steps = 0
        self.seed = seed
        np.random.seed(self.seed)
        self.goal_coords = self.random_empty_cell()

    def reset(self):
        self.num_steps = 0
        self.state = self.random_empty_cell()
        while self.state == self.goal_coords:
            self.state = self.random_empty_cell()
        return self.state

    def step(self, action):
        self.num_steps += 1
        next_state = self.move(self.state, action)

        current_distance = abs(self.state[0] - self.goal_coords[0]) + abs(self.state[1] - self.goal_coords[1])
        next_distance = abs(next_state[0] - self.goal_coords[0]) + abs(next_state[1] - self.goal_coords[1])

        reward = 1 if next_distance < current_distance else -1 
        done = next_state == self.goal_coords or self.num_steps >= self.max_steps

        self.state = next_state
        return self.state, reward, done, {}

    def is_wall(self, x, y):
        return self.grid_matrix[y][x] == 1

    def move(self, state, action):
        x, y = state
        if action == 0 and y > 0 and not self.is_wall(x, y - 1):
            return x, y - 1
        elif action == 1 and y < len(self.grid_matrix) - 1 and not self.is_wall(x, y + 1):
            return x, y + 1
        elif action == 2 and x > 0 and not self.is_wall(x - 1, y):
            return x - 1, y
        elif action == 3 and x < len(self.grid_matrix[0]) - 1 and not self.is_wall(x + 1, y):
            return x + 1, y
        return state

    def render(self, mode='human'):
        output_grid = copy.deepcopy(self.grid_matrix)
        x, y = self.state
        output_grid[y][x] = 'X'
        for row in output_grid:
            print(' '.join(map(str, row)))
        time.sleep(0.5)

    def random_empty_cell(self):
        while True:
            x = np.random.randint(len(self.grid_matrix[0]))
            y = np.random.randint(len(self.grid_matrix))
            if not self.is_wall(x, y):
                return x, y

grid_matrix = [
    [1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,0,0,1,0,0,0,0,0,1],
    [1,0,0,0,0,0,1,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,1,0,0,0,0,0,1],
    [1,0,0,0,0,0,1,0,0,0,0,0,1],
    [1,1,0,1,1,1,1,0,0,0,0,0,1],
    [1,0,0,0,0,0,1,1,1,0,1,1,1],
    [1,0,0,0,0,0,1,0,0,0,0,0,1],
    [1,0,0,0,0,0,1,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,1,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1]
]

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
goal_coords = (11, 1)

def value_iteration(env, discount_factor=0.90, theta=0.01):
    states = [(x, y) for x in range(len(env.grid_matrix[0])) for y in range(len(env.grid_matrix))]
    actions = [UP, DOWN, LEFT, RIGHT]
    value_map = np.zeros_like(env.grid_matrix, dtype=np.float32)
    policy_map = np.full((len(env.grid_matrix), len(env.grid_matrix[0])), None)

    while True:
        delta = 0
        for state in states:
            if state == env.goal_coords or env.is_wall(*state):
                continue

            v = value_map[state[1], state[0]]
            max_value = float('-inf')
            for action in actions:
                next_state = env.move(state, action)
                reward = 100 if next_state == env.goal_coords else 0
                value = reward + discount_factor * value_map[next_state[1], next_state[0]]
                if value > max_value:
                    max_value = value
                    policy_map[state[1], state[0]] = action
            value_map[state[1], state[0]] = max_value
            delta = max(delta, abs(v - max_value))

        if delta < theta:
            break
    
    return policy_map

def expert_policy(state, policy):
    action = policy[state[1], state[0]]
    return action

def generate_dataset_formatted(env, transitions=10000):
    data = {
        'main_key': {
            'states': np.zeros((transitions, 2), dtype=np.float32),
            'actions': np.zeros((transitions, 1), dtype=np.float32),
            'rewards': np.zeros((transitions, 1), dtype=np.float32),
            'next_states': np.zeros((transitions, 2), dtype=np.float32),
            'terminations': np.zeros((transitions, 1), dtype=np.bool_)
        }
    }

    for i in range(transitions):
        state = env.reset()
        policy_map = value_iteration(env)

        action = expert_policy(state, policy_map)
        next_state, reward, done, _ = env.step(action)

        data['main_key']['states'][i] = np.array(state, dtype=np.float32)
        data['main_key']['actions'][i] = np.array([action], dtype=np.float32)
        data['main_key']['rewards'][i] = np.array([reward], dtype=np.float32)
        data['main_key']['next_states'][i] = np.array(next_state, dtype=np.float32)
        data['main_key']['terminations'][i] = np.array([done], dtype=np.bool_)

    return data


env = GridWorldEnv(grid_matrix)
formatted_dataset = generate_dataset_formatted(env, transitions=10000)
with open('expert_data.pkl', 'wb') as file:
    pickle.dump(formatted_dataset, file)

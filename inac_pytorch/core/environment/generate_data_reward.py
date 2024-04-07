import numpy as np
import pickle
import gym
from gym import spaces

class GridWorldEnv(gym.Env):
    def __init__(self, grid_matrix, goal_coords, seed=np.random.randint(int(1e5))):
        super(GridWorldEnv, self).__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple((spaces.Discrete(len(grid_matrix)), spaces.Discrete(len(grid_matrix[0]))))
        self.grid_matrix = grid_matrix
        self.goal_coords = goal_coords
        self.state = None

    def reset(self):
        self.state = (1, 11)
        return self.state

    def step(self, action):
        next_state = self.move(self.state, action)
        current_distance = abs(self.state[0] - self.goal_coords[0]) + abs(self.state[1] - self.goal_coords[1])
        next_distance = abs(next_state[0] - self.goal_coords[0]) + abs(next_state[1] - self.goal_coords[1])
        reward = 100 if next_state == self.goal_coords else 1 if next_distance < current_distance else -1
        done = next_state == self.goal_coords
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
        pass

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

def value_iteration(grid, goal, discount_factor=0.90, theta=0.01):
    states = [(x, y) for x in range(len(grid[0])) for y in range(len(grid))]
    actions = [UP, DOWN, LEFT, RIGHT]
    value_map = np.zeros_like(grid, dtype=np.float32)
    policy = np.full((len(grid), len(grid[0])), None)
    env = GridWorldEnv(grid_matrix, goal_coords)
    while True:
        delta = 0
        for state in states:
            if state == goal or env.is_wall(*state):
                continue

            v = value_map[state[1], state[0]]
            max_value = float('-inf')
            for action in actions:
                next_state = env.move(state, action)
                reward = 1 if next_state == goal else 0
                value = reward + discount_factor * value_map[next_state[1], next_state[0]]
                if value > max_value:
                    max_value = value
                    policy[state[1], state[0]] = action
            value_map[state[1], state[0]] = max_value
            delta = max(delta, abs(v - max_value))

        if delta < theta:
            break
    
    return policy

def expert_policy(state, policy):
    action = policy[state[1], state[0]]
    return action

policy_map = value_iteration(grid_matrix, goal_coords, discount_factor=0.9, theta=0.01)

def generate_dataset_formatted(env, policy_map, transitions=10000):
    data = {
        'states': np.zeros((transitions, 2), dtype=np.float32),
        'actions': np.zeros((transitions, 1), dtype=np.float32),
        'rewards': np.zeros((transitions, 1), dtype=np.float32),
        'next_states': np.zeros((transitions, 2), dtype=np.float32),
        'terminations': np.zeros((transitions, 1), dtype=np.bool_)
    }

    state = (1, 11)  # Starting state
    reset_count = 0
    dataset_action_counts = {UP: 0, DOWN: 0, LEFT: 0, RIGHT: 0}
    for i in range(transitions):
        action = expert_policy(state, policy_map)
        dataset_action_counts[action] += 1
        next_state = env.move(state, action)
        reward = 1 if next_state == goal_coords else 0
        done = next_state == goal_coords
        data['states'][i] = np.array(state, dtype=np.float32)
        data['actions'][i] = np.array([action], dtype=np.float32)
        data['rewards'][i] = np.array([reward], dtype=np.float32)
        data['next_states'][i] = np.array(next_state, dtype=np.float32)
        data['terminations'][i] = np.array([done], dtype=np.bool_)
        if done:
            state = (1, 11) 
            reset_count += 1
        else:
            state = next_state
    print("Action Distribution in Dataset:", dataset_action_counts)
    print(f"State was reset {reset_count} times")
    return data

env = GridWorldEnv(grid_matrix, goal_coords)
policy_map = value_iteration(grid_matrix, goal_coords, discount_factor=0.9, theta=0.01)
formatted_dataset = generate_dataset_formatted(env, policy_map)

with open('expert_data.pkl', 'wb') as file:
    pickle.dump(formatted_dataset, file)

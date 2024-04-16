import numpy as np
import pickle
from grid_env import GridWorldEnv

def random_action(env):
    return np.random.choice([GridWorldEnv.UP, GridWorldEnv.DOWN, GridWorldEnv.LEFT, GridWorldEnv.RIGHT])

def generate_dataset_formatted(env, transitions=10000, train_ratio=0.8):
    experiences = []
    data = {
        'states': [],
        'actions': [],
        'rewards': [],
        'next_states': [],
        'terminations': []
    }
    empty_cells = env.get_empty_cells()

    for _ in range(transitions):
        start_index = np.random.choice(len(empty_cells))
        goal_index = np.random.choice(len(empty_cells))
        while goal_index == start_index:
            goal_index = np.random.choice(len(empty_cells))
        start, goal = empty_cells[start_index], (11,1)
        env.state = start

        while True:
            action = random_action(env)
            if action is None:
                print("No valid action found, skipping this state:", env.state)
                break
            next_state, reward, done, _ = env.step(action)

            experience = (np.array(env.state), action, reward, np.array(next_state), done)
            experiences.append(experience)

            data['states'].append(np.array(env.state))
            data['actions'].append(action)
            data['rewards'].append(reward)
            data['next_states'].append(np.array(next_state))
            data['terminations'].append(done)

            if done:
                break
            env.state = next_state

    for key in data:
        data[key] = np.array(data[key])

    train_size = int(len(experiences) * train_ratio)
    train_experiences = experiences[:train_size]
    test_experiences = experiences[train_size:]

    train_data = {k: v[:train_size] for k, v in data.items()}
    test_data = {k: v[train_size:] for k, v in data.items()}

    return (train_experiences, test_experiences), ({'pkl': train_data}, {'pkl': test_data})


if __name__ == '__main__':
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
    env = GridWorldEnv(grid_matrix)
    (train_experiences, test_experiences), (train_data_dict, test_data_dict) = generate_dataset_formatted(env, transitions=10000)

    with open('core/train_data_random.pkl', 'wb') as f:
        pickle.dump((train_experiences, {'pkl': train_data_dict}), f)
    with open('core/test_data_random.pkl', 'wb') as f:
        pickle.dump((test_experiences, {'pkl': test_data_dict}), f)

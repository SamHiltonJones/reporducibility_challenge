import pickle
import time
import numpy as np
import os
import matplotlib.pyplot as plt

def load_data(is_training, dataset_name):
    base_path = "core/"
    dataset_suffix = 'train_data.pkl' if is_training else 'test_data.pkl'
    dataset_file = f'{dataset_name}_{dataset_suffix}'  # Build the file name dynamically based on the dataset
    file_path = os.path.join(base_path, dataset_file)
    
    with open(file_path, 'rb') as file:
        loaded_experiences, loaded_data_dict = pickle.load(file)

    if 'pkl' in loaded_data_dict:
        data_dict = loaded_data_dict['pkl']
    else:
        data_dict = loaded_data_dict  

    for key, value in data_dict.items():
        if hasattr(value, 'shape'):
            print(f"Loaded data - {key} shape: {value.shape}")
        else:
            print(f"Loaded data - {key} is not an array")

    return loaded_experiences, data_dict


def run_steps(agent, max_steps, log_interval, eval_path, train_data, test_data):
    print("Starting run_steps method")
    start_time = time.time()
    evaluations = []

    for experience in train_data:
        agent.replay.feed(experience)

    agent.populate_returns(initialize=True)

    while agent.total_steps < max_steps:
        if agent.total_steps % log_interval == 0:
            elapsed_time = time.time() - start_time
            mean, median, min_, max_ = agent.log_file(elapsed_time=elapsed_time, test=True)
            evaluations.append(mean)
            print(f"Evaluation at step {agent.total_steps}: Mean Reward = {mean:.2f}, Median = {median:.2f}, Min = {min_:.2f}, Max = {max_:.2f}, Time elapsed = {elapsed_time:.2f} seconds")

            start_time = time.time()  

            if hasattr(agent, 'calculate_average_loss'):
                avg_loss = agent.calculate_average_loss()
                print(f"Average Loss at step {agent.total_steps}: {avg_loss:.4f}")

        agent.step()

        if max_steps and agent.total_steps >= max_steps:
            break

    agent.save()
    np.save(os.path.join(eval_path, "evaluations.npy"), np.array(evaluations))
    # plt.figure(figsize=(10, 5))
    # plt.plot(agent.average_rewards, label='Average Reward')
    # plt.xlabel('Episodes')
    # plt.ylabel('Average Reward')
    # plt.title('Average Reward vs. Episodes')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    print("Completed running steps.")
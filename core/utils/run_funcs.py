import pickle
import time
import numpy as np
import os
import matplotlib.pyplot as plt

def load_data(is_training):
    base_path = "/home/sam/jack_and_sam/reproducibility_challenge/core/"
    dataset_file = 'train_data.pkl' if is_training else 'test_data.pkl'
    file_path = os.path.join(base_path, dataset_file)
    
    with open(file_path, 'rb') as f:
        return pickle.load(f)

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
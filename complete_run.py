import os
import json
import numpy as np
import matplotlib.pyplot as plt
from run_ac_offline import run_experiment

def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def run_multiple_experiments(learning_rates, num_runs=10):
    results = {}
    base_dir = 'results'

    for lr in learning_rates:
        lr_dir = os.path.join(base_dir, f'lr_{lr}')
        ensure_directory(lr_dir)

        all_rewards = []
        for run in range(num_runs):
            seed = np.random.randint(0, 10000)
            episode_rewards = run_experiment(learning_rate=lr, seed=seed)
            all_rewards.append(episode_rewards)

            rewards_filename = os.path.join(lr_dir, f'episode_rewards_run_{run}.json')
            with open(rewards_filename, 'w') as f:
                json.dump(episode_rewards, f)

        avg_rewards = np.mean(all_rewards, axis=0)
        results[lr] = avg_rewards.tolist() 
        avg_rewards_filename = os.path.join(lr_dir, 'average_rewards.json')
        with open(avg_rewards_filename, 'w') as f:
            json.dump(avg_rewards.tolist(), f)

        plt.figure()
        plt.plot(avg_rewards, label=f'LR: {lr}')
        plt.xlabel('Iteration')
        plt.ylabel('Average Return per Episode')
        plt.title(f'Average Return per Episode Over Iterations (LR: {lr})')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(lr_dir, 'avg_rewards.png'))
        plt.close()

    combined_results_filename = os.path.join(base_dir, 'combined_results.json')
    with open(combined_results_filename, 'w') as f:
        json.dump(results, f)

    return results

if __name__ == "__main__":
    learning_rates = [0.01, 0.003, 0.001]
    results = run_multiple_experiments(learning_rates)

    plt.figure(figsize=(10, 6))
    for lr, avg_rewards in results.items():
        plt.plot(avg_rewards, label=f'LR: {lr}')
    plt.xlabel('Iteration')
    plt.ylabel('Average Return per Episode')
    plt.title('Average Return per Episode Over Iterations for Different Learning Rates')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join('results', 'combined_avg_rewards.png'))
    plt.show()

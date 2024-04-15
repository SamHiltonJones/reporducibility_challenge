import numpy as np
import pickle
import os
from collections import Counter

# Path to the train_data.pkl file
data_path = '/home/sam/jack_and_sam/reproducibility_challenge/core/train_data.pkl'  # Change to the correct path

def analyze_data(data_path):
    with open(data_path, 'rb') as f:
        _, data = pickle.load(f)  # Load your train_data

    # Ensure that you're accessing the data dictionary correctly
    if 'pkl' in data:
        data = data['pkl']['pkl']
    
    states = data['states']
    actions = data['actions']
    rewards = data['rewards']
    next_states = data['next_states']
    terminations = data['terminations']
    
    # Basic statistics
    num_experiences = len(actions)
    unique_states = len(np.unique(states, axis=0))
    average_reward = np.mean(rewards)
    
    # Count the distribution of actions
    action_counts = Counter(actions)
    for action, count in action_counts.items():
        print(f"Action {action}: {count} occurrences ({count / num_experiences * 100:.2f}%)")

    # Print basic statistics
    print(f"\nTotal number of experiences: {num_experiences}")
    print(f"Unique states encountered: {unique_states}")
    print(f"Average reward per experience: {average_reward:.4f}")
    
    # Check for termination statistics
    terminations_count = Counter(terminations)
    for term_state, count in terminations_count.items():
        print(f"Termination state {term_state}: {count} occurrences ({count / num_experiences * 100:.2f}%)")
    
    # If applicable, you can also look at the transitions from state to state
    # This requires a bit more complex processing depending on what you need

if __name__ == "__main__":
    analyze_data(data_path)

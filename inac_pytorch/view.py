import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load the pickle file
with open('/home/sam/comp6258/inac_pytorch/expert_data.pkl', 'rb') as file:
    data = pickle.load(file)

# Analyze the structure
print("Data Keys:", data.keys())
data = data[list(data.keys())[0]]
print("Inside DataKeys: ", data.keys())

print("States shape:", data['states'].shape)
print("Actions shape:", data['actions'].shape)
print("Rewards shape:", data['rewards'].shape)
print("Next States shape:", data['next_states'].shape)
print("Terminations shape:", data['terminations'].shape)

unique_states = np.unique(data['states'], axis=0)
unique_actions = np.unique(data['actions'])
print("Unique states:", len(unique_states))
print("Unique actions:", len(unique_actions))

plt.hist(data['actions'], bins=len(unique_actions), alpha=0.7, color='blue')
plt.title('Action Distribution')
plt.xlabel('Actions')
plt.ylabel('Frequency')
plt.show()

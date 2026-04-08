import numpy as np
from environment import AdvancedGridWorld

env = AdvancedGridWorld()

# Grid size = 5x5 → states = 25
state_size = (5, 5)
action_size = 4  # up, down, left, right

# Initialize Q-table with zeros
q_table = np.zeros((state_size[0], state_size[1], action_size))

# Hyperparameters
alpha = 0.1   # learning rate
gamma = 0.9    # discount factor
epsilon = 0.2  # exploration rate
episodes = 500

for episode in range(episodes):
    state = env.reset()
    done = False

    while not done:
        x, y = state

        # exploration vs exploitation
        if np.random.rand() < epsilon:
            action = np.random.randint(action_size)
        else:
            action = np.argmax(q_table[x, y])

        next_state, reward, done, _ = env.step(action)
        nx, ny = next_state

        # Q-learning update
        old_value = q_table[x, y, action]
        next_max = np.max(q_table[nx, ny])

        new_value = old_value + alpha * (reward + gamma * next_max - old_value)
        q_table[x, y, action] = new_value

        state = next_state

print("Training complete!")

# Save Q-table as JSON (Pickle-safe)
import json
with open("q_table.json", "w") as f:
    json.dump(q_table.tolist(), f)

print("Q-table saved as q_table.json")

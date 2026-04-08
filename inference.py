import os
import json
import numpy as np
from environment import AdvancedGridWorld

env = AdvancedGridWorld()

with open("q_table.json", "r") as f:
    q_table = np.array(json.load(f))


def get_state_index(state):
    x, y = state
    return int(x), int(y)


def run():
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        s = get_state_index(state)
        action = int(np.argmax(q_table[s]))

        state, reward, done, _ = env.step(action)
        total_reward += reward

    return total_reward


if __name__ == "__main__":
    score = run()
    print("Final Score:", score)

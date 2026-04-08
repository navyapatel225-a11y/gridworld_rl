import numpy as np

class AdvancedGridWorld:
    def __init__(self, size=5):
        self.size = size
        self.reset()

    def reset(self):
        self.agent_pos = [0, 0]
        self.goal = [self.size - 1, self.size - 1]
        self.steps = 0
        return np.array(self.agent_pos)

    def step(self, action):
        # 0: up, 1: down, 2: left, 3: right
        if action == 0:
            self.agent_pos[1] -= 1
        elif action == 1:
            self.agent_pos[1] += 1
        elif action == 2:
            self.agent_pos[0] -= 1
        elif action == 3:
            self.agent_pos[0] += 1

        # bounds
        self.agent_pos[0] = max(0, min(self.size - 1, self.agent_pos[0]))
        self.agent_pos[1] = max(0, min(self.size - 1, self.agent_pos[1]))

        self.steps += 1

        done = self.agent_pos == self.goal

        # reward
        reward = -0.1
        if done:
            reward = 10

        return np.array(self.agent_pos), reward, done, {}

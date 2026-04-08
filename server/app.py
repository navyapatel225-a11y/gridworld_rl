import numpy as np
from typing import Tuple, Dict, Any

class AdvancedGridWorld:
    def __init__(self, size: int = 5):
        self.size = size
        self.reset()

    def reset(self) -> np.ndarray:
        self.agent_pos = [0, 0]
        self.goal = [self.size - 1, self.size - 1]
        self.steps = 0
        return np.array(self.agent_pos)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        # 0: up, 1: down, 2: left, 3: right
        if action == 0:
            self.agent_pos[1] -= 1
        elif action == 1:
            self.agent_pos[1] += 1
        elif action == 2:
            self.agent_pos[0] -= 1
        elif action == 3:
            self.agent_pos[0] += 1

        # bounds checking
        self.agent_pos[0] = max(0, min(self.size - 1, self.agent_pos[0]))
        self.agent_pos[1] = max(0, min(self.size - 1, self.agent_pos[1]))

        self.steps += 1

        done = self.agent_pos == self.goal

        # reward: -0.1 per step, +10 for reaching goal
        reward = 10.0 if done else -0.1

        info = {"steps": self.steps, "position": self.agent_pos.copy()}

        return np.array(self.agent_pos), reward, done, info

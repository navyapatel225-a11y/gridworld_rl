import os
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
from fastapi.middleware.cors import CORSMiddleware

# ========== ENVIRONMENT CLASS ==========
class AdvancedGridWorld:
    def __init__(self, size: int = 5):
        self.size = size
        self.reset()

    def reset(self) -> np.ndarray:
        self.agent_pos = [0, 0]
        self.goal = [self.size - 1, self.size - 1]
        self.steps = 0
        return np.array(self.agent_pos)

    def step(self, action: int):
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
        reward = 10.0 if done else -0.1
        info = {"steps": self.steps, "position": self.agent_pos.copy()}

        return np.array(self.agent_pos), reward, done, info

# ========== FASTAPI APP ==========
app = FastAPI(title="GridWorld RL Environment")

# Add CORS middleware for compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize environment
env = AdvancedGridWorld(size=5)

# ========== REQUEST/RESPONSE MODELS ==========
class StepRequest(BaseModel):
    action: int

class StepResponse(BaseModel):
    observation: List[int]
    reward: float
    done: bool
    info: Dict[str, Any]

# ========== API ENDPOINTS ==========
@app.get("/")
def root():
    return {
        "status": "running",
        "environment": "AdvancedGridWorld",
        "size": env.size,
        "endpoints": {
            "reset": {"method": "POST", "path": "/reset"},
            "step": {"method": "POST", "path": "/step"}
        }
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/reset")
def reset():
    obs = env.reset()
    return {"observation": obs.tolist()}

@app.post("/step")
def step(request: StepRequest):
    if request.action not in [0, 1, 2, 3]:
        raise HTTPException(status_code=400, detail="Action must be 0, 1, 2, or 3")
    
    obs, reward, done, info = env.step(request.action)
    return StepResponse(
        observation=obs.tolist(),
        reward=float(reward),
        done=bool(done),
        info=info
    )

# ========== RUN SERVER ==========
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

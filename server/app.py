import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
import sys

# Add parent directory to path to import environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import AdvancedGridWorld

# THIS MUST BE NAMED 'app' - it's what the root app.py imports
app = FastAPI(title="GridWorld RL Environment", description="OpenEnv compatible RL environment")

# Initialize environment
env = AdvancedGridWorld(size=5)

class StepRequest(BaseModel):
    action: int

class StepResponse(BaseModel):
    observation: List[int]
    reward: float
    done: bool
    info: Dict[str, Any]

@app.get("/")
def root():
    return {
        "status": "running",
        "environment": "AdvancedGridWorld",
        "size": env.size,
        "endpoints": ["/reset", "/step"]
    }

@app.post("/reset")
def reset():
    obs = env.reset()
    return {"observation": obs.tolist()}

@app.post("/step")
def step(request: StepRequest):
    if request.action not in [0, 1, 2, 3]:
        raise HTTPException(status_code=400, detail="Action must be 0-3")
    
    obs, reward, done, info = env.step(request.action)
    return StepResponse(
        observation=obs.tolist(),
        reward=float(reward),
        done=bool(done),
        info=info
    )

def main():
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()

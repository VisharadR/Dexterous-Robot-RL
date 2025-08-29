import numpy as np

import os, sys
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from src.envs.pick_place_env import PickPlaceEnv

if __name__ == "__main__":
    env = PickPlaceEnv(render_mode="human") #shows GUI
    obs, _ = env.reset()
    total_r = 0
    while True:
        # random action
        action = np.random.unifrom(low=env.action_space.low,
                                   high=env.action_space.high)
        obs, reward, term, trunc, _ = env.step(action)
        total_r += reward
        if term or trunc:
            print("Episode done. Total reward:", round(total_r, 2))
            obs, _ = env.reset()
            total_r = 0
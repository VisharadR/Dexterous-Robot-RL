import os, sys
import time 
import glob
import numpy as np
from stable_baselines3 import PPO

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from src.envs.pick_place_env import PickPlaceEnv

def find_latest_model():
    # Look for results/*/models/(best_model.zip | ppo_pickplace_final.zip | ckpt*.zip)
    candidate_runs = sorted(
        [d for d in glob.glob(os.path.join("results", "*")) if os.path.isdir(d)],
        key=os.path.getmtime,
        reverse=True,
    )
    for run_dir in candidate_runs:
        models_dir = os.path.join(run_dir, "models")
        if not os.path.isdir(models_dir):
            continue
        # Preference order: best_model -> final -> newest checkpoint
        best = os.path.join(models_dir, "best_model.zip")
        final = os.path.join(models_dir, "ppo_pickplace_final.zip")
        if os.path.isfile(best):
            return best
        if os.path.isfile(final):
            return final
        # fallback: latest ckpt_*.zip
        ckpts = sorted(glob.glob(os.path.join(models_dir, "ckpt*.zip")), key=os.path.getmtime, reverse=True)
        if ckpts:
            return ckpts[0]
    return None

if __name__ == "__main__":
    model_path = find_latest_model()
    if model_path is None:
        raise FileNotFoundError(
            "No trained model found. Run scripts/train_ppo.py first, "
            "or set MODEL_PATH manually."
        )
    print(f"Loading model from: {model_path}")

    env = PickPlaceEnv(render_mode="human")
    model = PPO.load(model_path)

    n_episodes = 10
    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_r = 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(action)
            ep_r += r
            time.sleep(1/120)  # slow down a bit for viewing
            if term or trunc:
                print(f"Episode {ep+1}: return={ep_r:.3f}")
                break

    env.close()
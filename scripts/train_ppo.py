import os, sys
import time 
import numpy as np
from typing import Callable

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)


import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.logger import configure

from src.envs.pick_place_env import PickPlaceEnv


# -------------- config ----------------
RUN_NAME = time.strftime("ppo_pickplace_%Y%m%d_%H%M%S")
LOG_DIR = os.path.join("results", RUN_NAME)
SAVE_DIR = os.path.join(LOG_DIR, "models")
EVAL_DIR = os.path.join(LOG_DIR, "eval")

TOTAL_TIMESTEPS = 1_000_000   # bump if training plateaus early
N_ENVS = 4                  # try 1 if your CPU is weak
SEED = 42
EVAL_EPISODES = 10
EVAL_FREQ = 10_000          # every X steps
# --------------------------------------

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

def make_env(render_mode=None) -> Callable[[], gym.Env]:
    def _thunk():
        env = PickPlaceEnv(render_mode=render_mode)
        env =Monitor(env)
        return env
    return _thunk

if __name__ == "__main__":
    # vectorized envs (headless)
    if N_ENVS > 1:
        env = SubprocVecEnv([make_env(render_mode=None) for _ in range(N_ENVS)])
    else:
        env = DummyVecEnv([make_env(render_mode=None)])

    # separate eval env (non-vectorized)
    eval_env = DummyVecEnv([make_env(render_mode=None)])

    # logger (TensorBoard & stdout)
    new_logger = configure(LOG_DIR, ["stdout", "tensorboard"])

    # policy_kwargs: small MLP; you can enlarge later
    policy_kwargs = dict(net_arch=[256, 256, 128])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=SEED,
        learning_rate=3e-4,
        n_steps=2048//max(1, N_ENVS),
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        policy_kwargs=policy_kwargs,
        tensorboard_log=LOG_DIR,
    )
    model.set_logger(new_logger)


    # callbacks: save checkpoints + best model on eval reward
    checkpoint_cb = CheckpointCallback(
        save_freq=50_000 // max(1, N_ENVS), # in steps
        save_path=SAVE_DIR,
        name_prefix="ckpt"
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=SAVE_DIR,
        log_path=EVAL_DIR,
        eval_freq=EVAL_FREQ // max(1, N_ENVS),
        n_eval_episodes=EVAL_EPISODES,
        deterministic=True,
        render=False
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[checkpoint_cb, eval_cb])

    # final save
    final_path = os.path.join(SAVE_DIR, "ppo_pickplace_final")
    model.save(final_path)
    print(f"Training complete. Saved model to: {final_path}")
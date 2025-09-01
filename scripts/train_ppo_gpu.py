import os, sys, time, argparse
import numpy as np
import torch
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.logger import configure

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from src.envs.pick_place_env import PickPlaceEnv


def make_env(render_mode=None, easy=True, dr=False):
    def _thunk():
        env = PickPlaceEnv(render_mode=render_mode, domain_randomization=dr)
        env.set_easy_mode(easy)
        return Monitor(env)
    return _thunk


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=1_200_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--easy", type=int, default=1)
    ap.add_argument("--dr", type=int, default=0)
    ap.add_argument("--n-envs", type=int, default=8, help="vectorized envs (CPU parallel)")
    ap.add_argument("--n-steps", type=int, default=1024, help="rollout length per env")
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda"])
    ap.add_argument("--compile", type=int, default=1, help="torch.compile policy (PyTorch 2.x)")
    args = ap.parse_args()


    # ------- device selection ---------
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if device == "cuda":
        # faster matmul kernels on Ampere+
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    run_name = time.strftime("ppo_pickplace_%Y%m%d_%H%M%S")
    LOG_DIR  = os.path.join("results", run_name)
    SAVE_DIR = os.path.join(LOG_DIR, "models")
    EVAL_DIR = os.path.join(LOG_DIR, "eval")
    os.makedirs(SAVE_DIR, exist_ok=True); os.makedirs(EVAL_DIR, exist_ok=True)


    # ---- envs (parallel CPU stepping) ----
    if args.n_envs > 1:
        env = SubprocVecEnv([make_env(render_mode=None, easy=bool(args.easy), dr=bool(args.dr))
                             for _ in range(args.n_envs)])
    else:
        env = DummyVecEnv([make_env(render_mode=None, easy=bool(args.easy), dr=bool(args.dr))])

    eval_env = DummyVecEnv([make_env(render_mode=None, easy=bool(args.easy), dr=bool(args.dr))])

    logger = configure(LOG_DIR, ["stdout", "tensorboard"])

    # Larger batch to amortize GPU compute
    policy_kwargs = dict(net_arch=[256, 256, 128])


    model = PPO(
        "MlpPolicy",
        env,
        device=device,                  # <<< GPU for nets
        verbose=1,
        seed=args.seed,
        learning_rate=3e-4,
        n_steps=max(64, args.n_steps // max(1, args.n_envs)),  # per-env rollout length
        batch_size= min(4096, args.n_envs * 1024 // 2),        # big-ish batches for GPU
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        policy_kwargs=policy_kwargs,
        tensorboard_log=LOG_DIR,
    )
    model.set_logger(logger)

    # optional: compile policy for extra speed (PyTorch 2.x)
    if device == "cuda" and int(args.compile) == 1:
        try:
            model.policy = torch.compile(model.policy, mode="max-autotune")  # PyTorch 2.2+
            print("[info] torch.compile enabled for policy.")
        except Exception as e:
            print("[warn] torch.compile not enabled:", e)

    ckpt_cb = CheckpointCallback(save_freq=50_000, save_path=SAVE_DIR, name_prefix="ckpt")
    eval_cb  = EvalCallback(eval_env, best_model_save_path=SAVE_DIR, log_path=EVAL_DIR,
                            eval_freq=max(10_000 // max(1, args.n_envs), 2000),
                            n_eval_episodes=10, deterministic=True)

    model.learn(total_timesteps=args.timesteps, callback=[ckpt_cb, eval_cb])

    model.save(os.path.join(SAVE_DIR, "ppo_pickplace_final"))
    print("Saved:", os.path.join(SAVE_DIR, "ppo_pickplace_final.zip"))
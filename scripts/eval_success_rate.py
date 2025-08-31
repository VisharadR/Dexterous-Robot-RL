import os, sys, argparse, glob, numpy as np
from stable_baselines3 import PPO

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from src.envs.pick_place_env import PickPlaceEnv

def find_latest_model():
    runs = sorted([d for d in glob.glob(os.path.join("results","*")) if os.path.isdir(d)],
                  key=os.path.getmtime, reverse=True)
    for run in runs:
        mdir = os.path.join(run, "models")
        if not os.path.isdir(mdir): continue
        for name in ["best_model.zip", "ppo_pickplace_final.zip"]:
            pth = os.path.join(mdir, name)
            if os.path.isfile(pth): return pth
        ckpts = sorted(glob.glob(os.path.join(mdir, "ckpt*.zip")), key=os.path.getmtime, reverse=True)
        if ckpts: return ckpts[0]
    return None

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--dr", type=int, default=0, help="domain randomization (0/1)")
    ap.add_argument("--easy", type=int, default=1, help="curriculum easy mode (0/1)")
    ap.add_argument("--mode", type=str, default="and", choices=["and","lift","near"],
                    help="success criterion")
    args = ap.parse_args()

    model_path = find_latest_model()
    if model_path is None:
        raise FileNotFoundError("No model found in results/*/models. Train first.")
    print("Evaluating:", model_path, "| mode:", args.mode, "| easy:", args.easy, "| dr:", args.dr)

    env = PickPlaceEnv(render_mode=None, domain_randomization=bool(args.dr))
    env.set_easy_mode(bool(args.easy))
    env.set_success_mode(args.mode)
    model = PPO.load(model_path)

    N = args.episodes
    succ = 0
    rets = []
    max_heights = []
    min_goal_dists = []

    for _ in range(N):
        obs, _ = env.reset()
        done = False; trunc = False; ep_r = 0.0
        max_h = 0.0
        min_d = 1e9
        while not (done or trunc):
            # track diagnostics
            cube = env._get_cube_pose()
            max_h = max(max_h, float(cube[2]))
            d = float(np.linalg.norm(cube[:2] - env.goal_xy))
            min_d = min(min_d, d)

            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, trunc, info = env.step(action)
            ep_r += r

        rets.append(ep_r)
        max_heights.append(max_h)
        min_goal_dists.append(min_d)
        succ += int(info.get("success", False))

    print(f"Success: {succ}/{N} = {succ/N:.2%}")
    print(f"Return:  mean={np.mean(rets):.3f}, median={np.median(rets):.3f}, std={np.std(rets):.3f}")
    print(f"Diag:    mean max height={np.mean(max_heights):.3f} m, mean min goal dist={np.mean(min_goal_dists):.3f} m")
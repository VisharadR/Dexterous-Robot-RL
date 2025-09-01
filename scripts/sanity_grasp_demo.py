import time, numpy as np
import pybullet as p

import os, sys
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from src.envs.pick_place_env import PickPlaceEnv


# ---- helpers ----
def wait_steps(n):
    for _ in range(n):
        p.stepSimulation()
        time.sleep(1/240)

def move_converge(env: PickPlaceEnv, target, yaw=0.0, settle=12):
    """Send IK target, then step a bit to let joints settle."""
    env._move_ee_abs(target, yaw=yaw)         # sets motors + a few settle steps (your env should do this)
    for _ in range(settle):
        p.stepSimulation()
        time.sleep(1/240)

def print_status(env, tag=""):
    ee, _ = env._get_ee_pose()
    cube = env._get_cube_pose()
    print(f"{tag}  ee={ee.round(3)}  cube={cube.round(3)}  z_gap={(ee[2]-cube[2]):.3f}")

if __name__ == "__main__":
    env = PickPlaceEnv(render_mode="human", domain_randomization=False)
    env.set_easy_mode(True)
    env.set_success_mode("lift")          # not required; just fyi for your env
    obs, _ = env.reset()

    # tune these if needed
    HOVER_H = 0.08       # hover height above cube
    GRASP_H = 0.035      # grasp height above cube (fingers clear)
    LIFT_H  = 0.18       # lift height above cube
    DESCENT_STEP = 0.006 # 6mm per step

    # get cube position
    cube = env._get_cube_pose()

    # 1) move to hover
    target = np.array([cube[0], cube[1], cube[2] + HOVER_H], dtype=np.float32)
    print_status(env, "[move to hover] before")
    move_converge(env, target, yaw=0.0, settle=18)
    print_status(env, "[move to hover] after")
    wait_steps(120)

    # 2) descend in small steps down to grasp height
    z_goal = cube[2] + GRASP_H
    z_now  = target[2]
    while z_now - z_goal > 1e-6:
        z_now = max(z_goal, z_now - DESCENT_STEP)
        target = np.array([cube[0], cube[1], z_now], dtype=np.float32)
        move_converge(env, target, yaw=0.0, settle=6)
    print_status(env, "[descended to grasp]")

    # 3) close gripper and lock it for a short while
    env._set_gripper(0.0)           # close
    # engage your lock if you added one; otherwise just wait
    wait_steps(120)

    # 4) lift
    target = np.array([cube[0], cube[1], cube[2] + LIFT_H], dtype=np.float32)
    move_converge(env, target, yaw=0.0, settle=24)
    print_status(env, "[after lift]")
    wait_steps(240)

    print("Holding pose. Press Ctrl+C in the terminal to exit, or close the GUI window.")
    try:
        while True:
            p.stepSimulation()
            time.sleep(1/240)
    except KeyboardInterrupt:
        pass

    env.close()
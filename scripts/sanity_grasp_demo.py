import time, numpy as np, pybullet as p
import os, sys
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ_ROOT not in sys.path: sys.path.insert(0, PROJ_ROOT)
from src.envs.pick_place_env import PickPlaceEnv

def step_wait(n=60):  # ~0.25s at 240Hz
    for _ in range(n):
        p.stepSimulation()
        time.sleep(1/240)

def move_settle(env: PickPlaceEnv, xyz, yaw=0.0, settle=18):
    env._move_ee_abs(np.asarray(xyz, np.float32), yaw=yaw)
    step_wait(settle)

def status(env, tag):
    ee, _ = env._get_ee_pose()
    cube = env._get_cube_pose()
    print(f"{tag}  ee={ee.round(3)}  cube={cube.round(3)}  z_gap={(ee[2]-cube[2]):.3f}")

def xy_center_over_cube(env: PickPlaceEnv, yaw=0.0, passes=8, gain=0.6, hover_h=0.08):
    """Iteratively center EE above cube in XY while staying at hover height."""
    for i in range(passes):
        ee,_  = env._get_ee_pose()
        cube  = env._get_cube_pose()
        tgt   = ee.copy()
        err   = cube[:2] - ee[:2]
        tgt[:2] += gain * err  # damped nudge toward cube center
        tgt[2]  = cube[2] + hover_h
        move_settle(env, tgt, yaw=yaw, settle=8)

def try_one_grasp(env: PickPlaceEnv, yaw=0.0,
                  hover_h=0.08, grasp_h=0.033,
                  descend_step=0.006, micro_lift=0.02):
    """Return True on successful micro-lift (object came up), else False."""
    # 1) hover above cube + XY centering
    cube0 = env._get_cube_pose()
    hover = np.array([cube0[0], cube0[1], cube0[2] + hover_h], np.float32)
    status(env, "[hover start]")
    move_settle(env, hover, yaw=yaw, settle=24)
    xy_center_over_cube(env, yaw=yaw, passes=8, gain=0.6, hover_h=hover_h)
    status(env, "[hover centered]")

    # 2) descend to grasp height
    z_goal = env._get_cube_pose()[2] + grasp_h
    while True:
        ee,_ = env._get_ee_pose()
        if ee[2] <= z_goal + 1e-6: break
        next_z = max(z_goal, ee[2] - descend_step)
        move_settle(env, [ee[0], ee[1], next_z], yaw=yaw, settle=6)
    status(env, "[at grasp height]")

    # 3) close and settle
    env._set_gripper(0.0)
    step_wait(90)

    # 4) micro-lift to verify grasp
    cube_before = env._get_cube_pose()
    target = np.array([cube_before[0], cube_before[1], cube_before[2] + micro_lift], np.float32)
    move_settle(env, target, yaw=yaw, settle=24)
    cube_after = env._get_cube_pose()
    lifted = (cube_after[2] - cube_before[2]) > 0.008  # >8 mm moved up

    print(f"[grasp check] lifted={lifted}  dz={cube_after[2]-cube_before[2]:.3f} m")
    return lifted

if __name__ == "__main__":
    env = PickPlaceEnv(render_mode="human", domain_randomization=False)
    env.set_easy_mode(True)
    env.set_success_mode("lift")
    obs, _ = env.reset()

    # parameters you can tweak quickly
    HOVER_H = 0.08
    GRASP_H = 0.033     # 3.3 cm above table → clear contact
    DZ      = 0.006     # 6 mm descent increments
    MICRO   = 0.02      # 2 cm micro-lift validation

    # Try up to two orientations (yaw 0°, then 90°) so finger gap aligns with cube
    success = try_one_grasp(env, yaw=0.0,  hover_h=HOVER_H, grasp_h=GRASP_H, descend_step=DZ, micro_lift=MICRO)
    if not success:
        print("[retry] reopening and trying again with yaw=+90°")
        env._set_gripper(0.06); step_wait(45)
        success = try_one_grasp(env, yaw=np.pi/2, hover_h=HOVER_H, grasp_h=GRASP_H, descend_step=DZ, micro_lift=MICRO)

    # Final lift if succeeded
    if success:
        cube_now = env._get_cube_pose()
        final = np.array([cube_now[0], cube_now[1], cube_now[2] + 0.18], np.float32)
        move_settle(env, final, yaw=0.0, settle=48)
        status(env, "[after full lift]")
        print("✅ Success: holding cube. (Ctrl+C to exit)")
    else:
        print("❌ Grasp failed twice. Try: GRASP_H=0.030, increase finger force, or reduce cube mass slightly.")

    try:
        while True:
            p.stepSimulation()
            time.sleep(1/240)
    except KeyboardInterrupt:
        pass

    env.close()
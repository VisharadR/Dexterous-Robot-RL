import os, math, time, numpy as np
import gymnasium as gym
from gymnasium import spaces


import pybullet as p
import pybullet_data


class PickPlaceEnv(gym.Env):
    """
    Minimal PyBullet pick-and-place env (Franka-like arm with a simple gripper).
    Observation: low-dim (ee pose + object pose).
    Action: 3D ee delta + open/close gripper (4-dim).
    Reward: +1 when cube is within goal zone & lifted above table.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # action: [dx, dy, dz, grip] where grip in [-1,1] (close / open)
        self.action_space = spaces.Box(low=np.array([-0.02, -0.02, -0.02, -1.0]),
                                       high=np.array([ 0.02,  0.02,  0.02,  1.0]),
                                       dtype=np.float32)

        # obs: [ee(xyz), ee(yaw), obj(xyz)]
        high = np.array([np.inf]*7, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.physics_client = None
        self.time_step = 1./240.
        self.max_steps = 200
        self.step_count = 0

        # internal ids
        self.robot_id = None
        self.ee_link_id = None
        self.table_id = None
        self.cube_id = None
        self.goal_xy = np.array([0.5, 0.0], dtype=np.float32)  # simple goal on table

        self._connect()

    # ---------- setup ----------
    def _connect(self):
        if self.render_mode == "human":
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        p.setTimeStep(self.time_step)
        p.setGravity(0, 0, -9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

    def _load_world(self):
        p.resetSimulation()
        p.setTimeStep(self.time_step)
        p.setGravity(0, 0, -9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # plane + table
        p.loadURDF("plane.urdf")
        table_shift = [0.5, 0.0, -0.65]
        self.table_id = p.loadURDF("table/table.urdf", basePosition=table_shift)

        # robot: Franka Panda (in pybullet_data)
        robot_start_pos = [0.0, 0.0, 0.0]
        robot_start_ori = p.getQuaternionFromEuler([0, 0, 0])
        self.robot_id = p.loadURDF("franka_panda/panda.urdf",
                                   basePosition=robot_start_pos,
                                   baseOrientation=robot_start_ori,
                                   useFixedBase=True)

        # end-effector link index (panda_hand)
        self.ee_link_id = 11  # panda_hand link index in the URDF

        # open the gripper initially
        self._set_gripper(opening=0.04)

        # cube to pick
        cube_start = [0.5, 0.05, 0.02]  # on table
        self.cube_id = p.loadURDF("cube_small.urdf", basePosition=cube_start,
                                  globalScaling=1.0)

        # goal marker (visual only)
        self.goal_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.05, length=0.001,
                                               rgbaColor=[0, 1, 0, 0.5])
        self.goal_body = p.createMultiBody(baseMass=0,
                                           baseCollisionShapeIndex=-1,
                                           baseVisualShapeIndex=self.goal_visual,
                                           basePosition=[self.goal_xy[0], self.goal_xy[1], 0.02])

        # move EE to a good start pose
        self._move_ee_abs([0.4, 0.0, 0.3], yaw=0.0)

    # ---------- helpers ----------
    def _get_ee_pose(self):
        state = p.getLinkState(self.robot_id, self.ee_link_id)
        pos = np.array(state[0], dtype=np.float32)
        euler = p.getEulerFromQuaternion(state[1])
        yaw = np.float32(euler[2])
        return pos, yaw

    def _get_cube_pose(self):
        pos, orn = p.getBasePositionAndOrientation(self.cube_id)
        return np.array(pos, dtype=np.float32)

    def _ik(self, target_pos, yaw=0.0):
        # Keep fixed orientation (downward facing hand)
        target_ori = p.getQuaternionFromEuler([math.pi, 0, yaw])
        joint_poses = p.calculateInverseKinematics(self.robot_id, self.ee_link_id,
                                                   target_pos, target_ori,
                                                   maxNumIterations=200,
                                                   residualThreshold=1e-3)
        return joint_poses

    def _move_ee_abs(self, xyz, yaw=0.0):
        joint_poses = self._ik(xyz, yaw)
        for j, q in enumerate(joint_poses[:7]):  # first 7 joints are arm
            p.setJointMotorControl2(self.robot_id, j, p.POSITION_CONTROL, q, force=200)
        for _ in range(10):
            p.stepSimulation()

    def _set_gripper(self, opening: float):
        # panda_finger_joint1 = 9, panda_finger_joint2 = 10
        opening = float(np.clip(opening, 0.0, 0.08))
        p.setJointMotorControl2(self.robot_id, 9, p.POSITION_CONTROL, opening, force=50)
        p.setJointMotorControl2(self.robot_id,10, p.POSITION_CONTROL, opening, force=50)
        for _ in range(5):
            p.stepSimulation()

    # ---------- gym API ----------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self._load_world()

        # randomize cube XY a bit for diversity
        jitter = np.random.uniform(low=[-0.07, -0.07], high=[0.07, 0.07])
        cube_pos = np.array([0.5, 0.05, 0.02]) + np.append(jitter, 0.0)
        p.resetBasePositionAndOrientation(self.cube_id, cube_pos, [0,0,0,1])

        obs = self._get_obs()
        info = {}
        return obs, info

    def _get_obs(self):
        ee_pos, ee_yaw = self._get_ee_pose()
        cube_pos = self._get_cube_pose()
        obs = np.concatenate([ee_pos, [ee_yaw], cube_pos]).astype(np.float32)
        return obs

    def step(self, action):
        self.step_count += 1
        action = np.clip(action, self.action_space.low, self.action_space.high)
        dpos = action[:3]
        grip_cmd = action[3]  # -1 close, +1 open

        ee_pos, ee_yaw = self._get_ee_pose()
        target = ee_pos + dpos
        # workspace clamp
        target[0] = np.clip(target[0], 0.2, 0.7)
        target[1] = np.clip(target[1], -0.3, 0.3)
        target[2] = np.clip(target[2], 0.02, 0.5)

        self._move_ee_abs(target, yaw=ee_yaw)

        # simple gripper mapping
        if grip_cmd < 0:
            self._set_gripper(0.0)   # close
        else:
            self._set_gripper(0.06)  # open

        # step physics a bit
        for _ in range(4):
            p.stepSimulation()

        obs = self._get_obs()
        reward, terminated = self._compute_reward_done()
        truncated = self.step_count >= self.max_steps
        info = {}
        return obs, reward, terminated, truncated, info

    def _compute_reward_done(self):
        cube = self._get_cube_pose()
        # success if cube is near goal XY and lifted
        near_goal = np.linalg.norm(cube[:2] - self.goal_xy) < 0.08
        lifted = cube[2] > 0.15
        reward = 0.0
        if lifted:
            reward += 0.5
        if near_goal and lifted:
            reward += 0.5
        done = bool(near_goal and lifted)
        # small time penalty to encourage efficiency
        reward -= 0.001
        return float(reward), done

    def render(self):
        if self.render_mode == "human":
            time.sleep(self.time_step)
        else:
            # return an RGB array if needed later
            width, height, view, proj, _, _ = p.getDebugVisualizerCamera()
            img = p.getCameraImage(width, height)[2]
            return np.array(img)

    def close(self):
        if p.isConnected():
            p.disconnect()

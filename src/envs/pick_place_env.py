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

    def __init__(self, render_mode=None, domain_randomization: bool = False):
        super().__init__()
        self.render_mode = render_mode
        self.domain_randomization = bool(domain_randomization)

        # speed knobs
        self.action_repeat = 2         # repeat same action this many frames
        self.substeps = 1              # physics substeps per frame (keep low for speed)
        self.fast_dynamics = True      # use faster (less accurate) solver
        self.control_settle_steps = 60   # 6–12 works well

        # Small deltas for stable control
        self.action_space = spaces.Box(
            low=np.array([-0.01, -0.01, -0.01, -1.0], dtype=np.float32),
            high=np.array([ 0.01,  0.01,  0.01,  1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # [ee(x,y,z), ee_yaw, cube(x,y,z)]
        high = np.array([np.inf] * 7, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # Sim config
        self.time_step = 1.0 / 240.0
        self.max_steps = 300
        self.step_count = 0

        # IDs
        self.physics_client = None
        self.robot_id = None
        self.ee_link_id = 11  # panda_hand link index in pybullet_data/franka_panda
        self.arm_joint_indices = list(range(7))  # 0..6 are Panda arm joints
        self.joint_lower = None
        self.joint_upper = None
        self.joint_range = None
        self.joint_rest  = np.array([0.0, -0.4, 0.0, -1.8, 0.0, 1.8, 0.8], dtype=np.float32)  # elbow-down, wrist bent
        self.joint_damping = [0.05]*7  # light damping helps IK stability
        self.table_id = None
        self.cube_id = None

        # Goal and curriculum
        self.goal_xy = np.array([0.5, 0.0], dtype=np.float32)
        self._easy_mode = True
        self._easy_jitter_xy = 0.02  # 2 cm (start easy)
        self._hard_jitter_xy = 0.07  # 7 cm (later)
        self._grasp_locked_steps = 0         # counts down when we’ve closed near the cube
        self._grasp_lock_horizon = 30        # keep closed ~30 steps after closing near object
        self.approach_height = 0.035         # target hover height above cube before closing


        # Success logic (can be staged)
        self.success_mode = "lift" # "and" | "and" | "lift" | "near"
        self.lift_thresh = 0.06   # was 0.12; stage-1 target ~6 cm
        self.success_bonus = 8.0  # a bit more pop when we do lift

        # Reward weights / thresholds
        self.w_reach = 0.25
        self.w_goal = 0.25
        self.w_lift = 2.0
        self.w_time = 0.0005
        self.success_bonus = 6.0
        self.near_thresh = 0.10   # start lenient; tighten later
        self.lift_thresh = 0.12   # start lenient; tighten later

        # For potential-based shaping
        self._last_dist_ee_cube = None
        self._last_dist_cube_goal = None

        self._connect()

    # --------------------------------------------------------------------- #
    # Setup
    # --------------------------------------------------------------------- #
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

        if self.fast_dynamics:
        # fewer solver iterations, faster contact handling
            p.setPhysicsEngineParameter(
                fixedTimeStep=self.time_step,
                numSolverIterations=20,           # default ~50
                numSubSteps=self.substeps,        # we also control substeps externally
                collisionFilterMode=1,
                enableConeFriction=0,
                deterministicOverlappingPairs=1,
            )

        # Ground + table
        p.loadURDF("plane.urdf")
        table_shift = [0.5, 0.0, -0.65]
        self.table_id = p.loadURDF("table/table.urdf", basePosition=table_shift)

        # Robot (fixed base)
        flags = p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT
        self.robot_id = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=[0.0, 0.0, 0.0],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
            flags=flags
        )

        lower, upper = [], []
        for j in self.arm_joint_indices:
            info = p.getJointInfo(self.robot_id, j)
            lower.append(info[8])  # lower limit
            upper.append(info[9])  # upper limit
        self.joint_lower = np.array(lower, dtype=np.float32)
        self.joint_upper = np.array(upper, dtype=np.float32)
        self.joint_range = (self.joint_upper - self.joint_lower)

        # Initial gripper open
        self._set_gripper(opening=0.05)

        # Cube to manipulate
        self.cube_id = p.loadURDF("cube_small.urdf", basePosition=[0.5, 0.05, 0.02], globalScaling=1.0)

        # --- Improve graspability physics ---
        # lighter cube + higher friction, no bounce
        p.changeDynamics(self.cube_id, -1, mass=0.015,
                         lateralFriction=1.4, restitution=0.0,
                         rollingFriction=0.002, spinningFriction=0.002)
        # table friction so cube doesn't skate
        p.changeDynamics(self.table_id, -1, lateralFriction=1.0, restitution=0.0)

        # Visual goal ring (for GUI)
        goal_vis = p.createVisualShape(
            p.GEOM_CYLINDER, radius=0.05, length=0.001, rgbaColor=[0, 1, 0, 0.5]
        )
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=goal_vis,
            basePosition=[self.goal_xy[0], self.goal_xy[1], 0.02],
        )

        # Move EE to start pose
        self._move_ee_abs([0.4, 0.0, 0.30], yaw=0.0)

        # Mild domain randomization (camera jitter in GUI)
        if self.domain_randomization and self.render_mode == "human":
            try:
                p.resetDebugVisualizerCamera(
                    cameraDistance=1.2,
                    cameraYaw=np.random.uniform(-15, 15),
                    cameraPitch=np.random.uniform(-20, -5),
                    cameraTargetPosition=[0.45, 0.0, 0.1],
                )
            except Exception:
                pass

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
    def _get_ee_pose(self):
        state = p.getLinkState(self.robot_id, self.ee_link_id)
        pos = np.array(state[0], dtype=np.float32)
        yaw = np.float32(p.getEulerFromQuaternion(state[1])[2])
        return pos, yaw

    def _get_cube_pose(self):
        pos, _ = p.getBasePositionAndOrientation(self.cube_id)
        return np.array(pos, dtype=np.float32)

    def _ik(self, target_pos, yaw=0.0):
        # wrist-down orientation
        target_ori = p.getQuaternionFromEuler([math.pi, 0, yaw])
        # Fallback if limits not initialized yet
        if self.joint_lower is None:
            return p.calculateInverseKinematics(
                self.robot_id, self.ee_link_id, target_pos, target_ori,
                maxNumIterations=300, residualThreshold=1e-3
            )

        q = p.calculateInverseKinematics(
            bodyUniqueId=self.robot_id,
            endEffectorLinkIndex=self.ee_link_id,
            targetPosition=target_pos,
            targetOrientation=target_ori,
            lowerLimits=self.joint_lower.tolist(),
            upperLimits=self.joint_upper.tolist(),
            jointRanges=self.joint_range.tolist(),
            restPoses=self.joint_rest.tolist(),
            # jointDamping=self.joint_damping,
            maxNumIterations=300,
            residualThreshold=1e-3
        )
        return q

    def _move_ee_abs(self, xyz, yaw=0.0):
        joint_poses = self._ik(xyz, yaw)
        for j, q in enumerate(joint_poses[:7]):  # first 7 joints are arm
            p.setJointMotorControl2(self.robot_id, j, p.POSITION_CONTROL, targetPosition=q, force=300, positionGain=0.3, velocityGain=1.0, maxVelocity=2.0)
        for _ in range(self.control_settle_steps):
            p.stepSimulation()

    def _set_gripper(self, opening: float):
        # Stronger finger force helps lifting
        force = 220
        opening = float(np.clip(opening, 0.0, 0.08))
        p.setJointMotorControl2(self.robot_id, 9, p.POSITION_CONTROL, opening, force=force)
        p.setJointMotorControl2(self.robot_id, 10, p.POSITION_CONTROL, opening, force=force)
        for _ in range(5):
            p.stepSimulation()

    # --------------------------------------------------------------------- #
    # Gym API
    # --------------------------------------------------------------------- #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self._load_world()
        self._grasp_locked_steps = 0

        # Curriculum-controlled jitter
        jitter_xy = self._easy_jitter_xy if self._easy_mode else self._hard_jitter_xy
        jitter = np.random.uniform(low=[-jitter_xy, -jitter_xy], high=[jitter_xy, jitter_xy])
        cube_pos = np.array([0.5, 0.05, 0.03]) + np.append(jitter, 0.0)

        # Optional small sensor noise
        if self.domain_randomization:
            noise = np.random.normal(0.0, 0.001, size=3)  # ~1 mm
            cube_pos = cube_pos + noise

        p.resetBasePositionAndOrientation(self.cube_id, cube_pos, [0, 0, 0, 1])

        # Initialize potentials for shaping
        ee_pos, _ = self._get_ee_pose()
        self._last_dist_ee_cube = float(np.linalg.norm(ee_pos - cube_pos))
        self._last_dist_cube_goal = float(np.linalg.norm(cube_pos[:2] - self.goal_xy))

        obs = self._get_obs()
        info = {}
        return obs, info

    def _get_obs(self):
        ee_pos, ee_yaw = self._get_ee_pose()
        cube_pos = self._get_cube_pose()
        return np.concatenate([ee_pos, [ee_yaw], cube_pos]).astype(np.float32)

    def step(self, action):
        self.step_count += 1
        action = np.clip(action, self.action_space.low, self.action_space.high)
        dpos = action[:3]; grip_cmd = action[3]

        ee_pos, ee_yaw = self._get_ee_pose()
        target = ee_pos + dpos
        target[0] = np.clip(target[0], 0.2, 0.7)
        target[1] = np.clip(target[1], -0.3, 0.3)
        target[2] = np.clip(target[2], 0.012, 0.5)

        # set joint targets once
        self._move_ee_abs(target, yaw=ee_yaw)

        # auto-close heuristic + lock (keep your existing logic)
        cube_now = self._get_cube_pose()
        xy_dist = np.linalg.norm((target[:2] - cube_now[:2]))
        z_above = target[2] - cube_now[2]
        # if xy_dist < 0.025 and 0.02 <= z_above <= 0.06:
        if xy_dist < 0.03 and z_above > 0.05:
            # gently descend toward grasp height
            target[2] -= 0.01 # 8 mm per control step
            self._move_ee_abs(target, yaw=ee_yaw) # re-send pose and settle
            grip_cmd = -1.0
            self._grasp_locked_steps = max(self._grasp_locked_steps, self._grasp_lock_horizon)
        if self._grasp_locked_steps > 0:
            grip_cmd = -1.0
            self._grasp_locked_steps -= 1

        # map grip once
        if grip_cmd < 0: self._set_gripper(0.0)
        else:            self._set_gripper(0.06)

        # physics stepping (fast)
        for _ in range(self.action_repeat):
            for __ in range(self.substeps):
                p.stepSimulation()

        obs = self._get_obs()
        reward, success = self._compute_reward_done()
        truncated = self.step_count >= self.max_steps

        # update potentials (unchanged)
        ee_pos2, _ = self._get_ee_pose()
        cube2 = self._get_cube_pose()
        self._last_dist_ee_cube = float(np.linalg.norm(ee_pos2 - cube2))
        self._last_dist_cube_goal = float(np.linalg.norm(cube2[:2] - self.goal_xy))

        info = {"success": success}
        return obs, reward, bool(success), truncated, info
    
    
    # --------------------------------------------------------------------- #
    # Reward & success
    # --------------------------------------------------------------------- #
    def _compute_reward_done(self):
        ee_pos, _ = self._get_ee_pose()
        cube = self._get_cube_pose()

        dist_ee_cube = np.linalg.norm(ee_pos - cube)
        dist_cube_goal = np.linalg.norm(cube[:2] - self.goal_xy)

        # Potential-based improvements (positive if improving)
        d_reach = 0.0 if self._last_dist_ee_cube is None else (self._last_dist_ee_cube - dist_ee_cube)
        d_goal = 0.0 if self._last_dist_cube_goal is None else (self._last_dist_cube_goal - dist_cube_goal)

        reach_reward = self.w_reach * d_reach
        goal_reward = self.w_goal * d_goal
        lift_reward = self.w_lift * max(0.0, cube[2] - 0.03)  # absolute lift above table

        # Encourage good approach: be directly above cube at a small hover height
        xy_err = np.linalg.norm((ee_pos[:2] - cube[:2]))
        z_err  = abs((cube[2] + self.approach_height) - ee_pos[2])

        approach_reward = 0.15 * ( -xy_err ) + 0.15 * ( -z_err )


        # Simple grasp bonus: if closed and slightly lifted
        finger_l = p.getJointState(self.robot_id, 9)[0]
        finger_r = p.getJointState(self.robot_id, 10)[0]
        gripper_closed = (finger_l < 0.01 and finger_r < 0.01)
        lift_bias = 0.0
        if gripper_closed:
            lift_bias = 0.5 * max(0.0, cube[2] - 0.03)  # stronger reward once grasped
        grasp_bonus = 0.2 if (gripper_closed and cube[2] > 0.05) else 0.0

        # Success checks (staged)
        near_goal = dist_cube_goal < self.near_thresh
        lifted = cube[2] > self.lift_thresh

        if self.success_mode == "and":
            success = (near_goal and lifted)
        elif self.success_mode == "lift":
            success = lifted
        else:  # "near"
            success = near_goal

        reward = (reach_reward + goal_reward + lift_reward +
                    grasp_bonus + approach_reward + lift_bias -
                    self.w_time)
        if success:
            reward += self.success_bonus

        return float(reward), bool(success)

    # --------------------------------------------------------------------- #
    # Config helpers
    # --------------------------------------------------------------------- #
    def set_easy_mode(self, flag: bool):
        """Enable/disable curriculum easy mode (smaller object jitter)."""
        self._easy_mode = bool(flag)

    def set_success_mode(self, mode: str):
        """Set success criterion: 'and' | 'lift' | 'near'."""
        assert mode in ("and", "lift", "near")
        self.success_mode = mode

    def set_thresholds(self, near_thresh: float = None, lift_thresh: float = None):
        """Adjust success thresholds (meters)."""
        if near_thresh is not None:
            self.near_thresh = float(near_thresh)
        if lift_thresh is not None:
            self.lift_thresh = float(lift_thresh)

    # --------------------------------------------------------------------- #
    # Rendering / teardown
    # --------------------------------------------------------------------- #
    def render(self):
        if self.render_mode == "human":
            time.sleep(self.time_step)
        else:
            # Return an RGB frame if needed
            width, height, _, _, _, _ = p.getDebugVisualizerCamera()
            img = p.getCameraImage(width, height)[2]
            return np.array(img)

    def close(self):
        if p.isConnected():
            p.disconnect()
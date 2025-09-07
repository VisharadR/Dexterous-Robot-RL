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

        # Motion smoothing
        self.ee_filter_alpha = 0.35 # 0..1 (higher = snappier); start ~0.35
        self.ee_prev_target = np.array([0.4, 0.0, 0.30], dtype=np.float32)
        self.cart_rate_limit = np.array([0.012, 0.012, 0.010], dtype=np.float32)

        # PD for joint motors (milder)
        self.joint_force = 220
        self.joint_pos_gain = 0.15
        self.joint_vel_gain = 0.6
        self.joint_max_vel = 1.0

        # auto-descend gating 
        self._descend_cooldown = 0
        self._descend_cooldown_max = 12


        # IDs
        self.physics_client = None
        self.robot_id = None
        self.ee_link_id = 8  # panda_hand link index in pybullet_data/franka_panda
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

    def _find_link_index(self, names=("panda_hand", "panda_hand_tcp", "panda_link8")):
        for j in range(p.getNumJoints(self.robot_id)):
            link_name = p.getJointInfo(self.robot_id, j)[12].decode()
            if link_name in names:
                return j
        # fallback: keep existing value
        return self.ee_link_id

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
            basePosition=[0,0,0],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
            flags=flags
        )
        self.ee_link_id = 11 #self._find_link_index()
        self._auto_calibrate_ee_link()

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
        # seed the smoothing state
        ee0, _ = self._get_ee_pose()
        self.ee_prev_target = ee0.astype(np.float32)

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
    def _candidate_ee_indices(self):
        # Find plausible EE link indices by name, plus a few nearby fallbacks.
        names= ("panda_hand", "panda_hand_tcp", "panda_link8", "hand", "tool0")
        found = []
        for j in range(p.getNumJoints(self.robot_id)):
            link_name = p.getJointInfo(self.robot_id, j)[12].decode()
            joint_name = p.getJointInfo(self.robot_id, j)[1].decode()
            if any(n in link_name or n in joint_name for n in names):
                found.append(j)
        # common pandas: 8..12 often relevant - add as fallbacks (dedup)
        for j in (8, 9, 10, 11, 12):
            if 0 <= j < p.getNumJoints(self.robot_id) and j not in found:
                found.append(j)
        return found

        
    def _get_ee_pose(self):
        state = p.getLinkState(self.robot_id, self.ee_link_id)
        pos = np.array(state[0], dtype=np.float32)
        yaw = np.float32(p.getEulerFromQuaternion(state[1])[2])
        return pos, yaw

    def _get_cube_pose(self):
        pos, _ = p.getBasePositionAndOrientation(self.cube_id)
        return np.array(pos, dtype=np.float32)
    
    def _ik_nullspace(self, target_pos, target_ori):
        # Build limits if needed
        if self.joint_lower is None:
            lowers, uppers = [], []
            for j in range(7):
                ji = p.getJointInfo(self.robot_id, j)
                lowers.append(ji[8])
                uppers.append(ji[9])
            self.joint_lower = lowers
            self.joint_upper = uppers
            self.joint_range = [u - l for l, u in zip(lowers, uppers)]
            self.joint_rest = [0.0, -0.6, 0.0, -2.2, 0.0, 2.2, 0.8]
        return p.calculateInverseKinematics(
            bodyUniqueId=self.robot_id,
            endEffectorLinkIndex=self.ee_link_id,
            targetPosition=target_pos,
            targetOrientation=target_ori,
            lowerLimits=self.joint_lower,
            upperLimits=self.joint_upper,
            jointRanges=self.joint_range,
            restPoses=self.joint_rest,
            maxNumIterations=300,
            residualThreshold=1e-3,
        )
    
    def _auto_calibrate_ee_link(self):
        # Pick th e link index that actually tracks a test target best.
        target = [0.45, 0.05, 0.25]
        target_ori = p.getQuaternionFromEuler([math.pi, 0, 0.0])
        best = None
        best_err = 1e9

        # save current states to restote later
        saved = [p.getJointState(self.robot_id, j)[0] for j in range(7)]

        for cand in self._candidate_ee_indices():
            # temporarily use this candidate as EE link for IK
            self.ee_link_id = cand
            q = self._ik_nullspace(target, target_ori)

            # apply q (temporarily) and step a bit to get forward pose
            for j, qj in enumerate(q[:7]):
                p.resetJointState(self.robot_id, j, qj)
            for _ in range(10):
                p.stepSimulation()

            pos, _ = p.getLinkState(self.robot_id, cand)[:2]
            err = np.linalg.norm(np.array(pos) - np.array(target))

            if err < best_err:
                best_err, best = err, cand

        # restore original states
        for j, qj in enumerate(saved):
            p.resetJointState(self.robot_id, j, qj)
        for _ in range(2):
            p.stepSimulation()

        # lock in the best candidate
        if best is not None:
            self.ee_link_id = best
            # print once so you can see it
            print(f"[env] Chosed ee_link_id = {best} (err={best_err:.3f}m)")


    def _ik(self, target_pos, yaw=0.0):
        # wrist-down orientation
        target_ori = p.getQuaternionFromEuler([math.pi, 0, yaw])
        # Fallback if limits not initialized yet
        if self.joint_lower is None:
            lower, upper = [], []
            for j in range(7):
                ji = p.getJointInfo(self.robot_id, j)
                lower.append(ji[8])
                upper.append(ji[9])
            self.joint_lower = lower
            self.joint_upper = upper
            self.joint_range = [u - l for l, u in zip(lower, upper)]
            # a clearly bent elbow/wrist
            self.joint_rest = [0.0, -0.6, 0.0, -2.2, 0.0, 2.2, 0.8]
        # if self.joint_lower is None:
        #     return p.calculateInverseKinematics(
        #         self.robot_id, self.ee_link_id, target_pos, target_ori,
        #         maxNumIterations=300, residualThreshold=1e-3
        #     )

        q = p.calculateInverseKinematics(
            bodyUniqueId=self.robot_id,
            endEffectorLinkIndex=self.ee_link_id,
            targetPosition=target_pos,
            targetOrientation=target_ori,
            # lowerLimits=self.joint_lower.tolist(),
            # upperLimits=self.joint_upper.tolist(),
            # jointRanges=self.joint_range.tolist(),
            # restPoses=self.joint_rest.tolist(),
            lowerLimits=self.joint_lower,
            upperLimits=self.joint_upper,
            jointRanges=self.joint_range,
            restPoses=self.joint_rest,
            # solver=p.IK_DLS,    #robust
            # jointDamping=self.joint_damping,
            maxNumIterations=300,
            residualThreshold=1e-3
        )
        return q
    
    def _servo_to_pose(self, target_xyz, yaw=0.0, blend_steps=6):
        # Smoothly move toward IK pose with small steps to avoid oscillation
        # 1) rate-limit the cartesian change
        delta = np.asarray(target_xyz, dtype=np.float32) - self.ee_prev_target
        delta = np.clip(delta, -self.cart_rate_limit, self.cart_rate_limit)
        filtered = self.ee_prev_target + delta
        # 2) low_pass toward the filtered target
        filtered - self.ee_prev_target + self.ee_filter_alpha * ( filtered - self.ee_prev_target)
        self.ee_prev_target = filtered.copy()

        # 3) compute IK once for this micro-target
        q = self._ik(filtered, yaw)

        # 4) blend in joint-space over a few tiny steps (prevents snap)
        for k in range(blend_steps):
            w = (k+1)/blend_steps
            for j, qj in enumerate(q[:7]):
                p.setJointMotorControl2(
                    self.robot_id, j, p.POSITION_CONTROL,
                    targetPosition=qj,
                    force=self.joint_force,
                    positionGain=self.joint_pos_gain,
                    velocityGain=self.joint_vel_gain,
                    maxVelocity=self.joint_max_vel
                )
            p.stepSimulation()

    def _move_ee_abs(self, xyz, yaw=0.0):
        self._servo_to_pose(np.asarray(xyz, dtype=np.float32), yaw=yaw, blend_steps=6)
        # target_ori = p.getQuaternionFromEuler([math.pi, 0, yaw])
        # q = self._ik_nullspace(xyz, target_ori)
        # joint_poses = self._ik(xyz, yaw)
        # for j, q in enumerate(joint_poses[:7]):  # first 7 joints are arm
        #     p.setJointMotorControl2(self.robot_id, j, p.POSITION_CONTROL, targetPosition=q, force=300, positionGain=0.3, velocityGain=1.0, maxVelocity=2.0)
        # for _ in range(self.control_settle_steps):
        #     p.stepSimulation()

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
        # --- bookkeeping ---
        self.step_count += 1

        # --- action parsing ---
        action = np.clip(action, self.action_space.low, self.action_space.high)
        dpos = action[:3].astype(np.float32)
        grip_cmd = float(action[3])

        # --- current poses ---
        ee_pos, ee_yaw = self._get_ee_pose()
        cube_now = self._get_cube_pose()

        # --- target pose (workspace clamp + permissive Z) ---
        target = (ee_pos + dpos).astype(np.float32)
        target[0] = np.clip(target[0], 0.20, 0.70)
        target[1] = np.clip(target[1], -0.30, 0.30)
        target[2] = np.clip(target[2], 0.012, 0.50)  # allow the wrist to get close to the table

        # --- approach diagnostics ---
        xy_dist = float(np.linalg.norm(target[:2] - cube_now[:2]))
        z_above = float(target[2] - cube_now[2])

        # --- gated auto-descend (prevents oscillation) ---
        if getattr(self, "_descend_cooldown", None) is None:
            self._descend_cooldown = 0
        if self._descend_cooldown > 0:
            self._descend_cooldown -= 1
        if xy_dist < 0.030 and z_above > 0.055 and self._descend_cooldown == 0:
            target[2] = max(0.012, target[2] - 0.010)  # 1 cm nudge down
            self._descend_cooldown = getattr(self, "_descend_cooldown_max", 12)

        # --- keep yaw stable near the cube to avoid wrist wiggle ---
        if xy_dist < 0.060:
            ee_yaw = 0.0

        # --- auto-close heuristic + grasp-lock (helps early learning) ---
        if np.linalg.norm((target[:2] - cube_now[:2])) < 0.025 and 0.020 <= (target[2] - cube_now[2]) <= 0.060:
            grip_cmd = -1.0  # try to grasp when nicely aligned
            self._grasp_locked_steps = max(getattr(self, "_grasp_locked_steps", 0),
                                        getattr(self, "_grasp_lock_horizon", 30))
        if getattr(self, "_grasp_locked_steps", 0) > 0:
            grip_cmd = -1.0
            self._grasp_locked_steps -= 1

        # --- gripper command ---
        if grip_cmd < 0.0:
            self._set_gripper(0.0)   # close
        else:
            self._set_gripper(0.06)  # open

        # --- smooth Cartesian move (your _move_ee_abs should be the servo version) ---
        self._move_ee_abs(target, yaw=ee_yaw)

        # --- observation, reward, termination ---
        obs = self._get_obs()
        reward, success = self._compute_reward_done()
        truncated = self.step_count >= self.max_steps
        done = bool(success)

        # --- update potentials for delta shaping (reach/goal improvements) ---
        ee_pos2, _ = self._get_ee_pose()
        cube2 = self._get_cube_pose()
        self._last_dist_ee_cube = float(np.linalg.norm(ee_pos2 - cube2))
        self._last_dist_cube_goal = float(np.linalg.norm(cube2[:2] - self.goal_xy))

        # --- info (light diag) ---
        info = {"success": done, "xy_dist": xy_dist, "z_above": z_above}

        return obs, float(reward), done, truncated, info
    
    
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
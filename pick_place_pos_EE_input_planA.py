# This code will attempt to have the EE pick cubes from the same predisclosed EE initial pos and drop at a vertically increasing EE final pos
# until user termination

#!/usr/bin/env python3

import numpy as np
from scipy.optimize import minimize

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


def q(a, b, c, d, e):
    return np.array([a, b, c, d, e], dtype=float)


def p(x, y, z):
    return np.array([x, y, z], dtype=float)


def fk_position(qv):
    q1, q2, q3, q4, q5 = qv

    s1, c1 = np.sin(q1), np.cos(q1)
    s2, c2 = np.sin(q2), np.cos(q2)
    s23, c23 = np.sin(q2 + q3), np.cos(q2 + q3)
    s234, c234 = np.sin(q2 + q3 + q4), np.cos(q2 + q3 + q4)

    x = (0.11257 * s2 + 0.0052 * s23 - 0.028 * c2 - 0.1349 * c23 - 0.1351 * c234 - 0.0306) * s1
    y = -0.11257 * s2 * c1 - 0.0052 * s23 * c1 + 0.028 * c2 * c1 + 0.1349 * c23 * c1 + 0.1351 * c234 * c1 + 0.0306 * c1 + 0.0452
    z = 0.028 * s2 + 0.1349 * s23 + 0.1351 * s234 + 0.11257 * c2 + 0.0052 * c23 + 0.119

    return np.array([x, y, z], dtype=float)


def solve_ik_position(
    target_xyz,
    q_seed,
    bounds,
    q_nominal,
    residual_tol=0.02,
    desired_tool_pitch=None,
    tool_pitch_weight=0.0
):
    """
    IK for position only, with an optional soft preference on tool pitch.

    Approximate tool pitch in the vertical plane:
        tool_pitch ~= q2 + q3 + q4

    desired_tool_pitch is in radians.
    More negative / more positive depends on your sign convention in sim
    """
    def objective(qv):
        pos_err = fk_position(qv) - target_xyz
        dq_seed = qv - q_seed
        dq_nom = qv - q_nominal

        cost = (
            1.0e4 * np.dot(pos_err, pos_err) +
            2.0e-1 * np.dot(dq_seed, dq_seed) +
            1.0e-4 * np.dot(dq_nom, dq_nom) +
            5.0e-2 * (qv[0] - q_seed[0])**2 +
            1.0e-2 * (qv[4] - q_seed[4])**2
        )

        if desired_tool_pitch is not None and tool_pitch_weight > 0.0:
            tool_pitch = qv[1] + qv[2] + qv[3]
            cost += tool_pitch_weight * (tool_pitch - desired_tool_pitch) ** 2

        return cost

    def clip_to_bounds(qv):
        q_clipped = qv.copy()
        for i, (lo, hi) in enumerate(bounds):
            q_clipped[i] = np.clip(q_clipped[i], lo, hi)
        return q_clipped

    candidate_seeds = []

    # stay close to current branch first
    candidate_seeds.append(q_seed.copy())

    for d in [0.05, -0.05, 0.10, -0.10]:
        for j in [0, 1, 2, 3]:
            q_try = q_seed.copy()
            q_try[j] += d
            candidate_seeds.append(clip_to_bounds(q_try))

    # fallback seeds
    candidate_seeds.append(q_nominal.copy())
    q_mid = np.array([(a + b) * 0.5 for a, b in bounds], dtype=float)
    candidate_seeds.append(q_mid)

    rng = np.random.default_rng(42)
    for _ in range(6):
        qr = np.array([rng.uniform(a, b) for a, b in bounds], dtype=float)
        candidate_seeds.append(qr)

    best_q = None
    best_residual = np.inf

    for q0 in candidate_seeds:
        result = minimize(
            objective,
            q0,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 400, 'ftol': 1.0e-10, 'disp': False}
        )

        if not result.success:
            continue

        q_sol = result.x
        residual = np.linalg.norm(fk_position(q_sol) - target_xyz)

        if residual < best_residual:
            best_residual = residual
            best_q = q_sol

    if best_q is None:
        raise RuntimeError(f"IK failed completely for target {target_xyz}")

    if best_residual > residual_tol:
        raise RuntimeError(
            f"IK residual too large ({best_residual:.4f} m) for target {target_xyz}"
        )

    return best_q   


class StackCubesFromSource(Node):
    def __init__(self):
        super().__init__('stack_cubes_from_source')

        # ---------------- BASIC SETTINGS ----------------
        self._dt = 0.04

        self._joint_names_expected = [
            "Shoulder_Rotation",
            "Shoulder_Pitch",
            "Elbow",
            "Wrist_Pitch",
            "Wrist_Roll",
        ]

        self._topic_keys = [
            "shoulder_rotation",
            "shoulder_pitch",
            "elbow",
            "wrist_pitch",
            "wrist_roll",
        ]

        # ---------------- GRIPPER SETTINGS ----------------
        self._GRIPPER_OPEN = 0.5
        self._GRIPPER_CLOSED = -0.3

        # ---------------- TOOL PITCH PREFERENCE ----------------
        # Approximate tool pitch = q2 + q3 + q4
        #
        # These are soft preferences used by IK.
        # Pick/place gets a stronger downward preference than travel.
        #
        # If the gripper angles the wrong way, flip the sign or adjust magnitude.
        # Good starting range is roughly -1.6 to -0.8 rad.
        self._DESIRED_TOOL_PITCH_PICK_PLACE = -1.35
        self._DESIRED_TOOL_PITCH_TRAVEL = -1.10

        self._TOOL_PITCH_WEIGHT_PICK_PLACE = 8.0
        self._TOOL_PITCH_WEIGHT_TRAVEL = 2.0

        # ---------------- HOME / NOMINAL POSE ----------------
        self._HOME_Q = q(
            0.0,
            -0.161,
            0.026,
            -0.0297,
            0.0
        )
        self._HOME_XYZ = fk_position(self._HOME_Q)

        # ---------------- JOINT LIMITS ----------------
        self._BOUNDS = [
            (-2.00,  2.00),
            (-1.57,  1.57),
            (-1.58,  1.58),
            (-1.57,  1.57),
            (-3.142, 3.142),
        ]

        # =====================================================
        # USER INPUT
        # =====================================================

        # Same source location for every cube
        self._SOURCE_PICK = p(-0.15, 0.25, 0.01)
        self._SOURCE_ABOVE = p(-0.15, 0.25, 0.10)

        # Disabled transit for now
        self._TRANSIT = None

        # Stack destination XY and base Z
        self._STACK_XY = np.array([0.15, 0.25], dtype=float)
        self._STACK_BASE_Z = 0.01

        # Geometry / planning constants
        self._CUBE_HEIGHT = 0.03
        self._APPROACH_CLEARANCE = 0.06

        # Segment durations
        self._T_MOVE = 2.5
        self._T_LOWER = 1.5
        self._T_GRIP = 1.0
        self._T_LIFT = 1.5
        self._T_TRANSIT = 1.6

        # For finite number, set this to cube stack #
        # If None, it runs until user termination
        self._MAX_CUBES = None
        # ---------------- CYCLE EXECUTION STATE ----------------
        self._placed_count = 0
        self._current_cycle_waypoints = []
        self._current_cycle_segments = []
        self._segment_index = 0
        self._segment_start_time = None
        self._finished = False
        self._last_logged_segment = None

        # ---------------- FEEDBACK STATE ----------------
        self._last_fb_pos = None
        self._last_fb_vel = None
        self._last_fb_time = None

        self._last_cmd_pos = None
        self._last_cmd_vel = None

        self._seen_joint_states = False
        self._printed_joint_state_names = False

        # Build first cycle
        self.build_next_cycle(start_from_home=True)

        # ---------------- ROS PUB/SUB ----------------
        self._cmd_pub = self.create_publisher(JointTrajectory, 'joint_cmds', 10)

        qos_joint_states = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )

        self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            qos_joint_states
        )

        # Debug publishers
        self._cmd_pos_pubs = {}
        self._fb_pos_pubs = {}
        self._pos_err_pubs = {}

        self._cmd_vel_pubs = {}
        self._fb_vel_pubs = {}
        self._vel_err_pubs = {}

        for key in self._topic_keys:
            self._cmd_pos_pubs[key] = self.create_publisher(Float64, f'/debug/{key}/cmd_pos', 10)
            self._fb_pos_pubs[key] = self.create_publisher(Float64, f'/debug/{key}/fb_pos', 10)
            self._pos_err_pubs[key] = self.create_publisher(Float64, f'/debug/{key}/pos_err', 10)

            self._cmd_vel_pubs[key] = self.create_publisher(Float64, f'/debug/{key}/cmd_vel', 10)
            self._fb_vel_pubs[key] = self.create_publisher(Float64, f'/debug/{key}/fb_vel', 10)
            self._vel_err_pubs[key] = self.create_publisher(Float64, f'/debug/{key}/vel_err', 10)

        self._joint_state_dt_pub = self.create_publisher(Float64, '/debug/joint_state_dt', 10)
        self._stack_count_pub = self.create_publisher(Float64, '/debug/stack_count', 10)

        self._timer = self.create_timer(self._dt, self.timer_callback)
        self._status_timer = self.create_timer(1.0, self.status_callback)

        self.get_logger().info('Stack-cubes node started.')

    # -------------------------------------------------
    # DYNAMIC STACK TARGETS
    # -------------------------------------------------
    def current_stack_place_xyz(self):
        z_place = self._STACK_BASE_Z + self._placed_count * self._CUBE_HEIGHT
        return p(self._STACK_XY[0], self._STACK_XY[1], z_place)

    def current_stack_above_xyz(self):
        place_xyz = self.current_stack_place_xyz()
        return p(place_xyz[0], place_xyz[1], place_xyz[2] + self._APPROACH_CLEARANCE)

    # -------------------------------------------------
    # CYCLE BUILDING
    # -------------------------------------------------
    def build_cycle_cartesian_waypoints(self, start_xyz):
        place_xyz = self.current_stack_place_xyz()
        place_above_xyz = self.current_stack_above_xyz()

        waypoints = [
            {
                "name": "move_to_source_above",
                "xyz": self._SOURCE_ABOVE.copy(),
                "grip": self._GRIPPER_OPEN,
                "duration": self._T_MOVE,
            },
            {
                "name": "lower_to_source_pick",
                "xyz": self._SOURCE_PICK.copy(),
                "grip": self._GRIPPER_OPEN,
                "duration": self._T_LOWER,
            },
            {
                "name": "close_gripper",
                "xyz": self._SOURCE_PICK.copy(),
                "grip": self._GRIPPER_CLOSED,
                "duration": self._T_GRIP,
            },
            {
                "name": "lift_from_source",
                "xyz": self._SOURCE_ABOVE.copy(),
                "grip": self._GRIPPER_CLOSED,
                "duration": self._T_LIFT,
            },
        ]

        if self._TRANSIT is not None:
            waypoints.append({
                "name": "transit",
                "xyz": self._TRANSIT.copy(),
                "grip": self._GRIPPER_CLOSED,
                "duration": self._T_TRANSIT,
            })
        else:
            # move horizontally in x first while keeping same y and high z
            x_transport = p(
                place_above_xyz[0],
                self._SOURCE_ABOVE[1],
                max(self._SOURCE_ABOVE[2], place_above_xyz[2])
            )

            if np.linalg.norm(x_transport - self._SOURCE_ABOVE) > 1.0e-12:
                waypoints.append({
                    "name": "transport_x",
                    "xyz": x_transport,
                    "grip": self._GRIPPER_CLOSED,
                    "duration": self._T_MOVE,
                })

        waypoints.extend([
            {
                "name": "move_to_stack_above",
                "xyz": place_above_xyz,
                "grip": self._GRIPPER_CLOSED,
                "duration": self._T_MOVE,
            },
            {
                "name": "lower_to_stack_place",
                "xyz": place_xyz,
                "grip": self._GRIPPER_CLOSED,
                "duration": self._T_LOWER,
            },
            {
                "name": "open_gripper",
                "xyz": place_xyz,
                "grip": self._GRIPPER_OPEN,
                "duration": self._T_GRIP,
            },
            {
                "name": "retreat_from_stack",
                "xyz": place_above_xyz,
                "grip": self._GRIPPER_OPEN,
                "duration": self._T_LIFT,
            },
        ])

        return waypoints
    
    def solve_cycle_joint_waypoints(self, cartesian_waypoints, q_start):
        joint_waypoints = []
        q_prev = q_start.copy()

        for wp in cartesian_waypoints:
            target_xyz = wp["xyz"]
            name = wp["name"]

            self.get_logger().info(
                f'Trying IK for {name}: xyz = {np.round(target_xyz, 4).tolist()}'
            )

            same_xyz = False
            if len(joint_waypoints) > 0:
                prev_xyz = joint_waypoints[-1]["xyz"]
                same_xyz = np.linalg.norm(target_xyz - prev_xyz) < 1.0e-12

            if same_xyz:
                q_wp = q_prev.copy()
            else:
                # Stronger downward orientation at grasp/release related waypoints
                if (
                    "lower_to_source_pick" in name or
                    "close_gripper" in name or
                    "lower_to_stack_place" in name or
                    "open_gripper" in name
                ):
                    desired_tool_pitch = self._DESIRED_TOOL_PITCH_PICK_PLACE
                    tool_pitch_weight = self._TOOL_PITCH_WEIGHT_PICK_PLACE
                else:
                    desired_tool_pitch = self._DESIRED_TOOL_PITCH_TRAVEL
                    tool_pitch_weight = self._TOOL_PITCH_WEIGHT_TRAVEL

                q_wp = solve_ik_position(
                    target_xyz,
                    q_seed=q_prev,
                    bounds=self._BOUNDS,
                    q_nominal=self._HOME_Q,
                    residual_tol=0.02,
                    desired_tool_pitch=desired_tool_pitch,
                    tool_pitch_weight=tool_pitch_weight
                )

            joint_waypoints.append({
                "name": wp["name"],
                "xyz": target_xyz.copy(),
                "q": q_wp.copy(),
                "grip": wp["grip"],
                "duration": wp["duration"],
            })

            q_prev = q_wp.copy()

        return joint_waypoints
    def build_segments_from_joint_waypoints(self, joint_waypoints, q_start, g_start):
        segments = []
        q_prev = q_start.copy()
        g_prev = g_start

        for wp in joint_waypoints:
            segments.append({
                "name": wp["name"],
                "q0": q_prev.copy(),
                "q1": wp["q"].copy(),
                "g0": g_prev,
                "g1": wp["grip"],
                "duration": wp["duration"],
            })
            q_prev = wp["q"].copy()
            g_prev = wp["grip"]

        return segments

    def build_next_cycle(self, start_from_home=False):
        if self._MAX_CUBES is not None and self._placed_count >= self._MAX_CUBES:
            self._finished = True
            return

        if start_from_home or self._last_cmd_pos is None:
            q_start = self._HOME_Q.copy()
            g_start = self._GRIPPER_OPEN
            start_xyz = self._HOME_XYZ.copy()
        else:
            q_start = self._last_cmd_pos.copy()
            g_start = self._GRIPPER_OPEN
            start_xyz = fk_position(q_start)

        self._current_cycle_waypoints = self.build_cycle_cartesian_waypoints(start_xyz)
        joint_waypoints = self.solve_cycle_joint_waypoints(self._current_cycle_waypoints, q_start)
        self._current_cycle_segments = self.build_segments_from_joint_waypoints(joint_waypoints, q_start, g_start)

        self._segment_index = 0
        self._segment_start_time = None
        self._last_logged_segment = None

        self.get_logger().info(
            f'Prepared cycle for cube #{self._placed_count + 1}: '
            f'place_z={self.current_stack_place_xyz()[2]:.4f}'
        )

    # -------------------------------------------------
    # TRAJECTORY INTERPOLATION
    # -------------------------------------------------
    def cubic_blend(self, tau):
        s = 3.0 * tau**2 - 2.0 * tau**3
        ds_dtau = 6.0 * tau - 6.0 * tau**2
        return s, ds_dtau

    def evaluate_segment(self, seg, t_in_seg):
        T = seg["duration"]

        if T <= 1.0e-9:
            return seg["q1"].copy(), np.zeros(5), seg["g1"]

        tau = np.clip(t_in_seg / T, 0.0, 1.0)
        s, ds_dtau = self.cubic_blend(tau)

        q0 = seg["q0"]
        q1 = seg["q1"]
        dq = q1 - q0

        q_cmd = q0 + s * dq
        qdot_cmd = (ds_dtau / T) * dq
        g_cmd = seg["g0"] + s * (seg["g1"] - seg["g0"])

        return q_cmd, qdot_cmd, float(g_cmd)

    # -------------------------------------------------
    # DEBUG HELPERS
    # -------------------------------------------------
    def publish_scalar(self, pub, value):
        msg = Float64()
        msg.data = float(value)
        pub.publish(msg)

    def publish_joint_scalars(self, pubs, values):
        for i, key in enumerate(self._topic_keys):
            msg = Float64()
            msg.data = float(values[i])
            pubs[key].publish(msg)

    def extract_expected_vector(self, msg_names, values):
        if values is None or len(values) == 0:
            return None

        if msg_names is not None and len(msg_names) > 0:
            try:
                out = []
                for joint in self._joint_names_expected:
                    idx = msg_names.index(joint)
                    out.append(values[idx])
                return np.array(out, dtype=float)
            except ValueError:
                pass

        if len(values) >= 5:
            return np.array(values[:5], dtype=float)

        return None

    # -------------------------------------------------
    # MAIN LOOP
    # -------------------------------------------------
    def timer_callback(self):
        now_sec = self.get_clock().now().nanoseconds * 1e-9

        if self._finished:
            if self._last_cmd_pos is None:
                cmd_pos = self._HOME_Q.copy()
            else:
                cmd_pos = self._last_cmd_pos.copy()

            cmd_vel = np.zeros(5)
            gripper_cmd = self._GRIPPER_OPEN
            current_name = "finished_hold"

        else:
            if not self._current_cycle_segments:
                self._finished = True
                return

            while True:
                if self._segment_start_time is None:
                    self._segment_start_time = now_sec

                current_seg = self._current_cycle_segments[self._segment_index]
                current_name = current_seg["name"]
                t_in_seg = now_sec - self._segment_start_time

                if t_in_seg < current_seg["duration"]:
                    break

                # Advance to next segment
                self._segment_index += 1

                if self._segment_index < len(self._current_cycle_segments):
                    self._segment_start_time += current_seg["duration"]
                    continue

                # Finished one full cube cycle
                self._placed_count += 1
                self.build_next_cycle(start_from_home=False)

                if self._finished or not self._current_cycle_segments:
                    break

                # Start new cycle immediately from "now"
                self._segment_start_time = now_sec
                self._segment_index = 0
                continue

            if self._finished:
                if self._last_cmd_pos is None:
                    cmd_pos = self._HOME_Q.copy()
                else:
                    cmd_pos = self._last_cmd_pos.copy()

                cmd_vel = np.zeros(5)
                gripper_cmd = self._GRIPPER_OPEN
                current_name = "finished_hold"
            else:
                current_seg = self._current_cycle_segments[self._segment_index]
                current_name = current_seg["name"]
                t_in_seg = now_sec - self._segment_start_time
                cmd_pos, cmd_vel, gripper_cmd = self.evaluate_segment(current_seg, t_in_seg)

                if current_name != self._last_logged_segment:
                    self.get_logger().info(
                        f'Now executing segment: {current_name} '
                        f'(stack count = {self._placed_count})'
                    )
                    self._last_logged_segment = current_name

        self._last_cmd_pos = cmd_pos.copy()
        self._last_cmd_vel = cmd_vel.copy()

        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()

        point = JointTrajectoryPoint()
        point.positions = [
            float(cmd_pos[0]),
            float(cmd_pos[1]),
            float(cmd_pos[2]),
            float(cmd_pos[3]),
            float(cmd_pos[4]),
            float(gripper_cmd),
        ]

        msg.points = [point]
        self._cmd_pub.publish(msg)

        self.publish_joint_scalars(self._cmd_pos_pubs, cmd_pos)
        self.publish_joint_scalars(self._cmd_vel_pubs, cmd_vel)
        self.publish_scalar(self._stack_count_pub, self._placed_count)

        if self._last_fb_pos is not None:
            pos_err = cmd_pos - self._last_fb_pos
            self.publish_joint_scalars(self._pos_err_pubs, pos_err)

        if self._last_fb_vel is not None:
            vel_err = cmd_vel - self._last_fb_vel
            self.publish_joint_scalars(self._vel_err_pubs, vel_err)

    # -------------------------------------------------
    # FEEDBACK
    # -------------------------------------------------
    def joint_state_callback(self, msg: JointState):
        self._seen_joint_states = True

        if not self._printed_joint_state_names:
            self.get_logger().info(f'/joint_states names = {list(msg.name)}')
            self._printed_joint_state_names = True

        msg_names = list(msg.name)
        fb_pos = self.extract_expected_vector(msg_names, msg.position)

        if fb_pos is None:
            return

        t = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        if t == 0.0:
            t = self.get_clock().now().nanoseconds * 1e-9

        fb_vel = self.extract_expected_vector(msg_names, msg.velocity)

        if fb_vel is None:
            if self._last_fb_pos is not None and self._last_fb_time is not None:
                dt = t - self._last_fb_time
                if dt > 1.0e-6:
                    fb_vel = (fb_pos - self._last_fb_pos) / dt
                    self.publish_scalar(self._joint_state_dt_pub, dt)
        else:
            if self._last_fb_time is not None:
                dt = t - self._last_fb_time
                if dt > 1.0e-6:
                    self.publish_scalar(self._joint_state_dt_pub, dt)

        self._last_fb_pos = fb_pos.copy()
        self._last_fb_time = t

        self.publish_joint_scalars(self._fb_pos_pubs, fb_pos)

        if self._last_cmd_pos is not None:
            pos_err = self._last_cmd_pos - fb_pos
            self.publish_joint_scalars(self._pos_err_pubs, pos_err)

        if fb_vel is None:
            return

        self._last_fb_vel = fb_vel.copy()
        self.publish_joint_scalars(self._fb_vel_pubs, fb_vel)

        if self._last_cmd_vel is not None:
            vel_err = self._last_cmd_vel - fb_vel
            self.publish_joint_scalars(self._vel_err_pubs, vel_err)

    # -------------------------------------------------
    # STATUS
    # -------------------------------------------------
    def status_callback(self):
        self.get_logger().info(
            f'joint_states seen={self._seen_joint_states}, '
            f'placed_count={self._placed_count}, '
            f'finished={self._finished}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = StackCubesFromSource()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
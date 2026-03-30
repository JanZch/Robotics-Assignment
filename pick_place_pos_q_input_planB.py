#!/usr/bin/env python3

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class PickPlaceTrajectory(Node):
    def __init__(self):
        super().__init__('pick_place_trajectory')

        # ---------------- SETTINGS ----------------
        self._dt = 0.04  # 25 Hz, same as your earlier examples

        # Expected joint names from /joint_states
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
        # IMPORTANT:
        # These are placeholders. Adjust depending on your robot/sim.
        # For some robots larger = more open, for others the opposite.
        self._GRIPPER_OPEN = 0.5
        self._GRIPPER_CLOSED = 0.1

        # ---------------- KEY JOINT POSES ----------------
        # Replace/tune these in simulation.
        # They are example poses to demonstrate the structure.
        self._HOME = np.array([
            np.deg2rad(0.0),
            np.deg2rad(105.0),
            np.deg2rad(-70.0),
            np.deg2rad(-60.0),
            np.deg2rad(0.0)
        ], dtype=float)

        self._PICK_ABOVE = np.array([
            np.deg2rad(20.0),
            np.deg2rad(85.0),
            np.deg2rad(-55.0),
            np.deg2rad(-75.0),
            np.deg2rad(0.0)
        ], dtype=float)

        self._PICK = np.array([
            np.deg2rad(20.0),
            np.deg2rad(95.0),
            np.deg2rad(-68.0),
            np.deg2rad(-68.0),
            np.deg2rad(0.0)
        ], dtype=float)

        self._PLACE_ABOVE = np.array([
            np.deg2rad(-25.0),
            np.deg2rad(82.0),
            np.deg2rad(-50.0),
            np.deg2rad(-78.0),
            np.deg2rad(0.0)
        ], dtype=float)

        self._PLACE = np.array([
            np.deg2rad(-25.0),
            np.deg2rad(94.0),
            np.deg2rad(-66.0),
            np.deg2rad(-70.0),
            np.deg2rad(0.0)
        ], dtype=float)

        # ---------------- PLAN ----------------
        self._plan = self.build_plan()
        self._segment_index = 0
        self._segment_start_time = None
        self._plan_finished = False
        self._last_logged_segment = None

        # ---------------- INTERNAL STATE ----------------
        self._last_fb_pos = None
        self._last_fb_vel = None
        self._last_fb_time = None

        self._last_cmd_pos = None
        self._last_cmd_vel = None

        self._seen_joint_states = False
        self._printed_joint_state_names = False

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

        # Debug publishers: positions
        self._cmd_pos_pubs = {}
        self._fb_pos_pubs = {}
        self._pos_err_pubs = {}

        # Debug publishers: velocities
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

        self._timer = self.create_timer(self._dt, self.timer_callback)
        self._status_timer = self.create_timer(1.0, self.status_callback)

        self.get_logger().info('Pick-and-place position trajectory node started.')

    # -------------------------------------------------
    # PLAN DEFINITION
    # -------------------------------------------------
    def build_plan(self):
        """
        Each segment contains:
        - name
        - q0: start joint pose
        - q1: end joint pose
        - g0: start gripper value
        - g1: end gripper value
        - duration
        """
        return [
            {
                'name': 'hold_home',
                'q0': self._HOME, 'q1': self._HOME,
                'g0': self._GRIPPER_OPEN, 'g1': self._GRIPPER_OPEN,
                'duration': 1.5
            },
            {
                'name': 'move_to_pick_above',
                'q0': self._HOME, 'q1': self._PICK_ABOVE,
                'g0': self._GRIPPER_OPEN, 'g1': self._GRIPPER_OPEN,
                'duration': 2.5
            },
            {
                'name': 'lower_to_pick',
                'q0': self._PICK_ABOVE, 'q1': self._PICK,
                'g0': self._GRIPPER_OPEN, 'g1': self._GRIPPER_OPEN,
                'duration': 1.5
            },
            {
                'name': 'close_gripper',
                'q0': self._PICK, 'q1': self._PICK,
                'g0': self._GRIPPER_OPEN, 'g1': self._GRIPPER_CLOSED,
                'duration': 1.0
            },
            {
                'name': 'lift_from_pick',
                'q0': self._PICK, 'q1': self._PICK_ABOVE,
                'g0': self._GRIPPER_CLOSED, 'g1': self._GRIPPER_CLOSED,
                'duration': 1.5
            },
            {
                'name': 'move_to_place_above',
                'q0': self._PICK_ABOVE, 'q1': self._PLACE_ABOVE,
                'g0': self._GRIPPER_CLOSED, 'g1': self._GRIPPER_CLOSED,
                'duration': 3.0
            },
            {
                'name': 'lower_to_place',
                'q0': self._PLACE_ABOVE, 'q1': self._PLACE,
                'g0': self._GRIPPER_CLOSED, 'g1': self._GRIPPER_CLOSED,
                'duration': 1.5
            },
            {
                'name': 'open_gripper',
                'q0': self._PLACE, 'q1': self._PLACE,
                'g0': self._GRIPPER_CLOSED, 'g1': self._GRIPPER_OPEN,
                'duration': 1.0
            },
            {
                'name': 'retreat_from_place',
                'q0': self._PLACE, 'q1': self._PLACE_ABOVE,
                'g0': self._GRIPPER_OPEN, 'g1': self._GRIPPER_OPEN,
                'duration': 1.5
            },
            {
                'name': 'return_home',
                'q0': self._PLACE_ABOVE, 'q1': self._HOME,
                'g0': self._GRIPPER_OPEN, 'g1': self._GRIPPER_OPEN,
                'duration': 2.5
            },
        ]

    # -------------------------------------------------
    # TRAJECTORY INTERPOLATION
    # -------------------------------------------------
    def cubic_blend(self, tau):
        """
        Smooth cubic with zero velocity at tau=0 and tau=1.
        """
        s = 3.0 * tau**2 - 2.0 * tau**3
        ds_dtau = 6.0 * tau - 6.0 * tau**2
        return s, ds_dtau

    def evaluate_segment(self, seg, t_in_seg):
        T = seg['duration']

        if T <= 1e-9:
            return seg['q1'].copy(), np.zeros(5), seg['g1']

        tau = np.clip(t_in_seg / T, 0.0, 1.0)
        s, ds_dtau = self.cubic_blend(tau)

        q0 = seg['q0']
        q1 = seg['q1']
        dq = q1 - q0

        q_cmd = q0 + s * dq
        qdot_cmd = (ds_dtau / T) * dq

        g_cmd = seg['g0'] + s * (seg['g1'] - seg['g0'])

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

        # First try exact joint-name matching
        if msg_names is not None and len(msg_names) > 0:
            try:
                out = []
                for joint in self._joint_names_expected:
                    idx = msg_names.index(joint)
                    out.append(values[idx])
                return np.array(out, dtype=float)
            except ValueError:
                pass

        # Fallback: use first 5 values
        if len(values) >= 5:
            return np.array(values[:5], dtype=float)

        return None

    # -------------------------------------------------
    # MAIN CONTROL LOOP
    # -------------------------------------------------
    def timer_callback(self):
        now_sec = self.get_clock().now().nanoseconds * 1e-9

        if self._segment_start_time is None:
            self._segment_start_time = now_sec

        if self._plan_finished:
            # Hold final pose
            final_seg = self._plan[-1]
            cmd_pos = final_seg['q1'].copy()
            cmd_vel = np.zeros(5)
            gripper_cmd = final_seg['g1']
            current_name = 'finished_hold'
        else:
            current_seg = self._plan[self._segment_index]
            current_name = current_seg['name']

            t_in_seg = now_sec - self._segment_start_time

            # Advance to next segment if finished
            while t_in_seg >= current_seg['duration']:
                self._segment_index += 1
                self._segment_start_time += current_seg['duration']

                if self._segment_index >= len(self._plan):
                    self._plan_finished = True
                    break

                current_seg = self._plan[self._segment_index]
                current_name = current_seg['name']
                t_in_seg = now_sec - self._segment_start_time

            if self._plan_finished:
                final_seg = self._plan[-1]
                cmd_pos = final_seg['q1'].copy()
                cmd_vel = np.zeros(5)
                gripper_cmd = final_seg['g1']
                current_name = 'finished_hold'
            else:
                cmd_pos, cmd_vel, gripper_cmd = self.evaluate_segment(current_seg, t_in_seg)

        if current_name != self._last_logged_segment:
            self.get_logger().info(f'Now executing segment: {current_name}')
            self._last_logged_segment = current_name

        self._last_cmd_pos = cmd_pos.copy()
        self._last_cmd_vel = cmd_vel.copy()

        # Publish to robot/sim
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

        # Debug command
        self.publish_joint_scalars(self._cmd_pos_pubs, cmd_pos)
        self.publish_joint_scalars(self._cmd_vel_pubs, cmd_vel)

        # Debug errors
        if self._last_fb_pos is not None:
            pos_err = cmd_pos - self._last_fb_pos
            self.publish_joint_scalars(self._pos_err_pubs, pos_err)

        if self._last_fb_vel is not None:
            vel_err = cmd_vel - self._last_fb_vel
            self.publish_joint_scalars(self._vel_err_pubs, vel_err)

    # -------------------------------------------------
    # FEEDBACK CALLBACK
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

        # Estimate velocity from position difference if needed
        if fb_vel is None:
            if self._last_fb_pos is not None and self._last_fb_time is not None:
                dt = t - self._last_fb_time
                if dt > 1e-6:
                    fb_vel = (fb_pos - self._last_fb_pos) / dt
                    self.publish_scalar(self._joint_state_dt_pub, dt)
        else:
            if self._last_fb_time is not None:
                dt = t - self._last_fb_time
                if dt > 1e-6:
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
            f'segment_index={self._segment_index}, '
            f'plan_finished={self._plan_finished}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = PickPlaceTrajectory()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
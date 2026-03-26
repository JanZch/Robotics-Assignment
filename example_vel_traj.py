import rclpy
import numpy as np
from rclpy.node import Node

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
# Something for compatibility
from rclpy.qos import QoSProfile, ReliabilityPolicy

# Copypasted jacobian matrix
def compute_jacobian(theta1, theta2, theta3, theta4, theta5):
    return np.array([[(0.11257*np.sin(theta2) + 0.0052*np.sin(theta2 + theta3) - 0.028*np.cos(theta2) - 0.1349*np.cos(theta2 + theta3) - 0.0601*np.cos(theta2 + theta3 + theta4) - 0.0306)*np.cos(theta1)
                    , (0.028*np.sin(theta2) + 0.1349*np.sin(theta2 + theta3) + 0.0601*np.sin(theta2 + theta3 + theta4) + 0.11257*np.cos(theta2) + 0.0052*np.cos(theta2 + theta3))*np.sin(theta1)
                    , (0.1349*np.sin(theta2 + theta3) + 0.0601*np.sin(theta2 + theta3 + theta4) + 0.0052*np.cos(theta2 + theta3))*np.sin(theta1)
                    , 0.0601*np.sin(theta1)*np.sin(theta2 + theta3 + theta4)
                    , 0]
                    , [(0.11257*np.sin(theta2) + 0.0052*np.sin(theta2 + theta3) - 0.028*np.cos(theta2) - 0.1349*np.cos(theta2 + theta3) - 0.0601*np.cos(theta2 + theta3 + theta4) - 0.0306)*np.sin(theta1)
                    , -(0.028*np.sin(theta2) + 0.1349*np.sin(theta2 + theta3) + 0.0601*np.sin(theta2 + theta3 + theta4) + 0.11257*np.cos(theta2) + 0.0052*np.cos(theta2 + theta3))*np.cos(theta1)
                    , -(0.1349*np.sin(theta2 + theta3) + 0.0601*np.sin(theta2 + theta3 + theta4) + 0.0052*np.cos(theta2 + theta3))*np.cos(theta1)
                    , -0.0601*np.sin(theta2 + theta3 + theta4)*np.cos(theta1)
                    , 0]
                    , [0
                    , -0.11257*np.sin(theta2) - 0.0052*np.sin(theta2 + theta3) + 0.028*np.cos(theta2) + 0.1349*np.cos(theta2 + theta3) + 0.0601*np.cos(theta2 + theta3 + theta4)
                    , -0.0052*np.sin(theta2 + theta3) + 0.1349*np.cos(theta2 + theta3) + 0.0601*np.cos(theta2 + theta3 + theta4)
                    , 0.0601*np.cos(theta2 + theta3 + theta4)
                    , 0]])
# Computes joint velocities from pose and desired cartesian velocity (xyz only)
def velocity_trajectory(pose, velocity):
    jacobian = compute_jacobian(*pose)
    inverse_jacobian = np.linalg.pinv(jacobian)
    return inverse_jacobian @ velocity


class ExampleTraj(Node):

    def __init__(self):
        super().__init__('example_trajectory')

        self._HOME = [np.deg2rad(0), np.deg2rad(70),
                     np.deg2rad(-40), np.deg2rad(-60),
                     np.deg2rad(0)]
        self._beginning = self.get_clock().now()
        self._publisher = self.create_publisher(JointTrajectory, 'joint_cmds', 10)
        timer_period = 0.04  # seconds
        self._timer = self.create_timer(timer_period, self.timer_callback)

        self._joint_names = [
            "Shoulder_Rotation",
            "Shoulder_Pitch",
            "Elbow",
            "Wrist_Pitch",
            "Wrist_Roll",
        ]

        # Stores latest joint positions
        self._pose = None 
        # Some compatibility stuff
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )
        # Subscribes to joint states
        self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            qos
        )

    # Runs when joint state gets published
    def joint_state_callback(self, msg: JointState):
        try:
            pose = []
            # Interprets message for joint angles and puts in array
            for joint in self._joint_names:
                idx = msg.name.index(joint)
                pose.append(msg.position[idx])

            self._pose = np.array(pose)

        except ValueError:
            # joint missing in message
            return

    def timer_callback(self):
        if self._pose is None:
            return  # wait until first joint message arrives

        # Clock
        now = self.get_clock().now()
        # Create message
        msg = JointTrajectory()
        msg.header.stamp = now.to_msg()

        dt = (now - self._beginning).nanoseconds * (1e-9)
        
        # Copy latest joint positions
        pose = self._pose.copy()
        # Desired velocity
        velocity = np.array([0.01, 0.0, 0.0])
        # Compute join velocities
        velocities = velocity_trajectory(pose, velocity)
        # Put velocites in point
        point = JointTrajectoryPoint()
        point.velocities = velocities.tolist()

        msg.points = [point]

        self._publisher.publish(msg)
        

def main(args=None):
    rclpy.init(args=args)

    example_traj = ExampleTraj()

    rclpy.spin(example_traj)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    example_traj.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
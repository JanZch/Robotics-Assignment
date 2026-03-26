import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from rclpy.qos import QoSProfile, ReliabilityPolicy


class JointStateReader(Node):

    def __init__(self):
        super().__init__('joint_state_reader')

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )

        self.create_subscription(
            JointState,
            '/joint_states',
            self.callback,
            qos
        )

    def callback(self, msg):
        for name, pos in zip(msg.name, msg.position):
            print(f"{name}: {pos}")


def main():
    rclpy.init()
    node = JointStateReader()
    rclpy.spin(node)
    rclpy.shutdown()
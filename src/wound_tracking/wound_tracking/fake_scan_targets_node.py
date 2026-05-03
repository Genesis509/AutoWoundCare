import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose
import numpy as np

from wound_tracking.scan_pose_publisher_node import _look_at_optical

C       = np.array([0.74, 0.25, 0.04])
FRONT   = np.array([-0.22, -0.13, 0.97])
RIGHT   = np.array([-0.08, 0.99, 0.11])
RADIUS  = 0.85
ANGLES  = [30.0, 0.0, -30.0]
FRAME   = 'base_link'


def _compute_poses():
    out = []
    for ang_deg in ANGLES:
        a = np.radians(ang_deg)
        pos = C + RADIUS * (np.cos(a) * FRONT + np.sin(a) * RIGHT)
        q = _look_at_optical(pos, C)
        p = Pose()
        p.position.x = float(pos[0])
        p.position.y = float(pos[1])
        p.position.z = float(pos[2])
        p.orientation.x = float(q[0])
        p.orientation.y = float(q[1])
        p.orientation.z = float(q[2])
        p.orientation.w = float(q[3])
        out.append(p)
    return out


class FakeScanTargetsNode(Node):

    def __init__(self):
        super().__init__('fake_scan_targets_node')
        self._poses = _compute_poses()
        self._pub = self.create_publisher(PoseArray, '/scan/target_poses', 10)
        self.create_timer(1.0, self._tick)
        self.get_logger().info(
            f'Fake scan targets ready: R={RADIUS} angles={ANGLES} C={C.tolist()}')

    def _tick(self):
        msg = PoseArray()
        msg.header.frame_id = FRAME
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.poses = self._poses
        self._pub.publish(msg)


def main():
    rclpy.init()
    n = FakeScanTargetsNode()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

"""
calibrated_tf_publisher_node.py

Reads eye_in_hand.json (T_camera_color_optical_frame → tool0) produced by
calibrate_eye_in_hand.py, inverts it, and publishes the static TF:

    tool0  →  camera_color_optical_frame

This replaces the hardcoded static_transform_publisher entries in real.launch.py
with calibration-derived values.

Falls back to hardcoded approximate values with a warning if the file is missing.
"""

import json
import os
import math
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster

CALIB_FILE   = os.path.expanduser(
    '~/Vision_Guided_Autonomous_Wound_Treatment_System/calibration/eye_in_hand.json')
PARENT_FRAME = 'tool0'
CHILD_FRAME  = 'camera_color_optical_frame'

# Fallback: approximate pre-calibration values
_FALLBACK_XYZ = (0.0, 0.04, 0.011)
_FALLBACK_RPY = (0.0, 0.0, math.pi)


def _rotmat_to_quat(R):
    """Shepperd method: 3×3 rotation matrix → (x, y, z, w)."""
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = 0.5 / math.sqrt(t + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s;  x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s;  z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s;  x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s;                  z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s;  x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s;  z = 0.25 * s
    n = math.sqrt(x*x + y*y + z*z + w*w)
    return x/n, y/n, z/n, w/n


def _rpy_to_quat(roll, pitch, yaw):
    cr, sr = math.cos(roll/2),  math.sin(roll/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cy, sy = math.cos(yaw/2),   math.sin(yaw/2)
    return (sr*cp*cy - cr*sp*sy,   # x
            cr*sp*cy + sr*cp*sy,   # y
            cr*cp*sy - sr*sp*cy,   # z
            cr*cp*cy + sr*sp*sy)   # w


class CalibratedTfPublisherNode(Node):

    def __init__(self):
        super().__init__('calibrated_tf_publisher')
        self._br = StaticTransformBroadcaster(self)
        self._publish()

    def _publish(self):
        ts = TransformStamped()
        ts.header.stamp    = self.get_clock().now().to_msg()
        ts.header.frame_id = PARENT_FRAME
        ts.child_frame_id  = CHILD_FRAME

        if os.path.isfile(CALIB_FILE):
            try:
                with open(CALIB_FILE) as f:
                    cal = json.load(f)

                # T_cam_to_tool0 maps points cam→tool0; translation column = cam
                # origin expressed in tool0. ROS TF parent=tool0, child=cam stores
                # child pose in parent — same quantity, no inversion.
                T_c2t = np.array(cal['T_cam_to_tool0'])
                R_c2t = T_c2t[:3, :3]
                t_c2t = T_c2t[:3, 3]

                qx, qy, qz, qw = _rotmat_to_quat(R_c2t)

                ts.transform.translation.x = float(t_c2t[0])
                ts.transform.translation.y = float(t_c2t[1])
                ts.transform.translation.z = float(t_c2t[2])
                ts.transform.rotation.x    = qx
                ts.transform.rotation.y    = qy
                ts.transform.rotation.z    = qz
                ts.transform.rotation.w    = qw

                consistency = cal.get("consistency")
                consistency_str = f'{consistency:.1f}' if consistency is not None else '?'
                self.get_logger().info(
                    f'Calibrated TF {PARENT_FRAME} → {CHILD_FRAME}  '
                    f't=[{t_c2t[0]:.4f}, {t_c2t[1]:.4f}, {t_c2t[2]:.4f}]  '
                    f'solver={cal.get("solver","?")}  '
                    f'consistency={consistency_str}  '
                    f'n_poses={cal.get("n_poses_used","?")}')

            except Exception as e:
                self.get_logger().error(
                    f'Failed to load {CALIB_FILE}: {e} — using fallback values')
                self._apply_fallback(ts)
        else:
            self.get_logger().warn(
                f'No calibration file at {CALIB_FILE} — using approximate fallback values. '
                f'Run calibrate_eye_in_hand.py to calibrate.')
            self._apply_fallback(ts)

        self._br.sendTransform(ts)

    def _apply_fallback(self, ts):
        qx, qy, qz, qw = _rpy_to_quat(*_FALLBACK_RPY)
        ts.transform.translation.x = _FALLBACK_XYZ[0]
        ts.transform.translation.y = _FALLBACK_XYZ[1]
        ts.transform.translation.z = _FALLBACK_XYZ[2]
        ts.transform.rotation.x    = qx
        ts.transform.rotation.y    = qy
        ts.transform.rotation.z    = qz
        ts.transform.rotation.w    = qw


def main(args=None):
    rclpy.init(args=args)
    node = CalibratedTfPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

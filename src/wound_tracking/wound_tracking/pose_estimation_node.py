"""
pose_estimation_node.py — SIM and REAL.

Uses the bbox ROI from feature_detector to deproject all valid depth pixels
inside the detection region, fits a plane via SVD, and publishes a full 6-DOF
approach pose whose tool-Z is normal to the detected surface.

Subscriptions:
  /wound/detection/pixels             (geometry_msgs/PointStamped)
  /wound/detection/roi                (sensor_msgs/RegionOfInterest)
  /camera/depth_registered/image_raw  (sensor_msgs/Image, 16UC1 mm, 960x540 aligned to RGB)
  /camera/color/camera_info           (sensor_msgs/CameraInfo)

Publications:
  /wound/target/pose         (geometry_msgs/PoseStamped, frame: base_link)
  /wound/target/marker       (visualization_msgs/Marker)
                               id=0: flat cylinder showing detected plane
                               id=1: arrow from approach pose to surface

Depth scale:
  - Sim:  Gz RGBD float32 metres              → scale = 1.0
  - Real: Kinect 2 registered depth uint16 mm → scale = 0.001
"""

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, RegionOfInterest
from geometry_msgs.msg import PointStamped, PoseStamped, Point, Quaternion
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge
import tf2_ros
import tf2_geometry_msgs  # noqa: F401 — registers PoseStamped with tf2

CAMERA_FRAME      = 'camera_color_optical_frame'
BASE_FRAME        = 'base_link'
EE_LINK           = 'tool0'
TARGET_OFFSET_M   = 0.15    # stand-off distance along approach axis
DEPTH_SCALE_REAL  = 0.001       # Kinect 2 registered depth: uint16 mm → metres
DEPTH_SCALE_SIM   = 1.0
MAX_DEPTH_AGE_S   = 0.2
TF_TIMEOUT_INIT   = 1.0
TF_TIMEOUT_STEADY = 0.1
MIN_PLANE_POINTS  = 20      # minimum valid depth pixels required for SVD


class PoseEstimationNode(Node):

    def __init__(self):
        super().__init__('pose_estimation')

        self._depth_scale = (DEPTH_SCALE_SIM
                             if self.get_parameter('use_sim_time').value
                             else DEPTH_SCALE_REAL)
        mode = 'SIM (float32 m)' if self.get_parameter('use_sim_time').value else 'REAL (Kinect2 registered uint16 mm)'
        self.get_logger().info(f'Depth mode: {mode}')

        self._bridge      = CvBridge()
        self._depth_image = None
        self._depth_stamp = None
        self._camera_info = None
        self._latest_roi  = None
        self._tf_ready    = False

        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self.create_subscription(PointStamped,    '/wound/detection/pixels', self._detection_cb, 10)
        self.create_subscription(RegionOfInterest,'/wound/detection/roi',    self._roi_cb,       10)
        self.create_subscription(Image,           '/camera/depth_registered/image_raw', self._depth_cb, 10)
        self.create_subscription(CameraInfo,      '/camera/color/camera_info', self._info_cb,    10)

        self._pub_pose   = self.create_publisher(PoseStamped, '/wound/target/pose',   10)
        self._pub_marker = self.create_publisher(Marker,      '/wound/target/marker', 10)

        self.get_logger().info('Pose estimation ready')

    def _depth_cb(self, msg: Image):
        self._depth_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self._depth_stamp = msg.header.stamp

    def _info_cb(self, msg: CameraInfo):
        self._camera_info = msg

    def _roi_cb(self, msg: RegionOfInterest):
        self._latest_roi = msg

    def _detection_cb(self, msg: PointStamped):
        if self._depth_image is None or self._camera_info is None or self._latest_roi is None:
            self.get_logger().warn('Waiting for depth / camera_info / roi',
                                   throttle_duration_sec=2.0)
            return

        det_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        dep_sec = self._depth_stamp.sec + self._depth_stamp.nanosec * 1e-9
        if abs(det_sec - dep_sec) > MAX_DEPTH_AGE_S:
            self.get_logger().warn('Depth frame stale', throttle_duration_sec=2.0)
            return

        K        = self._camera_info.k
        fx, fy   = K[0], K[4]
        cx, cy   = K[2], K[5]
        K_mat    = np.array(K, dtype=np.float64).reshape(3, 3)
        dist     = np.array(self._camera_info.d, dtype=np.float64)

        img_h, img_w = self._depth_image.shape[:2]
        roi = self._latest_roi
        u0  = max(0,     int(roi.x_offset))
        v0  = max(0,     int(roi.y_offset))
        u1  = min(img_w, u0 + int(roi.width))
        v1  = min(img_h, v0 + int(roi.height))

        patch  = self._depth_image[v0:v1, u0:u1].astype(np.float64)
        result = self._fit_plane(patch, u0, v0, fx, fy, cx, cy, K_mat, dist)

        if result is None:
            self.get_logger().warn('Plane fit failed — not enough valid depth points',
                                   throttle_duration_sec=2.0)
            return

        centroid_cam, normal_cam = result

        R_cam           = self._normal_to_rotation(normal_cam)
        qx, qy, qz, qw = self._rot_to_quat(R_cam)

        # Full pose in camera frame (surface point + approach orientation)
        pose_cam = PoseStamped()
        pose_cam.header.stamp       = msg.header.stamp
        pose_cam.header.frame_id    = CAMERA_FRAME
        pose_cam.pose.position.x    = float(centroid_cam[0])
        pose_cam.pose.position.y    = float(centroid_cam[1])
        pose_cam.pose.position.z    = float(centroid_cam[2])
        pose_cam.pose.orientation.x = float(qx)
        pose_cam.pose.orientation.y = float(qy)
        pose_cam.pose.orientation.z = float(qz)
        pose_cam.pose.orientation.w = float(qw)

        # Use the latest available transform rather than the exact image
        # timestamp.  In sim the robot_state_publisher TF can lag several
        # seconds behind camera frame stamps at startup, causing extrapolation
        # errors.  time=0 means "most recent" in tf2 and is safe here because
        # the robot moves slowly and poses are median-filtered downstream.
        pose_cam.header.stamp = rclpy.time.Time().to_msg()

        tf_timeout = TF_TIMEOUT_STEADY if self._tf_ready else TF_TIMEOUT_INIT
        try:
            # Transforms both position and orientation correctly
            pose_base = self._tf_buffer.transform(
                pose_cam, BASE_FRAME,
                timeout=rclpy.duration.Duration(seconds=tf_timeout))
            if not self._tf_ready:
                self._tf_ready = True
                self.get_logger().info('TF tree available — tracking active')
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as exc:
            self.get_logger().warn(f'TF error: {exc}', throttle_duration_sec=2.0)
            return
        pose_base.pose.orientation = self._minimize_roll(
            pose_base.pose.orientation)

        # Stand-off: pull back along approach direction (tool-Z in base_link)
        # so the robot arrives TARGET_OFFSET_M in front of the surface
        approach = self._quat_z_axis(pose_base.pose.orientation)

        target = PoseStamped()
        target.header               = pose_base.header
        target.pose.orientation     = pose_base.pose.orientation
        target.pose.position.x      = pose_base.pose.position.x - approach[0] * TARGET_OFFSET_M
        target.pose.position.y      = pose_base.pose.position.y - approach[1] * TARGET_OFFSET_M
        target.pose.position.z      = pose_base.pose.position.z - approach[2] * TARGET_OFFSET_M

        self._pub_pose.publish(target)
        self._publish_markers(pose_base, target)

    def _fit_plane(self, patch, u0, v0, fx, fy, cx, cy, K_mat, dist):
        """
        Deproject all valid depth pixels in the patch to 3D, filter outliers
        by depth range, then fit a plane with SVD.
        Applies cv2.undistortPoints so pixel coords match the distorted depth_reg
        grid while still deprojecting along correct rays.
        Returns (centroid_3d, unit_normal) in camera optical frame,
        or None if not enough points.
        """
        rows, cols = patch.shape[0], patch.shape[1]
        vs_grid, us_grid = np.mgrid[0:rows, 0:cols]
        us_abs = (us_grid + u0).ravel().astype(np.float64)
        vs_abs = (vs_grid + v0).ravel().astype(np.float64)
        d_flat = patch.ravel()

        if self._depth_scale == DEPTH_SCALE_SIM:
            valid = np.isfinite(d_flat) & (d_flat > 0.0)
        else:
            valid = d_flat > 0

        if valid.sum() < MIN_PLANE_POINTS:
            return None

        d_raw = d_flat[valid]
        us_v  = us_abs[valid]
        vs_v  = vs_abs[valid]

        # Reject pixels more than 2σ from the median depth (background / edges)
        med  = np.median(d_raw)
        std  = np.std(d_raw)
        keep = np.abs(d_raw - med) < 2.0 * std + 1e-9
        if keep.sum() < MIN_PLANE_POINTS:
            return None

        dm   = d_raw[keep] * self._depth_scale
        us_v = us_v[keep]
        vs_v = vs_v[keep]

        # Undistort pixel coords before deprojection: depth_reg is aligned to the
        # distorted color grid, so we pass the raw (distorted) pixel positions here
        # and get back ideal (undistorted) normalised directions.
        pts_dist = np.column_stack([us_v, vs_v]).astype(np.float32).reshape(-1, 1, 2)
        pts_und  = cv2.undistortPoints(pts_dist, K_mat, dist, P=K_mat)
        us_v = pts_und[:, 0, 0].astype(np.float64)
        vs_v = pts_und[:, 0, 1].astype(np.float64)

        X = (us_v - cx) * dm / fx
        Y = (vs_v - cy) * dm / fy
        Z = dm

        pts      = np.column_stack([X, Y, Z])
        centroid = pts.mean(axis=0)

        # SVD: normal = right singular vector with smallest singular value
        _, _, Vt = np.linalg.svd(pts - centroid, full_matrices=False)
        normal   = Vt[-1]

        # Camera optical frame: Z points forward (into scene).
        # A surface facing the camera has its normal pointing back (normal[2] < 0).
        if normal[2] > 0:
            normal = -normal

        return centroid, normal

    def _normal_to_rotation(self, normal_cam):
        """
        Build a rotation matrix whose tool-Z (third column) is the approach
        direction — pointing into the surface (opposite to the surface normal).
        tool-X is chosen to be stable and orthogonal.
        Returns a 3×3 numpy rotation matrix.
        """
        tool_z = -normal_cam          # into the surface = approach direction
        ref    = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(tool_z, ref)) > 0.9:
            ref = np.array([1.0, 0.0, 0.0])
        tool_x = np.cross(ref, tool_z);  tool_x /= np.linalg.norm(tool_x)
        tool_y = np.cross(tool_z, tool_x)
        return np.column_stack([tool_x, tool_y, tool_z])

    def _rot_to_quat(self, R):
        """Shepperd method: rotation matrix → (x, y, z, w)."""
        t = R[0, 0] + R[1, 1] + R[2, 2]
        if t > 0:
            s = 0.5 / np.sqrt(t + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s;  x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s;  z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s;  x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s;                  z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s;  x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s;  z = 0.25 * s
        n = np.sqrt(x*x + y*y + z*z + w*w)
        return x/n, y/n, z/n, w/n

    def _quat_z_axis(self, q):
        """Return the tool-Z axis (third column of rotation matrix) for a quaternion."""
        x, y, z, w = q.x, q.y, q.z, q.w
        return np.array([2*(x*z + y*w), 2*(y*z - x*w), 1 - 2*(x*x + y*y)])

    def _quat_to_rotmat(self, q):
        """Quaternion (geometry_msgs) → 3×3 rotation matrix."""
        x, y, z, w = q.x, q.y, q.z, q.w
        return np.array([
            [1 - 2*(y*y + z*z),   2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),       1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w),       2*(y*z + x*w),     1 - 2*(x*x + y*y)],
        ])

    def _minimize_roll(self, target_ori):
        """
        Replace the arbitrary roll component of *target_ori* with the roll
        that is closest to the current tool0 orientation.

        Keeps tool-Z (approach direction) from the target, but rebuilds
        tool-X / tool-Y to match the current wrist pose as closely as
        possible.  This prevents wrist_3 from spinning unnecessarily.

        Falls back to the original orientation if TF lookup fails.
        """
        try:
            tf = self._tf_buffer.lookup_transform(
                BASE_FRAME, EE_LINK,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05))
            cur_q = tf.transform.rotation
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return target_ori          # can't look up → keep original

        # Current tool0 rotation matrix in base_link
        R_cur = self._quat_to_rotmat(cur_q)
        cur_tool_x = R_cur[:, 0]      # current tool-X axis

        # New approach direction from target
        new_tool_z = self._quat_z_axis(target_ori)
        new_tool_z = new_tool_z / np.linalg.norm(new_tool_z)

        # Project current tool-X onto plane perpendicular to new approach
        proj = cur_tool_x - np.dot(cur_tool_x, new_tool_z) * new_tool_z
        norm = np.linalg.norm(proj)

        if norm < 1e-6:
            # Current tool-X is (anti-)parallel to new approach — degenerate.
            # Fall back to current tool-Y instead.
            cur_tool_y = R_cur[:, 1]
            proj = cur_tool_y - np.dot(cur_tool_y, new_tool_z) * new_tool_z
            norm = np.linalg.norm(proj)
            if norm < 1e-6:
                return target_ori      # fully degenerate — keep original

        new_tool_x = proj / norm
        new_tool_y = np.cross(new_tool_z, new_tool_x)

        R_new = np.column_stack([new_tool_x, new_tool_y, new_tool_z])
        qx, qy, qz, qw = self._rot_to_quat(R_new)

        return Quaternion(x=float(qx), y=float(qy), z=float(qz), w=float(qw))

    def _publish_markers(self, surface: PoseStamped, target: PoseStamped):
        now = self.get_clock().now().to_msg()

        # Flat cylinder lying in the detected plane (disc orientation = surface orientation)
        disc = Marker()
        disc.header.stamp    = now
        disc.header.frame_id = BASE_FRAME
        disc.ns = 'wound_tracking';  disc.id = 0
        disc.type   = Marker.CYLINDER
        disc.action = Marker.ADD
        disc.pose.position    = surface.pose.position
        disc.pose.orientation = surface.pose.orientation
        disc.scale.x = 0.08;  disc.scale.y = 0.08;  disc.scale.z = 0.004
        disc.color.r = 1.0;   disc.color.g = 0.5;   disc.color.a = 0.85
        disc.lifetime.sec = 1
        self._pub_marker.publish(disc)

        # Arrow: tail at approach pose, tip at surface — shows tool travel direction
        arrow = Marker()
        arrow.header.stamp    = now
        arrow.header.frame_id = BASE_FRAME
        arrow.ns = 'wound_tracking';  arrow.id = 1
        arrow.type   = Marker.ARROW
        arrow.action = Marker.ADD
        tail = Point()
        tail.x = target.pose.position.x
        tail.y = target.pose.position.y
        tail.z = target.pose.position.z
        tip = Point()
        tip.x = surface.pose.position.x
        tip.y = surface.pose.position.y
        tip.z = surface.pose.position.z
        arrow.points  = [tail, tip]
        arrow.scale.x = 0.01   # shaft diameter
        arrow.scale.y = 0.02   # head diameter
        arrow.color.g = 1.0;   arrow.color.a = 1.0
        arrow.lifetime.sec = 1
        self._pub_marker.publish(arrow)


def main(args=None):
    rclpy.init(args=args)
    node = PoseEstimationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
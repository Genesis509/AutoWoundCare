import os
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, PoseArray, Point
from visualization_msgs.msg import Marker, MarkerArray
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
import tf2_ros
import tf2_geometry_msgs  # noqa: F401
from ultralytics import YOLO

CAMERA_FRAME     = 'camera_color_optical_frame'
BASE_FRAME       = 'base_link'
DEPTH_SCALE_REAL = 0.001
DEPTH_SCALE_SIM  = 1.0
MIN_PLANE_POINTS = 20
TARGET_OFFSET_M  = 0.15
DETECTION_HZ     = 1.0
MODEL_PATH = os.path.expanduser(
    '~/Vision_Guided_Autonomous_Wound_Treatment_System/src/wound_tracking/models/best.pt')

WIN_NAME = 'YOLO Wound Pose'


class YoloPoseNode(Node):

    def __init__(self):
        super().__init__('yolo_pose_node')

        self._is_sim       = self.get_parameter('use_sim_time').value
        self._depth_scale  = DEPTH_SCALE_SIM if self._is_sim else DEPTH_SCALE_REAL

        self._bridge         = CvBridge()
        self._model          = YOLO(MODEL_PATH)
        self._latest_color   = None
        self._latest_depth   = None
        self._camera_info    = None
        self._display_frame  = None   # last annotated frame; shown in _color_cb
        self._tf_ready       = False

        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self.create_subscription(Image,      '/camera/color/image_raw',            self._color_cb, 10)
        self.create_subscription(Image,      '/camera/depth_registered/image_raw', self._depth_cb, 10)
        self.create_subscription(CameraInfo, '/camera/color/camera_info',          self._info_cb,  10)

        self._pub_markers    = self.create_publisher(MarkerArray,      '/wound/yolo/markers',     10)
        self._pub_poses      = self.create_publisher(PoseArray,        '/wound/yolo/poses',       10)
        self._pub_detections = self.create_publisher(Detection2DArray, '/wound/yolo/detections',  10)

        self.create_timer(1.0 / DETECTION_HZ, self._timer_cb)

        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_NAME, 960, 540)
        self.get_logger().info(f'yolo_pose_node ready  mode={"SIM" if self._is_sim else "REAL"}')

    def _color_cb(self, msg):
        try:
            self._latest_color = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'color bridge: {e}')
            return
        display = self._display_frame if self._display_frame is not None else self._latest_color
        cv2.imshow(WIN_NAME, display)
        cv2.waitKey(1)

    def _depth_cb(self, msg):
        self._latest_depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def _info_cb(self, msg):
        self._camera_info = msg

    def _timer_cb(self):
        if self._latest_color is None or self._latest_depth is None or self._camera_info is None:
            return

        frame = self._latest_color.copy()
        depth = self._latest_depth.copy()

        K      = self._camera_info.k
        fx, fy = K[0], K[4]
        cx, cy = K[2], K[5]
        K_mat  = np.array(K, dtype=np.float64).reshape(3, 3)
        dist   = np.array(self._camera_info.d, dtype=np.float64)

        img_h, img_w = depth.shape[:2]
        results      = self._model(frame, verbose=False)
        display      = frame.copy()

        wound_poses_cam = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x1 = max(0, x1);  y1 = max(0, y1)
                x2 = min(img_w, x2);  y2 = min(img_h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                conf  = float(box.conf[0])
                label = self._model.names[int(box.cls[0])]

                patch       = depth[y1:y2, x1:x2].astype(np.float64)
                plane_result = self._fit_plane(patch, x1, y1, fx, fy, cx, cy, K_mat, dist)

                if plane_result is not None:
                    centroid_cam, normal_cam = plane_result
                    R_cam = self._normal_to_rotation(normal_cam)
                    wound_idx = len(wound_poses_cam)
                    wound_poses_cam.append((centroid_cam, R_cam, x1, y1, x2, y2, conf, label))
                    box_color  = (0, 220, 0)
                    depth_str  = f'{centroid_cam[2]:.2f}m'
                    text       = f'#{wound_idx} {label} {conf:.2f} | {depth_str}'
                else:
                    box_color = (0, 160, 255)
                    text      = f'{label} {conf:.2f} | no depth'

                cv2.rectangle(display, (x1, y1), (x2, y2), box_color, 2)
                cv2.circle(display, ((x1 + x2) // 2, (y1 + y2) // 2), 4, box_color, -1)
                self._draw_label(display, text, (x1, y1), box_color)

        wound_count = len(wound_poses_cam)
        cv2.putText(display, f'{wound_count} wound(s)', (10, img_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        self._display_frame = display

        if not wound_poses_cam:
            return

        tf_timeout = 1.0 if not self._tf_ready else 0.1
        wound_poses_base = []
        det_array = Detection2DArray()
        det_array.header.stamp    = self.get_clock().now().to_msg()
        det_array.header.frame_id = CAMERA_FRAME

        for centroid_cam, R_cam, x1, y1, x2, y2, conf, label in wound_poses_cam:
            qx, qy, qz, qw = self._rot_to_quat(R_cam)
            pose_cam = PoseStamped()
            pose_cam.header.frame_id    = CAMERA_FRAME
            pose_cam.header.stamp       = rclpy.time.Time().to_msg()
            pose_cam.pose.position.x    = float(centroid_cam[0])
            pose_cam.pose.position.y    = float(centroid_cam[1])
            pose_cam.pose.position.z    = float(centroid_cam[2])
            pose_cam.pose.orientation.x = float(qx)
            pose_cam.pose.orientation.y = float(qy)
            pose_cam.pose.orientation.z = float(qz)
            pose_cam.pose.orientation.w = float(qw)
            try:
                pose_base = self._tf_buffer.transform(
                    pose_cam, BASE_FRAME,
                    timeout=rclpy.duration.Duration(seconds=tf_timeout))
                if not self._tf_ready:
                    self._tf_ready = True
                    self.get_logger().info('TF available — wound tracking active')
                wound_poses_base.append(pose_base)

                det = Detection2D()
                det.header = det_array.header
                det.bbox.center.position.x = float((x1 + x2) / 2)
                det.bbox.center.position.y = float((y1 + y2) / 2)
                det.bbox.size_x = float(x2 - x1)
                det.bbox.size_y = float(y2 - y1)
                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = label
                hyp.hypothesis.score    = conf
                hyp.pose.pose.position.z = float(centroid_cam[2])  # depth_m
                det.results.append(hyp)
                det_array.detections.append(det)

            except (tf2_ros.LookupException,
                    tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException) as e:
                self.get_logger().warn(f'TF: {e}', throttle_duration_sec=2.0)
                return

        self._publish_markers(wound_poses_base)
        self._publish_poses(wound_poses_base)
        self._pub_detections.publish(det_array)
        self.get_logger().info(f'{len(wound_poses_base)} wound(s) pose published',
                               throttle_duration_sec=2.0)

    def _fit_plane(self, patch, u0, v0, fx, fy, cx, cy, K_mat, dist):
        rows, cols = patch.shape[:2]
        vs_grid, us_grid = np.mgrid[0:rows, 0:cols]
        us_abs = (us_grid + u0).ravel().astype(np.float64)
        vs_abs = (vs_grid + v0).ravel().astype(np.float64)
        d_flat = patch.ravel()

        valid = (np.isfinite(d_flat) & (d_flat > 0.0)) if self._is_sim else (d_flat > 0)
        if valid.sum() < MIN_PLANE_POINTS:
            return None

        d_raw = d_flat[valid]
        us_v  = us_abs[valid]
        vs_v  = vs_abs[valid]

        med  = np.median(d_raw)
        std  = np.std(d_raw)
        keep = np.abs(d_raw - med) < 2.0 * std + 1e-9
        if keep.sum() < MIN_PLANE_POINTS:
            return None

        dm   = d_raw[keep] * self._depth_scale
        us_v = us_v[keep]
        vs_v = vs_v[keep]

        pts_dist = np.column_stack([us_v, vs_v]).astype(np.float32).reshape(-1, 1, 2)
        pts_und  = cv2.undistortPoints(pts_dist, K_mat, dist, P=K_mat)
        us_v = pts_und[:, 0, 0].astype(np.float64)
        vs_v = pts_und[:, 0, 1].astype(np.float64)

        X = (us_v - cx) * dm / fx
        Y = (vs_v - cy) * dm / fy
        Z = dm

        pts      = np.column_stack([X, Y, Z])
        centroid = pts.mean(axis=0)

        _, _, Vt = np.linalg.svd(pts - centroid, full_matrices=False)
        normal   = Vt[-1]

        if normal[2] > 0:
            normal = -normal

        return centroid, normal

    def _normal_to_rotation(self, normal_cam):
        tool_z = -normal_cam
        ref    = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(tool_z, ref)) > 0.9:
            ref = np.array([1.0, 0.0, 0.0])
        tool_x = np.cross(ref, tool_z);  tool_x /= np.linalg.norm(tool_x)
        tool_y = np.cross(tool_z, tool_x)
        return np.column_stack([tool_x, tool_y, tool_z])

    def _rot_to_quat(self, R):
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
        x, y, z, w = q.x, q.y, q.z, q.w
        return np.array([2*(x*z + y*w), 2*(y*z - x*w), 1 - 2*(x*x + y*y)])

    def _publish_markers(self, wound_poses_base):
        ma  = MarkerArray()
        now = self.get_clock().now().to_msg()

        # DELETEALL uses id=-1 to avoid colliding with wound disc id=0 in the same array
        delete_all = Marker()
        delete_all.header.frame_id = BASE_FRAME
        delete_all.header.stamp    = now
        delete_all.ns     = 'yolo_wounds'
        delete_all.id     = -1
        delete_all.action = Marker.DELETEALL
        ma.markers.append(delete_all)

        for i, pose_base in enumerate(wound_poses_base):
            approach = self._quat_z_axis(pose_base.pose.orientation)

            disc = Marker()
            disc.header.stamp    = now
            disc.header.frame_id = BASE_FRAME
            disc.ns     = 'yolo_wounds'
            disc.id     = i * 2
            disc.type   = Marker.CYLINDER
            disc.action = Marker.ADD
            disc.pose   = pose_base.pose
            disc.scale.x = 0.06;  disc.scale.y = 0.06;  disc.scale.z = 0.004
            disc.color.r = 1.0;   disc.color.g = 0.2;   disc.color.b = 0.2
            disc.color.a = 0.9
            disc.lifetime.sec = 3
            ma.markers.append(disc)

            tail = Point(
                x=pose_base.pose.position.x - approach[0] * TARGET_OFFSET_M,
                y=pose_base.pose.position.y - approach[1] * TARGET_OFFSET_M,
                z=pose_base.pose.position.z - approach[2] * TARGET_OFFSET_M)
            tip = Point(
                x=pose_base.pose.position.x,
                y=pose_base.pose.position.y,
                z=pose_base.pose.position.z)

            arrow = Marker()
            arrow.header.stamp    = now
            arrow.header.frame_id = BASE_FRAME
            arrow.ns     = 'yolo_wounds'
            arrow.id     = i * 2 + 1
            arrow.type   = Marker.ARROW
            arrow.action = Marker.ADD
            arrow.points  = [tail, tip]
            arrow.scale.x = 0.01
            arrow.scale.y = 0.02
            arrow.color.r = 1.0;  arrow.color.g = 0.6;  arrow.color.a = 1.0
            arrow.lifetime.sec = 3
            ma.markers.append(arrow)

        self._pub_markers.publish(ma)

    def _publish_poses(self, wound_poses_base):
        pa = PoseArray()
        pa.header.stamp    = self.get_clock().now().to_msg()
        pa.header.frame_id = BASE_FRAME
        for pose_base in wound_poses_base:
            pa.poses.append(pose_base.pose)
        self._pub_poses.publish(pa)

    @staticmethod
    def _draw_label(img, text, origin, color, font_scale=0.52, thickness=1):
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        x, y = origin
        y = y + th + baseline + 2  # draw inside top of bbox, never overlaps neighbours
        overlay = img.copy()
        cv2.rectangle(overlay, (x, y - th - baseline - 2), (x + tw + 4, y + baseline), color, cv2.FILLED)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
        cv2.putText(img, text, (x + 2, y - baseline), font, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)


def main(args=None):
    rclpy.init(args=args)
    node = YoloPoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

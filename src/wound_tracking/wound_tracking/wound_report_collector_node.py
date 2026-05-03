import os
import json
import time
from datetime import datetime
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseArray
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
from std_msgs.msg import String
from std_srvs.srv import Trigger

try:
    from wound_tracking.generate_wound_report import generate as _generate_pdf
    _PDF_AVAILABLE = True
except ImportError:
    _PDF_AVAILABLE = False

# Seconds to wait after arriving at a pose before capturing
# (robot settling + at least one fresh YOLO frame)
SETTLE_S      = 2.0

# Two wounds closer than this in 3D (base_link) are the same wound
DEDUP_DIST_M  = 0.08

VIS_THRESHOLD = 0.3

BODY_SEGMENTS = {
    'head':            (3,  6),
    'left_upper_arm':  (11, 13),
    'left_forearm':    (13, 15),
    'left_hand':       (15, 17),
    'right_upper_arm': (12, 14),
    'right_forearm':   (14, 16),
    'right_hand':      (16, 18),
    'left_chest':      (11, 23),
    'right_chest':     (12, 24),
    'abdomen':         (23, 24),
    'left_thigh':      (23, 25),
    'right_thigh':     (24, 26),
    'left_shin':       (25, 27),
    'right_shin':      (26, 28),
}


def _pt_seg_dist(p, a, b):
    ab = b - a
    denom = np.dot(ab, ab)
    if denom < 1e-9:
        return float(np.linalg.norm(p - a))
    t = np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0)
    return float(np.linalg.norm(p - a - t * ab))


def _body_location(cx, cy, landmark_poses):
    if len(landmark_poses) < 33:
        return 'unknown'
    pts = np.array([[p.position.x, p.position.y] for p in landmark_poses])
    vis = np.array([p.orientation.w for p in landmark_poses])
    wp  = np.array([cx, cy], dtype=float)
    best, best_d = 'unknown', float('inf')
    for name, (a, b) in BODY_SEGMENTS.items():
        if vis[a] < VIS_THRESHOLD or vis[b] < VIS_THRESHOLD:
            continue
        d = _pt_seg_dist(wp, pts[a], pts[b])
        if d < best_d:
            best_d, best = d, name
    return best


def _redness_index(crop_bgr):
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0
    b = crop_bgr[:, :, 0].astype(float)
    g = crop_bgr[:, :, 1].astype(float)
    r = crop_bgr[:, :, 2].astype(float)
    return float(np.mean(r) / (np.mean(g) + np.mean(b) + 1.0))


class WoundReportCollectorNode(Node):

    def __init__(self):
        super().__init__('wound_report_collector')

        self._bridge = CvBridge()

        self._latest_detections = None
        self._latest_poses      = None
        self._latest_skeleton   = None
        self._frozen_skeleton   = None   # set once on first detection, never overwritten
        self._latest_color      = None
        self._latest_depth      = None
        self._camera_info       = None

        # Scan-pose trigger state
        self._pending_pose      = None   # label string when waiting to capture
        self._pending_pose_time = 0.0

        # Accumulated unique wounds across all poses
        self._accumulated_wounds = []

        session_id = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self._session_dir = os.path.expanduser(f'~/wound_reports/{session_id}')
        os.makedirs(self._session_dir, exist_ok=True)
        self._session_data = {'session_id': session_id, 'wounds': []}
        self.get_logger().info(f'Session: {self._session_dir}')

        self.create_subscription(Detection2DArray, '/wound/yolo/detections',
                                 self._det_cb,      10)
        self.create_subscription(PoseArray,        '/wound/yolo/poses',
                                 self._poses_cb,    10)
        self.create_subscription(PoseArray,        '/skeleton/landmarks',
                                 self._skeleton_cb, 10)
        self.create_subscription(Image,            '/camera/color/image_raw',
                                 self._color_cb,    10)
        self.create_subscription(Image,            '/camera/depth_registered/image_raw',
                                 self._depth_cb,    10)
        self.create_subscription(CameraInfo,       '/camera/color/camera_info',
                                 self._info_cb,     10)
        self.create_subscription(String,           '/scan/at_pose',
                                 self._at_pose_cb,  10)

        # Timer drives the deferred capture so it doesn't block callbacks
        self.create_timer(0.5, self._capture_tick)

        self.create_service(Trigger, '/wound_report/capture', self._capture_srv)
        self.get_logger().info(
            'Collector ready — awaiting /scan/at_pose signals or call /wound_report/capture')

    def _det_cb(self, msg):
        self._latest_detections = msg

    def _poses_cb(self, msg):
        self._latest_poses = msg

    def _skeleton_cb(self, msg):
        self._latest_skeleton = list(msg.poses)
        # Freeze on first good detection (always frontal at initial pose).
        # Subsequent partial/wrong detections at L/C never overwrite this.
        if self._frozen_skeleton is None and len(msg.poses) >= 33:
            self._frozen_skeleton = list(msg.poses)
            self.get_logger().info('Skeleton frozen from initial pose')

    def _color_cb(self, msg):
        try:
            self._latest_color = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:
            pass

    def _depth_cb(self, msg):
        self._latest_depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def _info_cb(self, msg):
        self._camera_info = msg

    def _at_pose_cb(self, msg: String):
        label = msg.data
        if label == 'done':
            # Flush any pending capture that might not have fired yet
            if self._pending_pose is not None:
                pending = self._pending_pose
                self._pending_pose = None
                self._capture_pose(pending)
            self._finalize()
            return
        self._pending_pose      = label
        self._pending_pose_time = time.time()
        self.get_logger().info(
            f'Arrived at [{label}] — will capture in {SETTLE_S:.0f}s')

    def _capture_tick(self):
        if self._pending_pose is None:
            return
        if time.time() - self._pending_pose_time < SETTLE_S:
            return
        label = self._pending_pose
        self._pending_pose = None
        self._capture_pose(label)

    def _capture_srv(self, request, response):
        if not self._ready():
            response.success = False
            response.message = 'Camera data not ready'
            return response
        if not self._latest_detections or not self._latest_detections.detections:
            response.success = False
            response.message = 'No wound detections available'
            return response
        self._capture_pose('manual')
        response.success = True
        response.message = (f'Captured {len(self._accumulated_wounds)} '
                            f'wound(s) total — {self._session_dir}')
        return response

    def _ready(self):
        return (self._latest_color is not None and
                self._latest_depth is not None and
                self._camera_info  is not None)

    def _is_duplicate(self, pos: dict) -> bool:
        """True if pos is within DEDUP_DIST_M of any already-recorded wound."""
        for w in self._accumulated_wounds:
            ep = w.get('pose_base_link', {})
            if not ep:
                continue
            dist = ((ep['x'] - pos['x'])**2 +
                    (ep['y'] - pos['y'])**2 +
                    (ep['z'] - pos['z'])**2) ** 0.5
            if dist < DEDUP_DIST_M:
                return True
        return False

    def _capture_pose(self, label: str):
        """Snapshot current detections, deduplicate, append unique wounds."""
        if not self._ready():
            self.get_logger().warn(f'[{label}] Camera not ready — skipping capture')
            return
        if not self._latest_detections or not self._latest_detections.detections:
            self.get_logger().info(f'[{label}] No detections — nothing to record')
            return

        frame = self._latest_color.copy()
        depth = self._latest_depth.copy()
        K     = self._camera_info.k
        fx, fy = K[0], K[4]
        img_h, img_w = depth.shape[:2]

        detections = self._latest_detections.detections
        poses      = self._latest_poses.poses if self._latest_poses else []
        # Always use the frozen initial-pose skeleton for body location.
        # Falls back to whatever was last seen if freeze hasn't happened yet.
        landmarks  = self._frozen_skeleton or self._latest_skeleton or []

        new_count = 0
        for i, det in enumerate(detections):
            cx_b = det.bbox.center.position.x
            cy_b = det.bbox.center.position.y
            x1 = max(0,     int(cx_b - det.bbox.size_x / 2))
            y1 = max(0,     int(cy_b - det.bbox.size_y / 2))
            x2 = min(img_w, int(cx_b + det.bbox.size_x / 2))
            y2 = min(img_h, int(cy_b + det.bbox.size_y / 2))

            label_cls = det.results[0].hypothesis.class_id if det.results else 'unknown'
            conf      = float(det.results[0].hypothesis.score) if det.results else 0.0
            depth_m   = float(det.results[0].pose.pose.position.z) if det.results else 0.0

            if depth_m <= 0 and x2 > x1 and y2 > y1:
                patch = depth[y1:y2, x1:x2].astype(float)
                valid = patch[patch > 0]
                if valid.size > 0:
                    depth_m = float(np.median(valid)) * 0.001

            pose_data = {}
            if i < len(poses):
                p = poses[i]
                pose_data = {'x': round(p.position.x, 4),
                             'y': round(p.position.y, 4),
                             'z': round(p.position.z, 4)}

            # Skip if this wound was already recorded from a different pose
            if pose_data and self._is_duplicate(pose_data):
                self.get_logger().info(
                    f'  [{label}] det {i}: duplicate (within {DEDUP_DIST_M*100:.0f}cm) — skip')
                continue

            w_mm     = (x2 - x1) * depth_m / fx * 1000.0 if depth_m > 0 else 0.0
            h_mm     = (y2 - y1) * depth_m / fy * 1000.0 if depth_m > 0 else 0.0
            area_cm2 = w_mm * h_mm / 100.0

            crop      = frame[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else None
            wid       = len(self._accumulated_wounds)
            crop_name = f'wound_{wid}_crop.jpg'
            ctx_name  = f'wound_{wid}_context.jpg'

            if crop is not None and crop.size > 0:
                cv2.imwrite(os.path.join(self._session_dir, crop_name), crop)

            # Full-frame context: raw image with only this wound's bbox
            ctx_frame = frame.copy()
            cv2.rectangle(ctx_frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(ctx_frame, f'#{wid}', (x1, max(12, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2, cv2.LINE_AA)
            cv2.imwrite(os.path.join(self._session_dir, ctx_name), ctx_frame)

            self._accumulated_wounds.append({
                'id':             wid,
                'scan_pose':      label,
                'label':          label_cls,
                'confidence':     round(conf, 3),
                'bbox_px':        [x1, y1, x2, y2],
                'depth_m':        round(depth_m, 3),
                'width_mm':       round(w_mm, 1),
                'height_mm':      round(h_mm, 1),
                'area_cm2':       round(area_cm2, 2),
                'redness_index':  round(_redness_index(crop), 3),
                'body_location':  _body_location(cx_b, cy_b, landmarks),
                'pose_base_link': pose_data,
                'images': {
                    'crop':     crop_name,
                    'skeleton': ctx_name,   # PDF generator reads this key for the right-side image
                },
            })
            new_count += 1

        # Write intermediate JSON so data is safe even if node crashes
        self._flush_json()

        self.get_logger().info(
            f'[{label}] {new_count} new wound(s) | '
            f'{len(self._accumulated_wounds)} total | '
            f'skel={"yes" if landmarks else "no"}')

    def _finalize(self):
        self._flush_json()
        self.get_logger().info(
            f'Scan complete — {len(self._accumulated_wounds)} unique wound(s) | '
            f'{self._session_dir}')
        if _PDF_AVAILABLE:
            try:
                _generate_pdf(self._session_dir)
                self.get_logger().info('PDF report generated')
            except Exception as e:
                self.get_logger().warn(f'PDF generation failed: {e}')

    def _flush_json(self):
        self._session_data['wounds']       = self._accumulated_wounds
        self._session_data['last_updated'] = datetime.now().isoformat()
        with open(os.path.join(self._session_dir, 'session_data.json'), 'w') as f:
            json.dump(self._session_data, f, indent=2)


def main(args=None):
    rclpy.init(args=args)
    node = WoundReportCollectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

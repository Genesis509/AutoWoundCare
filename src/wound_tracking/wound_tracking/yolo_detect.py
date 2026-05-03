import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, BoundingBox2D
from geometry_msgs.msg import Pose2D
import cv2
import numpy as np
from ultralytics import YOLO
import os
import time

FOCAL_X  = 1081.37
FOCAL_Y  = 1081.37
CENTER_X = 959.5
CENTER_Y = 539.5

DEPTH_PATCH_RADIUS = 5
ANALYZE_INTERVAL   = 10

MODEL_PATH = os.path.expanduser(
    '~/Vision_Guided_Autonomous_Wound_Treatment_System/src/wound_tracking/models/best.pt')

class YoloDetectNode(Node):
    def __init__(self):
        super().__init__('yolo_detect_node')
        self.bridge = CvBridge()

        self.get_logger().info(f'Loading YOLO model from: {MODEL_PATH}')
        self.model = YOLO(MODEL_PATH)
        self.get_logger().info('YOLO model loaded.')

        self.latest_color_img   = None
        self.latest_depth_img   = None
        self.last_analysis_time = 0.0
        self.last_result_frame  = None   # annotated snapshot, never mutated after storage

        self.create_subscription(Image, '/camera/color/image_raw',            self.color_callback, 10)
        self.create_subscription(Image, '/camera/depth_registered/image_raw', self.depth_callback, 10)

        self._pub_detections = self.create_publisher(Detection2DArray, '/wound_tracking/detections', 10)

        self.create_timer(0.5, self.timer_callback)

        cv2.namedWindow('YOLO Wound Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('YOLO Wound Detection', 960, 540)
        self.get_logger().info(f'Ready. First analysis in {ANALYZE_INTERVAL}s ...')

    def depth_callback(self, msg):
        self.latest_depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def color_callback(self, msg):
        try:
            self.latest_color_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'CvBridge error: {e}')
            return

        # Build display: snapshot result OR live feed — never mutate stored frames
        display = self.last_result_frame.copy() if self.last_result_frame is not None \
                  else self.latest_color_img.copy()

        remaining = max(0.0, ANALYZE_INTERVAL - (time.time() - self.last_analysis_time))
        self._draw_countdown(display, remaining)

        cv2.imshow('YOLO Wound Detection', display)
        cv2.waitKey(1)

    def timer_callback(self):
        now = time.time()
        if now - self.last_analysis_time < ANALYZE_INTERVAL:
            return
        if self.latest_color_img is None or self.latest_depth_img is None:
            self.get_logger().warn('Waiting for camera data...')
            return

        self.last_analysis_time = now
        self.get_logger().info('── Snapshot analysis ──────────────────────')
        self._analyse(self.latest_color_img.copy(), self.latest_depth_img.copy())

    def _analyse(self, frame, depth):
        h, w  = frame.shape[:2]
        results = self.model(frame, verbose=False)

        det_array = Detection2DArray()
        det_array.header.stamp    = self.get_clock().now().to_msg()
        det_array.header.frame_id = 'camera_color_optical_frame'

        count = 0
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf  = float(box.conf[0])
                cls   = int(box.cls[0])
                label = self.model.names[cls]
                color = self._class_color(cls)

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                z_m = self._sample_depth(cx, cy, h, w, depth)

                if z_m > 0:
                    X = (cx - CENTER_X) * z_m / FOCAL_X
                    Y = (cy - CENTER_Y) * z_m / FOCAL_Y
                    dist_str = f'{z_m:.2f}m'
                    xyz_str  = f'X:{X:.2f} Y:{Y:.2f} Z:{z_m:.2f}'
                else:
                    dist_str = 'depth N/A'
                    xyz_str  = ''

                # Draw bbox and labels on the snapshot frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                self._draw_label(frame, f'{label} {conf:.2f} | {dist_str}', (x1, y1), color)
                if xyz_str:
                    self._draw_label(frame, xyz_str, (x1, y2 + 4), color, bg_alpha=0.4)
                cv2.circle(frame, (cx, cy), 4, color, -1)

                self.get_logger().info(
                    f'  [{label}] conf={conf:.2f}  box=({x1},{y1},{x2},{y2})  depth={dist_str}')

                # Build Detection2D
                d = Detection2D()
                d.header = det_array.header
                d.bbox.center.position.x = float(cx)
                d.bbox.center.position.y = float(cy)
                d.bbox.size_x = float(x2 - x1)
                d.bbox.size_y = float(y2 - y1)
                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = label
                hyp.hypothesis.score    = conf
                d.results.append(hyp)
                det_array.detections.append(d)
                count += 1

        ts = time.strftime('%H:%M:%S')
        cv2.putText(frame, f'Snapshot @ {ts}  |  {count} detection(s)',
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        self.get_logger().info(f'  Total: {count}  (next in {ANALYZE_INTERVAL}s)')
        self._pub_detections.publish(det_array)
        self.last_result_frame = frame   # store once, never touched again

    @staticmethod
    def _draw_countdown(img, seconds_remaining):
        h, w = img.shape[:2]
        bar_w = int(w * (1.0 - seconds_remaining / ANALYZE_INTERVAL))
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (bar_w, 6), (0, 200, 100), cv2.FILLED)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        cv2.putText(img, f'Next analysis in {seconds_remaining:.0f}s', (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 100), 1, cv2.LINE_AA)

    def _sample_depth(self, cx, cy, h, w, depth) -> float:
        r  = DEPTH_PATCH_RADIUS
        x0, x1 = max(cx - r, 0), min(cx + r + 1, w)
        y0, y1 = max(cy - r, 0), min(cy + r + 1, h)
        valid = depth[y0:y1, x0:x1].astype(np.float32)
        valid = valid[valid > 0]
        return 0.0 if valid.size == 0 else float(np.median(valid)) / 1000.0

    @staticmethod
    def _class_color(cls):
        palette = [(0,255,0),(0,128,255),(255,0,0),(0,255,255),
                   (255,0,255),(255,165,0),(0,200,130),(180,0,255)]
        return palette[cls % len(palette)]

    @staticmethod
    def _draw_label(img, text, origin, color, font_scale=0.55, thickness=1, bg_alpha=0.6):
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        x, y = origin
        y = max(y, th + baseline + 2)   # clamp to stay inside frame
        overlay = img.copy()
        cv2.rectangle(overlay, (x, y - th - baseline - 2), (x + tw + 4, y + baseline), color, cv2.FILLED)
        cv2.addWeighted(overlay, bg_alpha, img, 1 - bg_alpha, 0, img)
        cv2.putText(img, text, (x + 2, y - baseline), font, font_scale, (255,255,255), thickness, cv2.LINE_AA)


def main():
    rclpy.init()
    node = YoloDetectNode()
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
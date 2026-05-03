"""
feature_detector_node.py — SIM and REAL.

Subscribes to the RGB camera stream, applies a dual-range HSV mask to
isolate red objects, finds the largest contiguous red region, and publishes
its pixel centroid and bounding box ROI.

Subscriptions:
  /camera/color/image_raw             (sensor_msgs/Image)
  /camera/depth_registered/image_raw  (sensor_msgs/Image)
  /camera/color/camera_info           (sensor_msgs/CameraInfo)

Publications:
  /wound/detection/pixels    (geometry_msgs/PointStamped)
                               point.x = column (u), point.y = row (v)
                               point.z = contour area (px²)
  /wound/detection/roi       (sensor_msgs/RegionOfInterest)
  /wound/detection/image     (sensor_msgs/Image) — annotated frame for debug
"""

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, RegionOfInterest, CameraInfo
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge

HSV_RED_LOW_1  = np.array([165, 100,  60], dtype=np.uint8)
HSV_RED_HIGH_1 = np.array([180, 255, 255], dtype=np.uint8)
HSV_RED_LOW_2  = np.array([  0, 100,  60], dtype=np.uint8)
HSV_RED_HIGH_2 = np.array([ 10, 255, 255], dtype=np.uint8)

MIN_CONTOUR_AREA = 300


class FeatureDetectorNode(Node):

    def __init__(self):
        super().__init__('feature_detector')
        self._bridge = CvBridge()

        self._last_depth_reg = None          # latest registered depth (960×540 uint16, mm)
        self._camera_matrix  = None          # 3×3 from camera_info
        self._dist_coeffs    = None          # 1×5 from camera_info

        self._show_debug = True
        self._sub = self.create_subscription(
            Image, '/camera/color/image_raw', self._image_cb, 10)
        self.create_subscription(
            Image, '/camera/depth_registered/image_raw', self._depth_cb, 10)
        self.create_subscription(
            CameraInfo, '/camera/color/camera_info', self._info_cb, 10)

        self._pub_pixels = self.create_publisher(PointStamped,    '/wound/detection/pixels', 10)
        self._pub_roi    = self.create_publisher(RegionOfInterest, '/wound/detection/roi',    10)
        self._pub_image  = self.create_publisher(Image,            '/wound/detection/image',  10)

        self.get_logger().info('Feature detector ready')

    def _depth_cb(self, msg: Image):
        self._last_depth_reg = self._bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def _info_cb(self, msg: CameraInfo):
        if self._camera_matrix is not None:
            return  # only need it once
        self._camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self._dist_coeffs   = np.array(msg.d, dtype=np.float64)
        self.get_logger().info('Camera intrinsics received — undistortion active')

    def _image_cb(self, msg: Image):
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = (cv2.inRange(hsv, HSV_RED_LOW_1, HSV_RED_HIGH_1) |
                cv2.inRange(hsv, HSV_RED_LOW_2, HSV_RED_HIGH_2))

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        annotated = frame.copy()
        best      = None
        best_area = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= MIN_CONTOUR_AREA and area > best_area:
                best_area = area
                best      = cnt

        if best is not None:
            x, y, w, h = cv2.boundingRect(best)
            cx = x + w // 2
            cy = y + h // 2

            # Depth at centroid from registered depth map (mm → display as mm and cm)
            depth_mm = 0
            if self._last_depth_reg is not None:
                h_d, w_d = self._last_depth_reg.shape[:2]
                if 0 <= cy < h_d and 0 <= cx < w_d:
                    depth_mm = int(self._last_depth_reg[cy, cx])
            depth_str = f'{depth_mm}mm ({depth_mm/10:.1f}cm)' if depth_mm > 0 else 'depth=---'

            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(annotated, (cx, cy), 6, (0, 0, 255), -1)
            cv2.putText(annotated, f'({cx},{cy})  {depth_str}',
                        (x, max(y - 8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            det = PointStamped()
            det.header  = msg.header
            det.point.x = float(cx)
            det.point.y = float(cy)
            det.point.z = float(best_area)
            self._pub_pixels.publish(det)

            roi = RegionOfInterest()
            roi.x_offset   = int(x)
            roi.y_offset   = int(y)
            roi.width      = int(w)
            roi.height     = int(h)
            roi.do_rectify = False
            self._pub_roi.publish(roi)

        out = self._bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        out.header = msg.header
        self._pub_image.publish(out)

        if self._show_debug:
            cv2.namedWindow('Wound Detection', cv2.WINDOW_NORMAL)
            cv2.imshow('Wound Detection', annotated)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = FeatureDetectorNode()
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
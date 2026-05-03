#!/usr/bin/env python3
"""
Standalone OpenCV viewer for wound detection.

Subscribes to the raw camera image, runs HSV red detection locally,
and displays the annotated result in an OpenCV window.

Works identically in sim and real -- same topic, same logic.

Usage (sim):
  ros2 run wound_tracking view_detection

Usage (real):
  ros2 run wound_tracking view_detection
  (camera_driver publishes to the same /camera/color/image_raw topic)
"""

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# HSV thresholds for red (same as feature_detector_node)
HSV_RED_LOW_1  = np.array([165, 100,  60], dtype=np.uint8)
HSV_RED_HIGH_1 = np.array([180, 255, 255], dtype=np.uint8)
HSV_RED_LOW_2  = np.array([  0, 100,  60], dtype=np.uint8)
HSV_RED_HIGH_2 = np.array([ 10, 255, 255], dtype=np.uint8)

MIN_CONTOUR_AREA = 300


class DetectionViewer(Node):

    def __init__(self):
        super().__init__('detection_viewer')
        self._bridge = CvBridge()
        self.create_subscription(
            Image, '/camera/color/image_raw', self._image_cb, 10)
        self.get_logger().info('Detection viewer ready -- press Q to quit')

    def _image_cb(self, msg):
        # Passthrough then convert -- handles both sim (rgb8) and real (bgr8)
        raw = self._bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        if len(raw.shape) == 2:
            frame = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        elif raw.shape[2] == 4:
            frame = cv2.cvtColor(raw, cv2.COLOR_RGBA2BGR)
        elif msg.encoding in ('rgb8', 'rgb16'):
            frame = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
        else:
            frame = raw.copy()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Red mask
        mask = (cv2.inRange(hsv, HSV_RED_LOW_1, HSV_RED_HIGH_1) |
                cv2.inRange(hsv, HSV_RED_LOW_2, HSV_RED_HIGH_2))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find largest contour
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_area = 0.0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= MIN_CONTOUR_AREA and area > best_area:
                best_area = area
                best = cnt

        if best is not None:
            x, y, w, h = cv2.boundingRect(best)
            cx, cy = x + w // 2, y + h // 2

            # Bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Centroid
            cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
            # Crosshair
            cv2.line(frame, (cx - 15, cy), (cx + 15, cy), (0, 255, 255), 1)
            cv2.line(frame, (cx, cy - 15), (cx, cy + 15), (0, 255, 255), 1)
            # Label
            cv2.putText(
                frame,
                f'target ({cx}, {cy})  area={best_area:.0f}px',
                (x, max(y - 10, 16)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.putText(
                frame, 'NO DETECTION', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow('Wound Tracking - Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise SystemExit


def main(args=None):
    rclpy.init(args=args)
    node = DetectionViewer()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

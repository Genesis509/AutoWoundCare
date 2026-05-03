"""
camera_driver_node.py — REAL MODE ONLY.

Reads RGB and depth frames from two V4L2 devices and publishes them as
standard ROS2 camera topics consumed by the rest of the pipeline:

  /camera/color/image_raw    (sensor_msgs/Image, BGR8)
  /camera/depth/image_raw    (sensor_msgs/Image, 16UC1, SR300 Z16 units)
  /camera/color/camera_info  (sensor_msgs/CameraInfo, SR300 intrinsics)

In simulation this node is NOT launched — Gz + ros_gz_bridge publishes to
the same topic names directly.

Depth scale: raw_uint16 × DEPTH_SCALE_MM = depth in mm
             raw_uint16 × DEPTH_SCALE_M  = depth in metres
pose_estimation_node reads the scale from use_sim_time at init.
"""

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

RGB_DEVICE   = 2       # /dev/video2
DEPTH_DEVICE = 4       # /dev/video4  (SR300 Z16 depth stream)

WIDTH  = 640
HEIGHT = 480
FPS    = 30

# SR300 intrinsics (update with your calibration file values if available)
FX = 617.0
FY = 617.0
CX = 320.0
CY = 240.0

FRAME_ID_COLOR = 'camera_color_optical_frame'
FRAME_ID_DEPTH = 'camera_color_optical_frame'  # depth aligned to colour


class CameraDriverNode(Node):

    def __init__(self):
        super().__init__('camera_driver')
        self._bridge = CvBridge()
        self._rgb   = None
        self._depth = None
        self._rgb = cv2.VideoCapture(RGB_DEVICE, cv2.CAP_V4L2)
        self._rgb.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self._rgb.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
        self._rgb.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        self._rgb.set(cv2.CAP_PROP_FPS, FPS)
        if not self._rgb.isOpened():
            self.get_logger().error(f'Cannot open RGB device {RGB_DEVICE}')
            return
        for _ in range(5):   # warm-up: discard first frames
            self._rgb.read()
        self.get_logger().info(f'RGB camera opened  (/dev/video{RGB_DEVICE})')
        self._depth = cv2.VideoCapture(DEPTH_DEVICE, cv2.CAP_V4L2)
        self._depth.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'Z16 '))
        self._depth.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
        self._depth.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        self._depth.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        if not self._depth.isOpened():
            self.get_logger().error(f'Cannot open depth device {DEPTH_DEVICE}')
        else:
            for _ in range(5):
                self._depth.read()
            self.get_logger().info(f'Depth camera opened (/dev/video{DEPTH_DEVICE})')
        self._pub_rgb   = self.create_publisher(Image,      '/camera/color/image_raw',   10)
        self._pub_depth = self.create_publisher(Image,      '/camera/depth/image_raw',   10)
        self._pub_info  = self.create_publisher(CameraInfo, '/camera/color/camera_info', 10)
        self._info_msg = self._build_camera_info()

        self.create_timer(1.0 / FPS, self._timer_cb)
        self.get_logger().info('Camera driver ready')

    def _build_camera_info(self):
        msg = CameraInfo()
        msg.header.frame_id = FRAME_ID_COLOR
        msg.width  = WIDTH
        msg.height = HEIGHT
        msg.distortion_model = 'plumb_bob'
        msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        msg.k = [FX,  0.0, CX,
                 0.0, FY,  CY,
                 0.0, 0.0, 1.0]
        msg.r = [1.0, 0.0, 0.0,
                 0.0, 1.0, 0.0,
                 0.0, 0.0, 1.0]
        msg.p = [FX,  0.0, CX,  0.0,
                 0.0, FY,  CY,  0.0,
                 0.0, 0.0, 1.0, 0.0]
        return msg

    def _timer_cb(self):
        now = self.get_clock().now().to_msg()
        ret, frame = self._rgb.read()
        if ret:
            msg = self._bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            msg.header.stamp    = now
            msg.header.frame_id = FRAME_ID_COLOR
            self._pub_rgb.publish(msg)

            info = self._info_msg
            info.header.stamp = now
            self._pub_info.publish(info)
        if self._depth.isOpened():
            ret2, raw = self._depth.read()
            if ret2:
                depth = self._decode_z16(raw)
                if depth is not None:
                    msg2 = self._bridge.cv2_to_imgmsg(depth, encoding='16UC1')
                    msg2.header.stamp    = now
                    msg2.header.frame_id = FRAME_ID_DEPTH
                    self._pub_depth.publish(msg2)

    def _decode_z16(self, raw):
        """Convert raw V4L2 Z16 bytes to a (H x W) uint16 numpy array."""
        raw_bytes = raw.tobytes()
        n_pixels  = len(raw_bytes) // 2
        if n_pixels == 0:
            return None
        depth = np.frombuffer(raw_bytes, dtype=np.uint16, count=n_pixels)
        if n_pixels == WIDTH * HEIGHT:
            return depth.reshape(HEIGHT, WIDTH)
        self.get_logger().warn(
            f'Unexpected depth size: {n_pixels} pixels', throttle_duration_sec=5.0)
        return None

    def destroy_node(self):
        if self._rgb is not None and self._rgb.isOpened():
            self._rgb.release()
        if self._depth is not None and self._depth.isOpened():
            self._depth.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraDriverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

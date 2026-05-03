"""
kinect2_driver_node.py — Kinect 2 bridge receiver.

Receives RGB and depth frames from Windows-side kinect_bridge.py via ZMQ,
publishes as standard ROS2 camera topics:

  /camera/color/image_raw             (sensor_msgs/Image, BGR8,   960×540)
  /camera/depth/image_raw             (sensor_msgs/Image, 16UC1,  512×424)
  /camera/depth_registered/image_raw  (sensor_msgs/Image, 16UC1,  960×540, aligned to colour)
  /camera/color/camera_info           (sensor_msgs/CameraInfo)
"""

import threading
import time as pytime
import json
import os
import numpy as np
import cv2
import zmq
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from builtin_interfaces.msg import Time
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

WINDOWS_HOST   = '127.0.0.1'   # WSL2 mirrored networking
ZMQ_PORT       = 5555

RGB_W, RGB_H   = 960, 540      # downscaled by bridge
DEPTH_W        = 512
DEPTH_H        = 424

CALIB_FILE = os.path.expanduser(
    '~/Vision_Guided_Autonomous_Wound_Treatment_System/calibration/rgb_intrinsics.json')

# Kinect 2 approximate intrinsics (colour camera, scaled to 960x540) — fallback only
FX = 1081.0 * (960 / 1920)
FY = 1081.0 * (540 / 1080)
CX =  959.5 * (960 / 1920)
CY =  539.5 * (540 / 1080)

FRAME_ID_COLOR = 'camera_color_optical_frame'
FRAME_ID_DEPTH = 'camera_depth_optical_frame'


class Kinect2DriverNode(Node):

    def __init__(self):
        super().__init__('kinect2_driver')
        self._bridge        = CvBridge()
        self._pub_rgb       = self.create_publisher(Image,      '/camera/color/image_raw',            10)
        self._pub_depth     = self.create_publisher(Image,      '/camera/depth/image_raw',            10)
        self._pub_depth_reg = self.create_publisher(Image,      '/camera/depth_registered/image_raw', 10)
        self._pub_info      = self.create_publisher(CameraInfo, '/camera/color/camera_info',          10)
        self._info_msg      = self._build_camera_info()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

        self.get_logger().info(
            f'Kinect2 driver ready — tcp://{WINDOWS_HOST}:{ZMQ_PORT}')
    def _recv_loop(self):
        ctx  = zmq.Context()
        sock = ctx.socket(zmq.SUB)
        sock.connect(f'tcp://{WINDOWS_HOST}:{ZMQ_PORT}')
        sock.setsockopt(zmq.SUBSCRIBE, b'rgb')
        sock.setsockopt(zmq.SUBSCRIBE, b'depth')
        sock.setsockopt(zmq.SUBSCRIBE, b'depth_reg')
        sock.setsockopt(zmq.RCVTIMEO, 200)

        while not self._stop.is_set():
            try:
                topic, data = sock.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                pytime.sleep(0.002)
                continue
            except Exception as e:
                self.get_logger().error(f'ZMQ error: {e}')
                continue

            t = pytime.time()
            now = Time(sec=int(t), nanosec=int((t % 1) * 1e9))

            try:
                if topic == b'rgb':
                    buf   = np.frombuffer(data, dtype=np.uint8)
                    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                    if frame is None:
                        continue
                    
                    frame = cv2.flip(frame, 1)
                    msg = self._bridge.cv2_to_imgmsg(frame, encoding='bgr8')
                    msg.header.stamp    = now
                    msg.header.frame_id = FRAME_ID_COLOR
                    self._pub_rgb.publish(msg)

                    info = self._info_msg
                    info.header.stamp = now
                    self._pub_info.publish(info)

                elif topic == b'depth':
                    depth = np.frombuffer(data, dtype=np.uint16).reshape(DEPTH_H, DEPTH_W)
                    depth = cv2.flip(depth, 1)
                    msg = self._bridge.cv2_to_imgmsg(depth, encoding='16UC1')
                    msg.header.stamp    = now
                    msg.header.frame_id = FRAME_ID_DEPTH
                    self._pub_depth.publish(msg)

                elif topic == b'depth_reg':
                    depth_reg = np.frombuffer(data, dtype=np.uint16).reshape(RGB_H, RGB_W)
                    depth_reg = cv2.flip(depth_reg, 1)
                    msg = self._bridge.cv2_to_imgmsg(depth_reg, encoding='16UC1')
                    msg.header.stamp    = now
                    msg.header.frame_id = FRAME_ID_COLOR  # aligned to colour frame
                    self._pub_depth_reg.publish(msg)

            except Exception as e:
                self.get_logger().error(f'Publish error: {e}', throttle_duration_sec=5.0)

        sock.close()
        ctx.term()

    def _build_camera_info(self):
        fx, fy, cx, cy = FX, FY, CX, CY
        dist = [0.0, 0.0, 0.0, 0.0, 0.0]

        if os.path.isfile(CALIB_FILE):
            try:
                with open(CALIB_FILE) as f:
                    cal = json.load(f)
                K    = cal['rgb_K']
                dist = cal['rgb_dist']
                fx, cx = K[0][0], K[0][2]
                fy, cy = K[1][1], K[1][2]
                self.get_logger().info(
                    f'Loaded calibration from {CALIB_FILE}  '
                    f'(reproj err={cal.get("reprojection_error", "?")})')
            except Exception as e:
                self.get_logger().warn(f'Could not load calibration file: {e} — using defaults')
        else:
            self.get_logger().warn(
                f'No calibration file at {CALIB_FILE} — using approximate intrinsics')

        msg = CameraInfo()
        msg.header.frame_id  = FRAME_ID_COLOR
        msg.width            = RGB_W
        msg.height           = RGB_H
        msg.distortion_model = 'plumb_bob'
        msg.d = dist
        msg.k = [fx,  0.0, cx,
                 0.0, fy,  cy,
                 0.0, 0.0, 1.0]
        msg.r = [1.0, 0.0, 0.0,
                 0.0, 1.0, 0.0,
                 0.0, 0.0, 1.0]
        msg.p = [fx,  0.0, cx,  0.0,
                 0.0, fy,  cy,  0.0,
                 0.0, 0.0, 1.0, 0.0]
        return msg

    def destroy_node(self):
        self._stop.set()
        self._thread.join(timeout=2.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = Kinect2DriverNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

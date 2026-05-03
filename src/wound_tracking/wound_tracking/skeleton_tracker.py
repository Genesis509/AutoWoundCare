import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray, Pose
from cv_bridge import CvBridge
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
from mediapipe import Image as MpImage, ImageFormat

MODEL_PATH = os.path.expanduser(
    '~/Vision_Guided_Autonomous_Wound_Treatment_System/src/wound_tracking/models/pose_landmarker.task')

# Tasks API pose landmarker uses 33 landmarks, same indices as classic mediapipe
POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (11,12),(11,13),(13,15),(15,17),(15,19),(17,19),
    (12,14),(14,16),(16,18),(16,20),(18,20),
    (11,23),(12,24),(23,24),(23,25),(24,26),
    (25,27),(26,28),(27,29),(28,30),(29,31),(30,32),(27,31),(28,32)
]

class KinectSkeletonNode(Node):
    def __init__(self):
        super().__init__('kinect_skeleton_node')
        self.bridge = CvBridge()

        opts = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.IMAGE)
        self.landmarker = PoseLandmarker.create_from_options(opts)

        self.create_subscription(Image, '/camera/color/image_raw',            self.color_callback, 10)
        self.create_subscription(Image, '/camera/depth_registered/image_raw', self.depth_callback, 10)
        self.latest_depth = None
        self._pub_landmarks = self.create_publisher(PoseArray, '/skeleton/landmarks', 10)

    def depth_callback(self, msg):
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def color_callback(self, msg):
        if self.latest_depth is None:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        h, w = frame.shape[:2]

        mp_img = MpImage(image_format=ImageFormat.SRGB,
                         data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = self.landmarker.detect(mp_img)

        if result.pose_landmarks:
            lms = result.pose_landmarks[0]
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]

            for a, b in POSE_CONNECTIONS:
                if a < len(pts) and b < len(pts):
                    cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
            for pt in pts:
                cv2.circle(frame, pt, 4, (0, 0, 255), -1)

            # right wrist = landmark 16
            cx, cy = pts[16]
            if 0 <= cx < w and 0 <= cy < h:
                z_m = self.latest_depth[cy, cx] / 1000.0
                if z_m > 0:
                    pass

            pa = PoseArray()
            pa.header.stamp = self.get_clock().now().to_msg()
            pa.header.frame_id = 'camera_color_optical_frame'
            for lm in lms:
                u = int(lm.x * w)
                v = int(lm.y * h)
                d_raw = self.latest_depth[v, u] if (0 <= v < self.latest_depth.shape[0] and
                                                     0 <= u < self.latest_depth.shape[1]) else 0
                p = Pose()
                p.position.x = float(u)
                p.position.y = float(v)
                p.position.z = float(d_raw) / 1000.0
                p.orientation.w = float(getattr(lm, 'visibility', 1.0))
                pa.poses.append(p)
            self._pub_landmarks.publish(pa)

        cv2.imshow('Kinect Skeleton', frame)
        cv2.waitKey(1)

def main():
    rclpy.init()
    node = KinectSkeletonNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
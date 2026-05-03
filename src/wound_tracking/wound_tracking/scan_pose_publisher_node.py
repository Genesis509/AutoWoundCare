import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import PoseArray, Pose, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from std_srvs.srv import Trigger
import tf2_ros
from tf2_ros import TransformException
import numpy as np

VIS_TH       = 0.5
RADIUS       = 0.85
ANGLES_DEG   = [30.0, 0.0, -30.0]
LABELS       = ['L', 'C', 'R']
COLORS       = [(1.0, 0.2, 0.2), (0.2, 1.0, 0.2), (0.2, 0.2, 1.0)]
TARGET_FRAME = 'base_link'
CAM_FRAME    = 'camera_color_optical_frame'

POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (11,12),(11,13),(13,15),(15,17),(15,19),(17,19),
    (12,14),(14,16),(16,18),(16,20),(18,20),
    (11,23),(12,24),(23,24),(23,25),(24,26),
    (25,27),(26,28),(27,29),(28,30),(29,31),(30,32),(27,31),(28,32)
]


def _quat_from_R(m):
    t = m[0,0] + m[1,1] + m[2,2]
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        return np.array([(m[2,1]-m[1,2])*s, (m[0,2]-m[2,0])*s, (m[1,0]-m[0,1])*s, 0.25/s])
    if m[0,0] > m[1,1] and m[0,0] > m[2,2]:
        s = 2.0 * np.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2])
        return np.array([0.25*s, (m[0,1]+m[1,0])/s, (m[0,2]+m[2,0])/s, (m[2,1]-m[1,2])/s])
    if m[1,1] > m[2,2]:
        s = 2.0 * np.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2])
        return np.array([(m[0,1]+m[1,0])/s, 0.25*s, (m[1,2]+m[2,1])/s, (m[0,2]-m[2,0])/s])
    s = 2.0 * np.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1])
    return np.array([(m[0,2]+m[2,0])/s, (m[1,2]+m[2,1])/s, 0.25*s, (m[1,0]-m[0,1])/s])


def _look_at_optical(pos, target):
    z = target - pos
    z /= np.linalg.norm(z) + 1e-9
    world_up = np.array([0.0, 0.0, 1.0])
    x = np.cross(world_up, z)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(np.array([1.0, 0.0, 0.0]), z)
    x /= np.linalg.norm(x) + 1e-9
    y = np.cross(z, x)
    return _quat_from_R(np.column_stack([x, y, z]))


def _tfm_to_T(tfm):
    t = tfm.transform.translation
    q = tfm.transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    T = np.eye(4)
    T[:3, 3] = [t.x, t.y, t.z]
    T[:3, :3] = np.array([
        [1-2*(y*y+z*z),   2*(x*y-z*w),     2*(x*z+y*w)],
        [2*(x*y+z*w),     1-2*(x*x+z*z),   2*(y*z-x*w)],
        [2*(x*z-y*w),     2*(y*z+x*w),     1-2*(x*x+y*y)],
    ])
    return T


class ScanPosePublisher(Node):
    def __init__(self):
        super().__init__('scan_pose_publisher')

        self._K = None
        self._cached = None
        self._cam_pos_snapshot = None

        self._tf_buf = tf2_ros.Buffer()
        self._tf_lis = tf2_ros.TransformListener(self._tf_buf, self)

        self.create_subscription(CameraInfo, '/camera/color/camera_info', self._info_cb, 10)
        self.create_subscription(PoseArray,  '/skeleton/landmarks',       self._lm_cb,   10)

        self._pub_poses   = self.create_publisher(PoseArray,   '/scan/target_poses', 10)
        self._pub_markers = self.create_publisher(MarkerArray, '/scan/markers',      10)

        self.create_service(Trigger, '/scan/snapshot_reset', self._reset_srv)
        self.create_timer(1.0, self._tick)

        self.get_logger().info('scan_pose_publisher ready — waiting for full skeleton')

    def _info_cb(self, msg):
        self._K = np.array(msg.k).reshape(3, 3)

    def _reset_srv(self, req, res):
        self._cached = None
        self._cam_pos_snapshot = None
        res.success = True
        res.message = 'cache cleared'
        return res

    def _lm_cb(self, msg):
        if self._cached is not None or self._K is None:
            return
        if len(msg.poses) < 33:
            return
        for idx in (11, 12, 23, 24):
            p = msg.poses[idx]
            if p.orientation.w < VIS_TH or p.position.z <= 0:
                return

        try:
            tfm = self._tf_buf.lookup_transform(TARGET_FRAME, CAM_FRAME, rclpy.time.Time())
        except TransformException as e:
            self.get_logger().warn(f'TF not ready: {e}')
            return

        fx, fy = self._K[0, 0], self._K[1, 1]
        cx, cy = self._K[0, 2], self._K[1, 2]
        T = _tfm_to_T(tfm)

        pts = []
        for p in msg.poses:
            u, v, d = p.position.x, p.position.y, p.position.z
            if d <= 0:
                pts.append(np.array([np.nan, np.nan, np.nan]))
                continue
            X = (u - cx) * d / fx
            Y = (v - cy) * d / fy
            ph = np.array([X, Y, d, 1.0])
            pts.append((T @ ph)[:3])

        self._cached = np.array(pts)
        self._cam_pos_snapshot = T[:3, 3].copy()

        C = self._torso_center()
        self.get_logger().info(
            f'Snapshot cached — torso=[{C[0]:.2f},{C[1]:.2f},{C[2]:.2f}] '
            f'(reset via /scan/snapshot_reset)')

    def _torso_center(self):
        return np.mean(self._cached[[11, 12, 23, 24]], axis=0)

    def _compute_poses(self):
        C = self._torso_center()
        sh_mid = 0.5 * (self._cached[11] + self._cached[12])
        hp_mid = 0.5 * (self._cached[23] + self._cached[24])

        up = sh_mid - hp_mid
        up /= np.linalg.norm(up) + 1e-9

        front = self._cam_pos_snapshot - C
        front -= np.dot(front, up) * up
        front /= np.linalg.norm(front) + 1e-9

        right = np.cross(up, front)
        right /= np.linalg.norm(right) + 1e-9

        out = []
        for ang in ANGLES_DEG:
            t = np.radians(ang)
            dirv = np.cos(t) * front + np.sin(t) * right
            pos = C + RADIUS * dirv
            quat = _look_at_optical(pos, C)
            out.append((pos, quat))
        return C, out

    def _tick(self):
        if self._cached is None:
            return
        C, poses = self._compute_poses()
        self._publish_poses(poses)
        self._publish_markers(C, poses)

    def _publish_poses(self, poses):
        pa = PoseArray()
        pa.header.stamp = self.get_clock().now().to_msg()
        pa.header.frame_id = TARGET_FRAME
        for pos, q in poses:
            p = Pose()
            p.position.x, p.position.y, p.position.z = float(pos[0]), float(pos[1]), float(pos[2])
            p.orientation.x = float(q[0])
            p.orientation.y = float(q[1])
            p.orientation.z = float(q[2])
            p.orientation.w = float(q[3])
            pa.poses.append(p)
        self._pub_poses.publish(pa)

    def _publish_markers(self, C, poses):
        ma = MarkerArray()
        now = self.get_clock().now().to_msg()

        torso = Marker()
        torso.header.frame_id = TARGET_FRAME
        torso.header.stamp = now
        torso.ns = 'torso'
        torso.id = 0
        torso.type = Marker.SPHERE
        torso.action = Marker.ADD
        torso.pose.position.x, torso.pose.position.y, torso.pose.position.z = float(C[0]), float(C[1]), float(C[2])
        torso.pose.orientation.w = 1.0
        torso.scale.x = torso.scale.y = torso.scale.z = 0.05
        torso.color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0)
        ma.markers.append(torso)

        for i, p in enumerate(self._cached):
            if np.any(np.isnan(p)):
                continue
            m = Marker()
            m.header.frame_id = TARGET_FRAME
            m.header.stamp = now
            m.ns = 'skel_pts'
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = float(p[0]), float(p[1]), float(p[2])
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.025
            m.color = ColorRGBA(r=0.7, g=0.7, b=0.7, a=1.0)
            ma.markers.append(m)

        line = Marker()
        line.header.frame_id = TARGET_FRAME
        line.header.stamp = now
        line.ns = 'skel_lines'
        line.id = 0
        line.type = Marker.LINE_LIST
        line.action = Marker.ADD
        line.pose.orientation.w = 1.0
        line.scale.x = 0.01
        line.color = ColorRGBA(r=0.5, g=0.5, b=0.5, a=1.0)
        for a, b in POSE_CONNECTIONS:
            pa_pt = self._cached[a]
            pb_pt = self._cached[b]
            if np.any(np.isnan(pa_pt)) or np.any(np.isnan(pb_pt)):
                continue
            line.points.append(Point(x=float(pa_pt[0]), y=float(pa_pt[1]), z=float(pa_pt[2])))
            line.points.append(Point(x=float(pb_pt[0]), y=float(pb_pt[1]), z=float(pb_pt[2])))
        ma.markers.append(line)

        for i, (pos, q) in enumerate(poses):
            r, g, b = COLORS[i]

            arr = Marker()
            arr.header.frame_id = TARGET_FRAME
            arr.header.stamp = now
            arr.ns = 'scan_arrow'
            arr.id = i
            arr.type = Marker.ARROW
            arr.action = Marker.ADD
            arr.pose.orientation.w = 1.0
            arr.points = [
                Point(x=float(pos[0]),  y=float(pos[1]),  z=float(pos[2])),
                Point(x=float(C[0]),    y=float(C[1]),    z=float(C[2])),
            ]
            arr.scale.x = 0.015
            arr.scale.y = 0.03
            arr.scale.z = 0.03
            arr.color = ColorRGBA(r=r, g=g, b=b, a=0.9)
            ma.markers.append(arr)

            sph = Marker()
            sph.header.frame_id = TARGET_FRAME
            sph.header.stamp = now
            sph.ns = 'scan_pos'
            sph.id = i
            sph.type = Marker.SPHERE
            sph.action = Marker.ADD
            sph.pose.position.x, sph.pose.position.y, sph.pose.position.z = float(pos[0]), float(pos[1]), float(pos[2])
            sph.pose.orientation.w = 1.0
            sph.scale.x = sph.scale.y = sph.scale.z = 0.04
            sph.color = ColorRGBA(r=r, g=g, b=b, a=1.0)
            ma.markers.append(sph)

            txt = Marker()
            txt.header.frame_id = TARGET_FRAME
            txt.header.stamp = now
            txt.ns = 'scan_label'
            txt.id = i
            txt.type = Marker.TEXT_VIEW_FACING
            txt.action = Marker.ADD
            txt.pose.position.x = float(pos[0])
            txt.pose.position.y = float(pos[1])
            txt.pose.position.z = float(pos[2]) + 0.07
            txt.pose.orientation.w = 1.0
            txt.scale.z = 0.06
            txt.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            txt.text = LABELS[i]
            ma.markers.append(txt)

        self._pub_markers.publish(ma)


def main():
    rclpy.init()
    node = ScanPosePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

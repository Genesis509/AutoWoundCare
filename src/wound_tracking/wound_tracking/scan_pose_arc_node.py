"""scan_pose_arc_node — singularity-aware viewpoint selection.

Drop-in replacement for scan_pose_publisher.  Generates a dense arc of camera
candidates on a sphere around the torso, in the lateral plane through the
current EE direction, then filters by reachability + singularity and scores by
manipulability proxy + wrist_3 delta + angular spread.  Top 3 published as L,C,R
on /scan/target_poses (same topic as the legacy node) so scan_loop is unchanged.
"""
import threading
import time

import numpy as np

import rclpy
from rclpy.node import Node

import tf2_ros
from tf2_ros import TransformException

from sensor_msgs.msg import CameraInfo, JointState
from geometry_msgs.msg import PoseArray, Pose, PoseStamped, Point
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from std_srvs.srv import Trigger
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import RobotState

VIS_TH         = 0.5
RADIUS         = 0.78
RADIUS_STEPS   = [0.0, 0.05, 0.10, 0.15]   # fallback: step away if IK fails at ideal radius
THETA_MAX_DEG  = 60.0
N_SAMPLES      = 15
MIN_SPREAD_DEG = 25.0
MAX_SPREAD_DEG = 45.0

WRIST2_MIN_SIN       = 0.20
ELBOW_MIN_SIN        = 0.15
JOINT_LIMIT          = 2.0 * np.pi * 0.95
PATH_MANIP_N         = 10
PATH_MANIP_MIN       = 0.03
WRIST3_MAX_TRAVEL    = 2.5
WRIST3_INIT_MAX_DEG  = 25.0    # |q6 - q6_init| cap per target — prevents camera arming
ROLL_SAMPLES         = 8       # camera-roll sweep about optical Z
JOINT_MAX_TRAVEL     = 3.5
IK_TIMEOUT           = 0.10
IK_WAIT_SLACK        = 0.20

PLANNING_GROUP = 'ur_manipulator'
EE_LINK        = 'tool0'
TARGET_FRAME   = 'base_link'
CAM_FRAME      = 'camera_color_optical_frame'

JOINT_NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
               'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

LABELS = ['L', 'C', 'R']
COLORS = [(1.0, 0.2, 0.2), (0.2, 1.0, 0.2), (0.2, 0.4, 1.0)]

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


def _look_at_optical(pos, target, phi=0.0):
    """Build optical-frame quat: z points from pos to target, free roll phi about z."""
    z = target - pos
    z /= np.linalg.norm(z) + 1e-9
    world_up = np.array([0.0, 0.0, 1.0])
    x0 = np.cross(world_up, z)
    if np.linalg.norm(x0) < 1e-6:
        x0 = np.cross(np.array([1.0, 0.0, 0.0]), z)
    x0 /= np.linalg.norm(x0) + 1e-9
    y0 = np.cross(z, x0)
    c, s = np.cos(phi), np.sin(phi)
    x = c * x0 + s * y0
    y = -s * x0 + c * y0
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


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _path_min_manip(q_start, q_end, n=PATH_MANIP_N):
    """Min |sin q5|·|sin q3| along the linear joint-space segment Pilz PTP executes."""
    ts = np.linspace(0.0, 1.0, n)
    m = 1.0
    for t in ts:
        q = (1.0 - t) * q_start + t * q_end
        v = abs(np.sin(q[4])) * abs(np.sin(q[2]))
        if v < m:
            m = v
    return m


class ScanPoseArc(Node):
    def __init__(self):
        super().__init__('scan_pose_arc')

        self._K = None
        self._cached_pts = None
        self._ee_pos = None
        self._joints_now = None
        self._result_poses = None
        self._result_joints = None
        self._diag = []
        self._busy = False

        self._tf_buf = tf2_ros.Buffer()
        self._tf_lis = tf2_ros.TransformListener(self._tf_buf, self)

        self.create_subscription(CameraInfo, '/camera/color/camera_info', self._info_cb, 10)
        self.create_subscription(PoseArray,  '/skeleton/landmarks',       self._lm_cb,   10)
        self.create_subscription(JointState, '/joint_states',             self._js_cb,   10)

        self._pub_poses   = self.create_publisher(PoseArray,       '/scan/target_poses',  10)
        self._pub_joints  = self.create_publisher(JointTrajectory, '/scan/target_joints', 10)
        self._pub_markers = self.create_publisher(MarkerArray,     '/scan/markers',       10)

        self._ik = self.create_client(GetPositionIK, '/compute_ik')

        self.create_service(Trigger, '/scan/snapshot_reset', self._reset_srv)
        self.create_timer(1.0, self._tick)

        self.get_logger().info('scan_pose_arc ready — waiting for skeleton + joints + /compute_ik')

    def _info_cb(self, m):
        self._K = np.array(m.k).reshape(3, 3)

    def _js_cb(self, m):
        idx = {n: i for i, n in enumerate(m.name)}
        try:
            self._joints_now = np.array([m.position[idx[n]] for n in JOINT_NAMES])
        except KeyError:
            return

    def _reset_srv(self, req, res):
        self._cached_pts = None
        self._ee_pos = None
        self._result_poses = None
        self._result_joints = None
        self._diag = []
        res.success = True
        res.message = 'cache cleared'
        return res

    def _lm_cb(self, msg):
        if self._cached_pts is not None or self._busy or self._K is None:
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

        self._cached_pts = np.array(pts)
        self._ee_pos = T[:3, 3].copy()
        self._busy = True

        C = self._torso_C()
        self.get_logger().info(
            f'Snapshot cached — torso=[{C[0]:.2f},{C[1]:.2f},{C[2]:.2f}], starting arc compute')
        threading.Thread(target=self._compute, daemon=True).start()

    def _torso_C(self):
        return np.mean(self._cached_pts[[11, 12, 23, 24]], axis=0)

    def _compute(self):
        try:
            if not self._ik.wait_for_service(timeout_sec=10.0):
                self.get_logger().error('/compute_ik unavailable — is move_group running?')
                return
            if self._joints_now is None:
                self.get_logger().error('No /joint_states yet')
                return

            C = self._torso_C()
            sh_mid = 0.5 * (self._cached_pts[11] + self._cached_pts[12])
            hp_mid = 0.5 * (self._cached_pts[23] + self._cached_pts[24])
            u = sh_mid - hp_mid
            u /= np.linalg.norm(u) + 1e-9

            f = self._ee_pos - C
            f /= np.linalg.norm(f) + 1e-9
            r = np.cross(f, u)
            if np.linalg.norm(r) < 1e-6:
                r = np.cross(f, np.array([1.0, 0.0, 0.0]))
            r /= np.linalg.norm(r) + 1e-9

            thetas = np.linspace(-np.radians(THETA_MAX_DEG),
                                  np.radians(THETA_MAX_DEG), N_SAMPLES)

            ik_results = []
            for th in thetas:
                dirv = np.cos(th) * f + np.sin(th) * r
                ok, q, quat_used = False, None, None
                pos_used, reason = C + RADIUS * dirv, 'no_ik'
                for extra in RADIUS_STEPS:
                    pos = C + (RADIUS + extra) * dirv
                    ok, q, quat_used, reason = self._eval_ik_best_roll(pos, C)
                    if ok:
                        pos_used = pos
                        break
                if quat_used is None:
                    quat_used = _look_at_optical(pos_used, C)
                ep_manip = (abs(np.sin(q[4])) * abs(np.sin(q[2]))) if ok else 0.0
                ik_results.append({'th': th, 'pos': pos_used, 'quat': quat_used,
                                   'q': q, 'ok': ok, 'reason': reason,
                                   'ep_manip': ep_manip, 'picked': False})

            valid = [c for c in ik_results if c['ok']]

            for c in ik_results:
                tag = 'OK  ' if c['ok'] else 'REJ '
                self.get_logger().info(
                    f"{tag} θ={np.degrees(c['th']):+5.1f}° "
                    f"ep_m={c['ep_manip']:.3f} {c['reason']}")

            if not valid:
                self.get_logger().warn('No valid IK solutions — check skeleton, reset')
                self._diag = ik_results
                return

            # Phase 1: center = valid candidate closest to theta=0 (EE-torso axis)
            center = min(valid, key=lambda c: abs(c['th']))

            # Phase 2: pick most extreme L at least MIN_SPREAD_DEG away from
            # center, subject to sequence_metrics check
            min_spread = np.radians(MIN_SPREAD_DEG)
            max_spread = np.radians(MAX_SPREAD_DEG)
            left_cands  = sorted(
                [c for c in valid
                 if center['th'] - max_spread <= c['th'] < center['th'] - min_spread],
                key=lambda c: c['th'])           # most negative first (widest spread first)

            if not left_cands:
                self.get_logger().warn(
                    f'No left candidate {MIN_SPREAD_DEG}-{MAX_SPREAD_DEG}° from '
                    f'center θ={np.degrees(center["th"]):.1f}° — reset or widen arc')
                self._diag = ik_results
                return

            best_picks, best_unw, best_spread = None, None, -1.0
            for lc in left_cands:
                seq = self._sequence_metrics([lc['q'], center['q']])
                if seq is None:
                    continue
                spread = center['th'] - lc['th']
                if spread > best_spread:
                    best_spread = spread
                    best_picks = [lc, center]
                    _, _, _, best_unw = seq

            if best_picks is None:
                self.get_logger().warn('No valid L-C sequence — check joint limits, reset')
                self._diag = ik_results
                return

            # Phase 3: R — mirror of L on the positive-θ side.
            # Feasibility checked from C's unwrapped joints (the actual departure state).
            right_cands = sorted(
                [c for c in valid
                 if center['th'] + min_spread < c['th'] <= center['th'] + max_spread],
                key=lambda c: c['th'], reverse=True)   # most positive = widest spread first

            r_pick = None
            r_unw  = None
            q_C    = best_unw[1]   # C's unwrapped joints
            for rc in right_cands:
                q_R = q_C + np.array([_wrap(rc['q'][i] - q_C[i]) for i in range(6)])
                if np.any(np.abs(q_R) > JOINT_LIMIT):
                    continue
                delta = q_R - q_C
                if float(abs(delta[5])) > WRIST3_MAX_TRAVEL:
                    continue
                if float(np.max(np.abs(delta))) > JOINT_MAX_TRAVEL:
                    continue
                if _path_min_manip(q_C, q_R) < PATH_MANIP_MIN:
                    continue
                r_pick = rc
                r_unw  = q_R
                break

            if r_pick is None:
                self.get_logger().warn(
                    'No valid R candidate — using C joints as R fallback')
                r_pick = center
                r_unw  = best_unw[1].copy()
            else:
                r_pick['picked'] = True

            all_picks = best_picks + [r_pick]
            all_unw   = best_unw   + [r_unw]

            for c in all_picks:
                c['picked'] = True
            self.get_logger().info(
                f'Selected: L={np.degrees(best_picks[0]["th"]):+.1f}°  '
                f'C={np.degrees(center["th"]):+.1f}°  '
                f'R={np.degrees(r_pick["th"]):+.1f}°  '
                f'spread={np.degrees(best_spread):.1f}°')

            self._result_poses  = [(c['pos'], c['quat']) for c in all_picks]
            self._result_joints = all_unw
            self._diag = ik_results
            self.get_logger().info(
                f'Publishing {len(all_picks)} arc targets '
                f'(/scan/target_poses + /scan/target_joints)')
        finally:
            self._busy = False

    def _sequence_metrics(self, q_list):
        """Simulate executed sequence current_q → q_list[0] → q_list[1] → ...

        Returns (min_path_manip, sum_|Δq6|, max_max_|Δq|, unwrapped_qs) or None
        if any segment violates joint limits, travel caps, or path-singularity.
        """
        prev = self._joints_now.copy()
        unw = []
        pm_min, dw3_sum, mt_max = 1.0, 0.0, 0.0
        for q in q_list:
            q_unw = prev + np.array([_wrap(q[i] - prev[i]) for i in range(6)])
            if np.any(np.abs(q_unw) > JOINT_LIMIT):
                return None
            delta = q_unw - prev
            mt  = float(np.max(np.abs(delta)))
            dw3 = float(abs(delta[5]))
            if dw3 > WRIST3_MAX_TRAVEL or mt > JOINT_MAX_TRAVEL:
                return None
            pm = _path_min_manip(prev, q_unw)
            if pm < PATH_MANIP_MIN:
                return None
            pm_min = min(pm_min, pm)
            dw3_sum += dw3
            mt_max = max(mt_max, mt)
            unw.append(q_unw)
            prev = q_unw
        return pm_min, dw3_sum, mt_max, unw

    def _eval_ik_best_roll(self, pos, target):
        """IK over camera roll about optical Z. Returns (ok, q, quat, reason).

        Look direction (camera Z → torso C) is preserved exactly; only the
        in-plane camera roll varies, which maps to wrist_3 since tool0 Z is
        co-aligned with optical Z in this setup.  Picks the IK solution with
        the smallest |q6 − q6_init|, capped at WRIST3_INIT_MAX_DEG.
        """
        w3_max = np.radians(WRIST3_INIT_MAX_DEG)
        q5_init = self._joints_now[5]
        best = None
        last_reason = 'no_ik'
        for phi in np.linspace(-np.pi, np.pi, ROLL_SAMPLES, endpoint=False):
            quat = _look_at_optical(pos, target, phi)
            ok, q, reason = self._eval_ik(pos, quat)
            if not ok:
                last_reason = reason
                continue
            d6 = abs(_wrap(q[5] - q5_init))
            if d6 > w3_max:
                last_reason = f'w3_{int(np.degrees(d6))}d'
                continue
            if best is None or d6 < best[0]:
                best = (d6, q, quat)
        if best is None:
            return False, None, None, last_reason
        return True, best[1], best[2], 'ok'

    def _eval_ik(self, pos, quat):
        req = GetPositionIK.Request()
        req.ik_request.group_name   = PLANNING_GROUP
        req.ik_request.ik_link_name = EE_LINK
        req.ik_request.avoid_collisions = True

        ps = PoseStamped()
        ps.header.frame_id = TARGET_FRAME
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = float(pos[0])
        ps.pose.position.y = float(pos[1])
        ps.pose.position.z = float(pos[2])
        ps.pose.orientation.x = float(quat[0])
        ps.pose.orientation.y = float(quat[1])
        ps.pose.orientation.z = float(quat[2])
        ps.pose.orientation.w = float(quat[3])
        req.ik_request.pose_stamped = ps

        rs = RobotState()
        rs.joint_state.name     = list(JOINT_NAMES)
        rs.joint_state.position = [float(v) for v in self._joints_now]
        req.ik_request.robot_state = rs

        req.ik_request.timeout.sec = 0
        req.ik_request.timeout.nanosec = int(IK_TIMEOUT * 1e9)

        future = self._ik.call_async(req)
        deadline = time.time() + IK_TIMEOUT + IK_WAIT_SLACK
        while not future.done() and time.time() < deadline:
            time.sleep(0.005)
        if not future.done():
            return False, None, 'ik_timeout'

        resp = future.result()
        if resp.error_code.val != 1:
            return False, None, f'ik_err{resp.error_code.val}'

        idx = {n: i for i, n in enumerate(resp.solution.joint_state.name)}
        try:
            q = np.array([resp.solution.joint_state.position[idx[n]] for n in JOINT_NAMES])
        except KeyError:
            return False, None, 'joints_missing'

        if np.any(np.abs(q) > JOINT_LIMIT):
            return False, q, 'joint_limit'
        if abs(np.sin(q[4])) < WRIST2_MIN_SIN:
            return False, q, 'wrist_sing'
        if abs(np.sin(q[2])) < ELBOW_MIN_SIN:
            return False, q, 'elbow_sing'
        return True, q, 'ok'

    def _tick(self):
        if self._result_poses is None:
            return
        self._publish_poses()
        self._publish_joints()
        self._publish_markers()

    def _publish_joints(self):
        if not self._result_joints:
            return
        jt = JointTrajectory()
        jt.header.stamp = self.get_clock().now().to_msg()
        jt.header.frame_id = TARGET_FRAME
        jt.joint_names = list(JOINT_NAMES)
        for q in self._result_joints:
            pt = JointTrajectoryPoint()
            pt.positions = [float(v) for v in q]
            jt.points.append(pt)
        self._pub_joints.publish(jt)

    def _publish_poses(self):
        pa = PoseArray()
        pa.header.stamp = self.get_clock().now().to_msg()
        pa.header.frame_id = TARGET_FRAME
        for pos, q in self._result_poses:
            p = Pose()
            p.position.x, p.position.y, p.position.z = float(pos[0]), float(pos[1]), float(pos[2])
            p.orientation.x = float(q[0])
            p.orientation.y = float(q[1])
            p.orientation.z = float(q[2])
            p.orientation.w = float(q[3])
            pa.poses.append(p)
        self._pub_poses.publish(pa)

    def _publish_markers(self):
        ma = MarkerArray()
        now = self.get_clock().now().to_msg()
        C = self._torso_C()

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

        for i, p in enumerate(self._cached_pts):
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
            pa_pt = self._cached_pts[a]
            pb_pt = self._cached_pts[b]
            if np.any(np.isnan(pa_pt)) or np.any(np.isnan(pb_pt)):
                continue
            line.points.append(Point(x=float(pa_pt[0]), y=float(pa_pt[1]), z=float(pa_pt[2])))
            line.points.append(Point(x=float(pb_pt[0]), y=float(pb_pt[1]), z=float(pb_pt[2])))
        ma.markers.append(line)

        for i, e in enumerate(self._diag):
            m = Marker()
            m.header.frame_id = TARGET_FRAME
            m.header.stamp = now
            m.ns = 'arc_cloud'
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(e['pos'][0])
            m.pose.position.y = float(e['pos'][1])
            m.pose.position.z = float(e['pos'][2])
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.025
            if e['ok']:
                m.color = ColorRGBA(r=0.2, g=1.0, b=0.2, a=0.8)
            else:
                m.color = ColorRGBA(r=1.0, g=0.2, b=0.2, a=0.6)
            ma.markers.append(m)

            txt = Marker()
            txt.header.frame_id = TARGET_FRAME
            txt.header.stamp = now
            txt.ns = 'arc_label'
            txt.id = i
            txt.type = Marker.TEXT_VIEW_FACING
            txt.action = Marker.ADD
            txt.pose.position.x = float(e['pos'][0])
            txt.pose.position.y = float(e['pos'][1])
            txt.pose.position.z = float(e['pos'][2]) + 0.04
            txt.pose.orientation.w = 1.0
            txt.scale.z = 0.025
            txt.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.9)
            if e['ok']:
                txt.text = f"{np.degrees(e['th']):+.0f}\nm={e['ep_manip']:.2f}"
            else:
                txt.text = f"{np.degrees(e['th']):+.0f}\n{e['reason']}"
            ma.markers.append(txt)

        for i, (pos, q) in enumerate(self._result_poses):
            r, g, b = COLORS[i] if i < len(COLORS) else (1.0, 1.0, 0.0)

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
            sph.pose.position.x = float(pos[0])
            sph.pose.position.y = float(pos[1])
            sph.pose.position.z = float(pos[2])
            sph.pose.orientation.w = 1.0
            sph.scale.x = sph.scale.y = sph.scale.z = 0.05
            sph.color = ColorRGBA(r=r, g=g, b=b, a=1.0)
            ma.markers.append(sph)

            lbl = Marker()
            lbl.header.frame_id = TARGET_FRAME
            lbl.header.stamp = now
            lbl.ns = 'scan_label'
            lbl.id = i
            lbl.type = Marker.TEXT_VIEW_FACING
            lbl.action = Marker.ADD
            lbl.pose.position.x = float(pos[0])
            lbl.pose.position.y = float(pos[1])
            lbl.pose.position.z = float(pos[2]) + 0.08
            lbl.pose.orientation.w = 1.0
            lbl.scale.z = 0.06
            lbl.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            lbl.text = LABELS[i] if i < len(LABELS) else str(i)
            ma.markers.append(lbl)

        self._pub_markers.publish(ma)


def main():
    rclpy.init()
    node = ScanPoseArc()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
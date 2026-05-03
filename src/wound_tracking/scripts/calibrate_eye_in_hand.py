#!/usr/bin/env python3
"""
Eye-in-hand calibration — Kinect 2 on UR16e.

Camera pose from /camera/color/image_raw (ROS2 topic via kinect2_driver).
Robot EEF pose from TF2 base_link -> tool0 (ur_robot_driver).
Board: same checkerboard used for intrinsic calibration (9x6 inner corners, 25mm).

SPACE = capture pose pair   F = toggle freedrive   C = compute & save   Q = quit

Acceptance checks before a capture is stored:
  1. Robot is still — two TF samples ~150 ms apart agree within 2 mm / 0.5°.
  2. Frame is fresh — image stamp within 0.5 s of now.
  3. Gripper TF is looked up AT THE FRAME'S TIMESTAMP (sync, not "latest").
  4. Rotation diversity — total angle ≥15° AND ≥2 of (roll,pitch,yaw) differ
     by ≥15° from the nearest existing pose (prevents yaw-only clusters).

Output: ~/Vision_Guided.../calibration/eye_in_hand.json
        ~/Vision_Guided.../calibration/eye_in_hand_dataset/
"""
import threading
import json
import os
import sys
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from std_srvs.srv import Trigger
from cv_bridge import CvBridge
import tf2_ros
BASE_DIR   = os.path.expanduser('~/Vision_Guided_Autonomous_Wound_Treatment_System/calibration')
INTR_FILE  = os.path.join(BASE_DIR, 'rgb_intrinsics.json')
CALIB_FILE = os.path.join(BASE_DIR, 'eye_in_hand.json')
DATA_DIR   = os.path.join(BASE_DIR, 'eye_in_hand_dataset')
BOARD_W   = 9
BOARD_H   = 6
SQUARE_MM = 25.0
BASE_FRAME = 'base_link'
EE_FRAME   = 'tool0'

MIN_CAPTURES = 20
STILL_TRANS_MM     = 2.0      # gripper translation change between 2 TF samples
STILL_ROT_DEG      = 0.5      # gripper rotation change between 2 TF samples
STILL_DT_SEC       = 0.15     # interval between the 2 stillness samples
FRAME_MAX_AGE_SEC  = 0.5      # reject if camera image is older than this
MIN_TOTAL_ROT_DEG  = 15.0     # min total rotation vs nearest existing pose
MIN_AXES_DIVERSE   = 2        # min number of euler axes (of 3) with ≥15° diff
PER_AXIS_MIN_DEG   = 15.0

class CalibNode(Node):

    def __init__(self):
        super().__init__('eye_in_hand_calib')
        self._bridge = CvBridge()
        self.frame        = None
        self.frame_stamp  = None     # builtin_interfaces/Time from image header
        self.K            = None
        self.dist         = None
        self._lock        = threading.Lock()

        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self.create_subscription(Image,      '/camera/color/image_raw',   self._img_cb,  10)
        self.create_subscription(CameraInfo, '/camera/color/camera_info', self._info_cb, 10)

        # URScript + dashboard for freedrive toggle
        self._urscript_pub = self.create_publisher(
            String, '/urscript_interface/script_command', 10)
        self._dashboard_play = self.create_client(Trigger, '/dashboard_client/play')

    def enter_freedrive(self):
        """Push URScript that holds freedrive_mode() forever. Interrupts
        external_control.urp until exit_freedrive() is called."""
        s = String()
        s.data = (
            "def ros_freedrive():\n"
            "  freedrive_mode()\n"
            "  while (True):\n"
            "    sync()\n"
            "  end\n"
            "end\n"
        )
        self._urscript_pub.publish(s)

    def exit_freedrive(self):
        """Stop freedrive and ask the dashboard to re-Play external_control."""
        s = String()
        s.data = (
            "def ros_end_freedrive():\n"
            "  end_freedrive_mode()\n"
            "end\n"
        )
        self._urscript_pub.publish(s)
        # Restart external_control.urp so normal command streaming resumes
        if self._dashboard_play.wait_for_service(timeout_sec=1.5):
            self._dashboard_play.call_async(Trigger.Request())
        else:
            self.get_logger().warn(
                '/dashboard_client/play unavailable — press Play on the '
                'pendant to restore external_control')

    def _img_cb(self, msg):
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        with self._lock:
            self.frame = frame
            self.frame_stamp = msg.header.stamp

    def _info_cb(self, msg):
        if self.K is not None:
            return
        self.K    = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.dist = np.array(msg.d, dtype=np.float64)
        self.get_logger().info('Camera intrinsics received')

    def get_frame(self):
        """Returns (frame, stamp) or (None, None)."""
        with self._lock:
            if self.frame is None:
                return None, None
            return self.frame.copy(), self.frame_stamp

    def get_eef_pose(self, stamp=None):
        """Returns (R 3x3, t 3x1) at the given ROS time (or latest if None)."""
        try:
            if stamp is None:
                tf_time = rclpy.time.Time()
            else:
                tf_time = rclpy.time.Time.from_msg(stamp)
            tf = self._tf_buffer.lookup_transform(
                BASE_FRAME, EE_FRAME,
                tf_time,
                timeout=rclpy.duration.Duration(seconds=0.2))
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f'TF lookup failed: {e}')
            return None

        q = tf.transform.rotation
        t = tf.transform.translation
        R = _quat_to_rotmat(q.x, q.y, q.z, q.w)
        return R, np.array([[t.x], [t.y], [t.z]])

def _quat_to_rotmat(x, y, z, w):
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),     2*(x*z + y*w)  ],
        [    2*(x*y + z*w),   1 - 2*(x*x + z*z), 2*(y*z - x*w)  ],
        [    2*(x*z - y*w),   2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


def _rvec_to_rotmat(rvec):
    R, _ = cv2.Rodrigues(rvec)
    return R


def _rotmat_to_euler_deg(R):
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    if sy > 1e-6:
        roll  = np.degrees(np.arctan2( R[2, 1], R[2, 2]))
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        yaw   = np.degrees(np.arctan2( R[1, 0], R[0, 0]))
    else:
        roll  = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        yaw   = 0.0
    return roll, pitch, yaw


def _angle_between_rotmats(R1, R2):
    R_rel = R1.T @ R2
    cos_a = np.clip((np.trace(R_rel) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(cos_a))


def _wrap_180(d):
    """Wrap angle difference to [-180, 180]."""
    return (d + 180.0) % 360.0 - 180.0


def _stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9

def check_robot_still(node):
    """Sample EEF twice ~STILL_DT_SEC apart, pass if change is tiny."""
    p1 = node.get_eef_pose()
    if p1 is None:
        return False, 'TF not available'
    time.sleep(STILL_DT_SEC)
    p2 = node.get_eef_pose()
    if p2 is None:
        return False, 'TF not available (2nd sample)'
    R1, t1 = p1
    R2, t2 = p2
    dt = float(np.linalg.norm(t2 - t1) * 1000.0)
    dr = float(_angle_between_rotmats(R1, R2))
    if dt > STILL_TRANS_MM or dr > STILL_ROT_DEG:
        return False, f'robot moving (Δt={dt:.1f}mm Δr={dr:.2f}°)'
    return True, f'still (Δt={dt:.1f}mm Δr={dr:.2f}°)'


def check_rotation_diversity(R_new, R_existing):
    """Return (ok, note). Rejects if poses clump on one Euler axis."""
    if not R_existing:
        return True, '(first pose)'
    total_dists = [_angle_between_rotmats(R_new, R) for R in R_existing]
    min_total = min(total_dists)
    if min_total < MIN_TOTAL_ROT_DEG:
        return False, f'total rotation only {min_total:.1f}° from nearest'

    roll_n, pitch_n, yaw_n = _rotmat_to_euler_deg(R_new)
    # for EACH existing pose, count axes where the new pose differs by ≥threshold
    # require that at least MIN_AXES_DIVERSE axes differ from the CLOSEST pose
    nearest_idx = int(np.argmin(total_dists))
    R_near = R_existing[nearest_idx]
    r_e, p_e, y_e = _rotmat_to_euler_deg(R_near)
    dr = abs(_wrap_180(roll_n  - r_e))
    dp = abs(_wrap_180(pitch_n - p_e))
    dy = abs(_wrap_180(yaw_n   - y_e))
    axes_diverse = sum([dr >= PER_AXIS_MIN_DEG,
                        dp >= PER_AXIS_MIN_DEG,
                        dy >= PER_AXIS_MIN_DEG])
    if axes_diverse < MIN_AXES_DIVERSE:
        return False, (f'only {axes_diverse} axis differs vs nearest '
                       f'(Δroll={dr:.1f}° Δpitch={dp:.1f}° Δyaw={dy:.1f}°)')
    return True, f'rot_dist={min_total:.1f}° axes_diverse={axes_diverse}/3'

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    board_w = int(input(f'Board inner corners width  [{BOARD_W}]: ')  or BOARD_W)
    board_h = int(input(f'Board inner corners height [{BOARD_H}]: ') or BOARD_H)
    sq_mm   = float(input(f'Square size mm [{SQUARE_MM}]: ')          or SQUARE_MM)

    board_size = (board_w, board_h)
    sq_m       = sq_mm / 1000.0
    objp       = np.zeros((board_w * board_h, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_w, 0:board_h].T.reshape(-1, 2) * sq_m
    criteria   = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    if not os.path.exists(INTR_FILE):
        print(f'ERROR: intrinsics not found at {INTR_FILE}')
        sys.exit(1)
    with open(INTR_FILE) as f:
        intr = json.load(f)
    K    = np.array(intr['rgb_K'],   dtype=np.float64)
    dist = np.array(intr['rgb_dist'], dtype=np.float64)
    print(f'Loaded intrinsics  (reproj err={intr.get("reprojection_error","?")})')

    rclpy.init()
    node = CalibNode()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    print('\nWaiting for /camera/color/image_raw ...')
    while node.get_frame()[0] is None:
        time.sleep(0.05)
    print('Stream OK.  Waiting for TF ...')
    while node.get_eef_pose() is None:
        time.sleep(0.1)
    print('TF OK.\n')

    npz_path  = os.path.join(DATA_DIR, 'calibration_data.npz')
    json_path = os.path.join(DATA_DIR, 'poses.json')

    R_g2b, t_g2b = [], []
    R_t2c, t_t2c = [], []
    raw_data     = []

    if os.path.exists(npz_path):
        existing = np.load(npz_path, allow_pickle=False)
        R_g2b = [R for R in existing['R_g2b']]
        t_g2b = [t for t in existing['t_g2b']]
        R_t2c = [R for R in existing['R_t2c']]
        t_t2c = [t for t in existing['t_t2c']]
        print(f'Loaded {len(R_g2b)} existing poses from {npz_path}')
    else:
        print('No existing dataset found — starting fresh.')

    if os.path.exists(json_path):
        with open(json_path) as f:
            raw_data = json.load(f)

    existing_imgs = [
        int(os.path.splitext(f)[0].split('_')[1])
        for f in os.listdir(DATA_DIR)
        if f.startswith('capture_') and f.endswith('.jpg')
    ]
    _img_counter = [max(existing_imgs) + 1 if existing_imgs else 0]

    print(f'SPACE=capture  F=toggle freedrive  C=compute  Q=quit')
    print(f'Need >= {MIN_CAPTURES} diverse poses.')
    print(f'Vary ROLL and PITCH too — yaw-only clusters will be rejected.\n')

    freedrive = False

    while True:
        frame, stamp = node.get_frame()
        if frame is None:
            continue

        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, board_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK)
        if found:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        display = cv2.undistort(frame, K, dist)
        if found:
            cv2.drawChessboardCorners(display, board_size, corners, found)

        n = len(R_g2b)
        cv2.putText(display,
                    f'Poses: {n} | {"BOARD OK" if found else "no board"} | SPACE=cap F=freedrive C=go Q=quit',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if found else (0, 100, 255), 2)
        if freedrive:
            cv2.putText(display, 'FREEDRIVE ON — robot compliant',
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 200, 255), 2)
        cv2.imshow('Eye-in-Hand Calibration', display)
        key = cv2.waitKey(30) & 0xFF

        if key == ord(' ') and found:
            ok, msg = check_robot_still(node)
            if not ok:
                print(f'  REJECTED: {msg} — hold still and retry')
                continue
            frame2, stamp2 = node.get_frame()
            if frame2 is None or stamp2 is None:
                print('  REJECTED: no frame')
                continue
            now_sec   = node.get_clock().now().nanoseconds * 1e-9
            frame_sec = _stamp_to_sec(stamp2)
            age = now_sec - frame_sec
            if age > FRAME_MAX_AGE_SEC:
                print(f'  REJECTED: frame age {age*1000:.0f} ms > {FRAME_MAX_AGE_SEC*1000:.0f} ms')
                continue
            eef = node.get_eef_pose(stamp=stamp2)
            if eef is None:
                print('  REJECTED: TF lookup at frame stamp failed')
                continue
            R_robot, t_robot = eef
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            found2, corners2 = cv2.findChessboardCorners(
                gray2, board_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK)
            if not found2:
                print('  REJECTED: board lost after still check')
                continue
            corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
            ok_pnp, rvec, tvec = cv2.solvePnP(objp, corners2, K, dist)
            if not ok_pnp:
                print('  REJECTED: solvePnP failed')
                continue
            R_board = _rvec_to_rotmat(rvec)
            ok_div, note_div = check_rotation_diversity(R_robot, R_g2b)
            if not ok_div:
                print(f'  REJECTED: {note_div}')
                continue
            roll, pitch, yaw = _rotmat_to_euler_deg(R_robot)
            R_g2b.append(R_robot)
            t_g2b.append(t_robot)
            R_t2c.append(R_board)
            t_t2c.append(tvec)

            img_idx  = _img_counter[0]
            _img_counter[0] += 1
            img_file = f'capture_{img_idx:03d}.jpg'
            cv2.imwrite(os.path.join(DATA_DIR, img_file), frame2)

            raw_data.append({
                'id':              img_idx,
                'image_file':      img_file,
                'frame_stamp_sec': frame_sec,
                'R_gripper2base':  R_robot.tolist(),
                't_gripper2base':  t_robot.ravel().tolist(),
                'rpy_deg':         [roll, pitch, yaw],
                'rvec_target2cam': rvec.ravel().tolist(),
                'tvec_target2cam': tvec.ravel().tolist(),
            })
            print(f'  OK img={img_idx:03d} total={len(R_g2b):02d}  '
                  f't=[{",".join(f"{v:.3f}" for v in t_robot.ravel())}]  '
                  f'rpy=[{roll:.1f},{pitch:.1f},{yaw:.1f}]  '
                  f'age={age*1000:.0f}ms  {note_div}')

        elif key in (ord('f'), ord('F')):
            freedrive = not freedrive
            if freedrive:
                node.enter_freedrive()
                print('FREEDRIVE ON — move robot by hand. Press F again to exit.')
            else:
                node.exit_freedrive()
                print('FREEDRIVE OFF — dashboard /play sent to restart external_control.')

        elif key in (ord('c'), ord('C')):
            if freedrive:
                node.exit_freedrive()
                freedrive = False
            break
        elif key in (ord('q'), ord('Q')):
            if freedrive:
                node.exit_freedrive()
            cv2.destroyAllWindows()
            node.destroy_node()
            rclpy.shutdown()
            return

    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

    n = len(R_g2b)
    if n < 4:
        print(f'Only {n} total captures — need at least 4.')
        return

    np.savez(npz_path,
             R_g2b=np.array(R_g2b), t_g2b=np.array(t_g2b),
             R_t2c=np.array(R_t2c), t_t2c=np.array(t_t2c),
             K=K, dist=dist)
    with open(json_path, 'w') as f:
        json.dump(raw_data, f, indent=2)
    print(f'\nDataset saved to {DATA_DIR}')

    SOLVERS = {
        'TSAI':      cv2.CALIB_HAND_EYE_TSAI,
        'PARK':      cv2.CALIB_HAND_EYE_PARK,
        'HORAUD':    cv2.CALIB_HAND_EYE_HORAUD,
        'ANDREFF':   cv2.CALIB_HAND_EYE_ANDREFF,
        'DANIILIDIS': cv2.CALIB_HAND_EYE_DANIILIDIS,
    }

    rot_dists = []
    for i in range(len(R_g2b)):
        for j in range(i + 1, len(R_g2b)):
            rot_dists.append(_angle_between_rotmats(R_g2b[i], R_g2b[j]))
    print(f'\nRotation diversity: min={min(rot_dists):.1f}°  '
          f'mean={np.mean(rot_dists):.1f}°  max={max(rot_dists):.1f}°')
    if max(rot_dists) < 60.0:
        print('WARNING: max rotation spread < 60° — recapture with more tilt.')

    print(f'\nRunning {len(SOLVERS)} solvers on {n} pose pairs...\n')
    results = {}
    for name, method in SOLVERS.items():
        try:
            R_c2g, t_c2g = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=method)
            err = _reprojection_error(R_c2g, t_c2g, R_g2b, t_g2b, R_t2c, t_t2c, K, dist)
            results[name] = (R_c2g, t_c2g, err)
            print(f'  {name:<12}  consistency = {err:.2f}  t = {t_c2g.ravel().round(4)}')
        except Exception as e:
            print(f'  {name:<12}  FAILED: {e}')

    if not results:
        print('All solvers failed.')
        return

    best_name = min(results, key=lambda k: results[k][2])
    R_best, t_best, err_best = results[best_name]
    print(f'\nBest solver: {best_name}  (consistency={err_best:.2f})')

    T_c2g = np.eye(4)
    T_c2g[:3, :3] = R_best
    T_c2g[:3, 3]  = t_best.ravel()
    rvec_out, _ = cv2.Rodrigues(R_best)

    data = {
        'solver':                best_name,
        'consistency':           float(err_best),
        'n_poses':               n,
        'T_cam_to_tool0':        T_c2g.tolist(),
        'R_cam_to_tool0':        R_best.tolist(),
        't_cam_to_tool0_m':      t_best.ravel().tolist(),
        'rodrigues_cam_to_tool0': rvec_out.ravel().tolist(),
        'all_solver_consistency': {k: v[2] for k, v in results.items()},
    }
    with open(CALIB_FILE, 'w') as f:
        json.dump(data, f, indent=2)

    print(f'\nT_cam_to_tool0:\n{T_c2g}')
    print(f'Translation (m): {t_best.ravel()}')
    print(f'\nSaved -> {CALIB_FILE}')


def _reprojection_error(R_c2g, t_c2g, R_g2b_list, t_g2b_list, R_t2c_list, t_t2c_list, K, dist):
    T_c2g = np.eye(4)
    T_c2g[:3, :3] = R_c2g
    T_c2g[:3, 3]  = t_c2g.ravel()

    errors = []
    for i in range(len(R_g2b_list)):
        T_g2b = np.eye(4); T_g2b[:3, :3] = R_g2b_list[i]; T_g2b[:3, 3] = t_g2b_list[i].ravel()
        T_t2c = np.eye(4); T_t2c[:3, :3] = R_t2c_list[i]; T_t2c[:3, 3] = t_t2c_list[i].ravel()
        T_pred = T_g2b @ T_c2g @ T_t2c
        for j in range(i + 1, len(R_g2b_list)):
            T_g2b_j = np.eye(4); T_g2b_j[:3, :3] = R_g2b_list[j]; T_g2b_j[:3, 3] = t_g2b_list[j].ravel()
            T_t2c_j = np.eye(4); T_t2c_j[:3, :3] = R_t2c_list[j]; T_t2c_j[:3, 3] = t_t2c_list[j].ravel()
            T_pred_j = T_g2b_j @ T_c2g @ T_t2c_j
            T_rel = np.linalg.inv(T_pred) @ T_pred_j
            t_err = np.linalg.norm(T_rel[:3, 3]) * 1000
            R_err = np.degrees(np.arccos(np.clip((np.trace(T_rel[:3, :3]) - 1) / 2, -1, 1)))
            errors.append(t_err + R_err)
    return float(np.mean(errors)) if errors else 9999.0


if __name__ == '__main__':
    main()

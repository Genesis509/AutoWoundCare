"""
tune_eye_in_hand.py  —  Live GUI to tune eye-in-hand calibration.

Sliders for translation (mm) and rotation (deg) of T_cam_to_tool0.
Publishes tool0 → camera_color_optical_frame TF in real-time so you can
watch RViz respond as you drag.  Save button writes back to eye_in_hand.json.

Run:
    ros2 run wound_tracking tune_eye_in_hand
or:
    python3 src/wound_tracking/wound_tracking/tune_eye_in_hand.py
"""

import json
import math
import os
import threading
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster

CALIB_FILE = os.path.expanduser(
    '~/Vision_Guided_Autonomous_Wound_Treatment_System/calibration/eye_in_hand.json')

PARENT_FRAME = 'tool0'
CHILD_FRAME  = 'camera_color_optical_frame'

def euler_to_rotmat(roll_deg, pitch_deg, yaw_deg):
    """ZYX Euler angles (deg) → 3×3 rotation matrix R_cam_to_tool0."""
    r = math.radians(roll_deg)
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)

    Rz = np.array([[ math.cos(y), -math.sin(y), 0],
                   [ math.sin(y),  math.cos(y), 0],
                   [ 0,            0,            1]])
    Ry = np.array([[ math.cos(p), 0, math.sin(p)],
                   [ 0,           1, 0           ],
                   [-math.sin(p), 0, math.cos(p)]])
    Rx = np.array([[1, 0,           0           ],
                   [0, math.cos(r), -math.sin(r)],
                   [0, math.sin(r),  math.cos(r)]])
    return Rz @ Ry @ Rx


def rotmat_to_euler(R):
    """3×3 rotation matrix → (roll, pitch, yaw) degrees, ZYX convention."""
    sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)
    if sy > 1e-6:
        roll  = math.degrees(math.atan2( R[2, 1], R[2, 2]))
        pitch = math.degrees(math.atan2(-R[2, 0], sy))
        yaw   = math.degrees(math.atan2( R[1, 0], R[0, 0]))
    else:
        roll  = math.degrees(math.atan2(-R[1, 2], R[1, 1]))
        pitch = math.degrees(math.atan2(-R[2, 0], sy))
        yaw   = 0.0
    return roll, pitch, yaw


def rotmat_to_quat(R):
    """3×3 rotation matrix → (x, y, z, w) quaternion."""
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = 0.5 / math.sqrt(t + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s;  x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s;  z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s;  x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s;                  z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s;  x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s;  z = 0.25 * s
    n = math.sqrt(x*x + y*y + z*z + w*w)
    return x/n, y/n, z/n, w/n

class TfPublisherNode(Node):
    """Publishes a single static TF; call update() to refresh."""

    def __init__(self):
        super().__init__('tune_eye_in_hand')
        self._br = StaticTransformBroadcaster(self)
        self._lock = threading.Lock()
        self._ts = TransformStamped()
        self._ts.header.frame_id = PARENT_FRAME
        self._ts.child_frame_id  = CHILD_FRAME

    def update(self, tx_m, ty_m, tz_m, roll_deg, pitch_deg, yaw_deg):
        """Broadcast tool0→cam TF. T_cam_to_tool0 translation column =
        cam origin in tool0 = exactly what TF parent=tool0, child=cam stores."""
        R_c2t = euler_to_rotmat(roll_deg, pitch_deg, yaw_deg)
        t_c2t = np.array([tx_m, ty_m, tz_m])

        qx, qy, qz, qw = rotmat_to_quat(R_c2t)

        with self._lock:
            self._ts.header.stamp        = self.get_clock().now().to_msg()
            self._ts.transform.translation.x = float(t_c2t[0])
            self._ts.transform.translation.y = float(t_c2t[1])
            self._ts.transform.translation.z = float(t_c2t[2])
            self._ts.transform.rotation.x    = qx
            self._ts.transform.rotation.y    = qy
            self._ts.transform.rotation.z    = qz
            self._ts.transform.rotation.w    = qw
            self._br.sendTransform(self._ts)

class TunerGUI:

    # Slider ranges
    T_MIN_MM, T_MAX_MM = -300, 300   # translation mm
    R_MIN_DEG, R_MAX_DEG = -180, 180 # rotation deg

    def __init__(self, root: tk.Tk, node: TfPublisherNode):
        self.root = root
        self.node = node
        root.title('Eye-in-Hand Tuner  —  tool0 → camera_color_optical_frame')
        root.resizable(False, False)

        # Load initial values from file
        tx, ty, tz, roll, pitch, yaw = self._load_from_file()
        self.var_tx    = tk.DoubleVar(value=tx * 1000)     # m → mm
        self.var_ty    = tk.DoubleVar(value=ty * 1000)
        self.var_tz    = tk.DoubleVar(value=tz * 1000)
        self.var_roll  = tk.DoubleVar(value=roll)
        self.var_pitch = tk.DoubleVar(value=pitch)
        self.var_yaw   = tk.DoubleVar(value=yaw)
        pad = dict(padx=8, pady=4)

        # Title
        tk.Label(root, text='T_cam_to_tool0  (drag = live TF update)',
                 font=('Helvetica', 12, 'bold')).grid(
                     row=0, column=0, columnspan=3, pady=(10, 4))

        # Section headers
        tk.Label(root, text='Translation (mm)', font=('Helvetica', 10, 'bold'),
                 fg='#004488').grid(row=1, column=0, columnspan=3, **pad)
        self._make_slider(root, 'tx  (X, left−)', self.var_tx,
                          self.T_MIN_MM, self.T_MAX_MM, 2)
        self._make_slider(root, 'ty  (Y, front+)', self.var_ty,
                          self.T_MIN_MM, self.T_MAX_MM, 3)
        self._make_slider(root, 'tz  (Z, down+)', self.var_tz,
                          self.T_MIN_MM, self.T_MAX_MM, 4)

        tk.Label(root, text='Rotation (deg, ZYX Euler)', font=('Helvetica', 10, 'bold'),
                 fg='#004488').grid(row=5, column=0, columnspan=3, **pad)
        self._make_slider(root, 'roll  (X-axis)', self.var_roll,
                          self.R_MIN_DEG, self.R_MAX_DEG, 6)
        self._make_slider(root, 'pitch (Y-axis)', self.var_pitch,
                          self.R_MIN_DEG, self.R_MAX_DEG, 7)
        self._make_slider(root, 'yaw   (Z-axis)', self.var_yaw,
                          self.R_MIN_DEG, self.R_MAX_DEG, 8)

        # Status bar
        self.status_var = tk.StringVar(value='Loaded from file.')
        tk.Label(root, textvariable=self.status_var, fg='#336600',
                 font=('Courier', 9)).grid(row=9, column=0, columnspan=3, **pad)

        # Buttons
        btn_frame = tk.Frame(root)
        btn_frame.grid(row=10, column=0, columnspan=3, pady=(6, 10))
        tk.Button(btn_frame, text='💾  Save to JSON', width=18,
                  bg='#2a7a2a', fg='white', font=('Helvetica', 10, 'bold'),
                  command=self._save).pack(side=tk.LEFT, padx=6)
        tk.Button(btn_frame, text='↺  Reload from JSON', width=18,
                  command=self._reload).pack(side=tk.LEFT, padx=6)
        tk.Button(btn_frame, text='⬜  Reset to Identity', width=18,
                  command=self._reset).pack(side=tk.LEFT, padx=6)

        # Trace all vars → live publish
        for v in (self.var_tx, self.var_ty, self.var_tz,
                  self.var_roll, self.var_pitch, self.var_yaw):
            v.trace_add('write', self._on_change)

        # Initial publish
        self._publish()

    def _make_slider(self, parent, label, var, lo, hi, row):
        tk.Label(parent, text=label, width=18, anchor='e').grid(
            row=row, column=0, padx=(8, 2), pady=3)
        s = ttk.Scale(parent, from_=lo, to=hi, orient='horizontal',
                      variable=var, length=400)
        s.grid(row=row, column=1, padx=4, pady=3)
        # Numeric entry
        e = tk.Entry(parent, textvariable=var, width=9)
        e.grid(row=row, column=2, padx=(2, 8), pady=3)
        e.bind('<Return>', self._on_change)
        e.bind('<FocusOut>', self._on_change)

    def _on_change(self, *_):
        try:
            self._publish()
        except Exception:
            pass

    def _publish(self):
        tx = self.var_tx.get() / 1000.0   # mm → m
        ty = self.var_ty.get() / 1000.0
        tz = self.var_tz.get() / 1000.0
        roll  = self.var_roll.get()
        pitch = self.var_pitch.get()
        yaw   = self.var_yaw.get()
        self.node.update(tx, ty, tz, roll, pitch, yaw)
        self.status_var.set(
            f't=[{tx*1000:+.1f}, {ty*1000:+.1f}, {tz*1000:+.1f}] mm  '
            f'rpy=[{roll:+.1f}, {pitch:+.1f}, {yaw:+.1f}] deg'
        )

    def _load_from_file(self):
        """Return (tx_m, ty_m, tz_m, roll_deg, pitch_deg, yaw_deg)."""
        if not os.path.isfile(CALIB_FILE):
            return 0.0, 0.0, 0.0, 0.0, 0.0, 180.0  # default: R_z180
        with open(CALIB_FILE) as f:
            cal = json.load(f)
        T = np.array(cal['T_cam_to_tool0'])
        R = T[:3, :3]
        t = T[:3, 3]
        roll, pitch, yaw = rotmat_to_euler(R)
        return float(t[0]), float(t[1]), float(t[2]), roll, pitch, yaw

    def _save(self):
        tx = self.var_tx.get() / 1000.0
        ty = self.var_ty.get() / 1000.0
        tz = self.var_tz.get() / 1000.0
        roll  = self.var_roll.get()
        pitch = self.var_pitch.get()
        yaw   = self.var_yaw.get()

        R = euler_to_rotmat(roll, pitch, yaw)
        t = np.array([tx, ty, tz])

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3]  = t

        import cv2
        rvec, _ = cv2.Rodrigues(R)

        # Load existing file to preserve metadata, then overwrite geometry
        if os.path.isfile(CALIB_FILE):
            with open(CALIB_FILE) as f:
                data = json.load(f)
        else:
            data = {}

        data['solver']              = 'manual_tune_gui'
        data['T_cam_to_tool0']      = T.tolist()
        data['R_cam_to_tool0']      = R.tolist()
        data['t_cam_to_tool0_m']    = t.tolist()
        data['rodrigues_cam_to_tool0'] = rvec.ravel().tolist()
        data['note'] = (
            f'Tuned via tune_eye_in_hand GUI. '
            f't=[{tx*1000:.3f}, {ty*1000:.3f}, {tz*1000:.3f}] mm  '
            f'rpy=[{roll:.3f}, {pitch:.3f}, {yaw:.3f}] deg (ZYX)'
        )

        with open(CALIB_FILE, 'w') as f:
            json.dump(data, f, indent=2)

        self.status_var.set(f'Saved to {CALIB_FILE}')
        messagebox.showinfo('Saved', f'Written to:\n{CALIB_FILE}')

    def _reload(self):
        tx, ty, tz, roll, pitch, yaw = self._load_from_file()
        self.var_tx.set(tx * 1000)
        self.var_ty.set(ty * 1000)
        self.var_tz.set(tz * 1000)
        self.var_roll.set(roll)
        self.var_pitch.set(pitch)
        self.var_yaw.set(yaw)
        self.status_var.set('Reloaded from file.')

    def _reset(self):
        self.var_tx.set(0.0)
        self.var_ty.set(0.0)
        self.var_tz.set(0.0)
        self.var_roll.set(0.0)
        self.var_pitch.set(0.0)
        self.var_yaw.set(180.0)   # R_z180 default
        self.status_var.set('Reset to R_z180 default.')

def main(args=None):
    rclpy.init(args=args)
    node = TfPublisherNode()

    # ROS2 spin in background thread
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # Tkinter GUI in main thread
    root = tk.Tk()
    TunerGUI(root, node)
    try:
        root.mainloop()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

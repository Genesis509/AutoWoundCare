#!/usr/bin/env python3
"""Offline eye-in-hand re-solve + diagnostics on saved dataset.

Reads calibration/eye_in_hand_dataset/calibration_data.npz (produced by
calibrate_eye_in_hand.py) and reports:
  - Dataset diversity (rotation/translation spread, board pose spread)
  - All 5 OpenCV solver solutions
  - Proper AX=XB residual (rotation deg, translation mm) per solver
  - Reprojection on checkerboard corners if images still exist

Usage: /usr/bin/python3 src/wound_tracking/scripts/offline_eye_in_hand.py
"""
import json
import os
import sys

import cv2
import numpy as np

BASE = os.path.expanduser('~/Vision_Guided_Autonomous_Wound_Treatment_System/calibration')
NPZ  = os.path.join(BASE, 'eye_in_hand_dataset', 'calibration_data.npz')
POSES_JSON = os.path.join(BASE, 'eye_in_hand_dataset', 'poses.json')

SOLVERS = {
    'TSAI':       cv2.CALIB_HAND_EYE_TSAI,
    'PARK':       cv2.CALIB_HAND_EYE_PARK,
    'HORAUD':     cv2.CALIB_HAND_EYE_HORAUD,
    'ANDREFF':    cv2.CALIB_HAND_EYE_ANDREFF,
    'DANIILIDIS': cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def angle_between(R1, R2):
    R = R1.T @ R2
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(c))


def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).ravel()
    return T


def axxb_residual(R_c2g, t_c2g, R_g2b, t_g2b, R_t2c, t_t2c):
    """For each consecutive pair: A X = X B where
       A = T_g2b_i^-1 @ T_g2b_j  (gripper motion in base)
       B = T_t2c_i @ T_t2c_j^-1  (target motion in cam)
       X = T_c2g.
       Returns mean/median/max rotation and translation residuals."""
    X = make_T(R_c2g, t_c2g)
    rot_err, tr_err = [], []
    n = len(R_g2b)
    for i in range(n):
        Ti = make_T(R_g2b[i], t_g2b[i])
        Tti = make_T(R_t2c[i], t_t2c[i])
        for j in range(i + 1, n):
            Tj = make_T(R_g2b[j], t_g2b[j])
            Ttj = make_T(R_t2c[j], t_t2c[j])
            A = np.linalg.inv(Ti) @ Tj
            B = Tti @ np.linalg.inv(Ttj)
            D = np.linalg.inv(A @ X) @ (X @ B)
            rot_err.append(angle_between(D[:3, :3], np.eye(3)))
            tr_err.append(np.linalg.norm(D[:3, 3]) * 1000.0)
    return {
        'rot_mean_deg': float(np.mean(rot_err)),
        'rot_med_deg':  float(np.median(rot_err)),
        'rot_max_deg':  float(np.max(rot_err)),
        'tr_mean_mm':   float(np.mean(tr_err)),
        'tr_med_mm':    float(np.median(tr_err)),
        'tr_max_mm':    float(np.max(tr_err)),
    }


def board_in_base_spread(R_c2g, t_c2g, R_g2b, t_g2b, R_t2c, t_t2c):
    """If calibration is correct, board-in-base should be constant (board
    doesn't move). Spread reveals calibration quality directly."""
    X = make_T(R_c2g, t_c2g)
    origins = []
    for i in range(len(R_g2b)):
        T_g2b = make_T(R_g2b[i], t_g2b[i])
        T_t2c = make_T(R_t2c[i], t_t2c[i])
        T_t2b = T_g2b @ X @ T_t2c     # board in base
        origins.append(T_t2b[:3, 3])
    origins = np.array(origins)
    std = origins.std(axis=0) * 1000.0
    return {
        'board_origin_std_mm': std.tolist(),
        'board_origin_mean_m': origins.mean(axis=0).tolist(),
        'board_origin_spread_mm': float(np.linalg.norm(origins.max(0) - origins.min(0)) * 1000.0),
    }


def main():
    if not os.path.isfile(NPZ):
        print(f'NO DATASET at {NPZ}')
        sys.exit(1)
    d = np.load(NPZ)
    R_g2b = [d['R_g2b'][i] for i in range(len(d['R_g2b']))]
    t_g2b = [d['t_g2b'][i] for i in range(len(d['t_g2b']))]
    R_t2c = [d['R_t2c'][i] for i in range(len(d['R_t2c']))]
    t_t2c = [d['t_t2c'][i] for i in range(len(d['t_t2c']))]
    K     = d['K']
    dist  = d['dist']
    n = len(R_g2b)
    print(f'Dataset: {n} pose pairs')
    print(f'K focal ≈ {K[0,0]:.1f}, {K[1,1]:.1f}  center {K[0,2]:.1f},{K[1,2]:.1f}')
    print(f'dist = {dist.ravel()}')
    print()
    print('── GRIPPER ROTATION SPREAD (base→tool0) ──────────────────')
    pair_rots = [angle_between(R_g2b[i], R_g2b[j])
                 for i in range(n) for j in range(i + 1, n)]
    print(f'  pairwise angle  min={min(pair_rots):6.1f}°  '
          f'mean={np.mean(pair_rots):6.1f}°  max={max(pair_rots):6.1f}°')

    print('── GRIPPER TRANSLATION SPREAD ─────────────────────────────')
    ts = np.array([t.ravel() for t in t_g2b])
    print(f'  t_g2b std_xyz (mm) = {(ts.std(0) * 1000).round(1)}')
    print(f'  t_g2b spread (mm)  = {(np.linalg.norm(ts.max(0) - ts.min(0)) * 1000):.1f}')

    print('── BOARD-IN-CAM DISTANCE / POSE ──────────────────────────')
    tc = np.array([t.ravel() for t in t_t2c])
    print(f'  z_cam (board depth) range: {tc[:,2].min():.3f} .. {tc[:,2].max():.3f} m  '
          f'mean {tc[:,2].mean():.3f}')
    print(f'  (x,y)_cam offset range: '
          f'x {tc[:,0].min():+.3f}..{tc[:,0].max():+.3f}   '
          f'y {tc[:,1].min():+.3f}..{tc[:,1].max():+.3f}')
    pair_t_rot = [angle_between(R_t2c[i], R_t2c[j])
                  for i in range(n) for j in range(i + 1, n)]
    print(f'  board-in-cam rotation pairwise  min={min(pair_t_rot):6.1f}°  '
          f'mean={np.mean(pair_t_rot):6.1f}°  max={max(pair_t_rot):6.1f}°')
    print()
    results = {}
    print('── SOLVERS ────────────────────────────────────────────────')
    print(f'{"name":<12}{"tx (mm)":>9}{"ty (mm)":>9}{"tz (mm)":>9}'
          f'{"roll°":>8}{"pitch°":>8}{"yaw°":>8}'
          f'{"rot_med°":>10}{"tr_med mm":>11}{"rot_max°":>10}{"tr_max mm":>11}')
    for name, m in SOLVERS.items():
        try:
            R_c2g, t_c2g = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=m)
        except Exception as e:
            print(f'  {name:<12} FAILED: {e}')
            continue
        res = axxb_residual(R_c2g, t_c2g, R_g2b, t_g2b, R_t2c, t_t2c)
        spread = board_in_base_spread(R_c2g, t_c2g, R_g2b, t_g2b, R_t2c, t_t2c)
        # euler from R
        sy = np.sqrt(R_c2g[0, 0] ** 2 + R_c2g[1, 0] ** 2)
        roll  = np.degrees(np.arctan2(R_c2g[2, 1], R_c2g[2, 2]))
        pitch = np.degrees(np.arctan2(-R_c2g[2, 0], sy))
        yaw   = np.degrees(np.arctan2(R_c2g[1, 0], R_c2g[0, 0]))
        tx, ty, tz = t_c2g.ravel() * 1000
        print(f'{name:<12}{tx:9.1f}{ty:9.1f}{tz:9.1f}'
              f'{roll:8.1f}{pitch:8.1f}{yaw:8.1f}'
              f'{res["rot_med_deg"]:10.2f}{res["tr_med_mm"]:11.1f}'
              f'{res["rot_max_deg"]:10.2f}{res["tr_max_mm"]:11.1f}')
        results[name] = {'R': R_c2g, 't': t_c2g, 'res': res, 'spread': spread}
    print()
    if not results:
        print('ALL SOLVERS FAILED.')
        return
    print('── BOARD-IN-BASE STATIONARITY  (should be ~0) ────────────')
    for name, r in results.items():
        std = r['spread']['board_origin_std_mm']
        sp  = r['spread']['board_origin_spread_mm']
        print(f'  {name:<12}  std=[{std[0]:6.1f},{std[1]:6.1f},{std[2]:6.1f}] mm  '
              f'total spread={sp:6.1f} mm')

    best = min(results, key=lambda k: results[k]['spread']['board_origin_spread_mm'])
    print(f'\nBest by board stationarity: {best}')
    r = results[best]
    print(f'  t_cam_to_tool0 (mm) = {(r["t"].ravel() * 1000).round(2)}')
    print(f'  R_cam_to_tool0 =\n{np.round(r["R"], 4)}')


if __name__ == '__main__':
    main()
